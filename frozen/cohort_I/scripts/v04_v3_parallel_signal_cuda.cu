#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int VOCAB=151936, CAPACITY=5, PARTS=8, THREADS=256;
__device__ __forceinline__ uint64_t now_ns(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;}
__device__ __forceinline__ void insert(float v,float& first,float& second){
  if(isnan(v)||isnan(first)){first=NAN;second=NAN;return;}
  if(v>=first){second=first;first=v;}else if(v>second)second=v;
}
__device__ __forceinline__ bool terminal(const int64_t* meta,const int64_t* eos,int count){
  bool done=meta[0]==meta[1];for(int i=0;i<count;i++)done=done||meta[2]==eos[i];return done;
}
__global__ void begin_step(const int64_t* token,int64_t* meta,int64_t* starts,int n,int limit){
  starts[n-1]=(int64_t)now_ns();meta[0]=n;meta[1]=limit;meta[2]=token[0];
}
__global__ void copy_window(const __half* x,__half* window,int row,bool vectorized){
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(vectorized){if(j<VOCAB/8)reinterpret_cast<int4*>(window+row*VOCAB)[j]=reinterpret_cast<const int4*>(x)[j];}
  else if(j<VOCAB)window[row*VOCAB+j]=x[j];
}
__global__ void partial_top2(const __half* x,const __half* window,float* partials,
    const int64_t* meta,const int64_t* eos,int eos_count,bool batch){
  int n=(int)meta[0];
  if(batch&&!terminal(meta,eos,eos_count))return;
  int row=batch?blockIdx.x/PARTS:n-1;int part=blockIdx.x%PARTS;
  if(row>=n)return;
  const __half* values=batch?window+row*VOCAB:x;
  constexpr int CHUNK=(VOCAB+PARTS-1)/PARTS;
  int end=min(VOCAB,(part+1)*CHUNK);float first=-INFINITY,second=-INFINITY;
  for(int j=part*CHUNK+threadIdx.x;j<end;j+=blockDim.x)insert(__half2float(values[j]),first,second);
  __shared__ float a[THREADS],b[THREADS];a[threadIdx.x]=first;b[threadIdx.x]=second;__syncthreads();
  for(int stride=THREADS/2;stride;stride>>=1){
    if(threadIdx.x<stride){first=a[threadIdx.x];second=b[threadIdx.x];insert(a[threadIdx.x+stride],first,second);insert(b[threadIdx.x+stride],first,second);a[threadIdx.x]=first;b[threadIdx.x]=second;}
    __syncthreads();
  }
  if(threadIdx.x==0){int offset=(row*PARTS+part)*2;partials[offset]=a[0];partials[offset+1]=b[0];}
}
__device__ __forceinline__ void combine(const float* partials,int row,float& first,float& second){
  first=-INFINITY;second=-INFINITY;
  for(int i=0;i<PARTS*2;i++)insert(partials[row*PARTS*2+i],first,second);
}
__global__ void finish_step(const float* partials,__half* pairs,const int64_t* meta,const int64_t* eos,
    int eos_count,float* host_features,int64_t* packet,bool batch){
  int n=(int)meta[0];bool done=terminal(meta,eos,eos_count);float first,second;
  if(!batch){combine(partials,n-1,first,second);pairs[(n-1)*2]=__float2half(first);pairs[(n-1)*2+1]=__float2half(second);}
  if(done)for(int row=0;row<n;row++){
    if(batch)combine(partials,row,first,second);
    else{first=__half2float(pairs[row*2]);second=__half2float(pairs[row*2+1]);}
    float gap=first-second;host_features[row*2]=gap;host_features[row*2+1]=expf((-gap)*(1.0f/3.0f));
  }
  __threadfence_system();  // GPU memory ordering, not a host/device synchronization.
  packet[0]=(int64_t)(((now_ns()&((1ULL<<44)-1))<<18)|((uint64_t)meta[2]&((1ULL<<18)-1)));
}
void dispatch(torch::Tensor x,torch::Tensor token,torch::Tensor window,torch::Tensor pairs,
    torch::Tensor partials,torch::Tensor meta,torch::Tensor eos,torch::Tensor packet,
    int64_t mapped_features,int64_t mapped_starts,int64_t n,int64_t limit,bool batch){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat16&&x.numel()==VOCAB&&x.stride(-1)==1,"Expected last FP16 vocabulary row");
  TORCH_CHECK(n>=1&&n<=limit&&limit<=CAPACITY,"K5 window exceeded");
  TORCH_CHECK(partials.is_cuda()&&partials.scalar_type()==torch::kFloat32&&partials.numel()==CAPACITY*PARTS*2,"Invalid bounded partial buffer");
  auto stream=at::cuda::getCurrentCUDAStream();const __half* input=reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  __half* buffer=reinterpret_cast<__half*>(window.data_ptr<at::Half>());
  begin_step<<<1,1,0,stream>>>(token.data_ptr<int64_t>(),meta.data_ptr<int64_t>(),reinterpret_cast<int64_t*>(mapped_starts),n,limit);C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(batch){
    bool vectorized=((reinterpret_cast<uintptr_t>(input)|reinterpret_cast<uintptr_t>(buffer))&15)==0;
    int elements=vectorized?VOCAB/8:VOCAB;
    copy_window<<<(elements+THREADS-1)/THREADS,THREADS,0,stream>>>(input,buffer,n-1,vectorized);C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  partial_top2<<<batch?CAPACITY*PARTS:PARTS,THREADS,0,stream>>>(input,buffer,partials.data_ptr<float>(),meta.data_ptr<int64_t>(),eos.data_ptr<int64_t>(),eos.numel(),batch);C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish_step<<<1,1,0,stream>>>(partials.data_ptr<float>(),reinterpret_cast<__half*>(pairs.data_ptr<at::Half>()),meta.data_ptr<int64_t>(),eos.data_ptr<int64_t>(),eos.numel(),reinterpret_cast<float*>(mapped_features),packet.data_ptr<int64_t>(),batch);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void stamp_kernel(int64_t* out){out[0]=(int64_t)now_ns();}
void stamp(torch::Tensor out){stamp_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
int64_t mapped(torch::Tensor host){TORCH_CHECK(host.is_pinned(),"Mapped mailbox must be pinned");void* device=nullptr;C10_CUDA_CHECK(cudaHostGetDevicePointer(&device,host.data_ptr(),0));return reinterpret_cast<int64_t>(device);}
