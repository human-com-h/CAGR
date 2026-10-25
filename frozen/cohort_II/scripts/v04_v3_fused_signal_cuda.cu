#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int VOCAB=151936,CAPACITY=5,PARTS=8,THREADS=256,CHUNK=VOCAB/PARTS;
static_assert(VOCAB%PARTS==0 && CHUNK%8==0,"Exact aligned partition geometry required");
__device__ __forceinline__ uint64_t now_ns(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;}
__device__ __forceinline__ void insert(float v,float& first,float& second){
  if(isnan(v)||isnan(first)){first=NAN;second=NAN;return;}
  if(v>=first){second=first;first=v;}else if(v>second)second=v;
}
__device__ __forceinline__ bool terminal(int n,int limit,int64_t token,const int64_t* eos,int count){
  bool done=n==limit;for(int i=0;i<count;i++)done=done||token==eos[i];return done;
}
// Each block stamps its OWN beginning. The following stream-ordered finish
// kernel uses their minimum, so no block-0 scheduling assumption can hide work.
__global__ void fused_work(const __half* x,__half* window,float* partials,int64_t* starts,
    const int64_t* token,const int64_t* eos,int eos_count,int n,int limit,bool batch,bool vectorized){
  if(threadIdx.x==0)starts[blockIdx.x]=(int64_t)now_ns();
  int row=batch?blockIdx.x/PARTS:n-1;int part=blockIdx.x%PARTS;
  if(row>=n)return;
  bool done=terminal(n,limit,token[0],eos,eos_count);
  if(batch && row==n-1){
    if(vectorized){
      const int4* input=reinterpret_cast<const int4*>(x);int4* out=reinterpret_cast<int4*>(window+row*VOCAB);
      for(int j=part*(CHUNK/8)+threadIdx.x;j<(part+1)*(CHUNK/8);j+=blockDim.x)out[j]=input[j];
    }else{
      for(int j=part*CHUNK+threadIdx.x;j<(part+1)*CHUNK;j+=blockDim.x)window[row*VOCAB+j]=x[j];
    }
  }
  if(batch&&!done)return;
  // Current row comes directly from the immutable model logit input. Other
  // rows were copied by earlier completed token reads: no inter-block barrier.
  const __half* values=(!batch||row==n-1)?x:window+row*VOCAB;
  float first=-INFINITY,second=-INFINITY;
  for(int j=part*CHUNK+threadIdx.x;j<(part+1)*CHUNK;j+=blockDim.x)insert(__half2float(values[j]),first,second);
  __shared__ float a[THREADS],b[THREADS];a[threadIdx.x]=first;b[threadIdx.x]=second;__syncthreads();
  for(int stride=THREADS/2;stride;stride>>=1){
    if(threadIdx.x<stride){first=a[threadIdx.x];second=b[threadIdx.x];insert(a[threadIdx.x+stride],first,second);insert(b[threadIdx.x+stride],first,second);a[threadIdx.x]=first;b[threadIdx.x]=second;}
    __syncthreads();
  }
  if(threadIdx.x==0){int offset=(row*PARTS+part)*2;partials[offset]=a[0];partials[offset+1]=b[0];}
}
__device__ __forceinline__ void combine(const float* partials,int row,float& first,float& second){
  first=-INFINITY;second=-INFINITY;for(int i=0;i<PARTS*2;i++)insert(partials[row*PARTS*2+i],first,second);
}
__global__ void finish_step(const float* partials,__half* pairs,const int64_t* starts,const int64_t* token,
    const int64_t* eos,int eos_count,float* host_features,int64_t* host_starts,int64_t* packet,int n,int limit,bool batch){
  int blocks=batch?CAPACITY*PARTS:PARTS;int64_t start=starts[0];
  for(int i=1;i<blocks;i++)if(starts[i]<start)start=starts[i];host_starts[n-1]=start;
  bool done=terminal(n,limit,token[0],eos,eos_count);float first,second;
  if(!batch){combine(partials,n-1,first,second);pairs[(n-1)*2]=__float2half(first);pairs[(n-1)*2+1]=__float2half(second);}
  if(done)for(int row=0;row<n;row++){
    if(batch)combine(partials,row,first,second);
    else{first=__half2float(pairs[row*2]);second=__half2float(pairs[row*2+1]);}
    float gap=first-second;host_features[row*2]=gap;host_features[row*2+1]=expf((-gap)*(1.0f/3.0f));
  }
  __threadfence_system();
  packet[0]=(int64_t)(((now_ns()&((1ULL<<44)-1))<<18)|((uint64_t)token[0]&((1ULL<<18)-1)));
}
void dispatch(torch::Tensor x,torch::Tensor token,torch::Tensor window,torch::Tensor pairs,
    torch::Tensor partials,torch::Tensor starts,torch::Tensor eos,torch::Tensor packet,
    int64_t mapped_features,int64_t mapped_starts,int64_t n,int64_t limit,bool batch){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat16&&x.numel()==VOCAB&&x.stride(-1)==1,"Expected last FP16 vocabulary row");
  TORCH_CHECK(n>=1&&n<=limit&&limit<=CAPACITY,"K5 window exceeded");
  TORCH_CHECK(partials.is_cuda()&&partials.scalar_type()==torch::kFloat32&&partials.numel()==CAPACITY*PARTS*2,"Invalid bounded partial buffer");
  TORCH_CHECK(starts.is_cuda()&&starts.scalar_type()==torch::kInt64&&starts.numel()==CAPACITY*PARTS,"Invalid bounded clock scratch");
  auto stream=at::cuda::getCurrentCUDAStream();const __half* input=reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  __half* buffer=reinterpret_cast<__half*>(window.data_ptr<at::Half>());
  bool vectorized=((reinterpret_cast<uintptr_t>(input)|reinterpret_cast<uintptr_t>(buffer))&15)==0;
  fused_work<<<batch?CAPACITY*PARTS:PARTS,THREADS,0,stream>>>(input,buffer,partials.data_ptr<float>(),starts.data_ptr<int64_t>(),
    token.data_ptr<int64_t>(),eos.data_ptr<int64_t>(),eos.numel(),n,limit,batch,vectorized);C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish_step<<<1,1,0,stream>>>(partials.data_ptr<float>(),reinterpret_cast<__half*>(pairs.data_ptr<at::Half>()),starts.data_ptr<int64_t>(),
    token.data_ptr<int64_t>(),eos.data_ptr<int64_t>(),eos.numel(),reinterpret_cast<float*>(mapped_features),
    reinterpret_cast<int64_t*>(mapped_starts),packet.data_ptr<int64_t>(),n,limit,batch);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void stamp_kernel(int64_t* out){out[0]=(int64_t)now_ns();}
void stamp(torch::Tensor out){stamp_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
int64_t mapped(torch::Tensor host){TORCH_CHECK(host.is_pinned(),"Mapped mailbox must be pinned");void* device=nullptr;C10_CUDA_CHECK(cudaHostGetDevicePointer(&device,host.data_ptr(),0));return reinterpret_cast<int64_t>(device);}
