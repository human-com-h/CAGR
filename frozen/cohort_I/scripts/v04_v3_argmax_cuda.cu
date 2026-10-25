#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <limits.h>

constexpr int VOCAB=151936,CAPACITY=5,PARTS=8,THREADS=256,CHUNK=VOCAB/PARTS;
static_assert(VOCAB%PARTS==0 && CHUNK%8==0,"Exact bounded geometry");
__device__ __forceinline__ uint64_t now_ns(){uint64_t v;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;}
// Keep TWO VALUES (including duplicate maxima), but the greedy index is the
// LOWEST index of the largest value. Never use an unspecified topk index.
__device__ __forceinline__ void offer(float v,int idx,float& first,float& second,int& best){
  if(v>first || (v==first && idx<best)){second=first;first=v;best=idx;}
  else if(v>second)second=v;
}
__global__ void scan(const __half* x,float* partials,int* indices,int* invalid,int64_t* starts,int row,bool vectorized){
  int part=blockIdx.x,tid=threadIdx.x;if(tid==0)starts[part]=(int64_t)now_ns();
  float first=-INFINITY,second=-INFINITY;int best=INT_MAX,bad=0;
  if(vectorized){
    const int4* vec=reinterpret_cast<const int4*>(x);
    for(int j=part*(CHUNK/8)+tid;j<(part+1)*(CHUNK/8);j+=blockDim.x){
      int4 item=vec[j];const __half* h=reinterpret_cast<const __half*>(&item);
      #pragma unroll
      for(int k=0;k<8;k++){float v=__half2float(h[k]);bad|=!isfinite(v);offer(v,j*8+k,first,second,best);}
    }
  }else{
    for(int j=part*CHUNK+tid;j<(part+1)*CHUNK;j+=blockDim.x){float v=__half2float(x[j]);bad|=!isfinite(v);offer(v,j,first,second,best);}
  }
  __shared__ float a[THREADS],b[THREADS];__shared__ int ix[THREADS],flags[THREADS];
  a[tid]=first;b[tid]=second;ix[tid]=best;flags[tid]=bad;__syncthreads();
  for(int stride=THREADS/2;stride;stride>>=1){
    if(tid<stride){
      first=a[tid];second=b[tid];best=ix[tid];
      offer(a[tid+stride],ix[tid+stride],first,second,best);
      offer(b[tid+stride],INT_MAX,first,second,best);
      a[tid]=first;b[tid]=second;ix[tid]=best;flags[tid]|=flags[tid+stride];
    }
    __syncthreads();
  }
  if(tid==0){int p=row*PARTS+part;partials[2*p]=a[0];partials[2*p+1]=b[0];indices[p]=ix[0];invalid[p]=flags[0];}
}
__global__ void finish(const float* partials,const int* indices,const int* invalid,__half* pairs,
    const int64_t* starts,int64_t* pending,const int64_t* eos,int eos_count,
    float* features,int64_t* host_starts,int64_t* packet,int n,int limit){
  int row=n-1,best=INT_MAX,bad=0;float first=-INFINITY,second=-INFINITY;
  int64_t begin=starts[0];
  for(int part=0;part<PARTS;part++){
    int p=row*PARTS+part;if(starts[part]<begin)begin=starts[part];bad|=invalid[p];
    offer(partials[2*p],indices[p],first,second,best);offer(partials[2*p+1],INT_MAX,first,second,best);
  }
  host_starts[row]=begin;
  // Any non-finite vocabulary entry is rejected, including a hidden -Inf.
  // Sentinel is outside the vocabulary but fits the existing 18-bit packet.
  if(bad || best<0 || best>=VOCAB || !isfinite(first) || !isfinite(second)){
    pending[0]=-1;__threadfence_system();
    packet[0]=(int64_t)(((now_ns()&((1ULL<<44)-1))<<18)|((1ULL<<18)-1));return;
  }
  pending[0]=best;pairs[2*row]=__float2half(first);pairs[2*row+1]=__float2half(second);
  bool done=n==limit;for(int j=0;j<eos_count;j++)done=done||best==eos[j];
  if(done)for(int r=0;r<n;r++){
    float gap=__half2float(pairs[2*r])-__half2float(pairs[2*r+1]);
    features[2*r]=gap;features[2*r+1]=expf((-gap)*(1.0f/3.0f));
  }
  __threadfence_system();
  packet[0]=(int64_t)(((now_ns()&((1ULL<<44)-1))<<18)|((uint64_t)best));
}
void dispatch(torch::Tensor x,torch::Tensor pending,torch::Tensor pairs,torch::Tensor partials,
    torch::Tensor indices,torch::Tensor invalid,torch::Tensor starts,torch::Tensor eos,torch::Tensor packet,
    int64_t mapped_features,int64_t mapped_starts,int64_t n,int64_t limit,bool vectorized){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat16&&x.numel()==VOCAB&&x.stride(-1)==1,"Expected one last FP16 vocabulary row");
  TORCH_CHECK(n>=1&&n<=limit&&limit<=CAPACITY,"K5 window exceeded");
  TORCH_CHECK(pending.is_cuda()&&pending.scalar_type()==torch::kInt64&&pending.numel()==1,"Invalid pending output");
  TORCH_CHECK(pairs.is_cuda()&&pairs.scalar_type()==torch::kFloat16&&pairs.numel()==CAPACITY*2,"Invalid pair buffer");
  TORCH_CHECK(partials.is_cuda()&&partials.scalar_type()==torch::kFloat32&&partials.numel()==CAPACITY*PARTS*2,"Invalid bounded partials");
  TORCH_CHECK(indices.is_cuda()&&indices.scalar_type()==torch::kInt32&&indices.numel()==CAPACITY*PARTS,"Invalid index buffer");
  TORCH_CHECK(invalid.is_cuda()&&invalid.scalar_type()==torch::kInt32&&invalid.numel()==CAPACITY*PARTS,"Invalid nonfinite flags");
  TORCH_CHECK(starts.is_cuda()&&starts.scalar_type()==torch::kInt64&&starts.numel()==CAPACITY*PARTS,"Invalid clock scratch");
  TORCH_CHECK(eos.is_cuda()&&eos.scalar_type()==torch::kInt64&&packet.is_cuda()&&packet.scalar_type()==torch::kInt64&&packet.numel()==1,"Invalid EOS/packet");
  TORCH_CHECK(x.device()==pending.device()&&x.device()==partials.device()&&x.device()==pairs.device()&&x.device()==indices.device()&&x.device()==invalid.device()&&x.device()==starts.device()&&x.device()==eos.device()&&x.device()==packet.device(),"Device mismatch");
  auto stream=at::cuda::getCurrentCUDAStream();const __half* input=reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  bool aligned=(reinterpret_cast<uintptr_t>(input)&15)==0;
  scan<<<PARTS,THREADS,0,stream>>>(input,partials.data_ptr<float>(),indices.data_ptr<int>(),invalid.data_ptr<int>(),starts.data_ptr<int64_t>(),n-1,vectorized&&aligned);C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish<<<1,1,0,stream>>>(partials.data_ptr<float>(),indices.data_ptr<int>(),invalid.data_ptr<int>(),reinterpret_cast<__half*>(pairs.data_ptr<at::Half>()),
    starts.data_ptr<int64_t>(),pending.data_ptr<int64_t>(),eos.data_ptr<int64_t>(),eos.numel(),reinterpret_cast<float*>(mapped_features),
    reinterpret_cast<int64_t*>(mapped_starts),packet.data_ptr<int64_t>(),n,limit);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void stamp_kernel(int64_t* out){out[0]=(int64_t)now_ns();}
void stamp(torch::Tensor out){stamp_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
int64_t mapped(torch::Tensor host){TORCH_CHECK(host.is_pinned(),"Mapped mailbox must be pinned");void* device=nullptr;C10_CUDA_CHECK(cudaHostGetDevicePointer(&device,host.data_ptr(),0));return reinterpret_cast<int64_t>(device);}
