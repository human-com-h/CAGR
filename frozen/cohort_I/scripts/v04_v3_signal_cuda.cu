#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

// No CUDA events or host/device synchronization. Launches use PyTorch's stream.
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long value; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)); return value;
}
__global__ void capture_kernel(const __half* x,const int64_t* token,__half* window,
    int64_t* meta,int64_t* starts,int n,int limit,bool copy) {
  if(threadIdx.x==0) { starts[n-1]=(int64_t)now_ns();meta[0]=n;meta[1]=limit;meta[2]=token[0]; }
  __syncthreads();
  if(copy) for(int j=threadIdx.x;j<151936;j+=blockDim.x) window[(n-1)*151936+j]=x[j];
}
__device__ __forceinline__ void insert(float v,float& first,float& second) {
  if(isnan(v)||isnan(first)){first=NAN;second=NAN;return;}
  if(v>=first){second=first;first=v;}else if(v>second)second=v;
}
__global__ void terminal_top2(const __half* x,const __half* pairs,float* host_features,const int64_t* meta,
    const int64_t* eos,int eos_count,bool batch) {
  int n=(int)meta[0];bool terminal=n==meta[1];
  for(int i=0;i<eos_count;i++) terminal=terminal||meta[2]==eos[i];
  if(!terminal||blockIdx.x>=n)return;
  float first=-INFINITY,second=-INFINITY;
  if(batch)for(int j=threadIdx.x;j<151936;j+=blockDim.x)insert(__half2float(x[blockIdx.x*151936+j]),first,second);
  __shared__ float a[256],b[256];a[threadIdx.x]=first;b[threadIdx.x]=second;__syncthreads();
  for(int stride=128;stride;stride>>=1){
    if(threadIdx.x<stride){first=a[threadIdx.x];second=b[threadIdx.x];insert(a[threadIdx.x+stride],first,second);insert(b[threadIdx.x+stride],first,second);a[threadIdx.x]=first;b[threadIdx.x]=second;}
    __syncthreads();
  }
  if(threadIdx.x==0){
    float gap=batch ? a[0]-b[0] : __half2float(pairs[blockIdx.x*2])-__half2float(pairs[blockIdx.x*2+1]);
    host_features[blockIdx.x*2]=gap;
    host_features[blockIdx.x*2+1]=expf((-gap)*(1.0f/3.0f));
    __threadfence_system();
  }
}
__global__ void finish_kernel(const int64_t* meta,int64_t* packet) {
  // 44 timer bits wrap after ~4.89h, longer than the frozen 10800s run budget.
  const uint64_t mask=(1ULL<<44)-1;
  packet[0]=(int64_t)(((now_ns()&mask)<<18)|((uint64_t)meta[2]&((1ULL<<18)-1)));
}
__global__ void stamp_kernel(int64_t* out){out[0]=(int64_t)now_ns();}
void capture(torch::Tensor x,torch::Tensor token,torch::Tensor window,torch::Tensor meta,
             int64_t mapped_starts,int64_t n,int64_t limit,bool copy){
  TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat16&&x.numel()==151936&&x.stride(-1)==1,"Expected contiguous last FP16 vocabulary row");
  TORCH_CHECK(n>=1&&n<=5&&limit>=n&&limit<=5,"K5 window exceeded");
  capture_kernel<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<__half*>(x.data_ptr<at::Half>()),token.data_ptr<int64_t>(),reinterpret_cast<__half*>(window.data_ptr<at::Half>()),meta.data_ptr<int64_t>(),reinterpret_cast<int64_t*>(mapped_starts),n,limit,copy);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void reduce(torch::Tensor window,torch::Tensor pairs,torch::Tensor meta,torch::Tensor eos,int64_t mapped_features,bool batch){
  terminal_top2<<<5,256,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<__half*>(window.data_ptr<at::Half>()),reinterpret_cast<__half*>(pairs.data_ptr<at::Half>()),reinterpret_cast<float*>(mapped_features),meta.data_ptr<int64_t>(),eos.data_ptr<int64_t>(),eos.numel(),batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void finish(torch::Tensor meta,torch::Tensor packet){finish_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(meta.data_ptr<int64_t>(),packet.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
void stamp(torch::Tensor out){stamp_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();}
int64_t mapped(torch::Tensor host){TORCH_CHECK(host.is_pinned(),"Mapped mailbox must be pinned");void* device=nullptr;C10_CUDA_CHECK(cudaHostGetDevicePointer(&device,host.data_ptr(),0));return reinterpret_cast<int64_t>(device);}
