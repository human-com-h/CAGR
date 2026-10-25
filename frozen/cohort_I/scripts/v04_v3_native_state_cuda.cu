// Compile the EXACT previously T4-validated kernels. No kernel math changes.
#include "v04_v3_argmax_cuda.cu"
void dispatch_cached(const void* input,void* pending,void* pairs,void* partials,
    void* indices,void* invalid,void* starts,const void* eos,int eos_count,void* packet,
    int64_t mapped_features,int64_t mapped_starts,int n,int limit,bool vectorized,int device){
  auto stream=at::cuda::getCurrentCUDAStream(device);
  const __half* x=static_cast<const __half*>(input);
  bool aligned=(reinterpret_cast<uintptr_t>(x)&15)==0;
  scan<<<PARTS,THREADS,0,stream>>>(x,static_cast<float*>(partials),static_cast<int*>(indices),static_cast<int*>(invalid),static_cast<int64_t*>(starts),n-1,vectorized&&aligned);C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish<<<1,1,0,stream>>>(static_cast<float*>(partials),static_cast<int*>(indices),static_cast<int*>(invalid),static_cast<__half*>(pairs),
    static_cast<const int64_t*>(starts),static_cast<int64_t*>(pending),static_cast<const int64_t*>(eos),eos_count,
    reinterpret_cast<float*>(mapped_features),reinterpret_cast<int64_t*>(mapped_starts),static_cast<int64_t*>(packet),n,limit);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
