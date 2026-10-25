#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cmath>
#include "v04_v3_native_state_host.h"
namespace py=pybind11;
void dispatch_cached(const void*,void*,void*,void*,void*,void*,void*,const void*,int,void*,int64_t,int64_t,int,int,bool,int);
void stamp(torch::Tensor);
int64_t mapped(torch::Tensor);
class NativeState {
  // Strong owners, never borrowed logits or prefill views. Pointers bind ONCE.
  std::vector<torch::Tensor> owned;
  std::array<void*,9> ptr{};
  int device,eos_count;int64_t features_device,starts_device;
  float* features;int64_t* starts;cagr_state::Records records;
 public:
  explicit NativeState(std::vector<torch::Tensor> buffers):owned(std::move(buffers)){
    TORCH_CHECK(owned.size()==12,"Expected 12 fixed owned buffers");
    device=owned[0].get_device();
    const std::array<at::ScalarType,9> types={torch::kInt64,torch::kFloat16,torch::kFloat32,torch::kInt32,torch::kInt32,torch::kInt64,torch::kInt64,torch::kInt64,torch::kFloat16};
    const std::array<int64_t,9> sizes={1,10,80,40,40,40,owned[6].numel(),1,5*151936};
    for(int i=0;i<9;i++){
      const auto& t=owned[i];TORCH_CHECK(t.is_cuda()&&t.get_device()==device&&t.is_contiguous()&&t.scalar_type()==types[i]&&t.numel()==sizes[i],"Invalid fixed buffer ",i);
      ptr[i]=t.data_ptr();
    }
    TORCH_CHECK(owned[0].dim()==2&&owned[0].size(0)==1&&owned[0].size(1)==1,"Invalid pending shape");
    TORCH_CHECK(owned[8].dim()==2&&owned[8].size(0)==5&&owned[8].size(1)==151936,"Invalid K5 FP16 capacity");
    TORCH_CHECK(owned[9].device().is_cpu()&&owned[9].is_pinned()&&owned[9].is_contiguous()&&owned[9].scalar_type()==torch::kFloat32&&owned[9].numel()==10,"Invalid pinned features");
    TORCH_CHECK(owned[10].device().is_cpu()&&owned[10].is_pinned()&&owned[10].is_contiguous()&&owned[10].scalar_type()==torch::kInt64&&owned[10].numel()==5,"Invalid pinned starts");
    // CPU-only boundary timestamps are NEVER mapped or read from a device.
    TORCH_CHECK(owned[11].device().is_cpu()&&owned[11].is_contiguous()&&owned[11].scalar_type()==torch::kInt64&&owned[11].numel()==5,"Invalid boundary timestamps");
    eos_count=int(owned[6].numel());TORCH_CHECK(eos_count>0,"Empty EOS set");
    features=owned[9].data_ptr<float>();starts=owned[10].data_ptr<int64_t>();
    features_device=mapped(owned[9]);starts_device=mapped(owned[10]);
  }
  int step(torch::Tensor logits,int n,int limit,bool vectorized,int64_t host_start){
    // Logits safety remains checked EVERY call. No fixed-buffer Tensor args.
    TORCH_CHECK(logits.is_cuda()&&logits.get_device()==device&&logits.scalar_type()==torch::kFloat16&&logits.dim()==2&&logits.size(0)==1&&logits.size(1)==151936&&logits.stride(1)==1,"Expected same-device one FP16 vocabulary row");
    auto& r=records.next(n,limit);r.a=host_start;
    TORCH_CHECK(host_start>0&&host_start<=cagr_state::clock_ns(),"Invalid host boundary clock");
    dispatch_cached(logits.data_ptr(),ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5],ptr[6],eos_count,ptr[7],features_device,starts_device,n,limit,vectorized,device);
    r.b=cagr_state::clock_ns();
    auto packet=uint64_t(owned[7].item<int64_t>()); // ONE original token read; no added sync/event.
    r.z=cagr_state::clock_ns();r.s=starts[n-1];int token=cagr_state::decode(packet,r.s,r.e);
    r.decode=cagr_state::clock_ns()-r.z;
    return token;
  }
  py::tuple last_raw() const {
    TORCH_CHECK(records.count>0,"No signal record");const auto& r=records.rows[records.count-1];
    return py::make_tuple(r.a,r.b,r.z,r.s,r.e,r.decode);
  }
  py::dict finalize(int count,bool include_features){
    TORCH_CHECK(count==records.count&&count>=1&&count<=5,"Wrong native phase count");
    const auto* ends=owned[11].data_ptr<int64_t>();py::list output,values;int64_t charge=0;
    for(int i=0;i<count;i++){
      auto& r=records.rows[i];cagr_state::enclose(r,ends[i]);charge+=r.work+r.decode+r.capture;
      py::dict d;d["cpu_start_ns"]=r.a;d["cpu_end_ns"]=r.b;d["shared_read_done_ns"]=r.z;d["gpu_start_ns"]=r.s;d["gpu_end_ns"]=r.e;
      d["work_union_upper_ns"]=r.work;d["causal_offset_lower_ns"]=r.lo;d["causal_offset_upper_ns"]=r.hi;d["decode_cpu_ns"]=r.decode;d["capture_cpu_ns"]=r.capture;
      d["shared_token_waits"]=1;d["extra_device_syncs"]=0;d["native_entry_calls"]=1;d["native_kernel_launches"]=2;
      d["accounting"]="per-dispatch causal union upper; dictionaries deferred to stage end; minimum ALL block starts";output.append(d);
      if(include_features){float gap=features[2*i],u=features[2*i+1];TORCH_CHECK(gap>=0&&std::isfinite(gap+u),"Non-finite v3 feature");values.append(py::make_tuple(gap,u));}
    }
    py::dict result;result["records"]=output;result["values"]=values;result["charged_ns"]=charge;return result;
  }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("stamp",&stamp);m.def("mapped",&mapped);m.def("host_clock_ns",&cagr_state::clock_ns);
  py::class_<NativeState>(m,"NativeState").def(py::init<std::vector<torch::Tensor>>()).def("step",&NativeState::step).def("last_raw",&NativeState::last_raw).def("finalize",&NativeState::finalize);
}
