#include <torch/extension.h>
void dispatch(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
              torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
              int64_t,int64_t,int64_t,int64_t,bool);
void stamp(torch::Tensor);
int64_t mapped(torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("dispatch",&dispatch);m.def("stamp",&stamp);m.def("mapped",&mapped);}
