#include <torch/extension.h>
void capture(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,bool);
void reduce(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,bool);
void finish(torch::Tensor,torch::Tensor);
void stamp(torch::Tensor);
int64_t mapped(torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("capture",&capture);m.def("reduce",&reduce);m.def("finish",&finish);m.def("stamp",&stamp);m.def("mapped",&mapped);}
