#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "imgblend_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("imgblend_wrapper", &imgblend_wrapper_cpp, "imgblend_wrapper_cpp");
}
