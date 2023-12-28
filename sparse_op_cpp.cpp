#include <torch/extension.h>
#include <ATen/ops/mm_native.h>
// if vscode complains about the above include,
// try to modify `.vscode/c_cpp_properties.json`
#include <iostream>
#include <vector>
#include <fstream>
#include <pybind11/pybind11.h>

torch::jit::Function* _sparse_bsr_bsc_matmul;

torch::Tensor sparse_bsr_mm(const torch::Tensor& a, const torch::Tensor& b) {
  if (a.layout() == at::kSparseBsr && b.layout() == at::kSparseBsc) {
    auto output = (*_sparse_bsr_bsc_matmul)({a, b});
    return output.toTensor();
  }
  return at::native::_sparse_csr_mm(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_bsr_mm, "spgemv forward");
}

// https://pytorch.org/cppdocs/api/classtorch_1_1_library.html
TORCH_LIBRARY_IMPL(aten, SparseCsrCPU, m) {
  auto my_module = pybind11::module::import("pypose.sparse.ops");
  auto script_function = my_module.attr("bsr_bsc_matmul").cast<torch::jit::StrongFunctionPtr>();
  _sparse_bsr_bsc_matmul = script_function.function_;
  m.impl("mm", sparse_bsr_mm);
}
