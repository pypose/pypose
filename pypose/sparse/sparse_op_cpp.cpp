#include <iostream>
#include <torch/extension.h>
#include <ATen/ops/mm_native.h>
#include <pybind11/pybind11.h>

torch::jit::Function* _sparse_bsr_bsc_matmul;

torch::Tensor sparse_bsr_mm(const torch::Tensor& a, const torch::Tensor& b) {
  if (a.layout() == at::kSparseBsr && b.layout() == at::kSparseBsc) {
    auto output = (*_sparse_bsr_bsc_matmul)({a, b});
    return output.toTensor();
  }
  return at::native::_sparse_csr_mm(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY_IMPL(aten, SparseCsrCPU, m) {
  // hide stderr
  freopen("/dev/null","a",stderr);
  auto module = pybind11::module::import("pypose.sparse.ops");
  auto object = module.attr("bsr_bsc_matmul");
  auto script_function = object.cast<torch::jit::StrongFunctionPtr>();
  _sparse_bsr_bsc_matmul = script_function.function_;
  m.impl("mm", sparse_bsr_mm);
  // resume stderr
  freopen ("/dev/tty","a",stderr);
}


TORCH_LIBRARY_IMPL(aten, SparseCsrCUDA, m) {
  m.impl("mm", sparse_bsr_mm);
}
