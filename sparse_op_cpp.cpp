#include <torch/extension.h>
// if vscode complains about the above include,
// try to modify `.vscode/c_cpp_properties.json`
#include <iostream>
#include <vector>


// implement this function:
// https://github.com/nicknytko/numml/blob/60645b3c0f7fd3cc08f61dafb448515ed348d1ac/cpp/sparse_csr_cpu.cpp#L11
torch::Tensor spgemv_forward(torch::Tensor bsr, torch::Tensor dense_vec) {
  auto crow = bsr.crow_indices();
  auto col = bsr.col_indices();
  auto values = bsr.values();
  // create an accessor for the crow pointer
  auto crow_accessor = crow.accessor<int64_t, 1>();
  auto col_accessor = col.accessor<int64_t, 1>();

  // AT_DISPATCH_FLOATING_TYPES explained:
  // https://pytorch.org/tutorials/advanced/cpp_extension.html#:~:text=The%20main%20point%20of%20interest%20here%20is%20the
  AT_DISPATCH_FLOATING_TYPES(values.type(), "spgemv_forward_cpu", ([&] {
      const auto A_data_acc = values.accessor<scalar_t, 1>();
  }));

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spgemv_forward, "spgemv forward");
}
