#include <torch/extension.h>

torch::Tensor q4k_dequant(torch::Tensor w, int64_t in);
torch::Tensor q4k_gemm_naive(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);
torch::Tensor q4k_gemm_tiled(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4k_dequant", &q4k_dequant, "Q4_K packed [out, in/256*144] -> fp32 [out, in]", py::arg("w"),
          py::arg("in_features"));
    m.def("q4k_gemm_naive", &q4k_gemm_naive, "bf16 x [M, in] @ Q4_K W[out, in]^T (+ bias) -> bf16 [M, out]",
          py::arg("x"), py::arg("w"), py::arg("bias") = py::none());
    m.def("q4k_gemm_tiled", &q4k_gemm_tiled, "the same GEMM, tiled through shared memory", py::arg("x"),
          py::arg("w"), py::arg("bias") = py::none());
}
