#include <torch/extension.h>

torch::Tensor q4k_dequant(torch::Tensor w, int64_t in);
torch::Tensor q4k_gemm_naive(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);
torch::Tensor q4k_gemm_tiled(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);
torch::Tensor q4k_gemm_mma(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias, int64_t tile_n);
int64_t q4k_mma_tile_n(int64_t M, int64_t N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q4k_dequant", &q4k_dequant, "Q4_K packed [out, in/256*144] -> fp32 [out, in]", py::arg("w"),
          py::arg("in_features"));
    m.def("q4k_gemm_naive", &q4k_gemm_naive, "bf16 x [M, in] @ Q4_K W[out, in]^T (+ bias) -> bf16 [M, out]",
          py::arg("x"), py::arg("w"), py::arg("bias") = py::none());
    m.def("q4k_gemm_tiled", &q4k_gemm_tiled, "the same GEMM, tiled through shared memory", py::arg("x"),
          py::arg("w"), py::arg("bias") = py::none());
    m.def("q4k_gemm_mma", &q4k_gemm_mma,
          "the same GEMM on the tensor cores (bf16 mma.sync); tile_n 0 picks the tile (q4k_mma_tile_n), 64 or 128 "
          "forces one (tests)", py::arg("x"), py::arg("w"), py::arg("bias") = py::none(), py::arg("tile_n") = 0);
    m.def("q4k_mma_tile_n", &q4k_mma_tile_n, "the N width of the tile q4k_gemm_mma picks for an M x N output",
          py::arg("M"), py::arg("N"));
}
