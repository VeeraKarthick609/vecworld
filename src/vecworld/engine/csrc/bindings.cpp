#include <torch/extension.h>

torch::Tensor q4k_dequant(torch::Tensor w, int64_t in);
torch::Tensor q4k_gemm_naive(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);
torch::Tensor q4k_gemm_tiled(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);
torch::Tensor q4k_gemm_mma(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias, int64_t tile_n);
int64_t q4k_mma_tile_n(int64_t M, int64_t N);
std::vector<torch::Tensor> q4k_quantize_act(torch::Tensor x);
torch::Tensor q4k_gemm_w4a8(torch::Tensor xq, torch::Tensor s, torch::Tensor xsum, torch::Tensor w,
                            c10::optional<torch::Tensor> bias, int64_t tile_n, bool out_fp32);
torch::Tensor q4k_gemv(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias);

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
    m.def("q4k_quantize_act", &q4k_quantize_act,
          "bf16 x [M, K] -> (int8 xq [M, K], fp32 s [M], int32 xsum [M, K / 32]): per-token s = max|x| / 127, "
          "xq = round(x / s), xsum = sum of xq per 32-wide sub-block", py::arg("x"));
    m.def("q4k_gemm_w4a8", &q4k_gemm_w4a8,
          "(xq * s) @ Q4_K W[out, in]^T (+ bias) on the INT8 tensor cores (mma.sync s8) -> bf16 [M, out] (fp32 "
          "with out_fp32); tile_n as q4k_gemm_mma", py::arg("xq"), py::arg("s"), py::arg("xsum"), py::arg("w"),
          py::arg("bias") = py::none(), py::arg("tile_n") = 0, py::arg("out_fp32") = false);
    m.def("q4k_gemv", &q4k_gemv,
          "bf16 x [M <= 8, in] @ Q4_K W[out, in]^T (+ bias) -> bf16 [M, out]: the memory-bound GEMV for the "
          "modulation linears, each Q4_K byte read once", py::arg("x"), py::arg("w"), py::arg("bias") = py::none());
}
