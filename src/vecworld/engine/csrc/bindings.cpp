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
std::vector<torch::Tensor> glue_rnm(torch::Tensor res, c10::optional<torch::Tensor> y,
                                    c10::optional<torch::Tensor> gate, torch::Tensor shift, torch::Tensor scale,
                                    double eps, bool quantize);
torch::Tensor glue_gated_residual(torch::Tensor res, torch::Tensor y, torch::Tensor gate,
                                  c10::optional<torch::Tensor> out);
void glue_qk_norm_rope(torch::Tensor x, c10::optional<torch::Tensor> weight, torch::Tensor cos, torch::Tensor sin,
                       torch::Tensor out, int64_t offset, double eps);

void register_stream(pybind11::module& m);   // stream.cpp: SlotRing, host_pinned, graph_launch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_stream(m);
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
    m.def("glue_rnm", &glue_rnm,
          "res bf16 [M, H], y [M, H] | None, gate | None, shift, scale bf16 [B, H] -> (res_out = res + gate * y, "
          "x_mod = LayerNorm(res_out) * (1 + scale) + shift, then with quantize xq, s, xsum of x_mod as "
          "q4k_quantize_act, else empty)", py::arg("res"), py::arg("y"), py::arg("gate"), py::arg("shift"),
          py::arg("scale"), py::arg("eps") = 1e-6, py::arg("quantize") = false);
    m.def("glue_gated_residual", &glue_gated_residual,
          "res + gate * y, gate [B, H] broadcast per image, into out (bf16 [M, H]) if given",
          py::arg("res"), py::arg("y"), py::arg("gate"), py::arg("out") = py::none());
    m.def("glue_qk_norm_rope", &glue_qk_norm_rope,
          "x bf16 [B, S, heads * 128] -> out[:, offset:offset + S] (bf16 [B, S_out, heads, 128]) = rotary(RMSNorm(x) "
          "per head, cos / sin fp32 [S_out, 128] at positions offset..)", py::arg("x"), py::arg("weight"),
          py::arg("cos"), py::arg("sin"), py::arg("out"), py::arg("offset"), py::arg("eps"));
}
