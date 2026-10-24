// Q4_K kernels. Stage 1 (Task 3): the naive correctness reference.
//
// A Q4_K super-block holds 256 weights of one row in 144 bytes:
//   d (fp16) | dmin (fp16) | scales[12]: 8 x 6-bit sc, 8 x 6-bit m | qs[128]: 256 x 4-bit codes
// Weight i of sub-block j (32 weights each): w = (d * sc_j) * q_i - (dmin * m_j).
// qs is 4 runs of 32 bytes; run g holds sub-block 2g in its low nibbles and 2g+1 in its high nibbles.
// A row of in_features weights is in_features / 256 super-blocks, back to back.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int QK_K = 256;
constexpr int BLOCK_BYTES = 144;

// Sub-block j's 6-bit scale and min (gguf-py Q4_K.get_scale_min, llama.cpp get_scale_min_k4).
__device__ __forceinline__ void scale_min(int j, const uint8_t* s, int& sc, int& m) {
    if (j < 4) {
        sc = s[j] & 63;
        m = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// One super-block's per-sub-block multipliers, rounded as gguf-py rounds them (fp32 products).
struct SubScale {
    float d_sc, dm;
};

__device__ __forceinline__ SubScale sub_scale(const uint8_t* blk, int j) {
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(blk + 2));
    int sc, m;
    scale_min(j, blk + 4, sc, m);
    return {__fmul_rn(d, float(sc)), __fmul_rn(dmin, float(m))};
}

// w = d_sc * q - dm with two roundings (no FMA contraction), so it is bit-exact against gguf-py's numpy.
__device__ __forceinline__ float decode(SubScale s, int q) { return __fsub_rn(__fmul_rn(s.d_sc, float(q)), s.dm); }

// The code of weight k (0..255) of a super-block.
__device__ __forceinline__ int code(const uint8_t* blk, int k) {
    const int j = k >> 5, i = k & 31;
    return (blk[16 + (j >> 1) * 32 + i] >> ((j & 1) * 4)) & 0x0F;
}

// Full dequant to fp32, one thread per weight (tests check the decode against gguf-py with it).
__global__ void dequant_kernel(const uint8_t* __restrict__ w, float* __restrict__ out, int64_t rows, int in) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * in) return;
    const int64_t r = idx / in;
    const int k = int(idx % in);
    const uint8_t* blk = w + (r * (in / QK_K) + k / QK_K) * BLOCK_BYTES;
    const int kk = k % QK_K;
    out[idx] = decode(sub_scale(blk, kk >> 5), code(blk, kk));
}

// Naive GEMM, y[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n]): one thread per output element. Each thread walks its
// weight row super-block by super-block, decodes the sub-block scales in registers and dots 32 codes at a time
// against x, accumulating in fp32. No shared memory, no reuse beyond what L1/L2 catch: a correctness reference.
// A 16 x 16 thread block covers 16 outputs (threadIdx.x, along n) of 16 rows of x (threadIdx.y): its 16 weight rows
// and 16 x rows stay in L1 while it runs, and a warp (2 rows of 16) reads each x value as a broadcast. (Measured on
// the 3050 Ti at M = 1280: 16 x 16 runs ~0.85 TFLOP/s, 32 x 8 ~0.55, threads along m instead of n ~0.3.)
__global__ void gemm_naive_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                                  const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, int M,
                                  int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;
    const int nb = K / QK_K;
    const __nv_bfloat16* xr = x + int64_t(m) * K;
    const uint8_t* wr = w + int64_t(n) * nb * BLOCK_BYTES;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = wr + b * BLOCK_BYTES;
        const uint32_t* qs = reinterpret_cast<const uint32_t*>(blk + 16);
        const __nv_bfloat16* xb = xr + b * QK_K;
        for (int g = 0; g < 4; ++g) {
            const SubScale lo = sub_scale(blk, 2 * g), hi = sub_scale(blk, 2 * g + 1);
            const uint2* xl = reinterpret_cast<const uint2*>(xb + g * 64);       // 4 bf16 per load
            for (int i4 = 0; i4 < 8; ++i4) {             // 4 bytes of codes at a time: 4 low + 4 high nibbles
                const uint32_t q = qs[g * 8 + i4];
                const uint2 a = xl[i4], c = xl[8 + i4];   // x for the low nibbles, and 32 further for the high
                const float xa[4] = {__low2float(*reinterpret_cast<const __nv_bfloat162*>(&a.x)),
                                     __high2float(*reinterpret_cast<const __nv_bfloat162*>(&a.x)),
                                     __low2float(*reinterpret_cast<const __nv_bfloat162*>(&a.y)),
                                     __high2float(*reinterpret_cast<const __nv_bfloat162*>(&a.y))};
                const float xc[4] = {__low2float(*reinterpret_cast<const __nv_bfloat162*>(&c.x)),
                                     __high2float(*reinterpret_cast<const __nv_bfloat162*>(&c.x)),
                                     __low2float(*reinterpret_cast<const __nv_bfloat162*>(&c.y)),
                                     __high2float(*reinterpret_cast<const __nv_bfloat162*>(&c.y))};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int byte = (q >> (8 * e)) & 0xFF;
                    acc += decode(lo, byte & 0x0F) * xa[e];
                    acc += decode(hi, byte >> 4) * xc[e];
                }
            }
        }
    }
    if (bias != nullptr) acc += __bfloat162float(bias[n]);
    y[int64_t(m) * N + n] = __float2bfloat16(acc);
}

void check_packed(const torch::Tensor& w, int64_t in) {
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kUInt8 && w.dim() == 2 && w.is_contiguous(),
                "packed Q4_K weight must be a contiguous CUDA uint8 [out, in / 256 * 144] tensor");
    TORCH_CHECK(in > 0 && in % QK_K == 0, "Q4_K in_features must be a positive multiple of 256, got ", in);
    TORCH_CHECK(w.size(1) == in / QK_K * BLOCK_BYTES, "packed row is ", w.size(1), " bytes; in_features ", in,
                " needs ", in / QK_K * BLOCK_BYTES);
}

}  // namespace

torch::Tensor q4k_dequant(torch::Tensor w, int64_t in) {
    check_packed(w, in);
    const c10::cuda::CUDAGuard guard(w.device());
    auto out = torch::empty({w.size(0), in}, w.options().dtype(torch::kFloat32));
    const int64_t total = w.size(0) * in;
    if (total == 0) return out;
    const int threads = 256;
    dequant_kernel<<<(total + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        w.data_ptr<uint8_t>(), out.data_ptr<float>(), w.size(0), int(in));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor q4k_gemm_naive(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
                "x must be a contiguous CUDA bf16 [M, in] tensor");
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    check_packed(w, K);
    TORCH_CHECK(w.device() == x.device(), "x and the weight are on different devices");
    const __nv_bfloat16* b = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == torch::kBFloat16 && bias->dim() == 1 &&
                        bias->size(0) == N && bias->is_contiguous(),
                    "bias must be a contiguous bf16 [out] tensor on x's device");
        b = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
    }
    TORCH_CHECK(M <= 65535LL * 16, "M too large for the naive grid");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return y;
    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(), b,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), int(M), int(N), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
