// Q4_K kernels. Stage 1 (Task 3): the naive correctness reference. Stage 2 (Task 4): tiled through shared memory.
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

// The same, from a super-block's first 16 bytes held in registers (h.x = d | dmin, h.y..h.w = scales[12]). With j a
// compile-time constant (the tiled kernel unrolls over sub-blocks) this is all shifts and masks.
__device__ __forceinline__ uint32_t scale_byte(uint4 h, int i) {
    const uint32_t word = i < 4 ? h.y : i < 8 ? h.z : h.w;
    return (word >> (8 * (i & 3))) & 0xFF;
}

__device__ __forceinline__ SubScale sub_scale_regs(uint4 h, int j) {
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x >> 16)));
    int sc, m;
    if (j < 4) {
        sc = scale_byte(h, j) & 63;
        m = scale_byte(h, j + 4) & 63;
    } else {
        sc = (scale_byte(h, j + 4) & 0x0F) | ((scale_byte(h, j - 4) >> 6) << 4);
        m = (scale_byte(h, j + 4) >> 4) | ((scale_byte(h, j) >> 6) << 4);
    }
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

// Tiled GEMM (Task 4). A thread block computes a TILE_M x TILE_N (128 x 128) tile of y with 256 threads, each owning
// 8 x 8 outputs. It walks K one sub-block (BK = 32 weights) at a time:
//
//   global --16-byte loads--> registers --decode / bf16->fp32--> shared memory --float4 reads--> 8 x 8 fp32 FMAs
//
//   shared, per K step:   As[32][128] = x[m0 .. m0+128, k .. k+32]^T   (fp32 from bf16)
//                         Ws[32][128] = W[n0 .. n0+128, k .. k+32]^T   (fp32, decoded from Q4_K)
//   thread (ty, tx):      acc[8][8] += As[k][its 8 rows] * Ws[k][its 8 cols], k = 0..31   (2048 FMAs per K step)
//
// Each weight is decoded once per tile and then read by 16 threads from shared memory (128 rows of x use it),
// instead of once per output as in the naive kernel. A Q4_K run (32 bytes of qs) holds two sub-blocks, low nibbles
// for sub-block 2g and high ones for 2g+1: the run is loaded into registers once and decoded twice, so no weight
// byte is fetched twice by a block. While one K step computes, the next one's x and W are already being fetched
// into registers (register prefetch); only then are they decoded into shared memory.
//
// Thread (ty, tx) owns rows {ty*4 + 0..3, 64 + ty*4 + 0..3} and columns {tx*4 + 0..3, 64 + tx*4 + 0..3}: float4
// reads of shared memory, and a warp reads 16 different 16-byte pieces of Ws per float4 (no bank conflicts). Loads
// of x and W map consecutive lanes to consecutive rows, so their shared-memory stores also hit 32 banks.
//
// Tiles are visited in pairs of M tiles (GROUP_M = 2) going along N: blocks that run at the same time share their
// weight tiles as well as their x tiles in L2 (measured +10-25% on the K = 3072 layers over plain row order).
// Measured alternatives (docs/engine/02-q4k-tiled.md): 128 x 64, 64 x 128, 64 x 64, 32 x 128, 16 x 128 tiles,
// double-buffered shared memory, coalesced (row-split) x loads, other groupings.
constexpr int TILE_M = 128, TILE_N = 128, BK = 32, THREAD_M = 8, THREAD_N = 8, GROUP_M = 2;
constexpr int TILED_THREADS = (TILE_M / THREAD_M) * (TILE_N / THREAD_N);   // 256

// 2 blocks per SM caps registers at 128 a thread (a few bytes spill; still the fastest measured).
__global__ void __launch_bounds__(TILED_THREADS, 2)
    gemm_tiled_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                      const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, int M, int N, int K) {
    constexpr int BM = TILE_M, BN = TILE_N, TM = THREAD_M, TN = THREAD_N, T = TILED_THREADS;
    constexpr int XL = BM * 4 / T;              // x slots per thread: BM rows x 4 pieces of 8 bf16 (16 bytes)
    constexpr int WL = BN * 2 / T;              // W slots per thread: BN rows x 2 halves of a run (16 code bytes)
    static_assert(XL >= 1 && XL * T == BM * 4 && WL >= 1 && WL * T == BN * 2, "tile shape vs thread count");
    static_assert(TM % 4 == 0 && TN % 4 == 0, "4-wide fragments");
    constexpr int RM = BM * 4 / TM, RN = BN * 4 / TN;   // stride between a thread's 4-row (4-column) pieces
    __shared__ __align__(16) float As[BK][BM];
    __shared__ __align__(16) float Ws[BK][BN];

    // Which tile: pairs of M tiles walk along N together (see GROUP_M above).
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, L = blockIdx.x;
    const int first = L / (GROUP_M * num_n) * GROUP_M, gs = min(num_m - first, GROUP_M);
    const int m0 = (first + L % (GROUP_M * num_n) % gs) * BM, n0 = L % (GROUP_M * num_n) / gs * BN;
    const int tid = threadIdx.x, tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int nb = K / QK_K;
    const int64_t row_bytes = int64_t(nb) * BLOCK_BYTES;

    // Slot i of this thread: x row m = idx % BM, piece c = idx / BM; W row n = idx % BN, half h = idx / BN of a run.
    // Rows past M or N are never read: their registers hold zeros, which decode to 0 (d = dmin = 0).
    const __nv_bfloat16* xp[XL];
    const uint8_t* wp[WL];
    uint4 xr[XL], wq[WL], wh[WL];
#pragma unroll
    for (int i = 0; i < XL; ++i) {
        const int idx = tid + i * T, m = idx % BM, c = idx / BM;
        xp[i] = m0 + m < M ? x + int64_t(m0 + m) * K + c * 8 : nullptr;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int n = (tid + i * T) % BN;
        wp[i] = n0 + n < N ? w + int64_t(n0 + n) * row_bytes : nullptr;
    }
    auto load_x = [&](int b, int j) {           // x for sub-block j of super-block b
#pragma unroll
        for (int i = 0; i < XL; ++i)
            xr[i] = xp[i] ? *reinterpret_cast<const uint4*>(xp[i] + b * QK_K + j * BK) : make_uint4(0, 0, 0, 0);
    };
    auto load_q = [&](int b, int g) {           // this slot's 16 code bytes of run g
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            const int h = (tid + i * T) / BN;
            wq[i] = wp[i] ? *reinterpret_cast<const uint4*>(wp[i] + b * BLOCK_BYTES + 16 + g * 32 + h * 16)
                          : make_uint4(0, 0, 0, 0);
        }
    };
    auto load_h = [&](int b) {                  // d, dmin, scales[12]: the super-block's first 16 bytes
#pragma unroll
        for (int i = 0; i < WL; ++i)
            wh[i] = wp[i] ? *reinterpret_cast<const uint4*>(wp[i] + b * BLOCK_BYTES) : make_uint4(0, 0, 0, 0);
    };
    auto store = [&](int j) {                   // registers -> shared memory for sub-block j: convert x, decode W
#pragma unroll
        for (int i = 0; i < XL; ++i) {
            const int idx = tid + i * T, m = idx % BM, c = idx / BM;
            const uint32_t v[4] = {xr[i].x, xr[i].y, xr[i].z, xr[i].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&v[e]);
                As[c * 8 + 2 * e][m] = __low2float(p);
                As[c * 8 + 2 * e + 1][m] = __high2float(p);
            }
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            const int idx = tid + i * T, n = idx % BN, h = idx / BN;
            const SubScale s = sub_scale_regs(wh[i], j);
            const uint32_t v[4] = {wq[i].x, wq[i].y, wq[i].z, wq[i].w};
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int q = (v[e >> 2] >> (8 * (e & 3) + 4 * (j & 1))) & 0x0F;   // low nibble: even j
                Ws[h * 16 + e][n] = decode(s, q);
            }
        }
    };

    float acc[TM][TN] = {};
    load_h(0);
    load_q(0, 0);
    load_x(0, 0);
    store(0);
    __syncthreads();
    for (int b = 0; b < nb; ++b) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {            // unrolled: j, and so every scale-byte index, is a constant
            const int jn = (j + 1) & 7, bn = j < 7 ? b : b + 1;
            const bool next = bn < nb;
            if (next) {                          // fetch the next K step while this one computes
                load_x(bn, jn);
                if ((jn & 1) == 0) load_q(bn, jn >> 1);    // a new run every other step
                if (jn == 0) load_h(bn);                   // a new super-block every 8th
            }
#pragma unroll
            for (int k = 0; k < BK; ++k) {
                float a[TM], bb[TN];
#pragma unroll
                for (int c = 0; c < TM / 4; ++c) {
                    const float4 v = *reinterpret_cast<const float4*>(&As[k][c * RM + ty * 4]);
                    a[c * 4] = v.x, a[c * 4 + 1] = v.y, a[c * 4 + 2] = v.z, a[c * 4 + 3] = v.w;
                }
#pragma unroll
                for (int c = 0; c < TN / 4; ++c) {
                    const float4 v = *reinterpret_cast<const float4*>(&Ws[k][c * RN + tx * 4]);
                    bb[c * 4] = v.x, bb[c * 4 + 1] = v.y, bb[c * 4 + 2] = v.z, bb[c * 4 + 3] = v.w;
                }
#pragma unroll
                for (int r = 0; r < TM; ++r)
#pragma unroll
                    for (int s = 0; s < TN; ++s) acc[r][s] = fmaf(a[r], bb[s], acc[r][s]);
            }
            __syncthreads();                     // everyone is done reading this step's tiles
            if (next) store(jn);
            __syncthreads();
        }
    }

    // Epilogue: + bias, round to bf16, 4 outputs (8 bytes) per store where they are all inside N.
    const bool vec = (N & 3) == 0;
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int m = m0 + (r >> 2) * RM + ty * 4 + (r & 3);
        if (m >= M) continue;
#pragma unroll
        for (int c = 0; c < TN / 4; ++c) {
            const int n = n0 + c * RN + tx * 4;
            float o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                o[e] = acc[r][c * 4 + e] + (bias != nullptr && n + e < N ? __bfloat162float(bias[n + e]) : 0.f);
            __nv_bfloat16* yp = y + int64_t(m) * N + n;
            if (vec && n + 3 < N) {
                __nv_bfloat162 p0 = __floats2bfloat162_rn(o[0], o[1]), p1 = __floats2bfloat162_rn(o[2], o[3]);
                uint2 u;
                u.x = *reinterpret_cast<uint32_t*>(&p0);
                u.y = *reinterpret_cast<uint32_t*>(&p1);
                *reinterpret_cast<uint2*>(yp) = u;
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (n + e < N) yp[e] = __float2bfloat16(o[e]);
            }
        }
    }
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

// The arguments every GEMM takes, checked; returns the bias pointer (nullptr: none). The kernels read x and the packed
// weight with vector loads (8 and 16 bytes; rows are 16-byte multiples), so both base addresses must be 16-byte
// aligned: a view that starts mid-allocation is rejected here rather than faulting or reading the wrong bytes.
namespace {

const __nv_bfloat16* check_gemm(const torch::Tensor& x, const torch::Tensor& w,
                                const c10::optional<torch::Tensor>& bias) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
                "x must be a contiguous CUDA bf16 [M, in] tensor");
    check_packed(w, x.size(1));
    TORCH_CHECK(w.device() == x.device(), "x and the weight are on different devices");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned (vector loads)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "the packed weight must be 16-byte aligned (vector loads)");
    TORCH_CHECK(x.size(0) <= INT32_MAX && w.size(0) <= INT32_MAX, "M and out_features must fit in int32");
    if (!(bias.has_value() && bias->defined())) return nullptr;
    TORCH_CHECK(bias->device() == x.device() && bias->scalar_type() == torch::kBFloat16 && bias->dim() == 1 &&
                    bias->size(0) == w.size(0) && bias->is_contiguous(),
                "bias must be a contiguous bf16 [out] tensor on x's device");
    return reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
}

}  // namespace

torch::Tensor q4k_gemm_naive(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias) {
    const __nv_bfloat16* b = check_gemm(x, w, bias);
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
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

torch::Tensor q4k_gemm_tiled(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias) {
    const __nv_bfloat16* b = check_gemm(x, w, bias);
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    const int64_t tiles = (M + TILE_M - 1) / TILE_M * ((N + TILE_N - 1) / TILE_N);
    TORCH_CHECK(tiles <= INT32_MAX, "too many tiles for the tiled grid");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return y;
    gemm_tiled_kernel<<<unsigned(tiles), TILED_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(), b,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), int(M), int(N), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
