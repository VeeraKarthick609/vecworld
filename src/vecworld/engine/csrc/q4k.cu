// Q4_K kernels. Stage 1 (Task 3): the naive correctness reference. Stage 2 (Task 4): tiled through shared memory.
// Stage 3 (Task 5): tensor cores (bf16 mma.sync), decoding Q4_K into the B fragments in registers.
// Stage 4 (Task 5b): W4A8, per-token INT8 activations x the Q4_K codes on the INT8 tensor cores (mma.sync s8).
// Task 5c: a GEMV for M <= 8 rows (the modulation linears), streaming the Q4_K bytes once at memory bandwidth.
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

#include "w4a8_quant.cuh"

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

// Tensor-core GEMM (Task 5): y = x W^T (+ bias) on bf16 mma.sync.m16n8k16, the weights decoded from Q4_K in
// registers. A thread block computes a BM x BN tile of y; each warp a WM x WN piece of it. K advances BK = 64
// weights per stage (one 32-byte run of a super-block's codes: sub-blocks 2r and 2r+1).
//
//   global --cp.async (16 B)--> shared, STAGES deep:  A stage: x[m0 .. m0+BM, k .. k+64]  bf16, 128 B a row
//                                                     W stage: per weight row, the super-block header (16 B: d,
//                                                              dmin, scales) and run r (32 B), still packed
//   shared --ldmatrix.x4--> A fragments (registers)
//   shared --2 x 16-bit loads--> 4 code bytes --decode (exact gguf fp32, RN to bf16)--> B fragments (registers)
//   mma.sync m16n8k16: A (16 x 16 bf16) x B (16 x 8 bf16) += C (16 x 8 fp32), per warp
//
// The B fragment of m16n8k16 gives lane (g = lane / 4, t = lane % 4) column n = g at k = 2t, 2t+1, 2t+8, 2t+9 (of
// the 16). In a 64-weight run, byte i holds k = i (low nibble) and k = 32 + i (high nibble), so the bytes
// {2t, 2t+1, 2t+8, 2t+9} feed k16 step 0 (low nibbles) and step 2 (high), and the same bytes + 16 steps 1 and 3:
// two 16-bit shared loads give a lane the codes of two k16 steps, and all four k values of a fragment fall in one
// 32-weight sub-block, so one (d * sc, dmin * m) pair decodes them. No repacking: the kernel reads the GGUF layout.
// A row's 48 shared bytes put 8 rows (g) x 2 words (t) on 16 different banks: conflict-free. The x tile's 16-byte
// chunks are XOR-swizzled by row (chunk ^ row % 8), so ldmatrix's 8 row reads hit 8 different bank groups.
//
// Rows past M or N are zero-filled by cp.async (zero header: d = dmin = 0, so their weights decode to 0).
namespace mma {

constexpr int BK = 64;                       // weights per stage: one run (two sub-blocks)
constexpr int A_ROW = BK * 2;                // bytes per x row per stage (8 chunks of 16)
constexpr int W_ROW = 48;                    // bytes per weight row per stage: header 16 + run 32

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes global -> shared without passing through registers; src_bytes = 0 fills the 16 bytes with zeros.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Sub-blocks 2r and 2r+1's multipliers from a super-block header held in registers (h.x = d | dmin, h.y..h.w =
// scales[12]); r is the run (0..3). The same bit layout as scale_min, specialised to a pair of sub-blocks: their
// bytes sit next to each other, so one shift by (r & 1) * 16 lines both up.
__device__ __forceinline__ void run_scales(uint4 h, int r, SubScale& lo, SubScale& hi) {
    const int sh = (r & 1) * 16;
    const uint32_t Y = h.y >> sh, Z = h.z >> sh, Wd = h.w >> sh;
    uint32_t sc0, sc1, m0, m1;
    if (r < 2) {
        sc0 = Y & 63, sc1 = (Y >> 8) & 63, m0 = Z & 63, m1 = (Z >> 8) & 63;
    } else {
        sc0 = (Wd & 0xF) | ((Y >> 2) & 0x30), sc1 = ((Wd >> 8) & 0xF) | ((Y >> 10) & 0x30);
        m0 = ((Wd >> 4) & 0xF) | ((Z >> 2) & 0x30), m1 = ((Wd >> 12) & 0xF) | ((Z >> 10) & 0x30);
    }
    const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(h.x >> 16)));
    lo = {__fmul_rn(d, float(sc0)), __fmul_rn(dmin, float(m0))};
    hi = {__fmul_rn(d, float(sc1)), __fmul_rn(dmin, float(m1))};
}

// 4-bit code -> exact float without an int->float conversion: 0x4B000000 | q is the float 2^23 + q.
__device__ __forceinline__ float code_f(uint32_t v, uint32_t sel) {
    return __uint_as_float(__byte_perm(v, 0x4B000000u, sel)) - 8388608.f;
}

// decode() on a code already in a float (the same two roundings).
__device__ __forceinline__ float decode_qf(SubScale s, float q) { return __fsub_rn(__fmul_rn(s.d_sc, q), s.dm); }

// One B fragment (two bf16x2 registers) from 4 codes (bytes of v: k = 2t, 2t+1, 2t+8, 2t+9, already masked to
// 0..15): each weight decoded exactly as gguf-py does in fp32, then rounded to bf16.
__device__ __forceinline__ void decode_frag(uint32_t v, SubScale s, uint32_t& b0, uint32_t& b1) {
    const float w[4] = {decode_qf(s, code_f(v, 0x7440)), decode_qf(s, code_f(v, 0x7441)),
                        decode_qf(s, code_f(v, 0x7442)), decode_qf(s, code_f(v, 0x7443))};
    __nv_bfloat162 p0 = __floats2bfloat162_rn(w[0], w[1]), p1 = __floats2bfloat162_rn(w[2], w[3]);
    b0 = *reinterpret_cast<uint32_t*>(&p0);
    b1 = *reinterpret_cast<uint32_t*>(&p1);
}

template <int BM, int BN, int WM, int WN, int STAGES, int GROUP, int MIN_BLOCKS>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MIN_BLOCKS)
    gemm_mma_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, int M, int N, int K) {
    constexpr int WARPS_N = BN / WN, T = (BM / WM) * WARPS_N * 32;
    constexpr int MI = WM / 16, NI = WN / 8;            // m16 and n8 fragments per warp
    constexpr int A_STAGE = BM * A_ROW, W_STAGE = BN * W_ROW;
    constexpr int A_CHUNKS = BM * (A_ROW / 16), W_CHUNKS = BN * 3;
    static_assert(A_CHUNKS % T == 0, "x chunks per thread");
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Ws = smem + STAGES * A_STAGE;

    // Grouped tile order: GROUP M tiles walk along N together, so blocks running at once share x and W in L2.
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, L = blockIdx.x;
    const int first = L / (GROUP * num_n) * GROUP, gs = min(num_m - first, GROUP);
    const int m0 = (first + L % (GROUP * num_n) % gs) * BM, n0 = L % (GROUP * num_n) / gs * BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N, g = lane >> 2, t = lane & 3;
    const int64_t row_bytes = int64_t(K / QK_K) * BLOCK_BYTES;
    const int KT = K / BK;

    auto load = [&](int kt, int st) {           // stage st <- K step kt (super-block kt / 4, run kt % 4)
        uint8_t* a = As + st * A_STAGE;
#pragma unroll
        for (int i = 0; i < A_CHUNKS / T; ++i) {
            const int idx = tid + i * T, r = idx >> 3, c = idx & 7;
            const bool ok = m0 + r < M;
            const __nv_bfloat16* src = ok ? x + int64_t(m0 + r) * K + kt * BK + c * 8 : x;
            cp_async16(a + r * A_ROW + ((c ^ (r & 7)) << 4), src, ok ? 16 : 0);
        }
        uint8_t* ws = Ws + st * W_STAGE;
        const int64_t boff = int64_t(kt >> 2) * BLOCK_BYTES, roff = 16 + (kt & 3) * 32;
#pragma unroll
        for (int i = 0; i < (W_CHUNKS + T - 1) / T; ++i) {
            const int idx = tid + i * T;
            if (idx < W_CHUNKS) {
                const int r = idx / 3, part = idx % 3;
                const bool ok = n0 + r < N;
                const uint8_t* src = ok ? w + int64_t(n0 + r) * row_bytes + boff + (part ? roff + (part - 1) * 16 : 0) : w;
                cp_async16(ws + r * W_ROW + part * 16, src, ok ? 16 : 0);
            }
        }
    };

    float acc[MI][NI][4] = {};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();                         // stage kt landed for everyone; stage kt - 1 is free again
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_async_commit();

        const int st = kt % STAGES, r = kt & 3;
        const uint8_t* a = As + st * A_STAGE;
        const uint8_t* ws = Ws + st * W_STAGE;
        SubScale sc[NI][2];
#pragma unroll
        for (int j = 0; j < NI; ++j)
            run_scales(*reinterpret_cast<const uint4*>(ws + (wn * WN + j * 8 + g) * W_ROW), r, sc[j][0], sc[j][1]);
#pragma unroll
        for (int p = 0; p < 2; ++p) {            // bytes 16p .. 16p + 15 of the run
            uint32_t q[NI];
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const uint8_t* run = ws + (wn * WN + j * 8 + g) * W_ROW + 16 + 16 * p + 2 * t;
                q[j] = uint32_t(*reinterpret_cast<const uint16_t*>(run)) |
                       (uint32_t(*reinterpret_cast<const uint16_t*>(run + 8)) << 16);
            }
#pragma unroll
            for (int nib = 0; nib < 2; ++nib) {  // low nibbles: sub-block 2r, k16 step p; high: 2r+1, step p + 2
                const int s = p + 2 * nib;
                uint32_t b[NI][2];
#pragma unroll
                for (int j = 0; j < NI; ++j)
                    decode_frag((q[j] >> (4 * nib)) & 0x0F0F0F0Fu, sc[j][nib], b[j][0], b[j][1]);
                uint32_t af[MI][4];
#pragma unroll
                for (int i = 0; i < MI; ++i) {
                    const int row = wm * WM + i * 16 + (lane & 15), chunk = 2 * s + (lane >> 4);
                    ldmatrix_x4(af[i], a + row * A_ROW + ((chunk ^ (row & 7)) << 4));
                }
#pragma unroll
                for (int i = 0; i < MI; ++i)
#pragma unroll
                    for (int j = 0; j < NI; ++j) mma_bf16(acc[i][j], af[i], b[j][0], b[j][1]);
            }
        }
    }
    cp_async_wait<0>();

    // Epilogue: + bias, round to bf16, a pair of outputs (4 bytes) per store where both are inside N.
    const bool pair_ok = (N & 1) == 0;
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const int n = n0 + wn * WN + j * 8 + 2 * t;
        const float b0 = bias != nullptr && n < N ? __bfloat162float(bias[n]) : 0.f;
        const float b1 = bias != nullptr && n + 1 < N ? __bfloat162float(bias[n + 1]) : 0.f;
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + wm * WM + i * 16 + g + 8 * h;
                if (m >= M) continue;
                const float o0 = acc[i][j][2 * h] + b0, o1 = acc[i][j][2 * h + 1] + b1;
                __nv_bfloat16* yp = y + int64_t(m) * N + n;
                if (pair_ok && n + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(yp) = __floats2bfloat162_rn(o0, o1);
                } else {
                    if (n < N) yp[0] = __float2bfloat16(o0);
                    if (n + 1 < N) yp[1] = __float2bfloat16(o1);
                }
            }
    }
}

}  // namespace mma

// W4A8 (Task 5b): the activations quantized to INT8 per token, the Q4_K codes used as INT8 as they are, on the INT8
// tensor cores (mma.sync.m16n8k32 s8 x s8 -> s32). One instruction's k = 32 is exactly one Q4_K sub-block, so for
// token m, weight row n and sub-block j (x_i = s_m * xq_i, w_i = (d * sc_j) * q_i - dmin * m_j):
//
//   sum_i x_i w_i = s_m * [ (d * sc_j) * sum_i xq_i q_i  -  (dmin * m_j) * xsum_j ]      xsum_j = sum_i xq_i
//                                   \__ s32, one mma __/                 \__ from quantize_act __/
//
// summed over the sub-blocks in fp32, times s_m in the epilogue (docs/engine/04-w4a8.md).
//
// The tiling is the mma kernel's (K step BK = 64: one run, two sub-blocks; cp.async, 2 stages; grouped tile order;
// zero-filled ragged edges; warps side by side along N). What changes:
//   - x is int8: a stage row is 64 bytes (4 chunks of 16), XOR-swizzled by (row / 2) % 4 so ldmatrix's 8 rows hit
//     all 32 banks. ldmatrix.x4 on a 16 x 32 int8 tile hands out exactly the s8 A fragment (PTX ISA, "Matrix
//     Fragments for mma.m16n8k32": a lane (g, t) holds rows g and g + 8 at k = 4t .. 4t+3 and 16 + 4t .. 16 + 4t+3,
//     the same 16-byte pieces as the bf16 m16n8k16 tile, which is also 16 rows x 32 bytes).
//   - B needs no decode: its s8 fragment is column g at k = 4t .. 4t+3 and 16 + 4t .. 16 + 4t+3, i.e. words t and
//     4 + t of the 32-byte run, low nibbles for sub-block 2r, high nibbles for 2r + 1: two 32-bit loads and masks.
//   - The per-sub-block rescale works on the accumulator fragment (rows g, g + 8; columns 2t, 2t + 1), so each lane
//     unpacks the scales of its own 2 columns per n8 tile, and reads xsum for its rows (a stage also holds xsum).
//   - The int32 -> fp32 conversion is free: the mma's C input is 0x4B400000 (the float 2^23 + 2^22), so D's bits are
//     that float plus the integer sum (|sum| <= 32 * 127 * 15 < 2^22), and one subtract gives it exactly.
namespace w4a8 {

using mma::cp_async16;
using mma::cp_async_commit;
using mma::cp_async_wait;
using mma::ldmatrix_x4;
using mma::smem_addr;

constexpr int BK = 64;                       // weights per stage: one run (two sub-blocks)
constexpr int A_ROW = BK;                    // bytes per x row per stage (int8; 4 chunks of 16)
constexpr int W_ROW = 48;                    // header 16 + run 32, as in mma
constexpr int XS_ROW = 8;                    // the row's two sub-block sums, int32
constexpr int MAGIC = 0x4B400000;            // the float 2^23 + 2^22
constexpr float MAGIC_F = 12582912.f;

__device__ __forceinline__ void cp_async8(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(src_bytes));
}

// D = A (16 x 32 s8) B (32 x 8 s8) + C, s32.
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}

template <int BM, int BN, int WM, int WN, int STAGES, int GROUP, int MIN_BLOCKS, typename OutT>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MIN_BLOCKS)
    gemm_w4a8_kernel(const int8_t* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ xsum,
                     const uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ bias, OutT* __restrict__ y,
                     int M, int N, int K) {
    constexpr int WARPS_N = BN / WN, T = (BM / WM) * WARPS_N * 32;
    constexpr int MI = WM / 16, NI = WN / 8;
    constexpr int A_STAGE = BM * A_ROW, W_STAGE = BN * W_ROW, XS_STAGE = BM * XS_ROW;
    constexpr int A_CHUNKS = BM * (A_ROW / 16), W_CHUNKS = BN * 3;
    static_assert(A_CHUNKS % T == 0, "x chunks per thread");
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Ws = smem + STAGES * A_STAGE;
    uint8_t* Xs = Ws + STAGES * W_STAGE;

    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, L = blockIdx.x;
    const int first = L / (GROUP * num_n) * GROUP, gs = min(num_m - first, GROUP);
    const int m0 = (first + L % (GROUP * num_n) % gs) * BM, n0 = L % (GROUP * num_n) / gs * BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N, g = lane >> 2, t = lane & 3;
    const int64_t row_bytes = int64_t(K / QK_K) * BLOCK_BYTES;
    const int KT = K / BK, KS = K / 32;

    auto load = [&](int kt, int st) {
        uint8_t* a = As + st * A_STAGE;
#pragma unroll
        for (int i = 0; i < A_CHUNKS / T; ++i) {
            const int idx = tid + i * T, r = idx >> 2, c = idx & 3;
            const bool ok = m0 + r < M;
            const int8_t* src = ok ? xq + int64_t(m0 + r) * K + kt * BK + c * 16 : xq;
            cp_async16(a + r * A_ROW + ((c ^ ((r >> 1) & 3)) << 4), src, ok ? 16 : 0);
        }
        uint8_t* ws = Ws + st * W_STAGE;
        const int64_t boff = int64_t(kt >> 2) * BLOCK_BYTES, roff = 16 + (kt & 3) * 32;
#pragma unroll
        for (int i = 0; i < (W_CHUNKS + T - 1) / T; ++i) {
            const int idx = tid + i * T;
            if (idx < W_CHUNKS) {
                const int r = idx / 3, part = idx % 3;
                const bool ok = n0 + r < N;
                const uint8_t* src = ok ? w + int64_t(n0 + r) * row_bytes + boff + (part ? roff + (part - 1) * 16 : 0) : w;
                cp_async16(ws + r * W_ROW + part * 16, src, ok ? 16 : 0);
            }
        }
#pragma unroll
        for (int i = 0; i < (BM + T - 1) / T; ++i) {
            const int r = tid + i * T;
            if (r < BM) {
                const bool ok = m0 + r < M;
                const int* src = ok ? xsum + int64_t(m0 + r) * KS + kt * 2 : xsum;
                cp_async8(Xs + st * XS_STAGE + r * XS_ROW, src, ok ? 8 : 0);
            }
        }
    };

    float acc[MI][NI][4] = {};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_async_commit();

        const int st = kt % STAGES, r = kt & 3;
        const uint8_t* a = As + st * A_STAGE;
        const uint8_t* ws = Ws + st * W_STAGE;
        const int* xst = reinterpret_cast<const int*>(Xs + st * XS_STAGE);
        // This lane's accumulator columns 2t, 2t + 1 of each n8 tile: their (d * sc, dmin * m) for both sub-blocks.
        SubScale sc[NI][2][2];                   // [n8 tile][column 2t + e][sub-block 2r + nib]
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                mma::run_scales(*reinterpret_cast<const uint4*>(ws + (wn * WN + j * 8 + 2 * t + e) * W_ROW), r,
                                sc[j][e][0], sc[j][e][1]);
        // B fragments (column g of each n8 tile): words t and 4 + t of the run, low nibbles then high nibbles.
        uint32_t b[NI][2][2];
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const uint32_t* run = reinterpret_cast<const uint32_t*>(ws + (wn * WN + j * 8 + g) * W_ROW + 16);
            const uint32_t v0 = run[t], v1 = run[4 + t];
            b[j][0][0] = v0 & 0x0F0F0F0Fu, b[j][0][1] = v1 & 0x0F0F0F0Fu;
            b[j][1][0] = (v0 >> 4) & 0x0F0F0F0Fu, b[j][1][1] = (v1 >> 4) & 0x0F0F0F0Fu;
        }
#pragma unroll
        for (int i = 0; i < MI; ++i) {
            const int rb = wm * WM + i * 16 + g;
            const int2 s0 = *reinterpret_cast<const int2*>(xst + rb * 2);           // row g: both sub-blocks
            const int2 s1 = *reinterpret_cast<const int2*>(xst + (rb + 8) * 2);     // row g + 8
            const float xsf[2][2] = {{float(s0.x), float(s1.x)}, {float(s0.y), float(s1.y)}};   // [nib][row half]
#pragma unroll
            for (int nib = 0; nib < 2; ++nib) {
                uint32_t af[4];
                const int row = wm * WM + i * 16 + (lane & 15), chunk = 2 * nib + (lane >> 4);
                ldmatrix_x4(af, a + row * A_ROW + ((chunk ^ ((row >> 1) & 3)) << 4));
#pragma unroll
                for (int j = 0; j < NI; ++j) {
                    int d[4];
                    mma_s8(d, af, b[j][nib][0], b[j][nib][1], MAGIC);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const SubScale s = sc[j][e & 1][nib];
                        const float q = __int_as_float(d[e]) - MAGIC_F;       // the exact int32 sum
                        acc[i][j][e] = fmaf(s.d_sc, q, acc[i][j][e]);
                        acc[i][j][e] = fmaf(-s.dm, xsf[nib][e >> 1], acc[i][j][e]);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();

    // Epilogue: x token scale s_m, + bias, store (bf16 pairs, or fp32 for the tests).
    const bool pair_ok = (N & 1) == 0;
#pragma unroll
    for (int j = 0; j < NI; ++j) {
        const int n = n0 + wn * WN + j * 8 + 2 * t;
        const float b0 = bias != nullptr && n < N ? __bfloat162float(bias[n]) : 0.f;
        const float b1 = bias != nullptr && n + 1 < N ? __bfloat162float(bias[n + 1]) : 0.f;
#pragma unroll
        for (int i = 0; i < MI; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = m0 + wm * WM + i * 16 + g + 8 * h;
                if (m >= M) continue;
                const float sm = xs[m];
                const float o0 = fmaf(acc[i][j][2 * h], sm, b0), o1 = fmaf(acc[i][j][2 * h + 1], sm, b1);
                OutT* yp = y + int64_t(m) * N + n;
                if constexpr (std::is_same_v<OutT, float>) {
                    if (n < N) yp[0] = o0;
                    if (n + 1 < N) yp[1] = o1;
                } else if (pair_ok && n + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(yp) = __floats2bfloat162_rn(o0, o1);
                } else {
                    if (n < N) yp[0] = __float2bfloat16(o0);
                    if (n + 1 < N) yp[1] = __float2bfloat16(o1);
                }
            }
    }
}

// quantize_act: one block per token. s = max|x| / 127 (IEEE division), xq = rint(x / s) (half to even, as
// torch.round), clamped to +-127; a zero token gets s = 0, xq = 0. xsum[m, j] = sum of xq over sub-block j (32
// values = 4 consecutive 8-value chunks, held by 4 neighbouring lanes). A row of up to QT * QREG * 8 = 16384 values
// (every FLUX linear) is read once into registers; a longer one is read twice.
// max|x| is taken on the bf16 bit patterns of |x| (an integer max): the same value for finite x, but an inf or NaN
// wins instead of being dropped (fmaxf drops NaN). Such a token gets s = NaN and xq = 0, so its outputs come out NaN,
// as on the bf16 paths, rather than as quietly clamped numbers.
constexpr int QT = 256, QREG = 8;

using vw_w4a8::amax8;
using vw_w4a8::quantize8;

__global__ void __launch_bounds__(QT) quantize_kernel(const __nv_bfloat16* __restrict__ x, int8_t* __restrict__ xq,
                                                      float* __restrict__ s_out, int* __restrict__ xsum, int K) {
    const int m = blockIdx.x, tid = threadIdx.x;
    const uint4* xr = reinterpret_cast<const uint4*>(x + int64_t(m) * K);
    const int chunks = K / 8;
    const bool regs = chunks <= QT * QREG;
    uint4 v[QREG];
    uint32_t amax = 0;                           // bf16 bits of max |x|
    if (regs) {
#pragma unroll
        for (int i = 0; i < QREG; ++i) {
            const int c = tid + i * QT;
            v[i] = c < chunks ? xr[c] : make_uint4(0, 0, 0, 0);
            amax = amax8(v[i], amax);
        }
    } else {
        for (int c = tid; c < chunks; c += QT) amax = amax8(xr[c], amax);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = max(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    __shared__ uint32_t red[QT / 32];
    if ((tid & 31) == 0) red[tid >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int i = 1; i < QT / 32; ++i) amax = max(amax, red[i]);
    const float s = amax >= 0x7F80u ? __int_as_float(0x7FC00000) : __fdiv_rn(__uint_as_float(amax << 16), 127.f);
    if (tid == 0) s_out[m] = s;
    int8_t* qr = xq + int64_t(m) * K;
    int* sr = xsum + int64_t(m) * (K / 32);
    if (regs) {
#pragma unroll
        for (int i = 0; i < QREG; ++i) {
            const int c = tid + i * QT;
            if (i * QT < chunks) quantize8(c, c < chunks, v[i], s, qr, sr);   // uniform across the block
        }
    } else {
        for (int c0 = 0; c0 < chunks; c0 += QT) {
            const int c = c0 + tid;
            quantize8(c, c < chunks, c < chunks ? xr[c] : make_uint4(0, 0, 0, 0), s, qr, sr);
        }
    }
}

}  // namespace w4a8

// GEMV (Task 5c): y = x W^T (+ bias) for M <= MAX_M = 8 rows of x: the modulation linears, which see one row per
// image (a pass holds at most 8 windows). At M = 1 a linear does 2 FLOPs per weight, about 3.6 FLOPs per byte of
// Q4_K, far below the ridge of the roofline: the time is the time to read the weight once, so the kernel's only job
// is to stream the packed bytes at full bandwidth, each byte read once, and to keep the rest out of the way
// (docs/engine/05-modulation-gemv.md). The tensor cores don't help: a 16-row mma tile would be 15/16 padding.
//
//   a warp per weight row (= output column n), ROWS rows per warp; K walks in chunks of KC = 1024 = 4 super-blocks:
//
//     lane l = 8 * grp + part:  super-block grp (0..3) of the chunk, 16-byte piece `part` (0..7) of its codes
//                               (run g = part / 2, half h = part % 2), plus the super-block's 16-byte header
//     global --128-bit loads, one chunk ahead--> registers --decode (exact gguf fp32)--> FMA with x from shared
//
//   x (bf16, M rows x 1024 of the chunk) is staged in shared memory by the whole block, once per chunk, and read by
//   all its warps. Piece `part` holds weights k = 64 g + 16 h + i (low nibbles, sub-block 2g) and k + 32 (high
//   nibbles, 2g + 1), i = 0..15: two runs of 16 x values per row of x.
//   The 8 lanes of a quarter-warp (one 128-bit shared load phase) read 8 different (g, h); unswizzled their x pieces
//   would sit on only 2 of the 8 16-byte bank groups (4-way conflicts), so pieces are XOR-swizzled (swz;
//   measured on 18432 x 3072: M = 1 154 -> 173 GB/s, M = 8 63 -> 79 GB/s).
//   Each lane sums its 32 weights x M rows in fp32; a warp-shuffle reduction adds the 32 lanes; lane m writes row m.
// The super-blocks of a row are contiguous, so a warp's 32 code loads cover 4 x 128 bytes of one row; the headers
// are read by 8 lanes each (one request). Rows past N and super-blocks past K are skipped.
namespace gemv {

constexpr int MAX_M = 8;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int KC = 4 * QK_K;                 // weights per chunk: 4 super-blocks, 8 lanes each
constexpr int PIECES = KC / 8;               // 16-byte pieces of x per row per chunk

// Piece c = 32 grp + 8 g + 4 f + 2 h + e (f: high-nibble run +32 weights, e: second 8 of 16) -> its slot. Within a
// quarter-warp grp, e and f are fixed and (g, h) take all 8 values; c ^ (g << 1) ^ h maps them to 8 different
// 16-byte bank groups (bit 0 = e ^ h, bit 1 = h ^ g0, bit 2 = f ^ g1), and it is a bijection on each 8 pieces.
__device__ __forceinline__ int swz(int c) { return c ^ (((c >> 3) & 3) << 1) ^ ((c >> 1) & 1); }

__device__ __forceinline__ void bf16x8(uint4 v, float* f) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(&u[i]);
        f[2 * i] = __low2float(p);
        f[2 * i + 1] = __high2float(p);
    }
}

template <int M, int ROWS>
__global__ void __launch_bounds__(THREADS)
    gemv_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y, int N, int K) {
    __shared__ uint4 xs[M * PIECES];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = lane >> 3, part = lane & 7, g = part >> 1, h = part & 1;
    const int nb = K / QK_K, chunks = (nb + 3) / 4;
    const int n0 = (blockIdx.x * WARPS + warp) * ROWS;
    const int64_t row_bytes = int64_t(nb) * BLOCK_BYTES;

    uint4 hd[ROWS], qs[ROWS];
    auto load_w = [&](int c) {                // this lane's header and code piece of chunk c, for each row
        const int sb = c * 4 + grp;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            hd[r] = qs[r] = make_uint4(0, 0, 0, 0);
            if (n0 + r < N && sb < nb) {
                const uint8_t* blk = w + (n0 + r) * row_bytes + int64_t(sb) * BLOCK_BYTES;
                hd[r] = __ldg(reinterpret_cast<const uint4*>(blk));
                qs[r] = __ldg(reinterpret_cast<const uint4*>(blk + 16 + part * 16));
            }
        }
    };

    float acc[ROWS][M] = {};
    load_w(0);
    for (int c = 0; c < chunks; ++c) {
        if (c) __syncthreads();               // every warp is done with the previous chunk's x
        for (int i = threadIdx.x; i < M * PIECES; i += THREADS) {
            const int m = i / PIECES, p = i % PIECES, k = c * KC + p * 8;
            xs[m * PIECES + swz(p)] =
                k < K ? *reinterpret_cast<const uint4*>(x + int64_t(m) * K + k) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        uint4 hc[ROWS], qc[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) hc[r] = hd[r], qc[r] = qs[r];
        if (c + 1 < chunks) load_w(c + 1);    // the next chunk's bytes are in flight while this one computes
        if (n0 >= N || c * 4 + grp >= nb) continue;
        const int base = grp * 32 + g * 8 + h * 2;
#pragma unroll
        for (int f = 0; f < 2; ++f) {         // f = 0: low nibbles, sub-block 2g; f = 1: high, 2g + 1
            float wv[ROWS][16];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const SubScale s = sub_scale_regs(hc[r], 2 * g + f);
                const uint32_t v[4] = {qc[r].x, qc[r].y, qc[r].z, qc[r].w};
#pragma unroll
                for (int i = 0; i < 16; ++i) wv[r][i] = decode(s, (v[i >> 2] >> (8 * (i & 3) + 4 * f)) & 0x0F);
            }
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float xv[8];
                    bf16x8(xs[m * PIECES + swz(base + 4 * f + e)], xv);
#pragma unroll
                    for (int r = 0; r < ROWS; ++r)
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[r][m] = fmaf(wv[r][e * 8 + i], xv[i], acc[r][m]);
                }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[r][m] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][m], o);
    if (lane >= M) return;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int n = n0 + r;
        if (n >= N) break;
        float o = 0.f;
#pragma unroll
        for (int m = 0; m < M; ++m)
            if (lane == m) o = acc[r][m];     // a static index: acc stays in registers
        if (bias != nullptr) o += __bfloat162float(bias[n]);
        y[int64_t(lane) * N + n] = __float2bfloat16(o);
    }
}

}  // namespace gemv


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

// Two tile shapes (docs/engine/03-q4k-mma.md): 128 x 128 when there are enough tiles to fill the GPU many times over,
// 128 x 64 when there are few (M = 256: 2 M tiles), where the last wave of 128 x 128 tiles would leave SMs idle. Both
// put their warps (8 or 4) side by side along N, each warp 128 rows x 16 columns, so every weight is decoded once per
// block. Two pipeline stages; 2 blocks share an SM in both (44 KB and 38 KB of shared memory each, of 100 KB).
namespace {

template <int BM, int BN, int WM, int WN, int STAGES, int GROUP, int MIN_BLOCKS>
void launch_mma(const torch::Tensor& x, const torch::Tensor& w, const __nv_bfloat16* b, torch::Tensor& y) {
    const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
    auto kern = mma::gemm_mma_kernel<BM, BN, WM, WN, STAGES, GROUP, MIN_BLOCKS>;
    constexpr int smem = STAGES * (BM * mma::A_ROW + BN * mma::W_ROW);
    static_assert(smem <= 48 * 1024, "more dynamic shared memory needs cudaFuncSetAttribute");
    const int64_t tiles = (M + BM - 1) / BM * ((N + BN - 1) / BN);
    TORCH_CHECK(tiles <= INT32_MAX, "too many tiles for the mma grid");
    kern<<<unsigned(tiles), (BM / WM) * (BN / WN) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(), b,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), int(M), int(N), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

int64_t q4k_mma_tile_n(int64_t M, int64_t N) {
    const int64_t big_tiles = (M + 127) / 128 * ((N + 127) / 128);
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return big_tiles >= 8 * sms ? 128 : 64;     // >= 4 waves of 2 blocks per SM: 128 x 128
}

torch::Tensor q4k_gemm_mma(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias, int64_t tile_n) {
    const __nv_bfloat16* b = check_gemm(x, w, bias);
    const int64_t M = x.size(0), N = w.size(0);
    TORCH_CHECK(tile_n == 0 || tile_n == 64 || tile_n == 128, "tile_n must be 0 (pick), 64 or 128, got ", tile_n);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return y;
    if (tile_n == 0) tile_n = q4k_mma_tile_n(M, N);
    if (tile_n == 128)
        launch_mma<128, 128, 128, 16, 2, 4, 2>(x, w, b, y);
    else
        launch_mma<128, 64, 128, 16, 2, 4, 3>(x, w, b, y);   // 3 only caps registers (170); 2 blocks fit per SM
    return y;
}

// W4A8 (Task 5b). quantize_act: bf16 x [M, K] -> (int8 xq [M, K], fp32 s [M], int32 xsum [M, K / 32]).
std::vector<torch::Tensor> q4k_quantize_act(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
                "x must be a contiguous CUDA bf16 [M, K] tensor");
    TORCH_CHECK(x.size(1) % QK_K == 0, "K must be a multiple of 256, got ", x.size(1));
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned (vector loads)");
    TORCH_CHECK(x.size(0) <= INT32_MAX, "M must fit in int32");
    const int64_t M = x.size(0), K = x.size(1);
    const c10::cuda::CUDAGuard guard(x.device());
    auto xq = torch::empty({M, K}, x.options().dtype(torch::kInt8));
    auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
    auto xsum = torch::empty({M, K / 32}, x.options().dtype(torch::kInt32));
    if (M == 0) return {xq, s, xsum};
    w4a8::quantize_kernel<<<unsigned(M), w4a8::QT, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), xq.data_ptr<int8_t>(), s.data_ptr<float>(),
        xsum.data_ptr<int>(), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {xq, s, xsum};
}

namespace {

template <int BM, int BN, int WM, int WN, int STAGES, int GROUP, int MIN_BLOCKS, typename OutT>
void launch_w4a8(const torch::Tensor& xq, const torch::Tensor& s, const torch::Tensor& xsum, const torch::Tensor& w,
                 const __nv_bfloat16* b, torch::Tensor& y) {
    const int64_t M = xq.size(0), K = xq.size(1), N = w.size(0);
    auto kern = w4a8::gemm_w4a8_kernel<BM, BN, WM, WN, STAGES, GROUP, MIN_BLOCKS, OutT>;
    constexpr int smem = STAGES * (BM * w4a8::A_ROW + BN * w4a8::W_ROW + BM * w4a8::XS_ROW);
    static_assert(smem <= 48 * 1024, "more dynamic shared memory needs cudaFuncSetAttribute");
    const int64_t tiles = (M + BM - 1) / BM * ((N + BN - 1) / BN);
    TORCH_CHECK(tiles <= INT32_MAX, "too many tiles for the w4a8 grid");
    kern<<<unsigned(tiles), (BM / WM) * (BN / WN) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
        xq.data_ptr<int8_t>(), s.data_ptr<float>(), xsum.data_ptr<int>(), w.data_ptr<uint8_t>(), b,
        reinterpret_cast<OutT*>(y.data_ptr()), int(M), int(N), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename OutT>
void dispatch_w4a8(int64_t tile_n, const torch::Tensor& xq, const torch::Tensor& s, const torch::Tensor& xsum,
                   const torch::Tensor& w, const __nv_bfloat16* b, torch::Tensor& y) {
    if (tile_n == 128)
        launch_w4a8<128, 128, 128, 16, 2, 4, 2, OutT>(xq, s, xsum, w, b, y);
    else
        launch_w4a8<128, 64, 128, 16, 2, 4, 3, OutT>(xq, s, xsum, w, b, y);
}

}  // namespace

// y = (xq * s) W^T (+ bias) on the INT8 tensor cores: bf16 [M, out] (fp32 with out_fp32, for the tests). tile_n as
// q4k_gemm_mma (0: q4k_mma_tile_n picks).
torch::Tensor q4k_gemm_w4a8(torch::Tensor xq, torch::Tensor s, torch::Tensor xsum, torch::Tensor w,
                            c10::optional<torch::Tensor> bias, int64_t tile_n, bool out_fp32) {
    TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == torch::kInt8 && xq.dim() == 2 && xq.is_contiguous(),
                "xq must be a contiguous CUDA int8 [M, in] tensor");
    const int64_t M = xq.size(0), K = xq.size(1), N = w.size(0);
    check_packed(w, K);
    TORCH_CHECK(w.device() == xq.device(), "xq and the weight are on different devices");
    TORCH_CHECK(s.device() == xq.device() && s.scalar_type() == torch::kFloat32 && s.dim() == 1 && s.size(0) == M &&
                    s.is_contiguous(), "s must be a contiguous fp32 [M] tensor on xq's device");
    TORCH_CHECK(xsum.device() == xq.device() && xsum.scalar_type() == torch::kInt32 && xsum.dim() == 2 &&
                    xsum.size(0) == M && xsum.size(1) == K / 32 && xsum.is_contiguous(),
                "xsum must be a contiguous int32 [M, in / 32] tensor on xq's device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(xq.data_ptr()) % 16 == 0, "xq must be 16-byte aligned (vector loads)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(xsum.data_ptr()) % 8 == 0, "xsum must be 8-byte aligned (vector loads)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
                "the packed weight must be 16-byte aligned (vector loads)");
    TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX, "M and out_features must fit in int32");
    const __nv_bfloat16* b = nullptr;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->device() == xq.device() && bias->scalar_type() == torch::kBFloat16 && bias->dim() == 1 &&
                        bias->size(0) == N && bias->is_contiguous(),
                    "bias must be a contiguous bf16 [out] tensor on xq's device");
        b = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
    }
    TORCH_CHECK(tile_n == 0 || tile_n == 64 || tile_n == 128, "tile_n must be 0 (pick), 64 or 128, got ", tile_n);
    const c10::cuda::CUDAGuard guard(xq.device());
    auto y = torch::empty({M, N}, xq.options().dtype(out_fp32 ? torch::kFloat32 : torch::kBFloat16));
    if (M == 0 || N == 0) return y;
    if (tile_n == 0) tile_n = q4k_mma_tile_n(M, N);
    if (out_fp32)
        dispatch_w4a8<float>(tile_n, xq, s, xsum, w, b, y);
    else
        dispatch_w4a8<__nv_bfloat16>(tile_n, xq, s, xsum, w, b, y);
    return y;
}

// GEMV (Task 5c): bf16 x [M <= 8, in] @ Q4_K W[out, in]^T (+ bias) -> bf16 [M, out], memory-bound, bf16 activations
// (exact, as naive). One instantiation per M, so the per-row accumulators are registers.
namespace {

template <int M>
void launch_gemv(const torch::Tensor& x, const torch::Tensor& w, const __nv_bfloat16* b, torch::Tensor& y) {
    // Two weight rows per warp from M = 2: each x value read from shared memory feeds both (3050 Ti, 18432 x 3072:
    // M = 4 98 -> 118 GB/s, M = 8 62 -> 79; four rows measured no better). M = 1 is at the memory roof with one.
    constexpr int ROWS = M >= 2 ? 2 : 1;
    const int64_t K = x.size(1), N = w.size(0);
    const int64_t rows_per_block = int64_t(gemv::WARPS) * ROWS;
    gemv::gemv_kernel<M, ROWS><<<unsigned((N + rows_per_block - 1) / rows_per_block), gemv::THREADS, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr<uint8_t>(), b,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), int(N), int(K));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor q4k_gemv(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias) {
    const __nv_bfloat16* b = check_gemm(x, w, bias);
    const int64_t M = x.size(0), N = w.size(0);
    TORCH_CHECK(M <= gemv::MAX_M, "q4k_gemv takes at most 8 rows of x (GEMV_MAX_M), got ", M);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return y;
    switch (M) {
        case 1: launch_gemv<1>(x, w, b, y); break;
        case 2: launch_gemv<2>(x, w, b, y); break;
        case 3: launch_gemv<3>(x, w, b, y); break;
        case 4: launch_gemv<4>(x, w, b, y); break;
        case 5: launch_gemv<5>(x, w, b, y); break;
        case 6: launch_gemv<6>(x, w, b, y); break;
        case 7: launch_gemv<7>(x, w, b, y); break;
        default: launch_gemv<8>(x, w, b, y); break;
    }
    return y;
}
