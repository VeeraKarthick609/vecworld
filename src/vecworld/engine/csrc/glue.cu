// Glue fusion (Task 6): the elementwise work between a FLUX block's GEMMs, one pass over each token row.
//
//   residual_norm_modulate:  res_out = res + gate * y          (skipped when y is absent: res_out is res)
//                            x_mod   = LayerNorm(res_out) * (1 + scale) + shift
//   gated_residual:          out     = res + gate * y          (a block's last residual, with no norm after it)
//
// gate / shift / scale are [B, H] rows, one per image (window), each broadcast over that image's L = M / B tokens.
// They are chunks of one modulation output ([B, 6H] or [B, 3H]), so their rows are strided; only the last dimension
// must be contiguous.
//
// Rounding follows the reference (diffusers in bf16) step by step, so the fused blocks are bit-identical to the eager
// ones: gate * y is rounded to bf16, then the sum; LayerNorm takes torch's own fp32 Welford statistics of the bf16
// sum, in torch's order (row_stats), and rounds rstd * (x - mean); 1 + scale is rounded, then the product, then the
// shift.
//
// With quantize, the same pass also emits W4A8's quantize_act outputs for x_mod (bf16 bits of the rounded x_mod,
// the same arithmetic: w4a8_quant.cuh), so the next INT8 linears skip their own read of x_mod.
//
//   one block (GT = 128 threads) per token row; a row of H <= GT * GV * 4 = 8192 values lives in registers:
//
//     res, y, gate --(8-byte loads, thread t: t, t + 128, ...)--> v = bf16(res + bf16(gate * y)) --> res_out
//                                       mean, var: Welford per thread, warp shuffle-down, 4-warp tree (torch's order)
//                                       rstd = rsqrt(var + eps)
//     scale, shift --------------------> x = bf16(bf16(bf16(rstd * (v - mean)) * bf16(1 + scale)) + shift) --> x_mod
//                                       [quantize: s = max|x| / 127, xq = rint(x / s), xsum per 32 values]
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "w4a8_quant.cuh"

namespace {

constexpr int GT = 128, GV = 16;    // threads per row (torch's 32 x 4); 4-value loads per thread: H <= 8192

__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ void unpack8(uint4 v, float* f) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 p = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[e]));
        f[2 * e] = p.x;
        f[2 * e + 1] = p.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint32_t u[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const __nv_bfloat162 p = __floats2bfloat162_rn(f[2 * e], f[2 * e + 1]);
        u[e] = *reinterpret_cast<const uint32_t*>(&p);
    }
    return make_uint4(u[0], u[1], u[2], u[3]);
}

struct RnmArgs {
    const __nv_bfloat16 *res, *y, *gate, *shift, *scale;
    int64_t gate_ld, shift_ld, scale_ld;     // row strides of the [B, H] modulation rows, in elements
    __nv_bfloat16 *res_out, *x_out;
    int8_t* xq;
    float* s;
    int* xsum;
    int H, L;                                // values per row; tokens per image
    float eps;
};

__device__ __forceinline__ void unpack4(uint2 v, float* f) {
    const float2 p0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
    const float2 p1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
    f[0] = p0.x; f[1] = p0.y; f[2] = p1.x; f[3] = p1.y;
}

__device__ __forceinline__ uint2 pack4(const float* f) {
    const __nv_bfloat162 p0 = __floats2bfloat162_rn(f[0], f[1]), p1 = __floats2bfloat162_rn(f[2], f[3]);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&p0), *reinterpret_cast<const uint32_t*>(&p1));
}

// torch's LayerNorm statistics, operation for operation (ATen layer_norm_kernel.cu, vectorized_layer_norm_kernel on
// a contiguous bf16 row: 4-value loads, 128 threads = 32 x 4 warps, Welford folded per thread in load order, a
// shuffle-down combine per warp, then a shared-memory tree over the 4 warps). Matching its order makes the fused
// LayerNorm bit-identical to torch's (0 mismatches in 12.6 M values, docs/engine/06-glue-fusion.md); a two-pass mean
// and variance flipped ~3 in a million last bits, which W4A8 then turned into a different sample.
struct Welford {
    float mean, sigma2, count;
};

__device__ __forceinline__ Welford welford_add(float val, Welford c) {
    const float delta = val - c.mean;
    const float nc = c.count + 1.f;
    const float nm = c.mean + delta * (1.f / nc);
    return {nm, c.sigma2 + delta * (val - nm), nc};
}

__device__ __forceinline__ Welford welford_combine(Welford a, Welford b) {
    const float count = a.count + b.count;
    if (!(count > 0.f)) return {0.f, 0.f, 0.f};
    const float coef = 1.f / count;
    const float na = a.count * coef, nb = b.count * coef;
    const float delta = b.mean - a.mean;
    return {na * a.mean + nb * b.mean, a.sigma2 + b.sigma2 + delta * delta * a.count * nb, count};
}

// mean and rstd of a row whose per-thread Welford states are wd (all GT threads call this).
__device__ __forceinline__ void row_stats(Welford wd, int H, float eps, float& mean, float& rstd) {
    __shared__ float ms[2 * (GT / 32)], cb[GT / 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        wd = welford_combine(wd, Welford{__shfl_down_sync(0xFFFFFFFFu, wd.mean, o),
                                         __shfl_down_sync(0xFFFFFFFFu, wd.sigma2, o),
                                         __shfl_down_sync(0xFFFFFFFFu, wd.count, o)});
#pragma unroll
    for (int off = GT / 64; off > 0; off /= 2) {
        if (lane == 0 && warp >= off && warp < 2 * off) {
            ms[2 * (warp - off)] = wd.mean;
            ms[2 * (warp - off) + 1] = wd.sigma2;
            cb[warp - off] = wd.count;
        }
        __syncthreads();
        if (lane == 0 && warp < off) wd = welford_combine(wd, Welford{ms[2 * warp], ms[2 * warp + 1], cb[warp]});
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ms[0] = wd.mean;
        ms[1] = wd.sigma2 / float(H);
    }
    __syncthreads();
    mean = ms[0];
    rstd = rsqrtf(ms[1] + eps);
}

template <bool HAS_Y, bool QUANT>
__global__ void __launch_bounds__(GT) rnm_kernel(RnmArgs a) {
    const int m = blockIdx.x, tid = threadIdx.x, vecs = a.H / 4;
    const int64_t b = m / a.L;
    const uint2* rr = reinterpret_cast<const uint2*>(a.res + int64_t(m) * a.H);
    uint2 v[GV];                              // the bf16 residual sum (res_out), 4 values per load, torch's layout
    Welford wd{0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < GV; ++i) {
        const int c = tid + i * GT;
        if (c >= vecs) break;
        uint2 r = rr[c];
        if constexpr (HAS_Y) {
            float fr[4], fy[4], fg[4];
            unpack4(r, fr);
            unpack4(reinterpret_cast<const uint2*>(a.y + int64_t(m) * a.H)[c], fy);
            unpack4(reinterpret_cast<const uint2*>(a.gate + b * a.gate_ld)[c], fg);
#pragma unroll
            for (int e = 0; e < 4; ++e) fr[e] = fr[e] + rb(fg[e] * fy[e]);
            r = pack4(fr);
            reinterpret_cast<uint2*>(a.res_out + int64_t(m) * a.H)[c] = r;
        }
        v[i] = r;
        float f[4];
        unpack4(r, f);
#pragma unroll
        for (int e = 0; e < 4; ++e) wd = welford_add(f[e], wd);
    }
    float mean, rstd;
    row_stats(wd, a.H, a.eps, mean, rstd);
    uint32_t amax = 0;
#pragma unroll
    for (int i = 0; i < GV; ++i) {
        const int c = tid + i * GT;
        if (c >= vecs) break;
        float f[4], sc[4], sh[4];
        unpack4(v[i], f);
        unpack4(reinterpret_cast<const uint2*>(a.scale + b * a.scale_ld)[c], sc);
        unpack4(reinterpret_cast<const uint2*>(a.shift + b * a.shift_ld)[c], sh);
#pragma unroll
        for (int e = 0; e < 4; ++e) f[e] = rb(rb(rb(rstd * (f[e] - mean)) * rb(1.f + sc[e])) + sh[e]);
        v[i] = pack4(f);                      // now x_mod
        reinterpret_cast<uint2*>(a.x_out + int64_t(m) * a.H)[c] = v[i];
        if constexpr (QUANT) amax = vw_w4a8::amax4(v[i], amax);
    }
    if constexpr (QUANT) {
        // quantize_act's steps (q4k.cu quantize_kernel) on the bf16 x_mod held in registers: same s, xq, xsum bits
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = max(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
        __shared__ uint32_t mred[GT / 32];
        if ((tid & 31) == 0) mred[tid >> 5] = amax;
        __syncthreads();
        amax = mred[0];
#pragma unroll
        for (int i = 1; i < GT / 32; ++i) amax = max(amax, mred[i]);
        const float s = amax >= 0x7F80u ? __int_as_float(0x7FC00000) : __fdiv_rn(__uint_as_float(amax << 16), 127.f);
        if (tid == 0) a.s[m] = s;
        int8_t* qr = a.xq + int64_t(m) * a.H;
        int* sr = a.xsum + int64_t(m) * (a.H / 32);
#pragma unroll
        for (int i = 0; i < GV; ++i) {
            const int c = tid + i * GT;
            if (i * GT < vecs) vw_w4a8::quantize4(c, c < vecs, v[i], s, qr, sr);   // whole warps (H % 256 == 0)
        }
    }
}

// out = bf16(res + bf16(gate * y)), 8 values a thread, grid-stride over the M x H / 8 chunks.
__global__ void gated_residual_kernel(const __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ y,
                                      const __nv_bfloat16* __restrict__ gate, int64_t gate_ld,
                                      __nv_bfloat16* __restrict__ out, int64_t total, int chunks, int L) {
    for (int64_t k = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; k < total; k += int64_t(gridDim.x) * blockDim.x) {
        const int64_t row = k / chunks, c = k % chunks, b = row / L;
        float fr[8], fy[8], fg[8];
        unpack8(reinterpret_cast<const uint4*>(res)[k], fr);
        unpack8(reinterpret_cast<const uint4*>(y)[k], fy);
        unpack8(reinterpret_cast<const uint4*>(gate + b * gate_ld)[c], fg);
#pragma unroll
        for (int e = 0; e < 8; ++e) fr[e] = fr[e] + rb(fg[e] * fy[e]);
        reinterpret_cast<uint4*>(out)[k] = pack8(fr);
    }
}

// qk_norm_rope: an attention input's per-head RMSNorm and rotary embedding in one pass, written into the joint
// [B, S_out, heads, 128] q / k buffer at a sequence offset (text at 0, image after it, as diffusers concatenates them).
// diffusers does RMSNorm (one pass), then apply_rotary_emb in fp32 through reshape / unbind / stack / .float() /
// mul / mul / add / .to(bf16) (about 6 passes), then torch.cat of the text and image parts (one more): measured
// 147 + 19 ms of a 1.12 s pass. Here a warp takes one (token, head) row of 128 values, 4 per lane:
//   n   = bf16(w * (x * rstd)),  rstd = rsqrt(mean(x^2) + eps)        (torch's rms_norm order, one rounding)
//   out = bf16(n * cos + rot(n) * sin), rot(n)[2i] = -n[2i+1], rot(n)[2i+1] = n[2i]   (fp32 products and sum,
//         each rounded as torch rounds them, no FMA contraction)
constexpr int QKD = 128, QK_WARPS = 8;

__global__ void __launch_bounds__(QK_WARPS * 32) qk_norm_rope_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, const float* __restrict__ cosf,
        const float* __restrict__ sinf, __nv_bfloat16* __restrict__ out, int S, int nh, int s_out, int off,
        float eps, int64_t rows) {
    const int64_t row = int64_t(blockIdx.x) * QK_WARPS + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31, h = int(row % nh);
    const int64_t bs = row / nh, s = bs % S, b = bs / S, pos = off + s;
    const uint2 raw = reinterpret_cast<const uint2*>(x + bs * nh * QKD + h * QKD)[lane];
    float f[4];
    {
        const float2 p0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
        const float2 p1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
        f[0] = p0.x; f[1] = p0.y; f[2] = p1.x; f[3] = p1.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) ss += f[e] * f[e];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
    const float rstd = rsqrtf(ss / float(QKD) + eps);
    float n[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float wv = w ? __bfloat162float(w[4 * lane + e]) : 1.f;
        n[e] = w ? rb(__fmul_rn(wv, __fmul_rn(f[e], rstd))) : rb(__fmul_rn(f[e], rstd));
    }
    const float4 c = reinterpret_cast<const float4*>(cosf + pos * QKD)[lane];
    const float4 sn = reinterpret_cast<const float4*>(sinf + pos * QKD)[lane];
    const float o0 = __fadd_rn(__fmul_rn(n[0], c.x), __fmul_rn(-n[1], sn.x));
    const float o1 = __fadd_rn(__fmul_rn(n[1], c.y), __fmul_rn(n[0], sn.y));
    const float o2 = __fadd_rn(__fmul_rn(n[2], c.z), __fmul_rn(-n[3], sn.z));
    const float o3 = __fadd_rn(__fmul_rn(n[3], c.w), __fmul_rn(n[2], sn.w));
    const __nv_bfloat162 q0 = __floats2bfloat162_rn(o0, o1), q1 = __floats2bfloat162_rn(o2, o3);
    reinterpret_cast<uint2*>(out + ((b * s_out + pos) * nh + h) * QKD)[lane] =
        make_uint2(*reinterpret_cast<const uint32_t*>(&q0), *reinterpret_cast<const uint32_t*>(&q1));
}

bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

void check_rows(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), name,
                " must be a contiguous CUDA bf16 [M, H] tensor");
    TORCH_CHECK(x.size(1) % 8 == 0 && x.size(1) <= GT * GV * 4, name, ": H must be a multiple of 8, at most ",
                GT * GV * 4, ", got ", x.size(1));
    TORCH_CHECK(aligned16(x.data_ptr()), name, " must be 16-byte aligned (vector loads)");
    TORCH_CHECK(x.size(0) <= INT32_MAX, name, ": M must fit in int32");
}

// A [B, H] modulation row tensor: last dimension contiguous, rows 16-byte aligned (they may be strided).
void check_mod(const torch::Tensor& v, const torch::Tensor& res, int64_t B, const char* name) {
    TORCH_CHECK(v.is_cuda() && v.device() == res.device() && v.scalar_type() == torch::kBFloat16 && v.dim() == 2 &&
                    v.size(0) == B && v.size(1) == res.size(1) && v.stride(1) == 1,
                name, " must be a CUDA bf16 [B, H] tensor on res's device with a contiguous last dimension");
    TORCH_CHECK(aligned16(v.data_ptr()) && (B == 1 || (v.stride(0) * 2) % 16 == 0), name,
                " rows must be 16-byte aligned (vector loads)");
}

int64_t images(const torch::Tensor& res, const torch::Tensor& shift) {
    const int64_t B = shift.dim() == 2 ? shift.size(0) : 0;
    TORCH_CHECK(B >= 1 && res.size(0) % B == 0, "the ", res.size(0), " tokens don't split evenly over ", B,
                " images (B modulation rows)");
    return B;
}

}  // namespace

std::vector<torch::Tensor> glue_rnm(torch::Tensor res, c10::optional<torch::Tensor> y,
                                    c10::optional<torch::Tensor> gate, torch::Tensor shift, torch::Tensor scale,
                                    double eps, bool quantize) {
    check_rows(res, "res");
    const int64_t B = images(res, shift), M = res.size(0), H = res.size(1);
    check_mod(shift, res, B, "shift");
    check_mod(scale, res, B, "scale");
    const bool has_y = y.has_value() && y->defined();
    TORCH_CHECK(has_y == (gate.has_value() && gate->defined()), "y and gate come together (or neither)");
    if (has_y) {
        check_rows(*y, "y");
        TORCH_CHECK(y->sizes() == res.sizes() && y->device() == res.device(), "y must match res's shape and device");
        check_mod(*gate, res, B, "gate");
    }
    if (quantize) TORCH_CHECK(H % 256 == 0, "quantize: H must be a multiple of 256, got ", H);
    const c10::cuda::CUDAGuard guard(res.device());
    auto res_out = has_y ? torch::empty_like(res) : res;
    auto x_out = torch::empty_like(res);
    auto opt = res.options();
    auto xq = quantize ? torch::empty({M, H}, opt.dtype(torch::kInt8)) : torch::Tensor();
    auto s = quantize ? torch::empty({M}, opt.dtype(torch::kFloat32)) : torch::Tensor();
    auto xsum = quantize ? torch::empty({M, H / 32}, opt.dtype(torch::kInt32)) : torch::Tensor();
    if (M > 0) {
        auto bp = [](const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
        RnmArgs a{bp(res), has_y ? bp(*y) : nullptr, has_y ? bp(*gate) : nullptr, bp(shift), bp(scale),
                  has_y ? gate->stride(0) : 0, shift.stride(0), scale.stride(0),
                  reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(x_out.data_ptr()),
                  quantize ? xq.data_ptr<int8_t>() : nullptr, quantize ? s.data_ptr<float>() : nullptr,
                  quantize ? xsum.data_ptr<int>() : nullptr, int(H), int(M / B), float(eps)};
        auto stream = at::cuda::getCurrentCUDAStream();
        if (has_y && quantize) rnm_kernel<true, true><<<unsigned(M), GT, 0, stream>>>(a);
        else if (has_y) rnm_kernel<true, false><<<unsigned(M), GT, 0, stream>>>(a);
        else if (quantize) rnm_kernel<false, true><<<unsigned(M), GT, 0, stream>>>(a);
        else rnm_kernel<false, false><<<unsigned(M), GT, 0, stream>>>(a);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {res_out, x_out, xq, s, xsum};
}

torch::Tensor glue_gated_residual(torch::Tensor res, torch::Tensor y, torch::Tensor gate) {
    check_rows(res, "res");
    check_rows(y, "y");
    TORCH_CHECK(y.sizes() == res.sizes() && y.device() == res.device(), "y must match res's shape and device");
    const int64_t B = images(res, gate);
    check_mod(gate, res, B, "gate");
    const c10::cuda::CUDAGuard guard(res.device());
    auto out = torch::empty_like(res);
    const int chunks = int(res.size(1) / 8);
    const int64_t total = res.size(0) * chunks;
    if (total == 0) return out;
    const int threads = 256;
    const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, 4096);
    gated_residual_kernel<<<unsigned(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(res.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()), gate.stride(0),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total, chunks, int(res.size(0) / B));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void glue_qk_norm_rope(torch::Tensor x, c10::optional<torch::Tensor> weight, torch::Tensor cos, torch::Tensor sin,
                       torch::Tensor out, int64_t offset, double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 3 && x.is_contiguous() &&
                    x.size(2) % QKD == 0,
                "x must be a contiguous CUDA bf16 [B, S, heads * 128] tensor");
    const int64_t B = x.size(0), S = x.size(1), nh = x.size(2) / QKD;
    TORCH_CHECK(out.device() == x.device() && out.scalar_type() == torch::kBFloat16 && out.dim() == 4 &&
                    out.is_contiguous() && out.size(0) == B && out.size(2) == nh && out.size(3) == QKD,
                "out must be a contiguous bf16 [B, S_out, heads, 128] tensor on x's device");
    TORCH_CHECK(offset >= 0 && offset + S <= out.size(1), "rows ", offset, "..", offset + S, " don't fit out's ",
                out.size(1), " positions");
    for (const auto* t : {&cos, &sin})
        TORCH_CHECK(t->device() == x.device() && t->scalar_type() == torch::kFloat32 && t->dim() == 2 &&
                        t->is_contiguous() && t->size(0) == out.size(1) && t->size(1) == QKD,
                    "cos / sin must be contiguous fp32 [S_out, 128] on x's device");
    const bool has_w = weight.has_value() && weight->defined();
    if (has_w)
        TORCH_CHECK(weight->device() == x.device() && weight->scalar_type() == torch::kBFloat16 &&
                        weight->numel() == QKD && weight->is_contiguous(),
                    "weight must be a contiguous bf16 [128] tensor on x's device");
    TORCH_CHECK(aligned16(x.data_ptr()) && aligned16(out.data_ptr()) && aligned16(cos.data_ptr()) &&
                    aligned16(sin.data_ptr()),
                "x, out, cos and sin must be 16-byte aligned (vector loads)");
    const int64_t rows = B * S * nh;
    if (rows == 0) return;
    const c10::cuda::CUDAGuard guard(x.device());
    qk_norm_rope_kernel<<<unsigned((rows + QK_WARPS - 1) / QK_WARPS), QK_WARPS * 32, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        has_w ? reinterpret_cast<const __nv_bfloat16*>(weight->data_ptr()) : nullptr, cos.data_ptr<float>(),
        sin.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), int(S), int(nh), int(out.size(1)),
        int(offset), float(eps), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
