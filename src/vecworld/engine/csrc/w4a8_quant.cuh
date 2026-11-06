// W4A8's per-token activation quantize, the pieces shared by quantize_act (q4k.cu) and the fused glue pass
// (glue.cu), so both produce the same bits: xq = rint(x / s) clamped to +-127, xsum per 32-wide sub-block.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace vw_w4a8 {

// The bf16 bit patterns of max|x| over 8 values (an integer max: an inf or NaN wins instead of being dropped).
__device__ __forceinline__ uint32_t amax4(uint2 v, uint32_t amax) {
    return max(amax, max(max(v.x & 0x7FFFu, (v.x >> 16) & 0x7FFFu), max(v.y & 0x7FFFu, (v.y >> 16) & 0x7FFFu)));
}

__device__ __forceinline__ uint32_t amax8(uint4 v, uint32_t amax) {
    const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) amax = max(amax, max(u[e] & 0x7FFFu, (u[e] >> 16) & 0x7FFFu));
    return amax;
}

// One value: rint(x / s) clamped to +-127 (IEEE division, half to even); 0 when s is 0 (a zero token) or NaN.
__device__ __forceinline__ int quant1(float x, float s) {
    return s > 0.f ? int(fminf(fmaxf(rintf(__fdiv_rn(x, s)), -127.f), 127.f)) : 0;
}

// Quantize chunk c (8 values, v) and write it; every lane of the warp calls this together (the xsum shuffles).
// A 32-value sub-block is 4 consecutive chunks, held by 4 neighbouring lanes.
__device__ __forceinline__ void quantize8(int c, bool ok, uint4 v, float s, int8_t* qr, int* sr) {
    int sum = 0;
    if (ok) {
        const uint32_t u[4] = {v.x, v.y, v.z, v.w};
        uint32_t packed[2] = {0, 0};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[e]));
            const int q0 = quant1(f.x, s), q1 = quant1(f.y, s);
            sum += q0 + q1;
            packed[e >> 1] |= ((uint32_t(q0) & 0xFF) | ((uint32_t(q1) & 0xFF) << 8)) << (16 * (e & 1));
        }
        *reinterpret_cast<uint2*>(qr + c * 8) = make_uint2(packed[0], packed[1]);
    }
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2);
    if (ok && (c & 3) == 0) sr[c >> 2] = sum;
}

// The same for chunk c of 4 values (v: 4 bf16); a sub-block is 8 consecutive chunks on 8 neighbouring lanes.
__device__ __forceinline__ void quantize4(int c, bool ok, uint2 v, float s, int8_t* qr, int* sr) {
    int sum = 0;
    if (ok) {
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        const int q0 = quant1(f0.x, s), q1 = quant1(f0.y, s), q2 = quant1(f1.x, s), q3 = quant1(f1.y, s);
        sum = q0 + q1 + q2 + q3;
        *reinterpret_cast<uint32_t*>(qr + c * 4) =
            (uint32_t(q0) & 0xFF) | ((uint32_t(q1) & 0xFF) << 8) | ((uint32_t(q2) & 0xFF) << 16) | (uint32_t(q3) << 24);
    }
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 4);
    if (ok && (c & 7) == 0) sr[c >> 3] = sum;
}

}  // namespace vw_w4a8
