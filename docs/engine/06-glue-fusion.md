# 06: glue fusion, the work between the GEMMs in one pass (Task 6)

**What this is:** the FLUX blocks' non-GEMM "glue" rewritten as a few hand-written passes over memory, in place of
the dozens of small PyTorch ops diffusers runs. There are three kernels (`engine/csrc/glue.cu`) and patched block
forwards (`engine/blocks.py`):

- `residual_norm_modulate`: residual add, LayerNorm and modulate (`x * (1 + scale) + shift`) in one pass. With
  w4a8 it can also emit the INT8 quantization the next linears need.
- `gated_residual`: a block's last `res + gate * y`, which has no norm after it.
- `qk_norm_rope`: the attention inputs' per-head RMSNorm and rotary embedding in one pass. It writes straight into
  the joint text+image q / k buffer, so the `torch.cat`s that join them disappear too.

**The result is bit-identical to diffusers.** Every pass rounds where diffusers rounds and takes the LayerNorm
statistics in torch's own order. The fused blocks, the pictures and the gate walk are byte-for-byte the unfused ones.
So fusion is lossless even on the lossy w4a8 default. The step drops **1.12 → 0.93 s** and the walk step **47.0 → 39.5 s**.

- `VECWORLD_FUSE` unset: fused under every engine kernel (the default); not fused under `VECWORLD_KERNEL=off`, so the
  reference stays diffusers all the way.
- `VECWORLD_FUSE=0` / `off`: diffusers' block forwards under any kernel.
- `VECWORLD_FUSE=1` / `on`: fused under any kernel, `off` included (the fused blocks on diffusers' own linears, to
  test them apart from the engine's GEMMs).

## The problem, in plain words

A FLUX block is mostly big matrix multiplies (GEMMs). Between them sit small jobs: add the attention output back into
the residual stream, normalize each token (LayerNorm), scale and shift it by the timestep's modulation, normalize the
query and key heads (RMSNorm), rotate them by position (RoPE), apply GELU, glue tensors together (`cat`). Each is
cheap arithmetic, a handful of adds and multiplies per value.

In PyTorch each of those jobs is a separate **kernel launch**. Every launch reads its whole input from VRAM and
writes its whole output back. For a [1280 tokens × 3072] bf16 activation that is 7.9 MB in and 7.9 MB out, for maybe
one multiply per value. The GPU does ~100 multiply-adds in the time it takes to fetch one byte
(`05-modulation-gemv.md`, the roofline), so these ops spend all their time **moving data**, with the math units
idle. They are **memory-bound elementwise ops**.

```
 the unfused chain for one norm, diffusers (each box: read the whole tensor from VRAM, write it back)

   res ──┐
   y ────┼─[ gate*y ]──[ res + . ]──[ LayerNorm ]──[ * (1+scale) ]──[ + shift ]──> x_mod
   gate ─┘    r/w          r/w          r/w            r/w             r/w
                                                                   = 5 round trips, ~5 x 2 x 7.9 MB

 fused: one kernel, each value read once, kept in registers, written once

   res, y, gate, scale, shift ──[ v = res + gate*y ; LN stats ; x = LN(v)*(1+scale)+shift ]──> res_out, x_mod
                                         1 read of res and y, 1 write of each output
```

**Fusing saves round trips to memory, not arithmetic.** The fused kernel does exactly the same multiplies and adds,
so it cannot do fewer FLOPs. It just never writes the intermediate results out and reads them back. The time of a
memory-bound op is bytes ÷ bandwidth, so cutting the bytes cuts the time.

## Measure first: where the non-GEMM time actually went

The brief guessed "~0.4 s of glue", an old estimate from before Tasks 3 to 5c. So before writing anything I profiled
one tile512 transformer pass on the default (w4a8, Task 5c, AC). Every GPU kernel is attributed to the op that
launched it (torch profiler trace, kernel ↔ launch by correlation id; `profile_glue.py` in the task workspace). The
group-offload weight copies run on their own stream, overlapped, and are left out.

| group (one pass, 1 window) | before (unfused) ms | share of step | after (fused) ms |
|---|---|---|---|
| linears: GEMMs + GEMV + quantize_act | 717 | 64% | 715 (quantize 25.6 → 13.2; GEMMs +1%, see below) |
| **RoPE**, diffusers' `apply_rotary_emb` in fp32 | **147** | **13%** | → in `qk_norm_rope` |
| QK RMSNorm | 19 | 1.7% | → in `qk_norm_rope` |
| attention (flash SDPA) | 68 | 6.1% | 70 (not touched) |
| residual / LayerNorm / modulate / gates / cats (the brief's "glue") | 68 | 6.1% | → all fused passes together: **35** |
| GELU | 21 | 1.9% | 22 (not fused, see "Tried and not chosen") |
| outside the blocks (embedders, final norm) | 1.5 | 0.1% | 1.6 |
| **wall time of the pass** | **1.118 – 1.120 s** | | **0.925 – 0.929 s** |

Two surprises:

1. **The brief's glue (norm + modulate + residual) was only 68 ms**, not 0.4 s. A third of it was `torch.cat`.
2. **The biggest non-GEMM cost was RoPE, 147 ms, 13% of the step.** diffusers rotates q and k in fp32 through a chain
   of reshape / unbind / stack / `.float()` / mul / mul / add / `.to(bf16)`, about six passes per projection at
   twice the bytes (fp32), 114 times per pass. Its natural partner, the QK RMSNorm right before it, added 19 ms.

So the task fused what actually cost: the brief's residual / norm / modulate pass (the required interface), and also
RoPE with its RMSNorm.

## How the kernels work

```
 residual_norm_modulate: one block of 128 threads per token row (H = 3072 values)

   thread t holds 4-value pieces t, t+128, t+256, ... of the row in registers (6 pieces for H = 3072)
   1. load res (and y, gate) ─> v = bf16(res + bf16(gate*y)) ─> write res_out          (one 8-byte load each)
   2. LayerNorm statistics of v: Welford per thread ─> warp shuffle ─> 4-warp tree in shared memory
   3. x = bf16(bf16(bf16(rstd*(v-mean)) * bf16(1+scale)) + shift) ─> write x_mod        (scale, shift: row b)
   4. (w4a8) s = max|x|/127, xq = rint(x/s), xsum per 32 values ─> attached to x_mod for the next linears

 qk_norm_rope: one warp per (token, head) row of 128 values, 4 per lane
   load q/k (8 bytes a lane) ─> RMSNorm (warp sum of squares) ─> rotate pairs (2i, 2i+1) by cos/sin at the
   token's position ─> write into the joint [B, T+L, heads, 128] buffer at the stream's offset (text 0, image T)
```

- **One modulation row per image.** `shift`, `scale` and `gate` are [B, 3072] rows (B windows in a pass, up to 8).
  Token row m uses row m ÷ (tokens per image). They are chunks of one modulation output, so the kernel takes them
  strided (no copy).
- **Rounded where diffusers rounds.** Every bf16 rounding of the eager chain happens at the same point. The
  LayerNorm statistics replay torch's own Welford algorithm in torch's order (4-value loads, 128 threads as 32 × 4,
  shuffle-down, then a tree over 4 warps). A search over 15 layouts found exactly one that reproduces torch bit for
  bit (0 mismatches in 12.6 M values). The RMSNorm uses torch's `w * (x * rstd)` with one rounding; the rotary
  products and sum use fp32 without FMA contraction, as torch's three separate ops do. Result: `torch.equal` with
  diffusers for x_mod, q and k, and for whole blocks.
- **The fused quantize.** Under w4a8 a norm's output feeds INT8 linears (to_q, to_k, proj_mlp, ff.net.0), and each
  of them used to read x_mod again to quantize it. The norm already holds x_mod in registers, so it computes the same
  s / xq / xsum (shared code, `w4a8_quant.cuh`; bit-identical to `quantize_act`) and attaches them to x_mod.
  A w4a8 `Q4KLinear` handed that tensor uses them. A keep-listed linear on mma ignores them and reads bf16 x_mod.
- **Cats removed.** q / k are written into the joint buffer (no cat). In a single block the GELU writes its output
  straight into `proj_out`'s input buffer next to the attention output (no [attn | mlp] cat). A single block given
  the previous single block's two output halves reuses that buffer instead of re-concatenating. `torch.cat` went from
  26 ms to 2 ms per pass.
- **What stays diffusers'.** Every linear, the attention kernel (`dispatch_attention_fn`, called as the processor
  calls it), the GELU math and the modulation linears are diffusers' / the engine's own modules. A non-default
  attention processor (IP-adapter), fused qkv projections or attention kwargs fall back to `attn(...)` as is.

All three kernels run near the memory roof. Microbench, one call (`micro_glue.py`):

| pass | fused µs | GB/s (of its own bytes) | eager torch µs | speed-up |
|---|---|---|---|---|
| rnm, M = 1024, no residual, + quantize | 96 | 167 | 249 | 2.6× |
| rnm, M = 1024, residual, + quantize | 167 | 172 | 430 | 2.6× |
| rnm, M = 1280 (single block), + quantize | 117 | 172 | 309 | 2.6× |
| rnm, M = 5120 (4 windows), residual | 706 | 178 | 2103 | 3.0× |
| gated_residual, M = 1024 | 106 | 179 | 181 | 1.7× |
| qk_norm_rope, 1280 tokens × 24 heads | 97 | 176 | 1403 | **14.5×** |

167–179 GB/s is 89–96% of the 187 GB/s this GPU reads at (`05-modulation-gemv.md`), so there is little left to win
inside these kernels: their time is the bytes they must move.

## Amdahl: why the glue's share grows as the GEMMs shrink

Amdahl's law: if a part takes fraction p of the time and you make it k times faster, the whole becomes
1 / ((1 − p) + p / k) times faster. Speeding up one part only helps as much as that part's share.

The flip side is that every task which speeds up the GEMMs makes the non-GEMM share **bigger**:

```
                       step s   GEMM-ish (linears) share   everything else (non-GEMM) share
 Task 1, diffusers      2.79     dequant + matmul dominate  small
 Task 5c, before T6     1.12     717 ms of 1042 ms = 69%    325 ms = 31%   <- RoPE alone 13%
 Task 6, fused          0.93     715 ms of  843 ms = 85%    128 ms = 15%   (attention 70, glue 35, GELU 22)
```

Tasks 3 to 5c took the GEMMs from ~2 s to ~0.7 s, but the elementwise ops stayed the same, so by Task 5c they were
almost a third of the work. That is why fusing them now buys 0.19 s, a 17% faster step, although they were never
"expensive" in FLOPs. Taking the remaining 128 ms of non-GEMM work to zero would make the pass at most
843 / 715 = 1.18× faster again. From here the GEMMs are the lever again (Task 7 and on).

## Numbers (RTX 3050 Ti Laptop, AC)

All on the default (w4a8 + `FLUX_A16_KEEP`), unfused (`VECWORLD_FUSE=off`, i.e. Task 5c) and fused measured in the
same session (`bench/t6-unfused*.json`, `bench/t6-fuse*.json`, `bench/t6-fuse-peak.json`, `bench/t6-fuse-glue.json`):

| | unfused (Task 5c) | **fused (Task 6)** |
|---|---|---|
| step s (tile512 median pass) | 1.12 | **0.93** |
| achieved TFLOP/s (bf16-equivalent) | 15.78 | **19.00** |
| % of the 19.1 bf16 peak | 82.6 | **99.4** |
| walk step s (`sharp`) | 47.0 | **39.5** |
| walk: wall s per window pass | 1.18 | 0.99 |
| one pass at 1 / 2 / 4 windows, s per window | 1.126 / 1.106 / 1.101 | **0.930 / 0.909 / 0.917** |
| peak allocated GB at 1 / 2 / 4 windows | 1.018 / 1.119 / 1.323 | **1.011 / 1.106 / 1.296** |
| non-GEMM GPU time per pass (profile) | 325 ms | 128 ms |

"99.4% of the bf16 peak" is the nominal FLOP count divided by the time. 75% of the token FLOPs now run on the INT8
tensor cores (Task 5b), which are faster than bf16, so this bf16-equivalent rate could pass 100%. It does not mean
the GPU is at its limit.

Peak memory barely moves (−1 to −2%). The peak is set by the streamed block weights and the largest GEMM's
activations (the 12,288-wide MLP intermediate), not by the norm intermediates the fusion stops keeping. Task 9 should
not count on fusion for memory headroom.

## Correctness

| Test | Bar | Result |
|---|---|---|
| `test_rnm_matches_eager[M=1,333,1280,5120]` (M = 5120: 4 images) | max\|ours − eager fp32\| ≤ 2⁻⁷ · max\|eager fp32\|, both outputs | pass |
| `test_rnm_no_residual` | same, y = None; res_out is res (no copy) | pass |
| `test_rnm_broadcasts_per_image[B=1,2,8]` | each image's tokens use its own row (= separate calls) | bit-equal |
| `test_rnm_matches_the_reference_rounding` (B = 1 and 4) | bit-identical to the eager bf16 chain | `torch.equal` |
| `test_rnm_quantize_matches_quantize_act[333,1280]` | fused s / xq / xsum = quantize_act(x_mod) | `torch.equal` |
| `test_attached_quantized_input_only_feeds_w4a8` | w4a8 uses an attached quantization; mma ignores even a wrong one | yes |
| `test_qk_norm_rope_matches_diffusers` (B = 2, 256 + 1024 tokens, real FLUX rotary tables) | = rms_norm + apply_rotary_emb + cat | `torch.equal` |
| `test_patched_block_matches_unpatched[off, w4a8]` (double 0, single 0, B = 2 windows) | max\|Δ\| / max\|ref\| < 2e-2 | **0** (`torch.equal`) on both kernels |
| `test_flux_fused_image_matches[mma, w4a8]` (512, same seed, fused vs unfused) | PSNR ≥ 30 dB | identical pixels (168 dB = the 1e-12 floor) on both |
| quality gate: 6-spot `sharp` walk (w4a8, fused) | no spot above t6-base + 0.03 (ruling) | **all 6 spots byte-identical to t6-base** (0.73 / 1.15 / 0.90 / 1.08 / 0.88 / 0.58) |
| CPU: `test_fuse_defaults_on_with_an_engine_kernel`, `test_painter_patches_blocks_when_fused` | the `VECWORLD_FUSE` rule; patch after the swap, before group offload | pass |

## The gate

The walk gate (Global Constraints) for this task, as ruled on 2026-11-06: the base is `/tmp/walk/t6-base` (Task 5c's
code, w4a8 default, scores 0.73 / 1.15 / 0.90 / 1.08 / 0.88 / 0.58; the fixed 1.1 limit is reported, not gated,
since the base itself scores 1.15 on spot 1's fog edge). The fused walk passes if no spot rises more than 0.03
above it, and the user judges the sheet.

| spot | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| t5 (Task 5 mma, lossless) | 0.71 | 1.08 | 0.91 | 0.89 | 0.84 | 0.55 |
| t6-base (5c, w4a8, unfused) | 0.73 | 1.15 | 0.90 | 1.08 | 0.88 | 0.58 |
| **t6 (fused, w4a8)** | **0.73** | **1.15** | **0.90** | **1.08** | **0.88** | **0.58** |

The fused walk's six spots are byte-identical to t6-base: Δ = 0 everywhere. **Pass**, so `VECWORLD_FUSE` is on by
default. (The laptop's /tmp was cleared between the base walk and this task's resumption, so t6-base was regenerated
with `VECWORLD_FUSE=off`, giving the recorded scores and columns exactly, and t5 from a worktree at bd179b8, giving
its recorded scores exactly. Both walks are deterministic.) Sheets: `t6-vs-base-t5.png` (t5 | t6-base | t6) and
`t6-vs-t5.png`.

## Tried and not chosen

Every alternative tried or measured and not kept, with its numbers and why it lost.

**1. Two-pass LayerNorm statistics (the first version of `residual_norm_modulate`).** Mean, then the variance around
it, in fp32 from registers: the textbook way, and as fast as the final kernel. Measured:

- 2.8e-6 of x_mod values differed from torch's by one last bit;
- whole blocks: max-rel 9.0e-3 (one bf16 step of the largest output; 86% of outputs bit-equal);
- the picture, fused vs unfused, same seed: **30.36 dB under mma** (passes the brief's 30) but **25.17 dB under w4a8**,
  a different sample of the same scene (trunks moved), as with Task 5c's one-step modulation change (25.78 dB);
- the gate walk against t6-base: 0.73 / 1.13 / 0.94 / 1.04 / 0.96 / 0.79. Spots 2, 4 and 5 rose by +0.04 / +0.08 /
  +0.21, past the +0.03 the ruling allows, although the worst seam fell (1.15 → 1.13). **Failed the relative gate.**

It lost because it was lossy where it did not need to be: W4A8 turns one flipped last bit into different INT8
rounding downstream, so a different sample, and the walk's seam scores move by more than ±0.03 between samples
(Task 5c's rounding-only change moved five spots by +0.09 to +0.10). Replaced by torch's own Welford order:
bit-identical, and the same speed (M = 1024 rnm 94.6 → 96.4 µs, noise).

**2. Which Welford layout is torch's.** A standalone probe ran torch-style Welford with every combination of load
width and warps per row against `torch.nn.functional.layer_norm` on 4096 × 3072 bf16 rows (12.6 M outputs):

| vec \ warps | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 2 | 196 | 155 | 153 | 132 | 129 |
| **4** | 214 | 162 | **0** | 104 | 118 |
| 8 | 175 | 161 | 142 | 132 | 113 |

(mismatching outputs). Only 4-value loads with 4 warps (128 threads) reproduce torch, so that layout is the kernel's.

**3. Launch configurations.** Microbench µs per call (best of 3 × 50); every variant is memory-bound at 89–96% of the
read roof, so the differences are noise:

| shape | rnm 64 thr / qk 4 warps | **rnm 128 / qk 8 (kept)** | rnm 256 / qk 16 |
|---|---|---|---|
| rnm M = 1024, no residual, + quantize | 97.5 | 94.6 | 94.1 |
| rnm M = 1024, residual, + quantize | 167.8 | 165.3 | 164.3 |
| rnm M = 256, residual | 37.0 | 37.7 | 38.1 |
| rnm M = 1280, + quantize | 117.9 | 117.0 | 116.0 |
| rnm M = 5120, residual | 704.1 | 700.1 | 700.1 |
| qk_norm_rope, 1280 × 24 heads | 96.7 | 96.6 | 96.0 |

(Measured with the two-pass statistics; the kept kernel is the 128-thread one, which torch's Welford order also needs.)

**4. The fused quantize (kept).** Profiled at the two-pass stage: without it the linears took 718.3 ms and the glue
42.2 ms (wall 1.097 s); with it 705.7 and 44.4 ms (wall 1.084 s): net −10 ms. In the final build `quantize_act` drops
from 25.6 ms (228 launches) to 13.2 ms (38 launches, all for the single blocks' `proj_out`, see 5b).

**5. Fusion boundaries considered and rejected.**

- (a) **GELU into the GEMM epilogue**, 22 ms (2.4% of the fused step). It needs an epilogue in both Q4_K GEMMs (mma and
  w4a8), so it belongs with GEMM work, not here.
- (b) **Quantizing `proj_out`'s input in its producers**, 13.2 ms (1.4%). Its rows are [attention | GELU(mlp)] from two
  producers, and a per-token scale needs the max over the whole row, so both must finish first. That only becomes
  possible once (a) is done.
- (c) **A block's last residual fused into the next block's first norm** (across blocks). The `gated_residual` passes
  cost ~7.5 ms per pass (microbench × launches), and fusing could save at most about half (one write and one read),
  3–4 ms. It would cross diffusers' block loop and the group-offload hooks.
- (d) **The attention output written straight into `proj_out`'s buffer**, 3.6 ms of copies. Flash SDPA allocates its
  own output; this would need a custom attention call.
- (e) **The last cats**, 1.8 ms: the double blocks' text+image v, and the first single block's stream join.
- (f) **RoPE folded into the q / k GEMM epilogue** instead of its own pass. `qk_norm_rope` already runs at 176 GB/s
  for 97 µs per call (~11 ms per pass), and the RMSNorm needs a full head's sum of squares first.

**6. Profiled and not worth fusing** (share of the fused 0.93 s step): attention 70 ms (7.5%, already flash
attention, not glue); GELU 22 ms (2.4%, item 5a); the `proj_out` quantize 13.2 ms (1.4%, 5b); the modulation GEMV
11.6 ms (1.2%, Task 5c); the attention-output copy 3.6 ms (0.4%); the leftover cats 1.8 ms (0.2%); the
embedders and final norm outside the blocks 1.6 ms (0.2%); SiLU on the timestep embedding 0.2 ms.

**7. Seen, not chased: the GEMMs ran ~1% slower with the glue fused** (w4a8 428.8 → 435.5 ms, mma 251.3 → 255.3 ms
per pass; same shapes and launch counts). The likely cause is the denser schedule on a power- and heat-limited laptop
GPU (fewer idle gaps between GEMMs). It costs ~10 ms of the 190 ms won.

## What to read next

- `learning-ledger.md`: the roofline (why elementwise ops are memory-bound) and Amdahl.
- Task 7 (CUDA graphs): with fewer, larger kernels per block, launch overhead becomes the next small cost to remove.
  The pass still has ~80 ms where the GPU waits (wall 0.925 s vs ~845 ms of kernels).
