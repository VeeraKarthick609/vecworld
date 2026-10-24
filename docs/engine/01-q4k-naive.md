# 01: A hand-written Q4_K kernel, naive (Task 3)

**What this is:** the first CUDA kernel of the engine. It runs FLUX's Q4_K linears straight from the packed 4-bit
weights, without building a bf16 weight in VRAM. It is the **correctness reference** that every later stage
(tiled, `mma`, W4A8) is tested against. It is **not a speedup**: one 512 window pass takes about 23 s against the
baseline's 2.74 s.

Turn it on with `VECWORLD_KERNEL=naive`. (Unset, the painter runs the `mma` kernel since Task 5; `off` is the
diffusers reference.)

## The problem: 1.08 s of every 2.74 s step is unpacking weights

diffusers keeps the transformer's weights in Q4_K on the GPU. Each time a linear runs, it:

1. unpacks the *whole* weight to a bf16 tensor in VRAM, using a chain of about ten PyTorch ops (split, view,
   shift, mask, multiply, subtract, cast ...);
2. hands that tensor to cuBLAS.

Every op in step 1 reads its input from VRAM and writes its output back to VRAM. Only then does the next op start.
The minimum traffic for the unpack is about 0.16 s per step (est.): read 6.66 GB of Q4_K and write 23.7 GB of bf16
at 192 GB/s. The intermediate round trips make it about 7x that. The bench measured **1.08 s per step, 42%**
(`bench/baseline.json`, `blocks.512x512.dequant_s`). The GEMM then reads the 23.7 GB of bf16 again. All of this
happens every step, for weights that never change. (Ledger, Technique 1.)

The fix is to *fuse* the two: decode each weight in registers right where the multiply needs it. Then the bf16
weight never exists in memory.

## The Q4_K layout

```
 one row of a weight (in_features values) = in_features / 256 super-blocks, back to back

 Q4_K super-block: 256 weights in 144 bytes (4.5 bits per weight)
 +---------+---------+---------------------------+---------------------------------------+
 | d  fp16 | dmin f16| scales[12] = 96 bits      | qs[128] = 256 x 4-bit codes q (0..15) |
 |   2 B   |   2 B   | 8 x 6-bit sc, 8 x 6-bit m |  two codes per byte                   |
 +---------+---------+---------------------------+---------------------------------------+
 8 sub-blocks of 32 weights. Weight i of sub-block j:

        w = (d * sc_j) * q_i  -  (dmin * m_j)

 scales[12], as gguf-py's get_scale_min unpacks them:
   byte 0..3   EEAAAAAA ...   sc_0..3 = low 6 bits          (top 2 bits: high bits of sc_4..7)
   byte 4..7   eeaaaaaa ...   m_0..3  = low 6 bits          (top 2 bits: high bits of m_4..7)
   byte 8..11  eeeeEEEE ...   sc_4..7 low nibble, m_4..7 high nibble

 qs: 4 runs of 32 bytes; run g = low nibbles -> sub-block 2g, high nibbles -> sub-block 2g+1
```

## How the naive kernel works

`src/vecworld/engine/csrc/q4k.cu`, `gemm_naive_kernel`:

```
 y[m, n] = sum_k x[m, k] * W[n, k] (+ bias[n])           one thread per output element (m, n)

 thread (m, n):
   for each super-block b of row n:                       K / 256 of them
     read d, dmin, scales[12]                             registers
     for each run g (sub-blocks 2g, 2g+1):
       (d*sc, dmin*m) for both sub-blocks                 6-bit unpack in registers
       for 8 x (4 code bytes + 4 + 4 bf16 x values):     vector loads
         w_lo = d_sc_lo * (byte & 15) - dm_lo             decode, fp32
         w_hi = d_sc_hi * (byte >> 4) - dm_hi
         acc += w_lo * x[k] + w_hi * x[k + 32]            fp32 accumulator
   y[m, n] = bf16(acc + bias[n])

 thread blocks are 16 x 16: 16 outputs (n) of 16 rows (m); their 16 weight rows and 16 x rows stay in L1
```

- **Exact decode.** It uses `__fmul_rn` / `__fsub_rn`, so the compiler cannot fuse `d_sc * q - dm` into an FMA.
  The result is then bit-for-bit what `gguf.quants.dequantize` computes. `test_q4k_dequant_matches_gguf` checks
  this with `torch.equal` on real FLUX tensors (in_features 3072, 12288 and 15360). The check runs through a
  dequant kernel that shares the GEMM's device decode.
- **Swap.** `swap_linears` replaces only `nn.Linear`s whose weight is Q4_K and has the linear's own shape. That is
  every block linear (the file's 304 Q4_K tensors, which diffusers splits into more linears); the F16
  and F32 layers (embedders, final layer)
  stay on diffusers (Review Focus 1, `test_swap_skips_non_q4k`). `Q4KLinear` keeps the packed bytes and the bias as
  frozen *parameters* that share the GGUF parameters' storage. They have to be parameters: diffusers 0.41's
  streamed group offload copies a block's buffers to the GPU but never moves them back, so buffers ran out of VRAM
  by the 20th block.
- **Ragged shapes.** Every thread checks its own bounds, so any M and N work: M = 1, 7, 333 and 1792 with N = 1000
  are tested (`test_q4k_gemm_ragged`).

## Why it is slow (and that's expected)

There are no tensor cores, no shared memory and no reuse beyond what L1 happens to catch. Each thread re-decodes
its whole weight row on its own, so every weight is decoded M times (1280 per pass) instead of once per tile.
CUDA cores can do at most about 9.6 TFLOP/s fp32 (est.). The decode, conversions and loads cost several
instructions per multiply-add, so the kernel reaches about 0.85 TFLOP/s on isolated linears.

A first version reached 0.45 TFLOP/s. Two cheap changes, both still "one thread per output", got it to 0.85:
vector loads (4 code bytes and 4 + 4 bf16 x values per load) and 16 x 16 thread blocks instead of 32 x 8. Blocks
of 32 x 8 reached 0.55. Putting threads along m instead of n dropped it to 0.3.

## Numbers (on AC, `scripts/bench_flux.py --painter engine`, VECWORLD_KERNEL=naive)

Measured 2026-10-24 on mrjarvis, on AC (`power: ac` in every file), with the game server stopped. The benches
ran on this task's diff, before it was committed (`git: 6f01119-dirty`).

| | Reference (diffusers, Task 1) | Naive kernel (Task 3) | |
|---|---|---|---|
| Step: one 512 window pass, 1024 + 256 tokens (`tile512.s_per_pass_median`) | 2.79 s (baseline 2.74 s) | **22.98 s** | 8.2x slower |
| Achieved TFLOP/s (17.67 TFLOP per pass) | 6.33 (baseline 6.45) | **0.77** | |
| % of the measured 19.1 TFLOP/s bf16 peak | 33% (baseline 34%) | **4.0%** | |
| Walk step (`step.s`, `sharp`, 40 window passes) | 61.4 s | **945.1 s** (23.6 s per window pass) | 15.4x slower |
| Resident step, weights on the GPU (`blocks.512x512.resident_step_s`) | 2.59 s | 21.25 s (0.83 TFLOP/s) | |
| ... of which dequant | 1.08 s | **0.00 s** (no separate dequant) | gone |
| Double / single block, 512 | 61.9 / 37.1 ms | 399.5 / 359.5 ms | |
| Resident step at 1024x512 | 3.65 s | 40.05 s (0.84 TFLOP/s) | |
| Peak allocated, tile512 / walk step | 1.23 / 1.37 GB | 1.02 / 1.33 GB | no bf16 weight scratch |

Files: `bench/t3-naive.json` (tile512), `bench/t3-naive-step.json` (walk step) and `bench/t3-naive-blocks.json`.

The step went from 2.74 s to about 23 s, as predicted. The 1.08 s of dequant is gone, but the GEMM itself went
from cuBLAS on tensor cores (about 1.4 s with glue) to about 21 s on CUDA cores without reuse. Being slower is
this stage's job: it pins down *what* the answer is, so the later stages can chase *how fast*.

## Correctness

The kernel layer, against gguf-py and torch:

- Dequant: bit-exact against `gguf.quants.dequantize` on three real tensors (`torch.equal`).
- GEMM: `max|ours - x @ deq(W)^T| / max|ref| < 2e-2` on three real tensors, and on ragged M = 1, 7, 333 and 1792.

The model, against the diffusers GGUF reference. There are three gates, tight to loose, because small differences
compound:

| Gate | Bar | Naive | Exact gguf fp32 weights + torch.matmul |
|---|---|---|---|
| **One block** (double 0, single 0), primary, `test_q4k_block_matches_reference[kernel]` | max-rel < 2e-2 | 5.5e-3 .. 8.3e-3 (also blocks 9 and 20) | the same |
| One whole forward pass (57 blocks), `test_flux_forward_matches_reference` | rel L2 < 0.06 | 4.2% (max-rel 0.127) | 4.5% (max-rel 0.139) |
| The 4-step picture at seed 0, `test_flux_image_matches_reference` | PSNR >= 28 dB | 29.87 dB | 29.4 dB (naive vs it: 30.7 dB) |

**Why the bars loosen.** Diffusers dequantizes each weight and rounds it to bf16 (via fp16) before its GEMM. The
engine keeps the exact fp32 weight and sums in its own order. Neither is wrong, and per block they agree to under
1%. But each block's output is the next one's input, so the last-bit differences grow through 57 blocks to about
4% (rel L2) within one pass. Four denoising steps then grow them again, to about 30 dB on the picture. The plan's
own reference check, plain `torch.matmul` on the exact weights, lands in the same place (29.4 dB, 4.5%). So a 30 dB
picture bar, or a 2e-2 max-rel bar on the whole pass, measures rounding policy, not kernel correctness. The
per-block gate is the one that catches a kernel bug, and later stages reuse it by kernel name.

Two more checks:

- **Plumbing.** Diffusers' own dequant, emulated inside the swapped layers, reproduces the reference picture at
  168 dB. The swap and the streaming are exact.
- **By eye.** The pictures are the same forest, path, light shafts and ferns. They differ only in needle-level
  texture (mean |diff| 4.5 of 255). The test saves a reference | naive | 4x difference sheet under pytest's
  tmp_path.

## What to read next

- Task 4 (tiled): load a tile of packed weights and a tile of x into shared memory once, and reuse each decoded
  weight across many rows of x. That removes the M-fold re-decode.
- Task 5 (`mma`): put the decoded bf16 fragments straight into `mma.sync.m16n8k16`, the tensor cores
  (ledger Technique 1, "How the fused kernel works").
- `docs/engine/speed.md`: the running table.
