# Engine speed table

One row per engine task (Goal B). **Step s** = `tile512.s_per_pass_median` from `scripts/bench_flux.py --what
tile512` on AC: one 512x512 window, 1024 image + 256 text tokens, one transformer pass. **Achieved TFLOP/s** =
`flop_per_step(1024, 256)` (17.67 TFLOP, ledger section 1.6) / step s, the nominal count even when work runs in INT8
(so **bf16-equivalent** from Task 5b on). **% peak** is against the measured
19.1 TFLOP/s bf16 matmul peak (`bench/baseline.json`, blocks). **Walk step** = `step.s` from `--what step`, a `sharp`
walk step's wall time (not gated).

| Task | Flags | Step s | Achieved TFLOP/s | % bf16 peak | Walk step s | Peak alloc GB (tile512 / step) | Quality gate | Git |
|---|---|---|---|---|---|---|---|---|
| 0: baseline (diffusers + GGUF + group offload) | `--painter reference` | 2.74 | 6.45 | 34 | (not measured) | 1.24 / (not measured) | the reference | 895c75e |
| 1: FluxPainter on the diffusers reference | `--painter reference` | 2.79 | 6.33 | 33 | 61.4 | 1.23 / 1.37 | the reference | a47474a (tile512), 60252f7 (step) |
| 3: fused Q4_K, naive (correctness reference) | `--painter engine`, `VECWORLD_KERNEL=naive` | 22.98 | 0.77 | 4.0 | 945.1 | 1.02 / 1.33 | lossless: block max-rel 8.3e-3 (< 2e-2), forward rel L2 4.2% (< 6%), picture 29.87 dB (>= 28) | 6f01119-dirty (this task's diff, before commit) |
| 4: fused Q4_K, tiled (shared memory, CUDA cores) | `--painter engine`, `VECWORLD_KERNEL=tiled` | 3.77 | 4.69 | 24.5 | 156.3 | 1.02 / 1.33 | lossless: block max-rel 8.3e-3 (< 2e-2), forward rel L2 4.7% (< 6%), picture 29.05 dB (>= 28); vs naive max-rel <= 3.5e-3. M < 32 linears (the M = 1 modulation layers, ~1% of the pass) run the naive kernel | 49f56bb-dirty (this task's diff, before commit) |
| 5: fused Q4_K, tensor cores (bf16 `mma.sync`, W4A16) | `--painter engine`, `VECWORLD_KERNEL=mma` (unset = mma; `off` = diffusers) | **1.40** | **12.62** | **66.1** | 57.4 | 1.02 / 1.33 | lossless: block max-rel 8.3e-3 (< 2e-2), forward rel L2 4.4% (< 6%), picture 29.37 dB (>= 28); vs naive max-rel <= 6.3e-3. GEMM 90.6-93.4% of torch.matmul at M = 1024/1280, 92-105% at M = 256 (`bench/t5-gemm.json`). M < 7 linears run naive | 075df02-dirty (this task's diff, before commit) |
| 5b: W4A8, per-token INT8 activations on the INT8 tensor cores (`mma.sync` s8), 75.0% of token-facing FLOPs; **the default** | `--painter engine`, `VECWORLD_KERNEL=w4a8` (unset = w4a8 with `FLUX_A16_KEEP` on mma; `mma` = W4A16; `off` = diffusers) | **1.15** | **15.37** (bf16-equivalent) | **80.4** | 48.9 | 1.02 / 1.33 | lossy, judged by the walk gate: two walks at a2bdb67 pass `--check`, bit-identical (max seam 1.058 at spot 1, limit 1.1; Task 5 mma 1.080, diffusers 1.082) with the 8-pattern `FLUX_A16_KEEP` chosen by ablation (the study's 5 worst families + the double blocks' MLP-out and attention-outs; 8 lists walked, see 04-w4a8.md). Same-seed 512 image vs mma 25.22 dB PSNR (recorded, not gated). GEMM 1.45-1.72x mma, 135-156% of torch.matmul bf16 (`bench/t5b-w4a8-gemm.json`) | a2bdb67 (tile512; step and gemm a2bdb67-dirty: a rewritten bench JSON only) |
| 5c: modulation on a Q4_K GEMV (M <= 8 rows, bf16 x, every kernel) | `--painter engine` (default w4a8; `GEMV_MAX_M` = 8 on any kernel) | **1.12** | **15.78** (bf16-equivalent) | **82.6** | 47.6 | 1.02 / 1.33 | lossless path (bf16 x, fp32 sum): block max-rel 8.30e-3 / 5.49e-3 (< 2e-2) under naive / tiled / mma with the GEMV running the modulation, forward rel L2 4.65% (< 6%), picture 29.20-29.42 dB (>= 28); GEMV vs naive max-rel <= 2.9e-3 (tests), 4.9e-3 on the real modulation inputs (one bf16 step); the w4a8 walk gate was not re-walked. Modulation 35.1 -> 10.97 ms per pass at M = 1 (165 GB/s, 86% of the 192 GB/s spec), 38.8 -> 15.9 ms at M = 4 (`bench/t5c-gemv-blocks.json`). Same-seed 512 vs 5b: 30.17 dB under mma, 25.78 dB under w4a8 (a different sample, judged by eye) | 66dcb2f-dirty (this task's diff, before commit) |
| 6: glue fusion: residual + LayerNorm + modulate in one pass, q / k RMSNorm + RoPE in one pass (into the joint q / k), the w4a8 quantize fused into the norm, cats removed; **bit-identical** to diffusers | `--painter engine` (default w4a8; `VECWORLD_FUSE` unset = on under engine kernels, `off` = the 5c path) | **0.93** | **19.00** (bf16-equivalent) | **99.4** | **39.5** | 1.01 / 1.30 (one pass at 1 / 2 / 4 windows: 1.011 / 1.106 / 1.296; unfused 1.018 / 1.119 / 1.323) | lossless, bit for bit: blocks, x_mod, q / k and the 512 picture `torch.equal` to unfused under mma and w4a8; the 6-spot walk byte-identical to t6-base (0.73 / 1.15 / 0.90 / 1.08 / 0.88 / 0.58), relative gate (base + 0.03, ruling 2026-11-06) passed; fixed 1.1 limit reported, not gated (spot 1 1.15, as at base). Non-GEMM GPU time per pass 325 -> 128 ms (`bench/t6-fuse-glue.json`); same-session unfused: 1.12 s, 15.78 TFLOP/s, walk 47.0 s | ee44876-dirty (this task's diff, before commit) |
| 7: blocks streamed through 3 fixed-address weight slots (no group offload) and replayed as CUDA graphs, one per block and shape; **bit-identical** to eager and to group offload | `--painter engine` (default w4a8 + fused; `VECWORLD_GRAPHS` unset = graphs under engine kernels, `0` = eager on the same slots; `VECWORLD_KERNEL=off` keeps group offload) | **0.85** | **20.79** (bf16-equivalent) | **108.8** | **38.8** | 1.16 / 1.92 (one pass at 1 / 2 / 4 windows: 1.249 / 1.431 / 1.792; Task 6 1.011 / 1.106 / 1.296) | lossless, bit for bit: whole transformer passes `torch.equal` for graphs vs eager on slots vs group offload (batch 1, 2, 3; 1024 + 256 / 77 tokens); walk gate: GATE_TEXT. Same session: Task 6 0.92 s (0.918 / 0.922 / 0.924), eager on slots 0.85 (0.844 / 0.850 / 0.852), graphs 0.85 (0.848 / 0.854 / 0.857); nsys GPU idle per pass 102 ms (10.9%) -> 0.76 ms (0.09%); walk step 39.0 / 38.3 / 38.8 s | ee3baae-dirty (this task's diff, before commit) |

Task 3 is the correctness reference, **8.2x slower** than the baseline step: CUDA cores, one thread per output, no
reuse. See `docs/engine/01-q4k-naive.md`.

Task 4 tiles the same kernel through shared memory: **6.1x faster than naive**, dequant share 0 (resident step
3.58 s, of which dequant 0.00 s), but still **1.38x slower than the baseline step**: its GEMM runs on CUDA cores near
what cuBLAS's fp32 SGEMM gets from them (5.3 TFLOP/s measured; the fp32 peak is ~8), while the reference's runs on
tensor cores. See `docs/engine/02-q4k-tiled.md`.

Task 5 moves the GEMM onto the tensor cores, decoding Q4_K into `mma` fragments in registers: **1.96x faster than
the baseline step** (2.74 -> 1.40 s) and 2.7x faster than tiled, with the GEMM at 91-93% of cuBLAS on the same
weights pre-dequantized. The walk step gains only 7% over the reference (57.4 against 61.4 s): it batches 4
windows per pass, which already spread the reference's dequant. Since Task 5 the painter and the game run `mma` by
default (`VECWORLD_KERNEL=off` for diffusers). See `docs/engine/03-q4k-mma.md`.

Task 5b quantizes each linear's input per token to INT8 and runs the Q4_K codes on the INT8 tensor cores
(`mma.m16n8k32`, one instruction = one Q4_K sub-block): the GEMM runs ~1.5x faster than `mma` (~25 bf16-equivalent
TFLOP/s against ~17), and the step drops **1.40 -> 1.15 s** (the GEMM from ~0.99 to ~0.74 s per pass, est.). It is
lossy, so the walk gate admits it, not a match to the reference: with every linear on INT8 a borderline fog edge
(spot 1) crossed the seam limit. An ablation over 8 keep-lists found the pass needs the study's 5 worst families plus
the double blocks' residual-stream outputs (MLP-out and attention-outs) on `mma` (`config.FLUX_A16_KEEP`), 25% of the
token-facing FLOPs. The walk step drops 57.4 -> 48.9 s. See `docs/engine/04-w4a8.md`.

Task 5c moves the modulation linears (one row per image, 27% of the weights, 0.04% of the math) off the naive kernel
onto a GEMV that streams their 1.815 GB of Q4_K once per pass at 165 GB/s: **35.1 -> 11.0 ms per pass**, close to
the 9.7 ms the bytes allow. M = 1 is memory-bound, so here the 4-bit weights do buy time (3.6x fewer bytes than
bf16). The step drops **1.15 -> 1.12 s**: the brief's "0.29 s" was modulation's share of the dequant, already gone
since Task 3. See `docs/engine/05-modulation-gemv.md`.

Task 6 fuses the work between the GEMMs. Profiled first, the brief's norm / modulate / residual glue was 68 ms of the
1.12 s pass, and the biggest non-GEMM cost was diffusers' fp32 RoPE (147 ms, 13%) plus the q / k RMSNorm (19 ms).
Three memory-bound kernels at 167-179 GB/s (89-96% of the read roof) replace them, and the q / k / stream cats go:
non-GEMM GPU time falls 325 -> 128 ms and the step **1.12 -> 0.93 s** (walk step 47.0 -> 39.5 s). Every pass rounds
where diffusers rounds, and the LayerNorm replays torch's own Welford order, so the result is bit-identical even on
the lossy w4a8 default (a two-pass LayerNorm was tried first and moved the w4a8 sample). Now the GEMMs are 85% of
the GPU time (Amdahl). See `docs/engine/06-glue-fusion.md`.

Task 7 replaces diffusers' group offload with three fixed-address weight slots the blocks stream through (block i in
slot i % 3, copies on a side stream ordered by events) and replays each block as a CUDA graph. Profiled first, the
Task 6 pass left the GPU idle 10.9% of the time, but almost none of it was launch overhead (gaps under 100 us: 0.34%):
it was group offload's per-block syncs and its blocking moves at the start and end of each pass. The slots remove
those (idle 0.09%), and the step drops **0.92 -> 0.85 s** (20.8 bf16-equivalent TFLOP/s); eager on the slots is
already as fast as the graphs, which buy a free CPU rather than speed. A 2-slot ring left a 13.5 ms stall per pass
(57 blocks is odd), hence 3. The walk step, at 4 windows a pass, gains ~1% (39.0 -> 38.8 s). See
`docs/engine/07-cuda-graphs.md`.
