# Engine speed table

One row per engine task (Goal B). **Step s** = `tile512.s_per_pass_median` from `scripts/bench_flux.py --what
tile512` on AC: one 512x512 window, 1024 image + 256 text tokens, one transformer pass. **Achieved TFLOP/s** =
`flop_per_step(1024, 256)` (17.67 TFLOP, ledger section 1.6) / step s. **% peak** is against the measured
19.1 TFLOP/s bf16 matmul peak (`bench/baseline.json`, blocks). **Walk step** = `step.s` from `--what step`, a `sharp`
walk step's wall time (not gated).

| Task | Flags | Step s | Achieved TFLOP/s | % bf16 peak | Walk step s | Peak alloc GB (tile512 / step) | Quality gate | Git |
|---|---|---|---|---|---|---|---|---|
| 0: baseline (diffusers + GGUF + group offload) | `--painter reference` | 2.74 | 6.45 | 34 | (not measured) | 1.24 / (not measured) | the reference | 895c75e |
| 1: FluxPainter on the diffusers reference | `--painter reference` | 2.79 | 6.33 | 33 | 61.4 | 1.23 / 1.37 | the reference | a47474a (tile512), 60252f7 (step) |
| 3: fused Q4_K, naive (correctness reference) | `--painter engine`, `VECWORLD_KERNEL=naive` | 22.98 | 0.77 | 4.0 | 945.1 | 1.02 / 1.33 | lossless: block max-rel 8.3e-3 (< 2e-2), forward rel L2 4.2% (< 6%), picture 29.87 dB (>= 28) | 6f01119-dirty (this task's diff, before commit) |
| 4: fused Q4_K, tiled (shared memory, CUDA cores) | `--painter engine`, `VECWORLD_KERNEL=tiled` | 3.77 | 4.69 | 24.5 | 156.3 | 1.02 / 1.33 | lossless: block max-rel 8.3e-3 (< 2e-2), forward rel L2 4.7% (< 6%), picture 29.05 dB (>= 28); vs naive max-rel <= 3.5e-3. M < 32 linears (the M = 1 modulation layers, ~1% of the pass) run the naive kernel | 49f56bb-dirty (this task's diff, before commit) |
| 5: fused Q4_K, tensor cores (bf16 `mma.sync`, W4A16); **the default** | `--painter engine`, `VECWORLD_KERNEL=mma` (unset = mma; `off` = diffusers) | **1.40** | **12.62** | **66.1** | 57.4 | 1.02 / 1.33 | lossless: block max-rel 8.3e-3 (< 2e-2), forward rel L2 4.4% (< 6%), picture 29.37 dB (>= 28); vs naive max-rel <= 6.3e-3. GEMM 90.6-93.4% of torch.matmul at M = 1024/1280, 92-105% at M = 256 (`bench/t5-gemm.json`). M < 7 linears run naive | 075df02-dirty (this task's diff, before commit) |

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
