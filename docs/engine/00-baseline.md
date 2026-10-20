# FLUX baseline on AC (Task 0)

Measured 2026-10-20 on mrjarvis (RTX 3050 Ti 4 GB), **on AC** (`power: ac`, GPU power limit 80 W, PCIe Gen4 x8 under load; the link idles at Gen1). Game server stopped during runs. Pipeline: the spike's route (diffusers + `flux1-schnell-Q4_K_S.gguf` + block-level group offload with `use_stream`), cached T5 embeddings, 4 steps, no CFG. Reproduce with `scripts/bench_flux.py --painter reference --what {tile512,step,pano,blocks,bw}`; raw summary numbers are in `bench/baseline.json`.

## Spike (battery) vs AC

| Measurement | Spike (battery, 35 W, Gen2) | AC | Speedup |
|---|---|---|---|
| 512x512 per step | 17.1 s | **2.73 s** (3 repeats: 2.75 / 2.80 / 2.83) | 6.3x |
| 512x512, 4 steps | 69.6 s | 11.2 s | 6.2x |
| 1024x512, 4 steps | 92.1 s (about 23 s/step) | 16.2 s (3.9 s/step) | 5.7x |
| Peak VRAM at 512 (allocated / smi) | 1.24 GB / 2380 MiB | 1.24 GB / 2380 MiB | same |
| Peak VRAM at 1024x512 | 1.72 GB / 3542 MiB | 1.72 GB / 3542 MiB | same |
| Peak process RSS (FLUX) | 9.6 GB | 9.2 GB | |
| Load (transformer + pipeline + hooks) | 16.2 s | 20-40 s (varies, page cache) | |
| 2 windows, sequential | 139.1 s | 22.6 s | 6.2x |
| 8-window pano, sequential | about 557 s (extrapolated) | about 90 s (extrapolated) | 6.2x |
| 8-window pano, batch 2 | about 380 s (est.) | **68.2 s** (4 chunks, measured) | |
| 8-window pano, batch 4 | about 300 s (est.) | **56.3 s** (2 chunks, measured) | |
| Double block, weights resident | 461 ms | 62.0 ms | 7.4x |
| Single block, weights resident | 260 ms | 37.1 ms | 7.0x |
| Compute per step, no offload | 18.7 s | 2.59 s | 7.2x |
| bf16 matmul 4096^2 | 5.2 TFLOPS | 19.2 TFLOPS | 3.7x |
| H2D pinned | 3.41 GB/s | **13.46 GB/s** | 3.9x |
| H2D pageable | 3.39 GB/s | 11.08 GB/s | 3.3x |
| SSD O_DIRECT sequential read | 1.8-1.9 GB/s | **2.96 GB/s** | |
| Copy / compute overlap (1 GB copy + 0.29 s compute) | overlapped | 0.287 s both vs 0.286 s compute alone: fully hidden | |

## Time breakdown, one 512x512 step (weights resident, 2.59 s)

| Part | Time | Share |
|---|---|---|
| GGUF Q4_K dequant (19 x 31.0 ms + 38 x 12.9 ms) | 1.08 s | 41.8% |
| Attention (SDPA, 57 calls x 1.13 ms) | 0.06 s | 2.5% |
| GEMM and glue (remainder) | 1.44 s | 55.7% |
| Copy, per step (6.66 GB / 13.46 GB/s) | 0.49 s, hidden by the prefetch stream | not on the critical path |

At 1024x512 (resident 3.65 s): dequant 1.08 s (29.7%), attention 0.21 s (5.8%), GEMM and glue 2.36 s (64.5%). Dequant is a fixed cost per step, so its share falls as tokens rise.

Offload overhead: the measured step (2.73 s) is 0.14 s (about 5%) above resident compute (2.59 s). Streaming is essentially free on AC, as the spike found on battery.

## What changed against the spike's conclusions

- The spike's ratios hold. Dequant is still the largest single item (42% against the spike's 45-50%), and copy is still hidden.
- The compute-vs-I/O gap narrowed. On AC the copy is 0.49 s against 2.59 s compute (19%), and a cold SSD stream (6.66 GB / 2.96 GB/s = 2.25 s/step) is now almost as long as compute. I/O becomes a limit once compute drops below about 2.3 s/step, which fused dequant-GEMM kernels would do. The streamer work is no longer "worth about 0 s" for a cold-weights case.
- Batching windows still pays: batch 2 costs 4.28 s/step and batch 4 costs 7.06 s/step, against 2.73 s/step for one window, so a window costs 2.14 s/step at batch 2 and 1.77 s/step at batch 4.
- With the fused-kernel hope of removing most of the 1.08 s dequant, the floor is about 1.5 s/step at 512.

## Notes on method

- Block numbers use real-shaped inputs (256 text tokens) with CUDA events, median of 5. Dequant is timed on its own, so it is not subtracted from a fused run; the GEMM and glue row is the remainder.
- A FLOP-count GEMM estimate (11.9 B params x 2 x 1280 tokens / 19.2 TFLOPS = 1.59 s) is larger than the 1.44 s remainder. It over-counts, because text-side weights only see 256 tokens, so treat GEMM as about 1.4 s.
- Pano 8-window numbers for batch 2 and 4 are measured over all 8 windows. The sequential figure is extrapolated from 2 windows.
- Batch 4 runs close to the 4 GB limit (peak allocated 2.59 GB, smi 3666 MiB); the allocator logged OOM retries and recovered.
- `ssd_pagecache_gbps` in the JSON (2.8) is a plain read right after the O_DIRECT pass and was not served from cache; ignore it.
