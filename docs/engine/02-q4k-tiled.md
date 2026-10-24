# 02: The Q4_K kernel, tiled through shared memory (Task 4)

**What this is:** the second stage of the engine's Q4_K GEMM. It does the same arithmetic as the naive kernel
(`01-q4k-naive.md`): the same bit-exact decode and an fp32 sum on CUDA cores. What changes is *where the data
comes from*. Each block of threads stages a tile of x and a tile of decoded weights in shared memory, and every
weight is decoded once per tile instead of once per output. It still runs on CUDA cores, not tensor cores; that
comes in Task 5.

Turn it on with `VECWORLD_KERNEL=tiled`. (Unset, the painter runs the `mma` kernel since Task 5.)

## The problem: the naive kernel re-reads and re-decodes everything

In the naive kernel, thread (m, n) reads the whole weight row n and the whole x row m, and decodes all K weights
of row n by itself. A 512 window pass has M = 1280 rows, so every weight is decoded 1280 times, and every x value
is read N times. The arithmetic is cheap, but each multiply-add costs a decode, two conversions and some loads.
L1 catches part of the re-reading, but the kernel peaks at about 0.85 TFLOP/s.

The cure is **reuse**. A thread block decodes a tile of weights once, keeps it in fast on-chip memory, and lets
all of its threads use it.

## The memory hierarchy, and what each level is for here

```
 level            size on the 3050 Ti          speed              what the tiled kernel keeps there
 ---------------  ---------------------------  -----------------  ------------------------------------------------
 registers        64K x 32-bit per SM (20 SMs) every cycle        8 x 8 fp32 accumulators per thread; the next
                                                                  K step's raw bytes (prefetch)
 shared memory    up to 100 KB per SM          ~1-2 cycles/read   As[32][128] (x, fp32) and Ws[32][128] (decoded W)
                  (shared by a block's threads)
 L1 / L2          128 KB per SM / 2 MB          tens / hundreds    whatever is reused across blocks (see grouping)
 VRAM (GDDR6)     4 GB                          192 GB/s           the packed Q4_K weights and x, read in 16-byte
                                                                  vector loads
```

A good GPU kernel moves each byte as few levels as it can, and does as much arithmetic as possible per byte that
it moves. The naive kernel does about 1 FMA per value it loads. The tiled kernel loads 8 + 8 values from shared
memory for 64 FMAs, and loads each value from VRAM into shared memory only once per tile.

## How the tiled kernel works

`src/vecworld/engine/csrc/q4k.cu`, `gemm_tiled_kernel`:

```
 y tile 128 x 128 (one thread block, 256 threads = 16 x 16, each owns 8 x 8 outputs)

 for each K step (32 weights = one Q4_K sub-block):
   1. global -> registers   x:  128 rows x 64 bytes (16-byte loads, 2 per thread)
                            W:  128 rows x 16 code bytes + 16-byte super-block header (1 + 1 per thread)
                                [issued one step ahead: they arrive while step 2-3 of the previous step run]
   2. registers -> shared   As[k][m] = float(x[m, k])           (bf16 -> fp32)
                            Ws[k][n] = d_sc * q - dm            (the naive kernel's exact decode)
   3. shared -> registers   for k in 0..31: 2 float4 of As + 2 float4 of Ws -> 64 FMAs into acc[8][8]

 epilogue: + bias, round to bf16, 8-byte stores
```

- **Decode once per tile.** A weight decoded into `Ws` is read by the 16 threads that share its column. Those threads
  cover all 128 rows of x in the tile. Per K step the block decodes 128 x 32 = 4096 weights and does
  128 x 128 x 32 = 524,288 FMAs, so the decode is about 1% of the work (in the naive kernel it is 100%: one decode
  per FMA).
- **Each weight byte fetched once.** A Q4_K run of 32 bytes holds two sub-blocks, the low nibbles and the high
  nibbles. The kernel loads the run into registers once and decodes it twice, once for each of two K steps.
- **Coalescing.** The loads are 16-byte vectors. On x, each row's 64 bytes per K step are read by 2 threads, each
  making 2 loads of 16 bytes (pieces c and c + 2 of the row's four; a warp's load covers 32 rows, one piece each). On
  W, two threads read the 32 bytes of a run. A warp's load uses half of each 32-byte sector it touches; the other
  half is read by a neighbouring load (the next piece, or the other half of the run), usually from L1. Weight rows are 1.7 to 8.6 KB apart, so a warp can't read
  one long contiguous stretch of W without a re-layout of the weights. That re-layout is a Task 5 question. The
  alternative mapping for x, 4 lanes per row so a warp reads 512 contiguous bytes, measured the same (see below).
- **No bank conflicts.** Consecutive lanes store consecutive m (or n), so a warp's stores hit 32 different banks. The
  compute reads are float4: a thread owns rows `ty*4..+3` and `64+ty*4..+3`, and the same split for columns, so
  a warp's 16 column reads are 16 adjacent 16-byte pieces.
- **Register prefetch.** While step k computes, the loads for step k+1 are already in flight into registers. Only
  after the compute (and a barrier) are they decoded into shared memory.
- **Grouped tile order.** Blocks are numbered so that pairs of M tiles walk along N together (`GROUP_M = 2`). The
  blocks resident at the same time then share weight tiles *and* x tiles in L2. That was worth 10 to 25% on the
  K = 3072 layers.
- **Ragged edges.** Rows past M or N are never read. Their registers hold zeros, which decode to 0 (d = dmin = 0),
  and the epilogue checks bounds on every store. K is always a multiple of 256 (`check_packed`).
- **Small M goes to the naive kernel.** FLUX's modulation linears run on one vector per window (M = 1), and a
  128-row tile would be 127/128 padding. For a 3072 -> 18432 layer at M = 1 that is 4.0 ms tiled against 0.7 ms
  naive. `Q4KLinear` sends M < 32 (`TILED_MIN_M`) to the naive kernel. The measured crossover is M = 24 to 32. The
  extension's `q4k_gemm_tiled` itself always tiles, and the tests check it at M = 1 and 7 too.
- **Alignment.** Both kernels read with vector loads, so the extension now rejects a misaligned x or weight view
  (`TORCH_CHECK`, 16 bytes). `Q4KLinear` copies a misaligned *activation* (a fresh copy of an x that
  is already made per call; fresh CUDA allocations are aligned, so it fires only for exotic views). It raises,
  naming the layer, on a misaligned packed *weight*: a weight layout that breaks alignment is a bug to fix where the
  layout is made, not something to pay a weight-sized copy for on every call. `test_q4k_gemm_misaligned` covers
  both.

## Choosing the tile shape (measured)

On isolated FLUX linears at M = 1280, `bench_tiles.py` (scratch, not committed), on AC with the game server
stopped. The range is over five shapes: 3072 -> 3072, 3072 -> 12288, 3072 -> 18432, 12288 -> 3072 and
15360 -> 3072. Laptop clocks move numbers by about ±5% between runs.

| Variant (tile M x N, outputs per thread, threads) | TFLOP/s |
|---|---|
| naive (Task 3) | 0.6 .. 0.9 |
| 128 x 128, 8 x 8, 256, plain row order, no register cap | 3.2 .. 3.5 |
| 128 x 64, 8 x 8, 128 | 3.7 .. 5.0 |
| 64 x 128, 8 x 8, 128 | 2.0 .. 4.8 |
| 64 x 64, 4 x 8 / 8 x 4, 128 | 2.9 .. 3.9 |
| 32 x 128, 4 x 8, 128 | 2.1 .. 2.7 |
| 16 x 128, 4 x 8, 64 | 1.1 .. 2.3 |
| 128 x 64 / 64 x 128 / 64 x 64, double-buffered shared memory | 1.9 .. 4.3 (lower occupancy: worse) |
| 128 x 64, 4 blocks per SM forced (register cap 128) | 3.2 .. 3.9 |
| 128 x 64, coalesced x loads (4 lanes per row) | same as 128 x 64 (±1%) |
| 128 x 64, grouped order, GROUP_M = 2 / 4 / 10 | 4.7 .. 5.2 / 4.4 .. 5.1 / 3.0 .. 5.0 |
| 128 x 128, 2 blocks per SM (register cap 128), plain order | 3.5 .. 5.5 |
| 128 x 128, 2 blocks per SM, GROUP_M = 4 | 4.6 .. 5.7 |
| **128 x 128, 8 x 8, 256, 2 blocks per SM, GROUP_M = 2 (chosen)** | **5.2 .. 5.6** |
| for scale: cuBLAS fp32 SGEMM, 4096^3, TF32 off, measured (not the fp32 peak, which is ~8 TFLOP/s: 2560 cores x 2 x ~1.6 GHz sustained) | 5.3 |

So the tiled kernel runs a Q4_K GEMM about as fast as cuBLAS runs a plain fp32 GEMM on the same CUDA cores, and
6 to 7x faster than the naive kernel. It is still far from the 19.1 TFLOP/s of the bf16 tensor cores: that gap is
Task 5's.

## Numbers (on AC, `scripts/bench_flux.py --painter engine`, VECWORLD_KERNEL=tiled)

Measured 2026-10-24 on mrjarvis, on AC (`power: ac` in every file; `/sys/class/power_supply/*/online` = 1 before
and after each run), with the game server stopped. The benches ran on this task's diff before commit
(`git: 49f56bb-dirty`).

| | Reference (diffusers) | Naive (Task 3) | **Tiled (Task 4)** |
|---|---|---|---|
| Step: one 512 window pass (`tile512.s_per_pass_median`) | 2.74 s baseline (Task 1 painter: 2.79 s) | 22.98 s | **3.77 s** |
| Achieved TFLOP/s (17.67 TFLOP per pass) | 6.45 baseline (Task 1 painter: 6.33) | 0.77 | **4.69** |
| % of the measured 19.1 TFLOP/s bf16 peak | 34% baseline (Task 1 painter: 33%) | 4.0% | **24.5%** |
| Walk step (`step.s`, `sharp`, 40 window passes) | 61.4 s (Task 1 painter) | 945.1 s | **156.3 s** (3.91 s per window pass, 4.52 TFLOP/s) |
| Resident step, weights on the GPU (`blocks.512x512.resident_step_s`) | 2.59 s | 21.25 s | **3.58 s** (4.94 TFLOP/s) |
| ... of which dequant | 1.08 s (42%) | 0.00 s | **0.00 s** (no bf16 weight is ever written) |
| ... of which attention (SDPA) | 0.06 s | 0.07 s | 0.07 s |
| Double / single block, 512 | 61.9 / 37.1 ms | 399.5 / 359.5 ms | 65.7 / 61.4 ms |
| Resident step at 1024x512 | 3.65 s | 40.05 s | 6.55 s (5.11 TFLOP/s) |
| Peak allocated, tile512 / walk step | 1.23 / 1.37 GB | 1.02 / 1.33 GB | 1.02 / 1.33 GB |

Files: `bench/t4-tiled.json` (tile512), `bench/t4-tiled-step.json` (walk step) and `bench/t4-tiled-blocks.json`.

**What the numbers say.** The tiled kernel is **6.1x faster than the naive one** per step (22.98 -> 3.77 s), and
the dequant share is gone: 0.00 s, against the reference's 1.08 s. The step is still **1.38x slower than the
reference** (3.77 against 2.74 s), as the brief expected. The reference spends 1.08 s unpacking weights but then
runs its GEMMs on tensor cores at up to 19 TFLOP/s. This kernel unpacks for free but does its GEMMs on CUDA cores.
Their fp32 peak is about 8 TFLOP/s; cuBLAS's own fp32 SGEMM measured 5.3 TFLOP/s here. At 4.7 to 5.1 TFLOP/s on the
whole pass this kernel is close to what cuBLAS gets from the same cores, so further CUDA-core tuning could win
perhaps 10 to 30% but is unlikely to close the 1.38x gap. Tensor cores (Task 5) can: the same
GEMM work at even half the bf16 peak would take about 1.8 s.

A double and a single block do about the same FLOPs (about 113M weights each, over 1280 tokens). On CUDA cores they
take about the same time (66 and 61 ms). In the reference, the single block is much cheaper (37 ms), because its
few, large GEMMs (3072 -> 21504) suit cuBLAS's tensor cores best.

## Correctness

The same gates as Task 3, now parametrized over the kernel (`tests/test_q4k.py`, `KERNELS = ["naive", "tiled"]`),
plus a direct check against the naive kernel:

| Gate | Bar | Naive | Tiled |
|---|---|---|---|
| Dequant bit-exact vs gguf-py (shared decode) | `torch.equal` | yes | yes (the same `decode`) |
| GEMM vs `x @ deq(W)^T` on 3 real tensors, and ragged M = 1, 7, 32, 333, 1792 with N = 1000 | max-rel < 2e-2 | pass | pass |
| Odd N = 1001 with bias (tiled's per-element epilogue), M = 7 and 333, `test_q4k_gemm_odd_n` | max-rel < 2e-2 | pass | pass |
| `Q4KLinear("tiled")` picks naive at M = 1 and 31, tiled at M = 32 and 1280, `test_q4k_tiled_small_m_runs_naive` | exact call list | - | pass |
| **Tiled vs naive** directly (`test_q4k_gemm_matches_naive`): 3 real tensors at M = 256, and N = 1000 at M = 1, 7, 31, 128, 129, 333, 1792 and N = 1001 at M = 7, 333, calling the raw kernels | max-rel < 8e-3 (one bf16 step of the largest output) | - | 5.7e-4 .. 3.5e-3 |
| One block (double 0, single 0) vs diffusers, primary | max-rel < 2e-2 | 8.3e-3 / 5.5e-3 | 8.3e-3 / 5.5e-3 |
| One whole forward pass vs diffusers | rel L2 < 0.06 | 4.18% (max-rel 0.127) | 4.72% (max-rel 0.162) |
| 4-step picture at seed 0 vs diffusers | PSNR >= 28 dB | 29.87 dB | 29.05 dB |

The tiled kernel decodes every weight bit-exactly as the naive one does; only the order of the fp32 sum differs.
The forward pass lands between the naive kernel (4.2%) and plain `torch.matmul` on the exact weights (4.5%,
Task 3). By eye the picture is the same forest, path, light shafts and ferns; it differs only in needle texture.

## What to read next

- Task 5 (`mma`): feed the decoded bf16 fragments to `mma.sync.m16n8k16` instead of fp32 FMAs. The tensor cores
  do 16 x 8 x 16 multiply-adds per instruction, and the 3050 Ti's bf16 peak is 19.1 TFLOP/s, against ~8 fp32 on CUDA cores (5.3 measured for cuBLAS SGEMM).
  The tiling, decode-once and prefetch in this kernel carry over; the inner loop changes.
- `docs/engine/speed.md`: the running table.
