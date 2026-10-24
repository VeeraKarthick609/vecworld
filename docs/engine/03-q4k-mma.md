# 03: The Q4_K kernel on tensor cores (Task 5, W4A16)

**What this is:** the third stage of the engine's Q4_K GEMM, and since Task 5 the **default** for the FLUX painter
and the game. The arithmetic is the same as in the naive and tiled kernels (`01-q4k-naive.md`, `02-q4k-tiled.md`):
y = x W^T with W decoded from Q4_K. The difference is the unit that does the multiply-adds. The tiled kernel used the
CUDA cores, at about 5 TFLOP/s. This one uses the **tensor cores** through `mma.sync.m16n8k16` (bf16 in, fp32
accumulate), which the 3050 Ti runs at up to 19.1 TFLOP/s. The weights still never exist as a bf16 tensor in VRAM.
Each warp decodes exactly the 4-bit codes its next tensor-core instruction needs, into registers, just before
issuing it.

It runs by default. `VECWORLD_KERNEL=off` (or `reference`) puts the diffusers path back; `naive` and `tiled` pick
the earlier stages.

## The problem: 1.08 s of dequant, and then CUDA cores

The diffusers reference spends 1.08 s of its 2.74 s step unpacking Q4_K to bf16 in VRAM before every GEMM
(`00-baseline.md`). Task 4 removed that unpacking (dequant share 0), but it did its GEMMs on CUDA cores at ~5
TFLOP/s, so its step was 3.77 s, slower than the reference. The reference's GEMMs ran on tensor cores. To beat it
we need both: no unpacking pass **and** tensor cores.

## Warps, fragments and `mma`

A **warp** is 32 threads that issue each instruction together. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
is one warp-wide instruction: the warp multiplies a 16 x 16 bf16 tile of A (here x) by a 16 x 8 bf16 tile of B (here
W^T) and adds the 16 x 8 product into fp32 accumulators. That is 2048 multiply-adds per instruction. No single
thread holds a whole tile. Each thread holds a **fragment**, a few registers whose positions the PTX spec fixes.
With g = lane / 4 and t = lane % 4:

```
 A (16 x 16, x rows m, k)     4 regs = 8 bf16:  rows g and g+8, k = 2t, 2t+1, 2t+8, 2t+9
 B (16 x 8, k x column n)     2 regs = 4 bf16:  column n = g,  k = 2t, 2t+1  |  2t+8, 2t+9
 C (16 x 8, fp32)             4 floats:         rows g and g+8, columns 2t, 2t+1

         k ->  0 1 2 3 4 5 6 7 8 9 ...15
 B col n=g:    t=0 t=0 t=1 t=1 ...  (t=0 again at 8, 9) ...       lane 4g + t holds 4 values of column g
```

A is loaded from shared memory with **`ldmatrix.x4`**: each of the 32 lanes passes the address of one 16-byte row
piece (8 bf16), and the hardware hands every lane its A fragment. B is the interesting part, because B is not bf16
in memory. It is Q4_K.

## Decoding straight into the B fragment

A B fragment needs, for weight row n = g, the four weights at k = 2t, 2t+1, 2t+8, 2t+9 of a 16-wide k step. Q4_K
stores a super-block's 256 weights as 4 runs of 32 bytes. Byte i of run r holds weight 64r + i in its low nibble,
and weight 64r + 32 + i in its high nibble. So for the 64 weights of one run:

```
 run r, 32 bytes:   byte 0 .. 15                        byte 16 .. 31
   low nibbles:     k16 step 0 (k = 0..15 of the run)   step 1 (k = 16..31)       sub-block 2r
   high nibbles:    step 2 (k = 32..47)                 step 3 (k = 48..63)       sub-block 2r+1

 lane (g, t) needs bytes 2t, 2t+1 and 2t+8, 2t+9 (+16 for steps 1 and 3) of row g's run
   -> two 16-bit shared loads = 4 code bytes = the codes of two B fragments (low nibbles, high nibbles)
```

All four k values of a fragment fall inside one 32-weight sub-block, so one pair of multipliers (d * sc, dmin * m)
decodes them. Per weight, the decode is:

1. A byte permute (`__byte_perm(v, 0x4B000000, ...)`) builds the float 2^23 + q, so there is no int-to-float conversion.
2. One subtract gives q exactly.
3. `q * (d * sc) - dmin * m`, with the same two fp32 roundings as gguf-py (so the fp32 weight is bit-exact).
4. One `cvt.rn.bf16x2.f32` packs two weights into a register.

The fragment goes straight into `mma`. **No repacking:** the kernel reads the GGUF super-blocks as they are, so the
packed parameter, the dequant test (bit-exact against gguf-py) and the naive fallback all share one layout. A row's
48 shared bytes per stage (16 header + 32 run) put the 8 rows x 2 words a warp reads on 16 different banks, so there
are no bank conflicts.

## The kernel

```
 VRAM: x (bf16)                        VRAM: W (Q4_K, 144 B per 256 weights)
   | cp.async 16 B, zero-fill past M      | cp.async 16 B: header (d, dmin, scales) + run r, zero-fill past N
   v                                      v
 shared, 2 stages:  A[128][64] bf16 (chunks XOR-swizzled by row)  |  W[BN rows][48 B] still packed
   | ldmatrix.x4                          | 2 x 16-bit loads -> nibbles -> fp32 decode -> bf16x2
   v                                      v
 A fragments (8 m16 tiles per warp)    B fragments (2 n8 tiles per warp)
                  \                       /
                   mma.sync m16n8k16, fp32 accumulators (128 x 16 per warp = 64 floats per thread)
                                  |
           epilogue: + bias, round to bf16, store pairs of outputs
```

- **Tile:** 128 x 128 outputs per block, 8 warps, each warp a 128-row x 16-column strip (8 x 2 `mma` per k16 step).
  All warps sit side by side along N, so **every weight is decoded exactly once per block**. With 2 warps along M
  (64 x 32 warp tiles), each weight was decoded twice; that cost about 5% (table below).
- **K step:** 64 weights, one 32-byte run (two sub-blocks), so a stage's W bytes are one 16-byte header plus one
  run per row.
- **Pipeline:** `cp.async` copies global to shared without passing through registers. Two stages: while the warps
  compute stage k, stage k+1 is in flight. One `__syncthreads` per K step. Two blocks share an SM (2 x 44 KB of
  shared memory, 128 registers a thread, no spills), so one block's loads overlap the other's math too.
- **Tile order:** blocks go in groups of 4 M tiles walking along N, as in Task 4, so blocks running at the same time
  share x and W in L2. With 2 blocks per SM the group size made no measurable difference (2, 4, 8, 16 within 1%).
- **Few tiles:** when the 128 x 128 grid has fewer than 4 waves (8 x 20 SMs = 160 tiles), the launcher uses 128 x 64
  tiles (4 warps, 3 blocks per SM) so the last wave doesn't leave SMs idle. In FLUX that is the M = 256 text-stream
  linears with N = 3072 or 9216.
- **Ragged edges:** rows past M or N are zero-filled by `cp.async` (src-size 0). A zero header decodes to d = dmin = 0,
  so those weights are 0. The epilogue stores pairs where both are inside N, single values otherwise.

### Small M goes to naive

`Q4KLinear("mma")` hands calls with M < `MMA_MIN_M` = 7 to the naive kernel. A 128-row tensor-core tile at M = 1
does 128x the useful work. On FLUX's modulation linear (3072 -> 18432, M = windows per pass), measured:

| M | 1 | 2 | 4 | 6 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| naive, ms | 0.65 | 0.63 | 0.67 | 0.83 | 1.07 | 2.18 | 4.31 | 8.57 |
| mma, ms | 0.84 | 0.84 | 0.84 | 0.85 | 0.84 | 0.85 | 0.86 | 0.86 |
| tiled, ms | 3.69 | 3.69 | 3.71 | 3.73 | 3.73 | 3.78 | 3.82 | 3.92 |

They cross at M = 6..8. The mma time is flat because reading the 32 MB weight dominates. In FLUX only the modulation
linears are this small (M = 1..4). The raw `q4k_gemm_mma` always runs the tensor-core path, and the tests check it at
M = 1, 7 and 31 against naive.

## Why decoding in registers kills the 1.08 s

The reference's 1.08 s is memory traffic. Every step it reads 6.7 GB of Q4_K, writes 23.7 GB of bf16 weights to VRAM
through a chain of PyTorch ops (each writing and re-reading an intermediate), and then cuBLAS reads the bf16 weights
back. Here no weight is ever written. A block reads each Q4_K byte once from L2/VRAM into shared memory (4.5 bits per
weight), and the decode is register arithmetic on the CUDA cores, about 5 instructions per weight. Those are separate
units from the tensor cores, so they overlap with the `mma`s. Each weight is decoded once per 128-row M tile: 10
times a pass at M = 1280. The `blocks` bench shows the result: dequant 0.00 s, and the double block takes 23.8 ms
against the reference's 61.9 ms.

## The GEMM against cuBLAS (`--what gemm`)

`scripts/bench_flux.py --painter engine --what gemm` (VECWORLD_KERNEL=mma) times `Q4KLinear` on a real Q4_K weight
of each FLUX linear shape. It compares against `torch.matmul` (cuBLAS) on the **same weight already dequantized to
bf16**: cuBLAS's GEMM without the reference's dequant. Medians of 20 runs after warm-up, the better of two rounds
each. AC, server stopped, `bench/t5-gemm.json`:

| M | K | N | weight | mma ms | matmul ms | mma TFLOP/s | matmul TFLOP/s | % of matmul |
|---|---|---|---|---|---|---|---|---|
| 1024 | 3072 | 9216 | double_blocks.0.img_attn.qkv | 3.312 | 3.066 | 17.51 | 18.91 | 92.6 |
| 1024 | 3072 | 3072 | double_blocks.0.img_attn.proj | 1.135 | 1.058 | 17.03 | 18.27 | 93.2 |
| 1024 | 3072 | 12288 | double_blocks.0.img_mlp.0 | 4.449 | 4.156 | 17.38 | 18.60 | 93.4 |
| 1024 | 12288 | 3072 | double_blocks.0.img_mlp.2 | 4.571 | 4.157 | 16.91 | 18.60 | 90.9 |
| 1280 | 3072 | 21504 | single_blocks.0.linear1 | 9.697 | 8.944 | 17.44 | 18.91 | 92.2 |
| 1280 | 15360 | 3072 | single_blocks.0.linear2 | 6.948 | 6.294 | 17.39 | 19.19 | 90.6 |
| **M = 1024 / 1280** | | | | | | | | **min 90.6, overall 91.9** |
| 256 | 3072 | 9216 | double_blocks.0.img_attn.qkv | 0.867 | 0.847 | 16.71 | 17.12 | 97.6 |
| 256 | 3072 | 3072 | double_blocks.0.img_attn.proj | 0.301 | 0.316 | 16.05 | 15.27 | 105.1 |
| 256 | 3072 | 12288 | double_blocks.0.img_mlp.0 | 1.135 | 1.047 | 17.03 | 18.47 | 92.2 |
| 256 | 12288 | 3072 | double_blocks.0.img_mlp.2 | 1.166 | 1.085 | 16.57 | 17.81 | 93.1 |
| **M = 256** | | | | | | | | **min 92.2, overall 95.0** |

So the brief's bar, at least 85% of `torch.matmul` on the M = 1024 / 1280 shapes, is met on every shape (90.6 to
93.4%). At M = 256 the 128 x 64 tiles keep it at 92 to 105%. ("Overall" is total matmul time over total mma time
across the shapes.)

**What limits the rest.** Nsight Compute can't read the GPU's counters here without root (`ERR_NVGPUCTRPERM`), so
the limit was measured by ablation instead:

- A timing-only build that skips the decode (B fragments = raw code bits, wrong results) ran at 96 to 104% of
  `torch.matmul`. So the decode, the scale unpacking and the 16-bit code loads cost the remaining 5 to 8%. Each
  weight is still decoded 10 times a pass (once per 128-row M tile).
- The other ~4% to cuBLAS's level is the plain GEMM machinery: 2 pipeline stages instead of cuBLAS's deeper ones (3
  stages fit only 1 block per SM, which measured worse), and `mma`-to-`ldmatrix` scheduling.
- Rewriting the decode with one FMA instead of mul + sub (dropping gguf's exact rounding) changed nothing within
  noise, so the decode is not ALU-bound on its own. It costs issue slots and registers next to the `mma`s.

### Tile shapes tried (same harness, % of `torch.matmul` in the same run, M = 1024 / 1280 shapes)

| Block tile, warp tile, stages, blocks per SM | % of matmul |
|---|---|
| 128 x 128, 64 x 32 warps (2 x 4), 3 stages, 1 | 77 .. 82 |
| the same, 4 stages | 77 .. 89 |
| 128 x 128, 32 x 64 warps (4 x 2), 3 stages, 1 | 69 .. 81 |
| 128 x 256, 64 x 64 warps, 3 stages, 1 | 80 .. 89 |
| 64 x 128, 32 x 32 warps, 3 stages, 2 | 74 .. 87 |
| 64 x 64, 32 x 32 warps, 3 stages, 3 | 67 .. 79 |
| 128 x 128, 64 x 32 warps, **2 stages, 2 per SM** | 86 .. 101 |
| the same, GROUP 2 / 8 / 16 instead of 4 | within 1% |
| the same, FMA decode (one rounding) | within 1% |
| the same, no decode (timing only, wrong results) | 96 .. 110 |
| **128 x 128, 128 x 16 warps (1 x 8: each weight decoded once), 2 stages, 2 per SM (chosen)** | **90 .. 99** |
| the same, no decode (timing only) | 96 .. 104 |
| 128 x 64, 128 x 16 warps, 2 stages, 3 per SM (chosen when < 160 tiles) | 86 .. 95 (M = 256: 91 .. 105) |
| 128 x 64, 3 stages | 71 .. 77 (3 x 57 KB of shared memory: only 1 block fits per SM) |

The lesson: on this GPU, **two resident blocks per SM** mattered more than pipeline depth or tile size. The rest
came from not decoding a weight twice.

## Numbers (on AC, `scripts/bench_flux.py --painter engine`, VECWORLD_KERNEL=mma)

Measured 2026-10-24 on mrjarvis, on AC (`power: ac` in every file; ACAD online = 1 before and after each run),
with the game server stopped, on this task's diff before commit (`git: 075df02-dirty`).

| | Reference (diffusers) | Naive (Task 3) | Tiled (Task 4) | **mma (Task 5)** |
|---|---|---|---|---|
| Step: one 512 window pass (`tile512.s_per_pass_median`) | 2.74 s baseline (Task 1 painter: 2.79 s) | 22.98 s | 3.77 s | **1.40 s** |
| Achieved TFLOP/s (17.67 TFLOP per pass) | 6.45 | 0.77 | 4.69 | **12.62** |
| % of the measured 19.1 TFLOP/s bf16 peak | 34% | 4.0% | 24.5% | **66.1%** |
| Walk step (`step.s`, `sharp`, 40 window passes in batches of 4) | 61.4 s (Task 1 painter) | 945.1 s | 156.3 s | **57.4 s** (1.44 s per window pass, 12.27 TFLOP/s) |
| Resident step, weights on the GPU (`blocks.512x512.resident_step_s`) | 2.59 s | 21.25 s | 3.58 s | **1.30 s** (13.59 TFLOP/s) |
| ... of which dequant | 1.08 s (42%) | 0.00 s | 0.00 s | **0.00 s** |
| ... of which attention (SDPA) | 0.06 s | 0.07 s | 0.07 s | 0.07 s |
| Double / single block, 512 | 61.9 / 37.1 ms | 399.5 / 359.5 ms | 65.7 / 61.4 ms | **23.8 / 22.2 ms** |
| Resident step at 1024x512 | 3.65 s | 40.05 s | 6.55 s | **2.42 s** (13.83 TFLOP/s) |
| Peak allocated, tile512 / walk step | 1.23 / 1.37 GB | 1.02 / 1.33 GB | 1.02 / 1.33 GB | 1.02 / 1.33 GB |

Files: `bench/t5-mma.json` (tile512), `bench/t5-mma-step.json`, `bench/t5-mma-blocks.json`, `bench/t5-gemm.json`.

**What the numbers say.**

- The step is **1.96x faster than the reference** (2.74 -> 1.40 s), **2.7x faster than tiled** and 16x faster than
  naive. That is inside the ledger's estimate: 1.6 s if the GEMM matched cuBLAS, 1.8 to 2.2 s at 65 to 85% of it.
  The GEMM matches cuBLAS more closely than that estimate assumed (91 to 93%).
- Dequant is no longer in the picture at all: 0.00 s of a 1.30 s resident step. Attention is 0.07 s. The other
  ~1.2 s is the linears (now ~17 TFLOP/s each) plus the glue: norms, modulation, RoPE and GELU, which Task 6
  targets.
- The streamed step (1.40 s) is only 0.10 s above the resident one (1.30 s), so group offload's H2D copies of the
  Q4_K weights mostly overlap the compute.
- **The walk step gains much less: 57.4 s against the reference's 61.4 s (7%).** A walk step runs its windows 4 at a
  time (FLUX_BATCH = 4). The reference dequantizes once per batch pass, not once per window, so its dequant was
  already spread over 4 windows, and its batched GEMMs (M up to 4 x 1280) run at cuBLAS speed. Per window pass our 1.44 s
  is about the same as tile512's 1.40 s: this kernel gains nothing from batching, because it never paid a per-pass
  dequant. The rest of the walk step's wall time is VAE encode / decode and text. Window batching and the glue (Tasks
  6 and 7) are where the walk step gains next.

## Correctness

The same gates as Tasks 3 and 4, parametrized over the kernel (`tests/test_q4k.py`, `KERNELS = ["naive", "tiled",
"mma"]`), plus the direct check against naive:

| Gate | Bar | Naive | Tiled | **mma** |
|---|---|---|---|---|
| Dequant bit-exact vs gguf-py (GGUF layout, no repacking) | `torch.equal` | yes | yes | yes (same layout; the fragment decode uses the same two roundings) |
| GEMM vs `x @ deq(W)^T`: 3 real tensors (M = 256, bias), ragged M = 1, 7, 32, 333, 1792 with N = 1000, N = 1001 with bias at M = 7, 333 | max-rel < 2e-2 | pass | pass | pass |
| **mma vs naive** directly (`test_q4k_gemm_matches_naive`, raw kernels): 3 real tensors at M = 256; N = 1000 at M = 1, 7, 31, 128, 129, 333, 1792; N = 1001 at M = 7, 333 | max-rel < 8e-3 | - | 5.7e-4 .. 3.5e-3 | **2.9e-3 .. 6.3e-3** |
| Small M goes to naive (`test_q4k_mma_small_m_runs_naive`: naive at M = 1, 6; mma at 7, 1280) | exact call list | - | - | pass |
| Misaligned x copied, misaligned packed weight raises naming the module (`test_q4k_gemm_misaligned`) | as Task 4 | pass | pass | pass |
| One block (double 0, single 0) vs diffusers, primary | max-rel < 2e-2 | 8.3e-3 / 5.5e-3 | 8.3e-3 / 5.5e-3 | **8.30e-3 / 5.49e-3** |
| One whole forward pass vs diffusers | rel L2 < 0.06 | 4.18% | 4.72% | **4.36%** (max-rel 0.129) |
| 4-step picture at seed 0 vs diffusers | PSNR >= 28 dB | 29.87 dB | 29.05 dB | **29.37 dB** |
| Default kernel (`test_kernel_defaults_to_mma`): unset or empty -> mma, `off` / `reference` -> diffusers | exact | | | pass |

mma vs naive is a little looser than tiled vs naive: the tensor cores take bf16 inputs, so each decoded weight is
rounded to bf16 (diffusers' path rounds its weights to bf16 too). It is still under one bf16 step of the largest
output. By eye, the picture sheet shows the same forest, path, light shafts and ferns as the reference; the
difference image is fine needle and fern texture only.

## What to read next

- Task 5b (`w4a8`): the same structure on the INT8 tensor cores (`mma.m16n8k32.s8`), at twice the bf16 rate. The
  x side must be quantized per token, and the weights' codes are already 4-bit integers.
- Task 6: the glue (norms, modulation, RoPE, GELU) is now a larger share of the 1.30 s than the dequant ever was.
- Ledger Technique 1 (`learning-ledger.md`); `docs/engine/speed.md` for the running table.
