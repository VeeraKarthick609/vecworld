# 05: the modulation GEMV, reading Q4_K at memory speed (Task 5c)

**What this is:** a fifth Q4_K kernel, `q4k_gemv`, for linears that see **at most 8 rows** of input. In FLUX
those are exactly the **modulation** linears. Every `Q4KLinear` now sends calls with M <= `GEMV_MAX_M` = 8
flattened rows to it, whatever `VECWORLD_KERNEL` says (`naive`, `tiled`, `mma` or the default `w4a8`). It takes x in
bf16 and sums in fp32 from the exact gguf weights, so it is **lossless**: the w4a8 default never puts modulation on
INT8.

- `VECWORLD_KERNEL` unset (`w4a8`), `mma`, `tiled`, `naive`: M <= 8 runs the GEMV, M > 8 the kernel named.
- `VECWORLD_KERNEL=off`: diffusers; no `Q4KLinear` exists, so nothing changes there.

## The problem, in plain words

Every FLUX block starts with **modulation** (AdaLN). One small vector per image (the timestep plus the pooled CLIP
text) goes through a big linear, 3072 -> 18432 in a double block (twice: image and text streams) and 3072 -> 9216 in
a single block. Out come the shift, scale and gate numbers that steer that block's norms. These 76 linears hold
**3.23 B weights, 27% of the model** (1.815 GB of Q4_K), but each multiplies **one row** (one per window: 1 in a
tile512 pass, 4 in a walk pass). They do **0.04% of the math**.

So for every weight we fetch from VRAM we do one multiply-add. The GPU can do about 100 multiply-adds in the time it
takes to fetch one byte. Modulation leaves the math units almost idle and spends all its time **waiting for
memory**. The only question that matters is how fast the weight bytes can be streamed.

Before this task, Q4KLinear sent these M = 1 calls to the **naive** kernel (Task 3), because a 128-row tensor-core
tile at M = 1 is 127/128 padding (`03-q4k-mma.md`, "Small M goes to naive"). Naive is one thread per output, and
each thread walks its own 1728-byte weight row. Neighbouring threads read addresses 1728 bytes apart, so memory
requests are scattered. Measured at BASE (66dcb2f, the w4a8 default), the 76 linears cost **35.1 ms per pass at
51.7 GB/s**, about a quarter of what the memory can deliver.

## GEMV vs GEMM on the roofline

The **roofline** says a kernel is limited by one of two ceilings. One is the math rate. The other is the memory
bandwidth times how much math you do per byte you fetch, the **arithmetic intensity** (FLOP/byte). Measured on this
RTX 3050 Ti: bf16 tensor-core peak **19.1 TFLOP/s**, VRAM read **187 GB/s** (a 512 MB `torch.sum`; 192 GB/s on paper).
The **ridge** where the two meet is 19.1e12 / 187e9 = **~102 FLOP per byte**. Below it you are memory-bound.

A linear y = x W^T with M rows, K inputs and N outputs does 2 M K N FLOPs and must read the weight once:

```
 weight format      bytes per weight     M = 1 intensity       M = 1280 intensity
 bf16               2                    2 / 2      = 1.0      2 * 1280 / 2      = 1280  FLOP/B
 Q4_K               144 / 256 = 0.5625   2 / 0.5625 = 3.6      2 * 1280 / 0.5625 = 4551  FLOP/B
```

```
 TFLOP/s (log)
  19.1 |                              ___________________________ bf16 tensor-core roof (measured)
       |                           _/                    ^
       |                        _/                       |  M = 1280 GEMMs sit far right:
       |                     _/ ridge ~102 FLOP/B        |  compute-bound, 4-bit weights don't make them faster
   1.0 |                  _/                             |  (W4A16 saves memory, not time)
       |               _/
  0.67 |      Q4_K -> * <- the roof at 3.6 FLOP/B: 187 GB/s x 3.6 = 0.67 TFLOP/s
       |           _/ ^
  0.19 | bf16 -> * /  |     memory roof: slope = 187 GB/s
       |       _/     |     M = 1 sits far left: memory-bound, so every byte saved is time saved
       |_____/________|_____________________________________________________
            1.0      3.6         102                                  4551   FLOP/B (log)
```

On the right (the token GEMMs, M = 1024 or 1280) the roof is flat. Time is set by the math, and storing the weight in
4 bits instead of 16 changes nothing about the time (ledger, Technique 1). On the far left (M = 1) the roof is the
slope. Time = bytes / bandwidth, so **4-bit weights *do* make it faster**, by exactly the byte ratio:

```
 76 modulation linears, 3.23 B weights, one pass, at 187 GB/s:
   bf16:  6.45 GB / 187 GB/s = 34.5 ms        (what a plain bf16 GEMV would need at best)
   Q4_K:  1.815 GB / 187 GB/s =  9.7 ms        (the floor for this task)
   naive, measured (before):     35.1 ms        (51.7 GB/s: the 4-bit win was eaten by scattered reads)
   q4k_gemv, measured (after):   10.97 ms       (165 GB/s, 86% of the 192 GB/s spec, 88% of the measured 187)
```

So the job is to read every Q4_K byte once, in big coalesced chunks, and keep everything else (decode, x, the sum)
off the critical path.

## How the kernel works

```
 one warp = one weight row (= one output column n); 8 warps per block; K in chunks of 1024 (4 super-blocks)

 weight row n:  | sb0 144 B | sb1 144 B | sb2 144 B | sb3 144 B | sb4 ... (12 super-blocks for K = 3072)
                  \_______________ one chunk: 576 contiguous bytes _______/

 lane l = 8 grp + part:   grp  = which super-block of the chunk (0..3)
                          part = which 16-byte piece of its 128 code bytes (0..7), run g = part / 2, half h = part % 2
   each lane: 1 x 128-bit load of codes  +  1 x 128-bit load of the 16-byte header (d, dmin, scales; 8 lanes share it)
   -> 32 weights: 16 low nibbles (sub-block 2g) and 16 high nibbles (sub-block 2g + 1), decoded in registers
      exactly as gguf does (w = d*sc*q - dmin*m in fp32)
   -> FMA with the x values from shared memory, fp32, for each of the M rows

 x (M rows x 1024 of the chunk, bf16) is staged in shared memory once per chunk by the whole block and
 read by its 8 warps. The next chunk's weight bytes are loaded into registers while this one computes.

 end: each lane holds M partial sums -> warp-shuffle reduction (xor 16, 8, 4, 2, 1) -> lane m writes y[m, n] + bias
```

What makes it fast:

- **Coalesced 128-bit loads.** A warp's 32 code loads cover 4 x 128 contiguous bytes of one row, and the 8 lanes of a
  super-block share its header in one request. Every Q4_K byte crosses the bus once.
- **Bytes in flight.** Each warp requests its next chunk before it computes this one. At M = 1 a block takes 2 KB
  of shared memory and 79 registers a thread, so 3 blocks (24 warps) share an SM, enough outstanding loads to keep
  the memory busy (M >= 2: 128 registers, 2 blocks; no spills at any M).
- **No bank conflicts on x.** A 128-bit shared load is served 8 lanes at a time. Those 8 lanes (one super-block, all
  8 pieces) want x at offsets 0, 32, 128, 160, ... bytes, which fall on only 2 of the 8 16-byte bank groups (4-way
  conflicts). An XOR swizzle of x's 16-byte pieces (`swz`) spreads them over all 8. Measured: M = 1 154 -> 173 GB/s,
  M = 8 63 -> 79 GB/s (18432 x 3072).
- **Two rows per warp from M = 2.** At M = 4 and 8 the kernel starts to run out of instruction slots, not memory:
  every weight now meets M rows of x. With two weight rows per warp, each x value read from shared memory and
  converted to fp32 feeds both rows. Measured: M = 4 98 -> 118 GB/s, M = 8 62 -> 79 GB/s. Four rows were no better.
- **One template per M** (1..8), so the M accumulators per row live in registers.

Why not the tensor cores? An `mma` tile is at least 16 rows, so at M = 1 it would be 15/16 padding. It would not
cost more memory traffic, but it would not save any either, and memory traffic is the whole cost. Forced at M = 1, the
existing kernels measured 49.5 ms per pass (`mma`) and 31.8 ms (`w4a8`, with its quantize), against the GEMV's 11.0.

Ragged shapes: rows past N are skipped, and super-blocks past K (K a multiple of 256 but not of 1024) are skipped
lane by lane. M = 0 returns an empty output; M > 8 is refused (`q4k_gemv takes at most 8 rows`). x and the packed
weight must be 16-byte aligned, as for every kernel, and a misaligned weight still raises in `Q4KLinear`, naming the
module (Task 4's rule). Nothing is copied.

## What changed in dispatch

```
 Q4KLinear.forward, M = flattened rows:
   M <= 8                        -> q4k_gemv               (any kernel; bf16 x, exact)
   8 < M < 32 and kernel tiled   -> naive                  (tiled's 128-row tile is mostly padding: TILED_MIN_M)
   otherwise                     -> the kernel named (naive / tiled / mma / w4a8)
```

`MMA_MIN_M` = 7 is gone: mma and w4a8 handed M < 7 to naive, and every such M now goes to the GEMV first. The naive
kernel itself is unchanged and is still the correctness reference every kernel is checked against.

## Numbers (RTX 3050 Ti Laptop, AC)

Modulation per pass, the 76 linears each timed alone through Q4KLinear (`scripts/bench_flux.py --what blocks`,
`modulation`; before measured with the same method at BASE 66dcb2f):

| M (windows per pass) | before: naive | after: q4k_gemv | GB/s after | % of 192 GB/s spec |
|---|---|---|---|---|
| 1 (tile512) | 35.1 ms (51.7 GB/s) | **10.97 ms** | **165.4** | 86.2 |
| 4 (walk, FLUX_BATCH) | 38.8 ms (46.8 GB/s) | **15.94 ms** | 113.9 | 59.3 |

One linear, 3072 -> 18432 (`microbench`, the better of two 50-call medians):

| M | 1 | 4 | 8 |
|---|---|---|---|
| naive | 607 us | 677 us | 1096 us |
| mma (Task 5's table, M >= 7 went there) | 840 us | 840 us | 840 us |
| q4k_gemv | 184 us (173 GB/s) | 270 us (118 GB/s) | 402 us (79 GB/s) |

The whole pass (`bench/t5c-gemv.json`, `bench/t5c-gemv-step.json`):

| | Task 5b (same session, BASE) | Task 5c |
|---|---|---|
| step s (tile512 median pass) | 1.15 | **1.12** |
| achieved TFLOP/s (bf16-equivalent) | 15.37 | **15.78** |
| % of the 19.1 bf16 peak | 80.4 | **82.6** |
| walk step s (`sharp`) | 48.9 (the 5b row) | 47.6 |
| peak allocated GB (tile512 / step) | 1.02 / 1.33 | 1.02 / 1.33 |

The step falls by ~0.03 s, which is the ~24 ms of modulation saved (the step is reported to 0.01 s). The brief
expected "0.29 s -> ~0". The 0.29 s was the modulation share of the **dequant** that diffusers did every step. Tasks
3 to 5 already removed that dequant, so by Task 5b modulation cost 35 ms, not 0.29 s, and this task takes those 35 ms
to 11. Those 11 ms are close to the 9.7 ms floor set by the bytes. A walk pass already shares one read of them
among its 4 windows: M = 4 costs 16 ms, not 4 x 11.

## Correctness

| Test | Bar | Result |
|---|---|---|
| `test_q4k_gemv_matches_reference[M=1,2,4,8]` (3072 / 12288 / 15360 x real rows, N = 1001, K = 1280 (a partial K chunk); bias; leading dims; also no bias) | max-rel < 2e-2 vs gguf dequant + `torch.matmul` (Task 3's bar); vs naive < `NAIVE_BAR` 8e-3; Q4KLinear's output = the raw kernel's | 1.8e-3 .. 3.2e-3 vs reference; 0 .. 2.9e-3 vs naive |
| `test_gemv_dispatch` | M = 1, 2 x 4 (leading dims), 8 -> `q4k_gemv` on every kernel; M = 9 -> the kernel's own GEMM (tiled: naive below 32) | yes |
| `test_q4k_tiled_small_m_runs_naive` | 1 -> gemv, 31 -> naive, 32 and 1280 -> tiled | yes |
| `test_q4k_gemv_rejects` | M = 9 refused, M = 0 empty | yes |
| `test_q4k_gemm_misaligned[*]` (extended) | the raw GEMV rejects misaligned x and weight; Q4KLinear at M = 1 raises on a misaligned weight, naming it | yes |
| `test_q4k_block_matches_reference[double0/single0 x naive/tiled/mma]` (temb is one row: the GEMV runs) | max-rel < 2e-2 vs diffusers | double0 8.30e-3, single0 5.49e-3 on all three (Task 3's values) |
| `test_flux_forward_matches_reference[*]` | rel L2 < 0.06 | 4.65% on all three |
| `test_flux_image_matches_reference[*]` | PSNR >= 28 dB | naive 29.20, tiled 29.25, mma 29.42 dB; the mma sheet checked by eye |

The GEMV and naive decode the same exact fp32 weights and sum in fp32. Only the order of the sum differs, so their
bf16 outputs differ by rounding. On the real modulation inputs of a tile512 paint (all 76 linears, the first step's
conditioning), GEMV vs naive measured max-rel 4.9e-3 of the largest output, i.e. about one bf16 step of it.

**The w4a8 picture moves.** A same-seed tile512 under the default, Task 5c routing vs Task 5b's (one painter; only
the M <= 8 routing switched; both deterministic, bit-identical on a repeat):

- `VECWORLD_KERNEL=mma` (lossless): **30.17 dB**, the usual rounding-order divergence (naive vs the exact fp32 path
  measured 30.7 dB in Task 3).
- `w4a8` (default): **25.78 dB**. It is the same forest, path, ferns and light shafts, and some trunks sit elsewhere:
  a different sample of the same prompt, as w4a8 vs mma was (25.22 dB, `04-w4a8.md`). The INT8 activation rounding
  turns a modulation change of one bf16 step into different rounding decisions downstream, and four denoising steps
  carry that forward. No quality is lost. The sheet: 5b | 5c | 8x |difference|.

Full suite: 739 passed (default + gpu + ui).

## What to read next

- Task 6 (glue fusion): norms, modulate, RoPE and GELU. The modulation *linear* is now at memory speed. What remains
  around it is the elementwise "x * (1 + scale) + shift" work on 1280 tokens, which Task 6 fuses into its neighbours.
- `learning-ledger.md` section 1.6 (the 27% / 0.04% split) and the roofline section; `speed.md` for the row.
