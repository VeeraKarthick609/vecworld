# Engine learning ledger

A running study log for the FLUX engine work. Each entry explains one idea with this project's real numbers, so it can be revised before an interview. New entries go at the bottom; add a line to the index when you do.

**Conventions.** "Measured" numbers come from `bench/baseline.json` and `docs/engine/00-baseline.md` (on AC, 2026-10-20, git 895c75e) unless labelled otherwise. "(est.)" marks a back-of-envelope figure, with its assumptions stated next to it. "(battery)" marks a number from the 2026-10-20 spike, which ran throttled (35 W, PCIe Gen2). The spike report itself is no longer on disk; its numbers survive in the spec, `docs/superpowers/specs/2026-10-20-flux-engine-design.md` §2.

## Ledger index

| # | Date | Entry | Related plan tasks |
|---|---|---|---|
| 1 | 2026-10-24 | [Making FLUX faster: ops/sec on a 4 GB GPU](#entry-1-making-flux-faster-opssec-on-a-4-gb-gpu) | 3, 4, 5, 6, 7, 8 |
| 2 | 2026-10-25 | [Techniques from Tasks 5, 5b and 5c](#entry-2-techniques-from-tasks-5-5b-and-5c-tensor-cores-w4a8-the-modulation-gemv), with "Tried and not chosen" tables for every task; Task 6 (glue fusion) added 2026-11-06; Task 7 (slots and graphs) added 2026-11-08; Task 8 (first-block cache) added 2026-11-08; Task 9 (window batching) added 2026-11-08; Task 10a (block streamer) added 2026-11-08 | 2b, 3, 4, 5, 5b, 5c, 6, 7, 8, 9, 10a |

---

## Entry 1: Making FLUX faster: ops/sec on a 4 GB GPU

*2026-10-24. Study time: about 45 minutes. Prerequisites: matrix multiply, a rough idea of what a GPU thread and warp are. Updated 2026-10-24 with the walkthrough's plain-language versions.*

**Tags.** Every section carries one tag. **Must**: be able to explain it in an interview. **Understand**: know what it is and why it matters; the details can wait until the task that needs them. **Skip**: skip it for now; read the one-line takeaway only.

### Interview revision list

**Must.**
- The roofline: arithmetic intensity (AI) and the ridge point (1.4).
- GEMM cost: 2MKN FLOPs and its bytes (1.4).
- Counting FLOPs as weights × the tokens that actually meet them (1.6).
- Quantization basics: a scale, a zero-point or min, and per-group scales (Technique 1).
- W4A16 vs W8A8 vs W4A8 (Technique 2).
- Why 4-bit weights save memory but not compute, except at M=1, where a GEMM is memory-bound (Techniques 1 and 2).
- Batching amortizes per-pass costs (Technique 3).
- Where a 2.74 s step goes (Part 2): ~40% wasted unpacking, ~40% real math already near the ceiling, ~15% small-op overhead; each is fixed a different way.
- Entry 2, Task 5:
  - Warps and fragments.
  - `mma.sync` m16n8k16: 2,048 multiply-adds per warp instruction.
  - Decode Q4_K straight into B-fragments in registers; it reached 92% of cuBLAS.
  - `cp.async` double buffering.
  - Two blocks per SM beat a deeper pipeline.
  - The walk step gained only 7%, because batching had already spread the dequant.
- Entry 2, Task 5b:
  - Per-token INT8 scales contain outliers.
  - `m16n8k32`'s k = 32 is one Q4_K sub-block.
  - The kernel is issue-bound (~12 rescale slots per `mma`), so 1.5×, not 4×.
  - Per-layer error didn't predict the failure; the residual-stream writes stay bf16.
- Entry 2, Task 5c:
  - M = 1 is memory-bound, so 4-bit weights do speed it up (3.6× fewer bytes): 35 → 11 ms at 86% of bandwidth.
  - Re-measure before optimizing (0.29 s was stale; the real gain was 0.03 s).
- Entry 2, Task 6:
  - Fusion saves memory round trips, not FLOPs; the kernels run at 89–96% of the memory roof.
  - Amdahl: glue was 31% of GPU time after the GEMMs got fast.
  - Profile first: the planned 0.4 s of glue was 68 ms; RoPE in fp32 was the hidden 147 ms.
  - Bit-identical lossless changes, so the gate shows Δ = 0 (two-pass LayerNorm's ~3e-6 bit flips failed under W4A8).
  - "99% of bf16 peak" is bf16-equivalent: 75% of the math runs on INT8.
- Entry 2, Task 7:
  - Profile first: launch overhead was 0.34% of a pass, while the real 10.9% idle was group-offload syncs.
  - Fixed weight slots took the pass from 0.92 to 0.85 s; graphs tied with eager because the GPU was already saturated.
  - Ring 3, not 2: with 57 (odd) blocks, ring 2 wraps block 0 onto block 56's slot.
  - Graphs were opt-in at Task 7 (no speed gain, but capture latency and extra memory); the default since Task 10a-g.
  - Time async GPU work with CUDA events, not CPU clocks.
- Entry 2, Task 8:
  - Residual reuse skips blocks whose input barely moved.
  - A 4-step distilled model moves 20–80% per step, so there is little to skip.
  - Thresholds 0.3 and 0.25 failed the seam gate (up to +0.166); 0.2 passed vacuously, with 0 skips.
  - The cache would buy 1.32× (tile, panorama) and 1.63× (walk step) at 0.3, and still ships off.
  - On a chained walk, PSNR measures trajectory divergence, so the gate decides on seams.
- Entry 2, Task 9:
  - When compute-bound, batching is a speed tie (~0.9 s per window-step at batch 1–8).
  - It halves passes and per-block Python (panorama 456 → 228), which matters for SSD streaming.
  - cuBLAS picks its algorithm by M: one ulp became a new picture under W4A8 (25.7 dB); `rowwise` makes windows bit-identical for +0.16%.
  - The reference is pinned to batch 1.
- Entry 2, Patterns so far: roofline first; data in registers; overlap loads with math; gate on the real output; measure before trusting estimates.

**Understand.** The exact Q4_K bytes, `mma` tile shapes, the min-term algebra, CUDA graphs, block caching and glue fusion.

**Skip for now.** INT8/fp16 throughput ratios per GPU generation, and SmoothQuant internals.

**Contents**
- Part 1: Foundations (FLOPs, bandwidth, roofline, FLUX's FLOP count, today's achieved TFLOPS)
- Part 2: The six techniques
  1. Fused Q4_K dequant + tensor-core GEMM (W4A16)
  2. W4A8 on the INT8 tensor cores
  3. Window batching
  4. CUDA graphs
  5. Block caching
  6. Fused attention and norm/glue kernels
- Part 3: Putting it together (table and stacked estimate)
- Part 4: Open decisions

### The hardware, in one table

`Understand`

RTX 3050 Ti Laptop GPU (GA107, Ampere), in the laptop `mrjarvis`.

| Property | Value | Source |
|---|---|---|
| SMs (streaming multiprocessors, the GPU's "cores") | 20 | spec sheet |
| FP32 CUDA cores | 2560 (128 per SM) | spec sheet |
| Tensor cores (3rd gen) | 80 (4 per SM) | spec sheet |
| VRAM | 4 GB GDDR6, 128-bit bus | spec sheet |
| VRAM bandwidth | about 192 GB/s on paper (128 bit × 12 Gbps ÷ 8); **187 GB/s measured** (512 MB `torch.sum`, Task 5c) | spec sheet; `05-modulation-gemv.md` |
| L2 cache | 2 MB | spec sheet |
| Shared memory | up to 100 KB per SM | Ampere GA10x docs |
| bf16 matmul, 4096², measured | **19.1 TFLOPS** | baseline.json `bf16_matmul_tflops` |
| Power limit under load | 90 W | baseline.json `gpu_power` |
| Host to device (H2D), pinned | **13.46 GB/s** (PCIe Gen4 x8) | baseline.json |
| H2D, pageable | 10.69 GB/s | baseline.json |
| SSD, O_DIRECT sequential | 2.0 GB/s (2.96 in an earlier run) | baseline.json, 00-baseline.md |
| System RAM | 14 GB | plan |

---

## Part 1: Foundations

### 1.1 FLOPs vs FLOP/s

`Must`

- A **FLOP** is one floating-point operation (an add or a multiply). It is a count of **work**.
- **FLOP/s** (FLOPS) is a **rate**: work per second. TFLOPS means 10¹² FLOP/s.
- A multiply-accumulate (`c += a*b`, an "FMA" or "MAC") counts as **2 FLOPs**. So multiplying an `M×K` matrix by a `K×N` matrix costs `2·M·K·N` FLOPs.

The core relation is time = work ÷ rate. If a FLUX step costs W FLOPs and we run at R FLOP/s, the step takes W/R seconds. You can make it faster in only two ways: do less work (smaller W: caching, skipping) or do it at a higher rate (bigger R: better kernels, lower precision, less waiting).

### 1.2 What peak throughput this GPU has

`Understand`. Takeaway: our bf16 ceiling is the measured 19.1 TFLOPS; INT8 is faster. The per-generation ratios in the table are `Skip`.

**Tensor cores** are units inside each SM that execute a small matrix multiply (for example 16×8×16) as one instruction. They are much faster than the ordinary FP32 "CUDA cores", but only for matrix shapes and specific data types.

| Data type / mode | Peak (est.) | How estimated |
|---|---|---|
| FP32 on CUDA cores (no tensor cores) | about 9.6 TFLOPS (est.) | 2560 cores × 2 FLOP × ~1.87 GHz |
| bf16 in, fp32 accumulate (tensor cores) | **19.1 TFLOPS measured** | baseline 4096² matmul |
| fp16 in, fp16 accumulate (tensor cores) | about 38 TFLOPS (est.) | GeForce Ampere runs fp16-accumulate at 2× the fp32-accumulate rate |
| INT8 in, int32 accumulate (tensor cores) | about 76 TOPS (est.) | GeForce Ampere INT8 rate is 4× the bf16/fp32-acc rate (ratio from NVIDIA's GA102 whitepaper; 3090: 71 vs 284 dense) |

A note on the 19.1. Ampere GeForce parts do 512 bf16 FLOP per SM per clock with fp32 accumulate (est., from the GA102 ratio). 19.1 TFLOPS ÷ (20 SMs × 512) ≈ 1.87 GHz, and real matmuls never hit 100% of the instruction rate. So 19.1 means the GPU was boosting at or above about 1.9 GHz on AC. That is plausible at 90 W, but it is at the top of the range. **Treat 19.1 as our practical ceiling**, and log the SM clock (`nvidia-smi --query-gpu=clocks.sm`) next time it is measured. On battery the same matmul gave 5.2 TFLOPS (battery), which is why only AC numbers count.

### 1.3 Memory bandwidth

`Must`

**Memory bandwidth** is how many bytes per second can move between VRAM and the SMs: about 192 GB/s here (est.). Other links in our chain are much slower:

```
SSD --2.0-3.0 GB/s--> RAM (pinned) --13.5 GB/s--> VRAM --~192 GB/s (est.)--> SMs (registers, shared mem)
```

Every byte of weight a kernel needs must cross the VRAM→SM link at least once. For very large weight sets it must also cross the PCIe link (13.5 GB/s) and, from Task 8 on, the SSD link (2–3 GB/s).

### 1.4 Arithmetic intensity and the roofline

`Must`

**Arithmetic intensity (AI)** = FLOPs done ÷ bytes moved from memory, in FLOP/byte. It says how much work you get per byte you fetch.

The **roofline model** says a kernel's best possible rate is:

```
attainable FLOP/s = min( peak FLOP/s ,  AI × bandwidth )
```

- If AI is low, you are limited by the sloped part (bandwidth). That is **memory-bound**: the SMs wait for data.
- If AI is high, you hit the flat part (peak compute). That is **compute-bound**: the data arrives faster than you can use it.
- The **ridge point** where they meet is peak ÷ bandwidth = 19.1 TFLOPS ÷ 192 GB/s ≈ **100 FLOP/byte (est.)**. A kernel needs more than about 100 FLOP per byte to use the tensor cores fully.

**AI of a GEMM** (General Matrix Multiply). An `M×K` activation times a `K×N` weight costs `2MKN` FLOPs. It moves about `K·N·b_w` bytes of weight plus `M·K·2 + M·N·2` bytes of bf16 activations in and out, where `b_w` is bytes per weight. When the weight dominates, AI ≈ `2M / b_w`. **The token count M sets the intensity.**

**Where the bytes go, for one real GEMM** (K=3072, N=12288, M=1280, all bf16 at 2 B):

| Part | Elements | Bytes | Grows with M? |
|---|---|---|---|
| Weight W (K×N) | 37.7 M | 75.5 MB | no: read once, shared by every token |
| Input X (M×K) | 3.9 M | 7.9 MB | yes |
| Output Y (M×N) | 15.7 M | 31.5 MB | yes |
| **Total** | | **≈ 115 MB** | |

FLOPs = 2 × 1280 × 3072 × 12288 = 96.6 G, so AI ≈ 96.6 G / 115 MB ≈ **840 FLOP/B**.

**Check: what does batching 4 windows do to AI?** M becomes 5120. FLOPs grow 4× (386 G). Bytes grow only about 2×: the weight is still 75.5 MB, but the activations grow with M (X = 31.5 MB, Y = 126 MB), so the total is about 233 MB. That gives **AI ≈ 1660: about 2×, not 4×.** `2M/b_w` holds only while the weight dominates the bytes, and at M=1280 the activations are already about a third of them.

We were already compute-bound at M=1280, so batching doesn't change the regime. It helps because each pass's fixed costs (today the dequant, later the launches and SSD reads) are paid once for more tokens. See Technique 3.

Our cases:

| Kernel | M | Weight format | AI (FLOP/B) | Regime |
|---|---|---|---|---|
| Image-stream and single-block GEMMs (example: K=3072, N=12288) | 1280 | bf16 (after dequant) | ~840 (est.) | compute-bound |
| Same GEMM, fused from Q4_K | 1280 | 0.5625 B/weight | ~1600 (est.) | compute-bound |
| Text-stream GEMMs in double blocks | 256 | bf16 | ~230 (est.) | compute-bound, closer to the ridge |
| Modulation linears (one vector per image) | 1 | bf16 / Q4_K | ~1 / ~3.6 | deeply memory-bound (GEMV) |
| Attention (flash-style, per head, N=1280, d=128) | — | — | ~640 (est.) | compute-bound |
| Dequant kernel chain (today) | — | Q4_K → bf16 | ~0 useful FLOPs; it only moves bytes | memory-bound |
| Glue: RMSNorm, LayerNorm, modulate, GELU, RoPE, residual adds | — | — | ~1 | memory-bound |

#### ASCII roofline (log-log, not to scale)

```
 attainable TFLOP/s
  19.1 +                              ______________________________________  bf16 tensor-core roof (measured)
       |                             /   A 17.7 attention (SDPA, measured)
       |                            /         S 12.9 single block minus dequant
       |                           /          D 10.1 double block minus dequant
       |                          /
       |                         /
   1.0 +                        /
       |                       /
       |                      /
  0.19 +   g  m              /       slope = VRAM bandwidth, ~192 GB/s (est.)
       |  /                 /
       | x  Q (dequant: moves 30+ GB/step, does no useful FLOPs; runs at ~28 GB/s effective)
       +----+-------+-------+--------+--------+------------> arithmetic intensity (FLOP/byte)
            1      10      100      1000
                      ridge ≈ 100 (est.)

  g = glue kernels (~1 FLOP/B)     m = modulation GEMVs (M=1)
  A, S, D sit at ~600-1600 FLOP/B: far right of the ridge, so compute-bound.
```

Achieved rates for D and S use the measured block times with the separately measured dequant time taken out: a block at 1280 tokens is about 310 GFLOP (see 1.6). Double: 310 G ÷ (61.9 − 31.1) ms = 10.1 TFLOPS. Single: 310 G ÷ (37.1 − 13.0) ms = 12.9 TFLOPS. Attention: 20.1 GFLOP per block ÷ 1.14 ms = 17.7 TFLOPS. These include glue, so the pure GEMM rates are higher.

**The roofline lesson for FLUX:** our big GEMMs are compute-bound at 1280 tokens, so **weight bandwidth is not what limits them**. What hurts is (a) memory-bound work that exists *beside* the GEMMs (dequant, glue, modulation GEMVs) and (b) GEMMs running below the roof.

### 1.5 Compute-bound vs memory-bound, as a habit

`Must`

Before optimising a kernel, find which side of the ridge it sits on:

- **Compute-bound** (big GEMMs, attention): use faster math units (tensor cores, lower precision like INT8) and keep them fed (tiling, double buffering).
- **Memory-bound** (dequant, norms, GEMVs): move fewer bytes. Fuse kernels so intermediate tensors never go to VRAM, use smaller formats, and read each byte once.

Making a memory-bound kernel do fewer FLOPs gains nothing. Making a compute-bound kernel read fewer bytes gains nothing either. That is why the roofline matters.

### 1.6 FLUX's FLOPs per step, from the real shapes

`Must`

> **Plain version (read this first).**
>
> **The idea in one line:** the work in a step ≈ 2 × (the weights a token passes through) × (the number of tokens). Everything else is counting those two correctly.
>
> ```
> tokens in ──► [19 double blocks] ──► [38 single blocks] ──► image out
> ```
>
> - Every token passes through P ≈ 113 M weights per block, double or single.
> - 57 blocks × 2 × P ≈ 12.9 G operations per token.
> - 512 px → 1024 image tokens + 256 text tokens = N = 1280.
> - 12.9 G × 1280 ≈ 16.5 T (weights) + ~1.2 T (attention) ≈ **17.7 TFLOP per step**.
>
> Three takeaways:
> 1. 17.7 / 2.74 s = 6.45 TFLOP/s against a 19.1 peak, so **we use about 34% of the GPU**. That gap is what the engine chases.
> 2. The spike's "30 TFLOP" multiplied all weights by all tokens. Image tokens never meet text weights, and modulation sees 1 row, not 1280. **Count only the weights and tokens that actually meet.**
> 3. Modulation is 27% of the weights and about 0% of the math, yet it is unpacked every step (~0.29 s of 2.74 s). It is a cheap target. *(Update, 2026-10-25: that 0.29 s was dequant, which the fused kernels removed. By Task 5b modulation cost 35 ms, and Task 5c took it to 11 ms. See Entry 2, Task 5c.)*
>
> ```
>               share of WEIGHTS              share of MATH
> modulation    ███████ 27%                   ▏ ~0%
> token-facing  ████████████████████ 73%      ████████████████████████ ~100%
> ```
>
> The detail below shows where each of these numbers comes from.

**Architecture.** Hidden size `h = 3072` (24 heads × 128), MLP width 12288. There are 19 **double blocks**, where image and text tokens have separate weights but attend jointly, and 38 **single blocks**, where all tokens share weights. Every block has a **modulation** linear (AdaLN, adaptive layer norm). It turns the per-image conditioning vector (timestep + pooled CLIP) into shift, scale and gate values, so it sees **one row per image**, not one per token.

Weight counts (all exact):

| Piece | Shape | Params |
|---|---|---|
| Double block, one stream: QKV | 3072 → 9216 | 28.31 M |
| Double block, one stream: attention out | 3072 → 3072 | 9.44 M |
| Double block, one stream: MLP (two linears) | 3072 → 12288 → 3072 | 75.50 M |
| **Double block, one stream, total** | | **113.25 M (call this P)** |
| Double block, modulation (img + txt, 6×3072 each) | 3072 → 2 × 18432 | 113.25 M (= P) |
| Single block: linear1 (QKV + MLP-in fused) | 3072 → 21504 | 66.06 M |
| Single block: linear2 (attention out + MLP-out fused) | 15360 → 3072 | 47.19 M |
| **Single block, token-facing total** | | **113.25 M (= P again)** |
| Single block, modulation | 3072 → 9216 | 28.31 M (= P/4) |

A neat accident: every token-facing unit is exactly P = 113,246,208 params.

- Double block = 3P = 339.7 M params. At Q4_K's 0.5625 bytes/param that is **191 MB**, matching the spec's measured block size.
- Single block = 1.25P = 141.6 M → **80 MB**, also matching.
- Whole stack: 19 × 3P + 38 × 1.25P = 104.5P = **11.83 B params = 6.66 GB at Q4_K**, matching the measured 6.66 GB of blocks.

**Linear FLOPs.** In a double block, the image weights (P) see the image tokens and the text weights (P) see the text tokens. So the block does 2·P·(N_img + N_txt) = 2·P·N. A single block also does 2·P·N. Modulation is about 0 (one row). So:

```
linear FLOPs per step = 57 blocks × 2 × P × N = 114 × 113.25 M × N = 12.91 GFLOP per token
```

**Attention FLOPs** per block: QKᵀ is 2·N²·h and attention×V is another 2·N²·h, so 4·N²·3072 = 12,288·N². Over 57 blocks that is 700,416·N². Softmax, embedders and the output projection add under 0.1% and are dropped.

**The brief's case: 512 px with 512 text tokens.** 512 px → a 64×64 latent → 2×2 patches → N_img = 1024; N = 1024 + 512 = 1536.

```
linear    = 12.91 GFLOP × 1536          = 19.83 TFLOP
attention = 700,416 × 1536² = 700,416 × 2,359,296 = 1.65 TFLOP
total                                    ≈ 21.5 TFLOP per step
```

**What we actually run: 256 text tokens.** `FluxPainter` encodes T5 to `[1, 256, 4096]` (schnell's usual maximum), and the bench uses 256 too. So the measured 2.74 s/step is at N = 1024 + 256 = 1280:

```
linear    = 12.91 GFLOP × 1280          = 16.53 TFLOP
attention = 700,416 × 1280² = 700,416 × 1,638,400 = 1.15 TFLOP
total                                    ≈ 17.7 TFLOP per step
```

**The spike's "about 30 TFLOP" was an over-count of about 1.7×.** It used 11.9 B params × 2 × 1280 tokens. That counts the 3.23 B modulation params as if every token used them, and counts both streams of a double block against all 1280 tokens. The baseline doc already suspected this. The exact count shows the size of the error.

> **Side fact worth remembering:** modulation is 19P + 38 × P/4 = 28.5P = **3.23 B params, 27% of all weight bytes**, and it does only 6.5 GFLOP per step. Today every one of those weights is dequantized every step: about 27% of the 1.08 s dequant (est., since dequant time scales with params: 31.1 ms per 3P block and 13.0 ms per 1.25P block both give about 10.9 G params/s). That is about 0.29 s per step spent unpacking weights for 0.04% of the work.

### 1.7 Today's achieved TFLOPS

`Must`

```
512 px, measured step 2.74 s:     17.7 TFLOP / 2.74 s  = 6.45 TFLOPS  = 34% of the measured 19.1 peak
512 px, resident compute 2.59 s:  17.7 / 2.59          = 6.8 TFLOPS   = 36%
1024x512, 16.11 s / 4 = 4.03 s/step, N = 2048 + 256 = 2304:
   linear 12.91 G × 2304 = 29.7 T, attention 700,416 × 2304² = 3.7 T → 33.5 TFLOP
   33.5 / 4.03 = 8.3 TFLOPS = 43%
```

The bigger image runs closer to the roof because dequant is a fixed cost per step (1.08 s at both sizes), spread over more FLOPs.

**Rounding and assumptions.** The FLOP counts are exact for the matmuls and leave out softmax, norms and embedders (under 0.1%). "Peak" is our measured 19.1, not a datasheet number; against a datasheet-style estimate of 17–20 TFLOPS the percentage moves by a few points. If someone runs with 512 text tokens (21.5 TFLOP) the step will be slower too, so do not divide 21.5 by 2.74.

### 1.8 Why "achieved TFLOPS as % of peak" is a good headline

`Must`

- **It normalises.** s/step changes with resolution, batch and text length; % of peak does not. 34% at 512 and 43% at 1024×512 are directly comparable.
- **It shows the headroom.** 34% means that, at best, we could be about 3× faster without doing less work. A team at 85% has nowhere to go except changing the work (lower precision, caching).
- **It ties to the roofline.** If a compute-bound kernel is at 30% of peak, the kernel is the problem, not the hardware.
- **Caveats:** it is only as honest as the FLOP count (the spike's 30 TFLOP would have said 57%, wrongly). The peak must match the math you use: once W4A8 runs on INT8 tensor cores, report "bf16-equivalent TFLOPS" or % of INT8 peak, and say which. It is also not a user metric, so always pair it with s/step and an image-quality check.

<details><summary>Check yourself (Part 1)</summary>

**Q1.** A GEMM with M=256, K=3072, N=3072 and bf16 weights. Roughly what is its arithmetic intensity, and which side of the ridge is it on?

**A1.** FLOPs = 2·256·3072·3072 ≈ 4.8 G. Bytes ≈ weights 3072·3072·2 = 18.9 MB + activations in and out 2 × 256·3072·2 = 3.1 MB ≈ 22 MB. AI ≈ 220 FLOP/B, about 2× the ridge (~100). Compute-bound, but not by much: at M=64 it would be memory-bound.

**Q2.** Why does the 1024×512 step reach a higher % of peak than the 512 step?

**A2.** Dequant (1.08 s) is the same per step whatever the token count, so with more tokens the fixed cost is a smaller share. Bigger GEMMs and attention also use the tensor cores better.

</details>

---

## Part 2: The six techniques

`Must` for the breakdown below; each technique has its own tag.

> **Plain version: where the 2.74 s of a step goes** (`Must`)
>
> ```
> one step ≈ 2.74 s
> ┌──────────────────────────┬─┬─────────────────────────┬──────────┬────────┐
> │ dequant 1.08 s           │A│ GEMM ~1.05 s            │glue ~0.4 │offload │
> │ unpacking weights        │ │ the real multiplies     │small ops │0.15 s  │
> └──────────────────────────┴─┴─────────────────────────┴──────────┴────────┘
>                             └ attention 0.06 s
> ```
>
> 1. **Dequant, 1.08 s (42%).** Every step unpacks all 11.8 B 4-bit weights to bf16. It does no useful math: it only copies bytes, and it is the biggest line. About 0.29 s of it is modulation weights (§1.6). **Fix:** the fused Q4_K kernel unpacks 32 weights at a time in registers, bringing this to about 0 (Technique 1). It is the biggest and surest win.
> 2. **Attention (SDPA), 0.06 s (2.5%).** Tokens look at each other. 1.15 TFLOP / 0.065 s ≈ 17.7 TFLOP/s, close to the 19.1 roof. It is already near-optimal, so leave it alone.
> 3. **GEMM, ~1.05 s (est.).** The real 16.5 TFLOP of weight multiplies. This is an estimate because the measurement lumps it with glue. It assumes cuBLAS runs at ~16 TFLOP/s on the M=1280 shapes and ~12 on the M=256 text shapes (small matrices fill the GPU less well). That is already about 80% of the bf16 roof (16.5 / 1.05 ≈ 15.7 TFLOP/s), so bf16 can't go much faster. The only options are faster math (W4A8 on the INT8 tensor cores, up to ~4× our bf16-with-fp32-accumulate rate in theory, Technique 2) or less math (block caching, Technique 5).
> 4. **Glue, ~0.4 s (est., 0.25–0.55).** Norms, modulate `x·(1+scale)+shift`, GELU, RoPE, residual adds and reshapes. Each reads and writes the whole activation for about 1 operation per number, so it is memory-bound. It also launches thousands of tiny kernels, each with fixed overhead. It is a range because it is 1.44 s minus the GEMM estimate. **Fix:** fusion, e.g. norm + modulate + add in one pass (Technique 6), and CUDA graphs, which record the launches once and replay them (Technique 4).
> 5. **Offload, 0.15 s.** The measured 2.74 s step is 0.15 s above the 2.59 s resident compute; the four lines above add up to that 2.59 s (2.58 after rounding).
>
> | Line | Now | After the engine (rough, est.) | How |
> |---|---|---|---|
> | Dequant | 1.08 s | ~0 | fused Q4_K kernel (Technique 1) |
> | Attention | 0.06 s | 0.06 s | already near-optimal |
> | GEMM | ~1.05 s | ~0.4–0.8 s: ~0.4–0.6 if our W4A16 kernel matches cuBLAS, 0.6–0.8 from a first-kernel 1.25–1.6 s | W4A8 INT8, if it passes the quality gate (Technique 2) |
> | Glue | ~0.4 s | ~0.1–0.15 s | fusion + CUDA graphs (Techniques 6, 4) |
>
> Block caching (Technique 5, Option B only) would then skip whole steps of what remains.
>
> **One-line takeaway:** the step is ~40% wasted unpacking, ~40% real math already near the ceiling and ~15% small-op overhead (plus ~5% offload), and each part is fixed a different way.

Today's 512 px step, measured, weights resident (2.59 s), plus 0.15 s offload overhead = 2.74 s:

| Part | Time | Share |
|---|---|---|
| GGUF Q4_K dequant | 1.08 s | 42% |
| Attention (SDPA) | 0.06 s | 2.5% |
| GEMM + glue | 1.44 s | 56% |
| of which GEMM (est.) | ~1.05 s | assumes cuBLAS at ~16 TFLOPS on M=1280 shapes and ~12 on the M=256 text GEMMs |
| of which glue (est.) | ~0.4 s (0.25–0.55) | the remainder |

### Technique 1: Fused Q4_K dequant + tensor-core GEMM (W4A16)

`Must` for the idea; the layout and `mma` details are `Understand`. *W4A16 means 4-bit weights, 16-bit activations.* Plan Tasks 3 (naive), 4 (tiled), 5 (`mma`).

> **Q4_K in one paragraph.** Q4_K stores each weight as a 4-bit number (0–15) plus a small "ruler" shared by every 32 weights. Real weight = ruler × number (minus an offset). That is 4.5 bits per weight, 3.6× smaller than bf16. Today we unpack the whole layer to bf16 before each multiply; the fused kernel unpacks 32 at a time in registers instead.

**The problem, in our numbers.** diffusers' GGUF path keeps weights in Q4_K on the GPU. On every linear call it unpacks the *whole* weight to a bf16 tensor in VRAM with a chain of PyTorch ops, then hands that to cuBLAS. That takes **1.08 s per step, 42% of the step**. The minimum traffic for that unpack is reading 6.66 GB of Q4_K and writing 23.7 GB of bf16 (11.83 B × 2 B): 30.3 GB ÷ 192 GB/s ≈ 0.16 s (est.). The measured 1.08 s is about 7× that, because each PyTorch op writes its intermediate result to VRAM and the next op reads it back. Then the GEMM reads the 23.7 GB again. Every step, for every weight, even though the weights never change.

**Understand later (Task 3): the Q4_K super-block in full.** Q4_K (llama.cpp's "k-quant") stores 256 weights in a 144-byte super-block, 4.5 bits per weight:

```
 Q4_K super-block: 256 weights in 144 bytes
 +---------+---------+-------------------------+--------------------------------------+
 | d  fp16 | dmin f16| scales[12] = 96 bits    | qs[128] = 256 x 4-bit codes q (0..15)|
 |   2 B   |   2 B   | 8 x 6-bit sc, 8 x 6-bit m|  two codes per byte                  |
 +---------+---------+-------------------------+--------------------------------------+
 8 sub-blocks of 32 weights. For weight i in sub-block j:

        w_i = (d * sc_j) * q_i  -  (dmin * m_j)
              \__ scale __/        \__ offset __/
```

- Two levels of scale: `d` and `dmin` (fp16) per 256 weights, and 6-bit `sc_j`, `m_j` per 32 weights. That gives near-per-32 accuracy for 12 bytes of metadata instead of 32.
- The 6-bit values are packed: for j < 4, `sc = scales[j] & 63`, `m = scales[j+4] & 63`. For j ≥ 4, the low 4 bits come from `scales[j+4]` and the top 2 bits from bits 6–7 of `scales[j-4]` (sc) and `scales[j]` (m).
- In `qs`, each 32-byte run holds two sub-blocks: low nibbles are sub-block 2k, high nibbles are sub-block 2k+1.
- `Q4_K_S` files mix in a few other types. Review Focus 1 says those must stay on the reference path.

**How the fused kernel works.** Never build the bf16 weight in VRAM. Each thread block loads a tile of *packed* Q4_K bytes into shared memory, and each thread unpacks only the weights its tensor-core instruction needs, in registers, right before using them:

```
 VRAM: Q4_K weights (0.56 B/w) --cp.async--> shared mem (still packed) --> registers: nibble -> q
                                                                              q * (d*sc) - dmin*m -> bf16 pair
 VRAM: bf16 activations --cp.async--> shared mem --ldmatrix--> A fragments           |
                                                                                     v
                                                  mma.sync.m16n8k16 (bf16 x bf16 -> fp32 accumulators)
                                                                                     |
                                              epilogue: + bias, -> bf16, store to VRAM (once)
```

- **`mma.sync.aligned.m16n8k16`** is the PTX instruction where one warp (32 threads) multiplies a 16×16 bf16 A tile by a 16×8 bf16 B tile into a 16×8 fp32 accumulator. Each thread holds part of each operand in registers (a "fragment"). For B, thread `lane` holds column `lane/4` at k = `2·(lane%4)`, +1, +8 and +9. Those four k values always fall inside one 32-weight sub-block, so one (sc, m) pair serves them.
- **`cp.async`** copies global→shared without passing through registers, so the next tile can load while this one computes (double buffering).
- **Offline repacking** (as the Marlin kernel does): reorder the nibbles once at load time so that one 32-bit load gives a thread exactly the 8 codes for its fragments. This turns the decode into a few shifts, masks and two FMAs.
- **Decode cost.** Each weight is decoded once per M-tile. With a 128-row M-tile and M=1280, that is 10 times per step: about 8.6 B token-facing weights × 10 × ~4 ops ≈ 0.34 T ops on the CUDA cores (est.), about 0.07 s if it overlaps with the tensor cores (they are different units). For comparison, the separate dequant takes 1.08 s.

**Expected gain (est.).** The 1.08 s dequant disappears. The GEMM no longer reads bf16 weights, but it was compute-bound anyway, so it only speeds up as far as our kernel matches cuBLAS. A first hand-written `mma` kernel at 65–85% of cuBLAS (est.) puts GEMM at 1.25–1.6 s. Add glue 0.4, attention 0.06 and the remaining offload overhead about 0.1:

```
step ≈ 1.25-1.6 + 0.4 + 0.06 + 0.1 ≈ 1.8-2.2 s   →  midpoint ~1.9 s, a factor of ~1.45x on 2.74 s (est.)
```

The baseline doc's "~1.5 s floor" assumes cuBLAS-level GEMMs. Matching cuBLAS is the hard part.

**Cost and risk.**
- Effort: three tasks, and the naive Task 3 kernel will probably be *slower* than the reference. It runs one thread per output on CUDA cores (≤ 9.6 TFLOPS est.) with no reuse. That is expected; say so in the explainer.
- Correctness traps: 6-bit scale packing, nibble order, ragged M (1, 7, 333, 1792), `in_features` not a multiple of 256, non-Q4_K tensors (Review Focus 1 and 2).
- **Quality: none in principle.** It is the same maths reordered, so the tests require bit-exact dequant and bf16-level GEMM error. The fp32 accumulation order differs from cuBLAS, so expect tiny differences (PSNR ≥ 30 dB gate).
- Side benefits: no bf16 weight scratch buffer in VRAM, and the streamer can move raw Q4_K bytes straight into the kernel.

**In an interview.** "Our 4-bit model spent 42% of each step unpacking weights to bf16 in VRAM before every GEMM. That is pure memory traffic, and it was repeated every step. I fused the unpack into a tensor-core GEMM: Q4_K tiles go to shared memory still packed, each thread decodes just its `mma` fragment in registers, and the bf16 weight never exists in memory. The dequant cost went to near zero, and the remaining work is a compute-bound GEMM measured against cuBLAS."

<details><summary>Check yourself (Technique 1)</summary>

**Q1.** Why doesn't fusing dequant into the GEMM make the GEMM's *own* compute faster at M=1280?

**A1.** At M=1280 the GEMM is compute-bound (AI ~840 FLOP/B with bf16 weights, versus a ridge of ~100). Reading 4-bit instead of 16-bit weights raises the AI further but does not raise the tensor-core roof. The gain comes from deleting the separate memory-bound dequant pass, not from the GEMM itself.

**Q2.** Q4_K uses 144 bytes per 256 weights. What fraction of that is metadata, and what would the bits per weight be with one fp16 scale and one fp16 min per 32 weights instead?

**A2.** Metadata is 2 + 2 + 12 = 16 bytes, about 11%; the 4-bit codes are 128 bytes. With fp16 scale + min per 32 weights: 4 bits + 32 bits/32 = 5.0 bits per weight, against Q4_K's 4.5.

</details>

### Technique 2: W4A8 on the INT8 tensor cores

`Must` for the short version; the rescale algebra is `Understand`; SmoothQuant internals are `Skip`. *W4A8 means 4-bit weights, 8-bit integer activations.* Not in the plan.

> **Short version.**
> - **W4A16 (Technique 1):** 4-bit storage, bf16 math. It saves memory and memory traffic, not math. The multiply runs at the bf16 tensor-core rate either way. The exception is M=1 (GEMV), which is memory-bound, so smaller weights do speed it up.
> - **W4A8:** also quantize activations to INT8 on the fly (one scale per token), and run on the INT8 tensor cores. The theoretical ceiling is about 4× our bf16-with-fp32-accumulate rate, or 2× fp16-with-fp16-accumulate. The real gain is lower (est. 1.8–2.5× on the GEMM).
> - **A nice fit:** Ampere's INT8 `mma.m16n8k32` consumes exactly 32 values of K, which is one Q4_K sub-block, so one ruler per instruction.
> - **Risks:** activation outliers. Fixes: per-token scales, SmoothQuant, or keeping sensitive layers at bf16. It must re-pass the quality gates, including the `walk_sheet.py` seam check on walks.

**The problem, in our numbers.** After Technique 1 the step is mostly bf16 GEMM, about 1.25–1.6 s of a ~1.9 s step (est.), already compute-bound. The bf16 roof is 19.1 TFLOPS. The INT8 tensor cores on GeForce Ampere run at about 4× that rate, ~76 TOPS (est.). The only way through the roof is to do the multiply in a cheaper number format.

**How it works.**
- **Activation quantization, per token.** Before each linear, take each row (token) of the bf16 activation and find `s_t = max|x_t| / 127`, then store `x_int8 = round(x_t / s_t)`. Each token gets its own scale, so one loud token doesn't crush the precision of the others.
- **Integer weights.** Q4_K's codes q are already small unsigned integers (0..15), so they can go into INT8 fragments as they are.
- **Rescale per sub-block.** INT8 `mma.m16n8k32` multiplies along k = 32, exactly one Q4_K sub-block. For each sub-block j:

```
 Σ_i x_i w_i  =  s_t · [ (d·sc_j) · Σ_i xq_i·q_i  −  (dmin·m_j) · Σ_i xq_i ]
                         \_ int32 from one mma _/        \_ per-token sub-block sum, precomputed _/
```

- So: integer `mma` per 32-wide k-chunk, then one fp32 FMA per output per sub-block to apply the scales, accumulated in fp32. The per-token, per-sub-block sums `Σ xq_i` come from the quantize pass, which is what llama.cpp's MMQ kernels do with their Q8_1 activations.

**Expected gain (est.).** Rescaling costs about 1 fp32 FMA per 32 integer MACs, the quantize pass can be fused into the previous glue kernel, and real INT8 kernels rarely reach 4× in practice. Guess the GEMM portion gets 1.8–2.5× faster: 1.25–1.6 s → ~0.6–0.8 s. On a ~1.9 s step that is about **1.4×** (range 1.2–1.6×, est.).

**Cost and risk.**
- **Quality: this is the real risk.** Diffusion transformers have activation outliers: a few channels with values far larger than the rest. Published work quantizing FLUX (for example SVDQuant, 2024) found these need special handling at 4 bits; 8-bit per-token is milder, but not free. Sensitive spots are likely the MLP-out input (after GELU), attention-out and the first/last blocks.
- Mitigations: keep sensitive linears at W4A16 (a per-layer mixed policy), use per-token *and* per-group scales, or apply SmoothQuant-style scaling (move outlier magnitude from activations into weights offline).
- **Quality gate:** same seed, PSNR against the W4A16 image (pick a bar such as ≥ 30 dB) **and** your eye on the contact sheets. Reject if it fails.
- Effort: high. It is a second kernel family, plus a quantize kernel and a calibration study per layer.
- Aside: GeForce Ampere also runs fp16-in/fp16-accumulate at ~2× bf16 (est.). That is cheaper to try, but fp16 accumulators overflow easily, and FLUX activations have a wide range. Test before trusting it.

**In an interview.** "Once the 4-bit GEMM was compute-bound on bf16 tensor cores, the next roof was the number format, so I moved activations to INT8 with per-token scales and used the INT8 `mma`, whose k=32 matches Q4_K's 32-weight sub-block. Integer partial sums get the sub-block scale and min applied in fp32. It's a quality trade, so it shipped behind a per-layer policy and an image-quality gate, not as a default."

<details><summary>Check yourself (Technique 2)</summary>

**Q1.** Why per-token activation scales instead of one scale per tensor?

**A1.** Token magnitudes differ a lot (text vs image tokens, outlier tokens). One tensor-wide scale is set by the largest value, so most tokens would use only a few of the 255 levels and lose precision. A per-token scale costs one float per row and fits each token's range.

**Q2.** What is the `dmin·m_j·Σ xq_i` term for, and why can it be precomputed?

**A2.** Q4_K weights are `scale·q − offset`. The offset part of the dot product is `offset × (sum of the activations in that sub-block)`. That sum depends only on the activation, not on the output column, so compute it once per token per sub-block during quantization and reuse it for every output.

</details>

### Technique 3: Window batching

`Must`. Plan Task 6. A panorama is painted as several overlapping windows (8 for the full pano). Batching runs several windows through each transformer pass together.

**The problem, in our numbers.** Every pass over the 57 blocks costs some things once per pass, whatever the number of windows: today that is the 1.08 s dequant and the 0.15 s offload overhead. One window at a time pays them 8 times per step.

**How it works.**

```
 batch 1:  [weights in, dequant] -> win1 ;  [weights in, dequant] -> win2 ; ...   (8 passes per step)
 batch 4:  [weights in, dequant] -> win1 win2 win3 win4                           (2 passes per step)
           GEMM M grows from 1280 to 4 x 1280 = 5120 rows; the weight tile is reused across all of them
```

**Measured today (reference painter, AC):**

| Batch | s per pass | s per window-step | 8-window pano |
|---|---|---|---|
| 1 | 2.74 | 2.74 | ~91 s (extrapolated) |
| 2 | 4.29 | 2.15 | 68.3 s |
| 4 | 7.05 | **1.76** | **56.3 s** |

Check: if batching saved only the per-pass fixed costs, batch 4 would save 3 × (1.08 + 0.15) = 3.69 s per four windows, against a measured 4 × 2.74 − 7.05 = 3.91 s. **Today's batching gain is almost all dequant amortization.**

**Expected gain after Technique 1 (est.).** This is the honest part: **Techniques 1 and 3 attack the same 1.08 s.** Once dequant is fused away, the fixed per-pass cost left is small: modulation GEMVs (~1.8 GB of Q4_K at ~192 GB/s ≈ 0.01 s, est.), launch overhead, poorly shaped small text GEMMs (M=256), and tile-wave tails on 20 SMs. Batching then gives maybe **1.1–1.2×** per window when the weights come from RAM.

**Where batching becomes essential: streaming from the SSD (Task 8).** Once weights come off the SSD every pass, each pass reads 6.66 GB at 2.0–2.96 GB/s = **2.25–3.33 s per pass** (from measured SSD rates). For a single window, that is an I/O floor at or *above* today's 2.74 s/step, whatever the kernels do. At batch 4 the I/O is 0.56–0.83 s per window-step, below the compute time. So batching is what lets the kernel work show up in the SSD configuration.

**The 4 GB activation limit.** The plan caps peak allocation at 3.6 GB. Measured peaks: batch 2 = 1.52 GB, batch 4 = 2.59 GB, so each extra window costs about (2.59 − 1.52)/2 ≈ **0.53 GB** with today's eager glue. Fixed costs (≈ 0.46 GB implied, est.) plus the Task 7 device ring (2 slots × 191 MB ≈ 0.38 GB):

```
max batch ≈ (3.6 − 0.46 − 0.38) / 0.53 ≈ 5.2  →  batch 4 is safe, 5 marginal, 8 does not fit (est.)
```

Batch 4 already logged allocator OOM retries in the baseline. Technique 6 (fused glue) shrinks the per-window activation, which may make room for a bigger batch.

**Cost and risk.** Effort: low to medium (stacking per-window text embeddings, ragged last batch). **Quality: none**: batching changes the order of operations only, and the plan gates it at PSNR ≥ 35 dB against batch 1. It does not help one small walk step that has only one or two windows to paint.

**In an interview.** "Anything paid once per pass over the weights (dequant, PCIe or SSD transfer, launches) is amortized by putting more tokens through each pass. On our GPU batching four panorama windows cut the per-window step from 2.74 to 1.76 s, and the arithmetic showed that saving was almost exactly the dequant cost. That told me batching and kernel fusion overlap, and that batching matters most once weights stream from the SSD."

<details><summary>Check yourself (Technique 3)</summary>

**Q1.** If Technique 1 removes dequant completely, what happens to the measured 1.56× gain of batch 4 when weights are resident in RAM, and why?

**A1.** It mostly disappears, to maybe 1.1–1.2× (est.). That gain was the 1.08 s per-pass dequant spread over 4 windows. Once dequant is gone there is little fixed per-pass cost left to share, and the GEMMs were already compute-bound at M=1280.

**Q2.** With weights read from the SSD at 2.0 GB/s, what is the lowest possible per-window-step time at batch 1 and at batch 4, from I/O alone?

**A2.** One pass reads 6.66 GB: 6.66 / 2.0 = 3.33 s. Batch 1: 3.33 s per window-step. Batch 4: 3.33 / 4 = 0.83 s.

</details>

### Technique 4: CUDA graphs

`Understand`. Not in the plan.

**The problem.** Every GPU kernel is started by the CPU: Python → PyTorch dispatch → CUDA driver → GPU queue. That costs roughly 5–20 µs of CPU per op in eager PyTorch (est.) and a few µs on the GPU between kernels. If the CPU can't enqueue kernels faster than the GPU finishes them, the GPU sits idle in the gaps.

**How many launches a step makes (est.).**

| Source | Count | Kernels each | Kernels per step |
|---|---|---|---|
| GGUF linears: 19 double × 14 (img q,k,v,out,ff1,ff2; same for txt; 2 modulation) + 38 single × 6 (mod, proj_mlp, q, k, v, proj_out) + ~6 embed/out | ~500 | ~20 today (unpack op chain + matmul) | ~10,000 |
| Glue per block (norms, modulate, RoPE, GELU, gates, residuals, concat/split) | 57 | ~40 | ~2,300 |
| Attention | 57 | 1 | 57 |
| **Total today** | | | **~12,000–15,000 (est.)** |
| After Technique 1 | | | ~3,000 (est.) |
| After Technique 6 | | | ~1,000 (est.) |

At ~10 µs each, 3,000 launches is about 0.03 s of CPU time against a ~1.9 s GPU step, so most of it hides behind the GPU queue. Graphs matter more as the step gets shorter and when Python hooks force syncs, as the offload and streamer hooks may.

**How it works.** Record the whole sequence of kernels once ("capture"), then submit it with one call ("replay"). The GPU gets the dependency graph and runs it with no CPU in the loop.

```
 eager:  CPU |k1|k2|k3|k4|...            GPU   [k1] [k2] [k3]   ← gaps when the CPU falls behind
 graph:  CPU |replay|                    GPU   [k1][k2][k3][k4]...
```

**Capture constraints.**
- **Static shapes**: one graph per (batch, resolution). That means 3–4 graphs (512, 1024×512, batch 2 and 4), each with its own memory pool.
- **Static addresses**: inputs that change (latents, timestep, embeddings) must be copied into the same buffers before each replay.
- **No CPU syncs or CPU-side decisions inside**: no `.item()`, no Python branches on GPU values, no allocations outside the graph's pool.
- **Python hooks don't run on replay.** The Task 7 streamer's copies would have to be captured as memcpy nodes. That works for the pinned-RAM arena, because the slot and host addresses are fixed. It conflicts with Task 8, where io_uring reads are driven by the CPU, unless you capture one graph per block and let the CPU sequence them.
- `check_stop()` between steps is fine: replay one step at a time.

**Expected gain (est.).** 1.0–1.1×, with **1.05×** as a working guess. Measure first with Nsight Systems (`nsys`): if the GPU timeline shows no gaps, graphs gain nothing.

**Cost and risk.** No quality impact. Medium effort; brittle when shapes vary, and it fights the streamer design.

**In an interview.** "Each kernel launch costs CPU time, and a diffusion step in eager PyTorch launches thousands of small kernels, so the GPU can idle between them. CUDA graphs record the step once and replay it as one submission, but they require static shapes and addresses and no host syncs. I'd only adopt them after an Nsight timeline showed real gaps."

<details><summary>Check yourself (Technique 4)</summary>

**Q1.** Why does a CUDA graph help less when most kernels are long GEMMs?

**A1.** Launches are asynchronous: while a long GEMM runs, the CPU can queue the next many kernels. Gaps only appear when GPU kernels finish faster than the CPU can enqueue them, which happens with many tiny kernels or a host sync.

**Q2.** The pano runs at batch 4 and a walk step at batch 1. What does that mean for graphs?

**A2.** Shapes differ, so each needs its own captured graph (and its own memory pool), or the batch has to be padded to a fixed size and the padded rows' compute wasted.

</details>

### Technique 5: Block caching (TeaCache / First-Block-Cache style)

`Understand`. Not in the plan.

**The problem.** Between two denoising steps the input changes, but sometimes most blocks' *contribution* (the residual they add) barely changes. If we can detect that cheaply, we can skip recomputing them.

**How it works.**

```
 step t:   x ──block 0──> r0(t) ── compare with r0(t−1) ──┐
                                                          │ rel change = |r0(t) − r0(t−1)| / |r0(t−1)|
             small (< threshold)? ────────────────────────┤
               yes: out = x + cached residual of blocks 1..56   (skip 56 blocks)
               no:  run blocks 1..56, store their residual for next time
```

- **First-Block-Cache** uses block 0's output change as the signal.
- **TeaCache** estimates the output change from the timestep-modulated input, using a fitted polynomial, and skips once the accumulated estimate stays under a threshold.

**Why 4-step schnell may benefit less (be honest).** Those methods were shown on 20–50-step samplers, where neighbouring steps are close and 30–50% of steps can be skipped. Schnell is *distilled* to 4 steps: each step jumps about a quarter of the noise range, and the model was trained to make each of those 4 calls count.
- Step 1 can never be skipped, because there is nothing cached yet. Only steps 2–4 are candidates, and skipping one removes 25% of the compute.
- The best case is 1.33×, and that means dropping a full step of a 4-step model. A realistic estimate at acceptable quality is **1.0–1.1×** (est.).
- It might do better on our partial-strength walk steps, which start at a lower noise level where steps change less. That is a hypothesis to measure, not a claim.

**Cost and risk.**
- **Quality risk: high.** A skipped step in a distilled model can blur detail or shift the composition. Effect is image-dependent, and thresholds tuned on one prompt may fail on another.
- Batching interaction: with a batch, all windows must skip together, or the batch splits.
- Memory: cached residuals cost ~8 MB per window (1280 × 3072 × 2 B), which is small.
- Effort: low to medium. Quality judgement needs A/B contact sheets.

**In an interview.** "Block caching reuses a block range's residual from the previous step when a cheap probe, such as the first block's output change, says the input hasn't moved much. It's a big win on 30–50-step samplers, but on a 4-step distilled model every step is a large jump, so the opportunity is at most one skipped step and the quality risk is high. I'd measure it on our walk steps, not assume the published speedups."

<details><summary>Check yourself (Technique 5)</summary>

**Q1.** Why can't block caching skip the first step?

**A1.** It reuses a residual computed at the previous step; at step 1 no previous residual exists. It also needs block 0's previous output to compare against.

**Q2.** On a 50-step model a cache skips 40% of steps. Why doesn't that translate to schnell?

**A2.** Skipping relies on adjacent steps being similar. With 50 steps they are tiny moves; with 4 steps each is a quarter of the noise range and the distilled model relies on all of them. The maximum skippable fraction is also lower (at most 3 of 4, realistically 0–1).

</details>

### Technique 6: Fused attention and norm/glue kernels

`Understand`. Not in the plan as a task.

**The problem.** "Glue" means every non-GEMM op around the matmuls: LayerNorm, the AdaLN modulate (`x·(1+scale)+shift`), QK RMSNorm, RoPE, GELU, gating (`gate·y`), residual adds, and the split/concat of text and image tokens. Each is memory-bound (~1 FLOP/byte): it reads a whole activation from VRAM and writes it back. One [1280, 3072] bf16 tensor is 7.9 MB; the MLP hidden state is 31.5 MB.

**What fraction of the step is glue (est.).**

| Stage | Step (est.) | Glue | Glue share |
|---|---|---|---|
| Today | 2.74 s | ~0.4 s | ~15% |
| After Technique 1 | ~1.9 s | ~0.4 s | ~21% |
| After Techniques 1 + 2 | ~1.36 s | ~0.4 s | ~29% |

As the GEMMs speed up, glue becomes a larger share of the step.

**Attention is already done.** PyTorch SDPA is already a FlashAttention-style kernel: it never builds the N×N score matrix in VRAM and computes softmax in tiles in shared memory. It runs at **17.7 TFLOPS, 2.5% of the step** (measured). Writing our own buys nothing.

**How fusion works.** Fold each elementwise op into the kernel that produces or consumes its input, so the intermediate tensor is never written to VRAM:

```
 today:  LN -> write -> read -> modulate -> write -> read -> GEMM ...        (each arrow = VRAM traffic)
 fused:  [LN + modulate] one kernel -> GEMM
         GEMM epilogue: + bias, GELU, gate·y + residual  (applied in registers before the single store)
         [QK RMSNorm + RoPE] one kernel, writing q/k straight in the layout SDPA wants
```

**Expected gain (est.).** Glue from ~0.4 s to ~0.1 s, saving ~0.3 s, or **~1.12×** on a ~1.4–1.9 s step (range 1.05–1.2×). It also shrinks the activation memory per window (fewer live intermediates), which can raise the batch limit in Technique 3.

**Cost and risk.** Quality: essentially none, if internals run in fp32 as the reference does; expect tiny numerical differences. Effort: medium. There are many small kernels, each needing a correctness test, and epilogue fusion means editing the Technique 1 kernel. An alternative is `torch.compile` for the glue alone, which is cheaper to try but harder to learn from.

**In an interview.** "After the GEMMs got fast, norms, modulation, RoPE and activations, which are all memory-bound, became a fifth to a third of the step. I fused them into neighbouring kernels: norm plus modulate in one pass, and bias, GELU and the gated residual in the GEMM epilogue, so each activation crosses VRAM once. Attention I left alone, because SDPA's flash kernel was already at 93% of the measured bf16 roof."

<details><summary>Check yourself (Technique 6)</summary>

**Q1.** Why does fusing GELU into the GEMM epilogue save time even though GELU's FLOPs are tiny?

**A1.** GELU is memory-bound. Unfused, it reads and writes the 31.5 MB MLP hidden state (63 MB of traffic per block, ~0.33 ms at 192 GB/s, est.). In the epilogue, the values are already in registers, so it costs a few ALU ops and no extra traffic.

**Q2.** Why does glue's share of the step rise as you optimise?

**A2.** Its absolute time stays the same (~0.4 s est.) while the GEMMs shrink, so its fraction of a smaller total grows. That's Amdahl's law.

</details>

---

## Part 3: Putting it together

`Must` for the two reality checks at the end; the factor arithmetic is `Understand`.

| # | Technique | Expected speedup on the step (est.) | Quality risk | Effort | In the plan? |
|---|---|---|---|---|---|
| 1 | Fused Q4_K + tensor-core GEMM (W4A16) | **1.45×** (1.25–1.7) | none (same maths; PSNR ≥ 30 dB gate) | high | yes, Tasks 3–5 |
| 2 | W4A8 on INT8 tensor cores | **1.4×** (1.2–1.6) on top of 1 | **medium**: activation outliers; needs a gate | high | no |
| 3 | Window batching | measured 1.56× per window today; **1.15×** after 1 (RAM); essential with the SSD | none (PSNR ≥ 35 dB gate) | low–medium | yes, Task 6 |
| 4 | CUDA graphs | **1.05×** (1.0–1.1) | none | medium; conflicts with streamer hooks | no |
| 5 | Block caching | **1.1×** (1.0–1.33) | **high** on a 4-step distilled model | low–medium | no |
| 6 | Fused glue (attention is already flash) | **1.12×** (1.05–1.2) | none | medium | no |
| — | Pinned layer streamer, io_uring loader | not a speedup: keep the server under 12 GB RSS; Task 8 adds an I/O floor | none | high | yes, Tasks 7–8 |

### Stacked estimate: s per window-step at 512 px, weights streamed from pinned RAM

```
today                                   2.74 s
Option A (1, 3, 4, 6):  1.45 × 1.15 × 1.05 × 1.12 = 1.96x  →  2.74 / 1.96 = 1.40 s
  + 2 (W4A8, if it passes the gate):    1.96 × 1.4  = 2.75x  →  2.74 / 2.75 = 1.00 s
Option B (all six, + 5):                2.75 × 1.1  = 3.02x  →  2.74 / 3.02 = 0.91 s
```

Sanity check: 17.7 TFLOP at 1.40 s is 12.6 TFLOPS, 66% of the measured bf16 peak. At 1.00 s it would be 17.7 bf16-equivalent TFLOPS, only possible because INT8 runs above the bf16 roof.

**Stacked estimates are optimistic.** Each factor was estimated as if it alone changed. In reality they overlap: 1 and 3 attack the same dequant, and 4 and 6 both remove small-kernel overhead. Each new kernel also lands below its target at first. **A realistic outcome for Option A is 1.6–2.0 s per window-step (est.).**

**Two reality checks that change priorities:**

1. **The panorama has already banked most of Technique 1.** Today's batch-4 pano already costs 1.76 s per window-step (56.3 s ÷ 32). Option A's 1.40 s takes it only to about 45 s (est.), **1.26×**. The rest of the gain has to come from faster GEMMs (W4A8) and fused glue, not from removing dequant. Technique 1 still matters for single-window walk steps, for VRAM, and as the base for Technique 2.
2. **After Task 8 (SSD-sourced weights), one window cannot go below 2.25–3.33 s per step** (6.66 GB ÷ 2.0–2.96 GB/s), about today's 2.74 s, however good the kernels are. At batch 4 the I/O is 0.56–0.83 s per window-step, under the compute time. So in the final configuration, compute wins only show up in batched passes, unless some weights stay cached in RAM.

---

## Part 4: Open decisions (for you)

`Skip` for interview revision; these are project decisions, not study.

**Decision 1: quality policy for speedups.**

- **A. Quality fixed.** Every speedup must produce the same picture, within the PSNR gates already in the plan. Do Techniques 1, 3, 4 and 6. Add 2 only if it passes a quality gate (same seed, PSNR against W4A16, plus your eye on contact sheets). Expected: ~1.4 s per window-step stacked, 1.6–2.0 s realistic (est.); ~1.0 s if W4A8 passes.
- **B. Speed first.** Accept measured, visible quality losses. Do all six, including block caching, with thresholds tuned on contact sheets. Expected: ~0.9 s per window-step stacked (est.), with a lower and less predictable quality floor.

What would decide it: whether FLUX's quality (the reason for phase 2, see spec §1) survives W4A8 and caching on your three reference journeys. A quick W4A8 PSNR study per layer, before writing the full kernel, would settle Technique 2 cheaply.

**Decision 2: the step recipe, pending from Task 2b.** The A/B contact sheets are yours to judge (`FLUX_STEP_RECIPE`, default `"sharp"`).

- **`sharp`**: graded release, focus 1.0 (the band ahead painted from scratch), sides 0.75, back 0.9, feather 0.25. It keeps the layout of where you came from. After the fix round its walk seam scores are ≤ 1.08 (limit 1.1); soft light-shaft veils remain (spot 3).
- **`fresh`**: a new panorama from the step's prompt and seed every step; only the prompt blend carries the walk on. It has no seams to manage, but no visual continuity either, and it is exempt from the continuity gate.

It matters for speed too: the recipe decides how many windows and denoising passes a walk step runs, and so whether walk steps are single-window (I/O-bound after Task 8, Technique 3 can't help) or batch-friendly. Before choosing, it is worth logging passes and windows per walk step for each recipe.

---

*Next entry suggestion: after Task 3's bench, record the naive kernel's numbers against these predictions and note where the estimates were wrong.* (Done in Entry 2.)

---

## Entry 2: Techniques from Tasks 5, 5b and 5c (tensor cores, W4A8, the modulation GEMV)

*2026-10-25. Study time: about 40 minutes. Sources: `docs/engine/03-q4k-mma.md`, `04-w4a8.md`, `05-modulation-gemv.md`, `docs/engine/speed.md` and the Task 5, 5b and 5c implementer reports. All numbers are measured on AC unless marked (est.). Read Entry 1's Part 1 (roofline) first.*

### Progress so far

`Must`

One 512 px window, one transformer pass (1024 image + 256 text tokens, 17.67 TFLOP). The walk step is a `sharp` walk step's wall time: 40 window passes in batches of 4, plus VAE and text.

| Stage | Step s | Achieved TFLOP/s | % of 19.1 bf16 peak | Walk step s |
|---|---|---|---|---|
| Baseline: diffusers + GGUF dequant | **2.74** | 6.45 | 34% | 61.4 (Task 1 painter, same path) |
| Task 3: naive Q4_K kernel (correctness reference) | 22.98 | 0.77 | 4.0% | 945.1 |
| Task 4: tiled, shared memory, CUDA cores | 3.77 | 4.69 | 24.5% | 156.3 |
| **Task 5: tensor cores, bf16 `mma` (W4A16)** | **1.40** | 12.62 | 66.1% | 57.4 |
| **Task 5b: W4A8 on INT8 tensor cores (default)** | **1.15** | 15.37 (bf16-equivalent) | 80.4% | 48.9 |
| **Task 5c: modulation on a Q4_K GEMV** | **1.12** | 15.78 (bf16-equivalent) | 82.6% | 47.6 (47.0 when re-run in Task 6's session) |
| **Task 6: glue fusion (bit-identical)** | **0.93** | 19.00 (bf16-equivalent) | 99.4% (see Task 6 for why this isn't "at the limit") | **39.5** (against 47.0 in the same session) |
| **Task 7: fixed weight slots, eager by default (graphs opt-in)** | **0.85** (Task 6 re-run in the same session: 0.92) | 20.79 (bf16-equivalent) | 108.8% (bf16-equivalent, see Task 6) | **38.3** (Task 6 same session: 39.0) |
| **Task 8: first-block cache, off by default; step unchanged** | **0.86** (Task 7 in its own session: 0.85) | 20.55 | 107.6% | unchanged (39.2 s wall in Task 8's bench session) |
| *Task 8, would-be at threshold 0.3 (fails the quality gate, not shipped)* | *0.650 mean (1.32×)* | *27.19 nominal: work skipped, not done faster* | — | *24.6 s wall (1.59×; 1.63× per pass)* |
| **Task 9: window batching, 8 windows a pass (bit-identical to batch 1)** | **0.87** (1 window: batching can't change it) | 20.31 | 106.3% | **39.3** in 5 passes (was 10); panorama **29.8 s** in 4 passes (was 8); peak 2.05 GB |

From Task 5b on, "TFLOP/s" counts the same 17.67 TFLOP even though 75% of it runs as INT8, so it is **bf16-equivalent**: how fast a bf16 engine would need to be to match. It can pass the bf16 roof without breaking physics.

Through Task 5c the step was 2.45× faster than the baseline (2.74 → 1.12 s), but the walk step only 1.29× (61.4 → 47.6 s). After Task 6: step 2.95× (0.93 s), walk step 1.55× (39.5 s). After Task 7: step 3.2× (0.85 s), walk step 1.6× (38.3 s). Task 8 added no speed: its cache ships off. Task 9 added no speed either: it halves the passes, which pays off in Task 10. (The walk step drifts 38.3–39.3 s between sessions with heat.) Why the walk gains less is the first lesson of Task 5.

**How the Entry 1 estimates did.**

| Estimate (Entry 1) | Predicted | Measured | Why it differed |
|---|---|---|---|
| Technique 1 (fused W4A16) step | ~1.9 s (1.6–2.2) | **1.40 s** | the kernel reached 92% of cuBLAS; I assumed 65–85% |
| Technique 2 (W4A8) GEMM speedup | 1.8–2.5× | **1.45–1.72×** | the kernel is issue-bound (Task 5b below) |
| Technique 2 step factor | ×1.4 | **×1.22** (1.40 → 1.15) | the slower GEMM, plus 25% of the FLOPs kept on bf16 for quality |
| Modulation "cheap target" | ~0.29 s | **~0.03 s** | the 0.29 s was dequant, which Task 3 already removed (Task 5c below) |

---

### Tried and not chosen: how to read these tables

`Must` (the habit, not every row)

Every task below ends with a **"Tried and not chosen"** table: one row for each alternative we actually built or measured, what it measured, why we didn't pick it, and **when it might win instead**. That last column is the interview material. "Why didn't you do X?" is a common question, and "we measured X at N, and it loses here because of Y, but it would win if Z" is the strongest possible answer. Where a number was never recorded, the row says so rather than guessing.

---

### Task 2b: the walk-step recipe

`Understand`

> **Plain version.** A walk step moves the camera, then repaints the view. The question is *how much* of the old picture to keep in each part of the panorama: the band ahead (new ground), the sides and the band behind. Keep too little and the scene stops being continuous; keep too much, or keep it unevenly, and you get seams where a freshly painted band meets an old one. We tried six recipes and several variants of the winner. **`sharp`** won: paint ahead from scratch (focus 1.0), release the sides at 0.75 and the back at 0.9, all "graded" (each column released once at its own strength, then denoised with its neighbours), and feather the focus edge over 0.25 of a turn. Its worst seam on the gate walk is 1.08, against a limit of 1.1. You confirmed the pick on 2026-10-24.

**Tried and not chosen.** Seam score: the strongest full-height edge in a spot (`walk_sheet.py`; above 1.1 fails).

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| `t1` (Task 1's step): linear release. The band ahead is painted from scratch; the rest is held to the original until sigma 0.15; the feather is a crossfade | Walk-step seams 1.13–1.79. Visible double exposures at the feather, chevrons behind, scene jumps | The two regions never denoise together, so the feather is a blend of two different pictures, a seam by construction | When the camera barely moves and both sides already agree (small local edits, inpainting a small object) |
| `t1` with side strength 0.5 | "did not fix" the seams (no per-spot score archived) | Same mechanism: a softer hold doesn't make the regions denoise together | Not likely here; the fix had to change the release, not the strength |
| `repaint`: graded, side 0.5, back 0.5, focus 1.0, feather 0.15 (SD-Turbo's approach) | Per-recipe seam scores not archived. Root cause measured: `dolly()`'s mirror padding is 38% of rows at z 2.5, and the 0.5 back band keeps it, so chevrons | Back 0.5 keeps the mirror artefacts behind the camera | With a model or camera move that creates no mirror padding (small dolly), or SD-Turbo, where it worked |
| `carry`: focus 0.75, back 0.75 | Scores not archived. A related test, focus 0.95 on sharp, brought back the parent's blurred close-ups | Focus below 1.0 carries the dolly-magnified blur ahead into the new picture | Small forward moves, where the content ahead is already sharp and not magnified |
| `renew`: focus 0.9, back 0.9 | Scores not archived; same focus < 1.0 issue as carry | Same as carry | Same as carry |
| `fresh`: a new panorama each step; only the prompt blend carries the walk on | No seams by design. Continuity score 0.08 (the baseline that sharp's bar 0.12 sits just above); exempt from the continuity gate | No visual continuity: each step is a new place, so walking feels like teleporting | Deliberate scene changes (the journey drifting to a new place), or a "teleport" move |
| `sharp` as first proposed: side 0.5, feather 0.15 | Seams 1.36 / 1.38 / 1.13 at spots 2, 4, 5 (fail) | Columns released at 0.6–0.85 between a fresh band and an old view became haze ending in a hard full-height edge | — (it was fixed, not replaced) |
| Binary (stepped) releases instead of graded | Seam 1.79 on the same step | Harder edges: a step in strength makes a step in the picture | Never, for blending; graded release is strictly better here |
| Feather 0.25 alone | That step's edge 1.41 → 0.64, but the next walk had haze bands at the focus edge: 1.33 / 1.31 | Moved the problem rather than fixing it | Together with wider sides, which is what was chosen (0.82 / 0.80) |
| Side 0.75 alone | That step's edge 1.41 → 0.67 (single-step test, not walked alone) | Not tested on a whole walk alone; the combination was | — |

---

### Task 3: the naive Q4_K kernel

`Understand`

> **Plain version.** The simplest correct kernel: one thread per output number, each decoding its own whole weight row. Its job was to pin down *what* the right answer is, not to be fast. It removed the 1.08 s dequant but made the step **22.98 s** (8.2× slower), as expected. It stays as the reference every faster kernel is tested against.

**Tried and not chosen** (isolated FLUX linears, TFLOP/s):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| First version: 32×8 thread blocks, scalar loads | 0.45 TFLOP/s | Slowest; each load fetched one value | Never as a fast path; it is the clearest version to read |
| 32×8 blocks with vector loads | 0.55 | The 16×16 shape measured better | — |
| Threads laid along m (tokens) instead of n (outputs) | 0.30 | Measured worst of all | Not measured beyond this; the cause wasn't profiled |
| **Chosen:** 16×16 blocks + vector loads (4 code bytes, 4 + 4 bf16 x per load) | 0.85 | — | — |
| Naive as the engine's kernel | 22.98 s/step against 2.74 s | No reuse: every weight decoded 1,280 times per pass, on CUDA cores | It did win at M = 1 (0.65 ms against `mma`'s 0.84 ms on one modulation linear), so it ran modulation until Task 5c's GEMV |

---

### Task 4: the tiled kernel (shared memory, CUDA cores)

`Understand`

> **Plain version.** Same maths as naive, but each block of threads works on a 128×128 tile of outputs. It decodes each weight **once per tile** into shared memory, so 128 rows of x reuse it. That made it **6.1× faster than naive** (3.77 s), with the dequant gone. It is still **1.38× slower than the reference** (2.74 s), because it multiplies on CUDA cores, which top out around **5.3 TFLOP/s** here (cuBLAS's own fp32 SGEMM, measured), while the reference used tensor cores. So it stays a teaching stage.

**Tried and not chosen** (isolated linears at M = 1280, range over 5 FLUX shapes, TFLOP/s; clocks move ±5%):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **Tiled as the default kernel** | 3.77 s/step against the reference's 2.74 s | CUDA cores cap it at about cuBLAS SGEMM's 5.3 TFLOP/s; the reference's tensor cores do 19.1 | A GPU or data type without tensor-core support, or where fp32 math is required |
| 128×128 tiles, plain row order, no register cap | 3.2–3.5 | Only 1 block per SM, and no L2 sharing between neighbouring blocks | — |
| 128×64 tiles | 3.7–5.0 | Less reuse per weight than 128×128 once 2 blocks per SM fit | Smaller M, where a 128×128 grid would leave SMs idle |
| 64×128 tiles | 2.0–4.8 | Lower reuse of x | — |
| 64×64 tiles | 2.9–3.9 | Lower reuse both ways | Small matrices with too few 128-tiles to fill 20 SMs |
| 32×128 / 16×128 tiles | 2.1–2.7 / 1.1–2.3 | Little reuse of each decoded weight | Very small M (few tokens) |
| Double-buffered shared memory (128×64, 64×128, 64×64) | 1.9–4.3 | Twice the shared memory, fewer resident blocks: **occupancy** fell more than the overlap gained | When the copy doesn't go through registers (`cp.async`, Task 5), where double buffering *did* win |
| 128×64 with 4 blocks per SM forced (register cap 128) | 3.2–3.9 | The register cap cost more than the extra blocks gained | — |
| Coalesced x loads (4 lanes per row, a warp reads 512 contiguous bytes) | Same as 128×64 (±1%) | No gain for extra complexity; L1 already caught the other half of each sector | If x loads ever became the bottleneck |
| Grouped order GROUP_M = 4 / 10 | 4.4–5.1 / 3.0–5.0 (128×64), against 4.7–5.2 for 2 | Larger groups shared L2 less well here | Other shapes or bigger L2 |
| 128×128, 2 blocks per SM, GROUP_M = 4 | 4.6–5.7 | About the same as the chosen GROUP_M = 2 (5.2–5.6), less consistent | — |
| Tiled at M = 1 (modulation) | 4.0 ms against naive's 0.7 ms on 3072 → 18432 | A 128-row tile at M = 1 is 127/128 padding; crossover measured at M = 24–32 | M ≥ 32 |

---

### Task 5: the Q4_K kernel on tensor cores (W4A16)

`Must` for the plain version, warps, fragments, `mma` and double buffering. `Understand` for the decode trick, `ldmatrix` and swizzle. `Skip` the exact nibble-to-lane map.

> **Plain version.** Task 4 removed the 1.08 s dequant but did its multiplies on the ordinary CUDA cores (~5 TFLOP/s), so it was *slower* than the baseline (3.77 s). Task 5 keeps "never unpack the weights to memory" and moves the multiply onto the tensor cores. Each group of 32 threads unpacks only the 4-bit weights its next tensor-core instruction needs, into registers, right before issuing it. The result is 2.74 → **1.40 s**, with the GEMM at **92% of cuBLAS** on weights cuBLAS was handed already unpacked.

**Warps and fragments** (`Must`). A **warp** is 32 threads that issue each instruction together. `mma.sync.m16n8k16` (bf16 in, fp32 accumulate) is one *warp-wide* instruction: it multiplies a 16×16 tile of A (activations) by a 16×8 tile of B (weights) and adds the result into a 16×8 fp32 tile C. That is 2,048 multiply-adds in one instruction. No thread holds a whole tile. Each holds a **fragment**, a few registers at positions fixed by the PTX spec:

```
 lane = 4*g + t   (g = lane/4 = 0..7, t = lane%4 = 0..3)

 A 16x16 bf16:  4 regs per lane = rows g and g+8, k = 2t, 2t+1, 2t+8, 2t+9
 B 16x8  bf16:  2 regs per lane = column g,       k = 2t, 2t+1, 2t+8, 2t+9
 C 16x8  fp32:  4 floats        = rows g and g+8, columns 2t, 2t+1

            k:  0  1 | 2  3 | 4  5 | 6  7 | 8  9 |10 11 | ...
 B column g:   t=0   | t=1  | t=2  | t=3  | t=0  | t=1  | ...   (4 lanes share one column)
```

**Decoding straight into B fragments** (`Understand`). A lane needs four weights of one weight row at k = 2t, 2t+1, 2t+8, 2t+9. In Q4_K's layout those are bytes 2t, 2t+1, 2t+8, 2t+9 of a 32-byte run. Low nibbles feed one k16 step and high nibbles another, so **two 16-bit shared-memory loads give the codes for two fragments**. All four k values sit inside one 32-weight sub-block, so one (d·sc, dmin·m) pair decodes them. Per weight:

1. A byte permute builds the float 2²³ + q. A float with exponent 2²³ has a mantissa step of exactly 1, so q lands in the mantissa with **no int-to-float instruction**.
2. One subtract recovers q exactly.
3. `q·(d·sc) − dmin·m` in fp32, with gguf's own two roundings, so the weight is bit-exact.
4. `cvt.rn.bf16x2` packs two weights into one register, which goes straight into `mma`.

The kernel reads the GGUF bytes as they are (**no offline repacking**, unlike Entry 1's Marlin suggestion). That keeps one layout for the bit-exact dequant test, the naive fallback and the fast kernel.

**`ldmatrix` and swizzle: avoiding bank conflicts** (`Understand`). Shared memory is split into 32 **banks** of 4 bytes. If several lanes hit the same bank in one request, the accesses serialize (a **bank conflict**). `ldmatrix.x4` loads A fragments: each lane gives the address of one 16-byte row piece, and the hardware deals the fragments out. An x row in shared memory is 64 bf16 = 128 bytes, which is exactly all 32 banks, so the same column of 8 rows lands in the **same 4 banks**, an 8-way conflict. **Swizzle** fixes it by storing 16-byte chunk c of row r at position c XOR (row bits):

```
 plain layout (chunk index per row)      XOR-swizzled (chunk c stored at c ^ (r % 8))
 row 0:  0 1 2 3 4 5 6 7                 row 0:  0 1 2 3 4 5 6 7
 row 1:  0 1 2 3 4 5 6 7                 row 1:  1 0 3 2 5 4 7 6
 row 2:  0 1 2 3 4 5 6 7                 row 2:  2 3 0 1 6 7 4 5
 ...                                     ...
 ldmatrix reads chunk 0 of rows 0-7:     chunk 0 of rows 0-7 now sits in 8 different
 all in one bank group -> 8-way          bank groups -> 1 pass, no conflict
```

The writer and the reader apply the same XOR, so it costs one instruction. The weight side needs none: a row's 48 bytes per stage (16 B header + 32 B run) is an odd multiple of 16 B, so 8 rows already spread over 16 banks.

**`cp.async` double buffering** (`Must`). `cp.async` copies global memory to shared memory without passing through registers, and the thread doesn't wait for it. With two shared-memory **stages**, the warps compute on one while the next K-slice loads into the other:

```
 time ->
 stage 0:  [load K0]           [load K2]           [load K4]
 stage 1:            [load K1]           [load K3]
 tensor:             [mma K0 ] [mma K1 ] [mma K2 ] [mma K3 ] ...
           one __syncthreads per K step swaps the roles
```

**Choosing the tile by SM fill** (`Understand`). The main tile is 128×128 outputs per block. Its 8 warps sit side by side along N, each with a 128×16 strip, so **every weight is decoded exactly once per block**. With 2 warps along M, each weight was decoded twice, which cost ~5%. Two blocks fit per SM (2 × 44 KB of shared memory), so there are 40 slots on 20 SMs. Blocks run in **waves**, and a ragged last wave leaves SMs idle:

```
 M = 256, N = 3072 (a text-stream linear), 40 block slots:
   128x128 tiles: 2 x 24 = 48 tiles  -> wave 1: 40, wave 2: 8 (32 slots idle)  -> ~60% busy (est.)
   128x64  tiles: 2 x 48 = 96 tiles, each half the work -> 40 + 40 + 16       -> ~80% busy (est.)
 rule in the launcher: under 160 tiles (4 waves), use 128x64
```

At M = 256 that keeps the kernel at 92–105% of cuBLAS. **The tuning lesson from the tile sweep:** two resident blocks per SM mattered more than pipeline depth or tile size. Three stages fit only one block per SM and measured worse (77–89% vs 90–99%), because the second block's loads hide the first one's math.

**Where the last ~8% to cuBLAS goes** (`Understand`). Nsight Compute needs root here, so this was measured by ablation. A timing-only build that skips the decode (wrong results) ran at 96–104% of cuBLAS, so **the decode costs about 5–8%**: issue slots and registers next to the `mma`s, because each weight is decoded 10 times per pass (once per 128-row M tile). The other few percent is plain GEMM machinery (2 stages against cuBLAS's deeper pipelines).

**The honest part: the walk step** (`Must`). The walk step went only **61.4 → 57.4 s (7%)** while the tile went 2.74 → 1.40 s (1.96×). A walk step batches 4 windows per pass, so the reference's per-pass dequant was *already* spread over 4 windows (Entry 1, Technique 3 predicted this overlap), and its batched GEMMs ran at cuBLAS speed. Our kernel never paid a per-pass dequant, so batching doesn't help it either: 1.44 s per window pass in the walk against 1.40 s in the tile.

**In an interview.** "I wrote a W4A16 GEMM that decodes Q4_K straight into `mma.sync` B-fragments in registers. The fp16 weight never exists in memory, `cp.async` double-buffers the tiles, and swizzled `ldmatrix` feeds the activations. It reached 92% of cuBLAS on pre-dequantized weights. An ablation showed the decode costs the remaining 5–8%, and the tile sweep showed two resident blocks per SM mattered more than pipeline depth."

<details><summary>Check yourself (Task 5)</summary>

**Q1.** Why does `mma` at M = 1 run no faster than at M = 64 (0.84 ms at every M up to 64 on a 3072 → 18432 linear)?

**A1.** A 16-row `mma` tile (128-row block tile) does the same work whatever M is, and the time is set by reading the 32 MB weight once. At M = 1, 127/128 of the tile is padding.

**Q2.** Why did 3 pipeline stages measure *worse* than 2?

**A2.** Three stages use enough shared memory that only one block fits per SM. With two blocks per SM, one block's loads overlap the other's math, which hides latency better than a deeper pipeline in one block.

</details>

**Tried and not chosen** (`Understand`). % of `torch.matmul` (cuBLAS) on the same weights pre-dequantized, M = 1024/1280 shapes, same harness:

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| 128×128, warps 2×4 (64×32 each), **3 stages**, 1 block per SM | 77–82% | Three stages' shared memory fits only 1 block per SM, so nothing hides one block's latency | A GPU with more shared memory per SM, where 3 stages and 2 blocks both fit |
| The same, 4 stages | 77–89% | Same one-block limit | Same |
| 128×128, warps 4×2 (32×64), 3 stages | 69–81% | Worse warp shape and one block per SM | — |
| 128×256 tile, 64×64 warps, 3 stages | 80–89% | Big tile, one block per SM | Very large GEMMs on bigger GPUs |
| 64×128, 32×32 warps, 3 stages, 2 per SM | 74–87% | Smaller tile: each weight decoded more often, less reuse | — |
| 64×64, 32×32 warps, 3 stages, 3 per SM | 67–79% | Even less reuse | Small matrices |
| 128×128, warps 2×4, 2 stages, 2 per SM | 86–101% | Each weight decoded **twice** per block (2 warps along M): about 5% slower than the chosen 1×8 layout | — |
| **Chosen:** 128×128, warps 1×8 (128×16 each, each weight decoded once), 2 stages, 2 per SM | **90–99%** | — | — |
| 128×64 tile with 3 stages | 71–77% | 3 × 57 KB shared memory: 1 block per SM | — |
| 128×64 tile (2 stages, 2 per SM) | 86–95% (M = 256: 91–105%) | Not chosen for big grids; **chosen** under 160 tiles (fills the last wave better) | The M = 256 text linears, where it is used |
| Grouped tile order GROUP 2 / 8 / 16 instead of 4 | Within 1% | No difference with 2 blocks per SM | — |
| FMA decode (one rounding instead of mul + sub) | Within 1% | No speed gain, and it would lose gguf's bit-exact rounding | If the decode ever became ALU-bound |
| Offline repacking of the weights (Marlin-style; Entry 1 suggested it) | Not built. Upper bound: the whole decode, layout included, costs 5–8% (the no-decode ablation) | GGUF's layout already gives conflict-free 16-bit loads, and one layout keeps the bit-exact test and the naive fallback simple | Memory-bound small-M GEMMs, where one wide load per fragment matters more |
| `mma` at M = 1..6 | 0.84 ms at every M ≤ 64, against naive's 0.65 ms at M = 1 (3072 → 18432) | The 128-row tile is mostly padding; M < 7 went to naive (Task 5c later routed it to a GEMV) | M ≥ 7 |

> **Remember this:** keep the data in registers. Decode exactly what the next tensor-core instruction needs, overlap the next load with this math, and the "4-bit tax" shrinks to ~8% of a cuBLAS GEMM.

---

### Task 5b: W4A8 on the INT8 tensor cores

`Must` for per-token scales, why k = 32 fits, "issue-bound" and the study vs ablation lesson. `Understand` for the min-term algebra, the int-to-float trick and the gate method. `Skip` the v2 fp16 min-term experiment.

> **Plain version.** After Task 5 the GEMMs were at 92% of cuBLAS in bf16, so a better bf16 kernel couldn't win much. The INT8 tensor cores run at ~4× our bf16-with-fp32-accumulate rate (est., Entry 1 §1.2). The weights are already small integers (Q4_K codes 0..15), so only the activations had to shrink. Each token's activations are rounded to 8-bit integers with that token's own "ruler" (scale), and the multiply runs on INT8. We got **1.40 → 1.15 s**, with **75%** of the math on INT8. The other 25% stays bf16 because the picture needed it, and *which* 25% was the surprise.

**Per-token activation scales** (`Must`). For each token (row) t: s_t = max|x_t| / 127, xq = round(x_t / s_t) in −127..127, plus xsum (the sum of xq over each 32 values, for the min term). A loud token gets a coarse ruler, **but only for itself**. `test_w4a8_outlier_token` multiplies one token by 1000 and checks that **every other token's output is bit-identical**: the outlier is contained. One scale per tensor would let the loudest token set the step for everyone. The cost is rounding: FLUX tokens have a few big values (max |x| a median 25–50× the median |x|, and ~700–1400× in the worst text-MLP layers), so most values use only a handful of the 255 levels.

**INT8 `mma.m16n8k32`: k = 32 is one Q4_K sub-block** (`Must`). The INT8 instruction multiplies along k = 32, exactly one Q4_K sub-block. So one instruction's whole integer sum shares one (d·sc_j, dmin·m_j) pair. The codes go in **as they are** (0..15 fits a signed byte), with no decode: mask `& 0x0F0F0F0F` for low nibbles and `>> 4` for high ones.

**The min-term algebra** (`Understand`). With x_i = s·xq_i and w_i = (d·sc_j)·q_i − dmin·m_j, one sub-block's share of an output is:

```
 Σ x_i w_i  =  s · [ (d·sc_j) · Σ xq_i q_i   −   (dmin·m_j) · Σ xq_i ]
                               \_ int32, one mma _/          \_ xsum_j: from quantize_act, once per token _/
```

The xsum doesn't depend on the output column, so it is computed once and reused by every output. The kernel accumulates `(d·sc_j)·int_sum − (dmin·m_j)·xsum_j` in fp32 per sub-block, multiplies by s once in the epilogue, adds the bias and rounds to bf16. The integer part is exact, so the kernel matches "fake-quantized x × exact weight" to ~1e-6. **All the loss is in x's rounding, none in the kernel.**

**The integer-exact int-to-float trick** (`Understand`). The `mma`'s accumulator input C is set to the integer 0x4B400000, which is the bit pattern of the float 2²³ + 2²². The `mma` adds the integer dot product to that pattern, so the sum lands in the mantissa. One fp32 subtract of 12,582,912 then gives the exact sum as a float. It is exact as long as the sum stays within ±2²² (4,194,304). The largest possible is 32 × 127 × 15 = **60,960**, far below. The hardware int-to-float conversion runs at 1/8 of the FMA rate and would have dominated.

**Why 1.5×, not 4×: issue-bound** (`Must`). Around each INT8 `mma` (~16 cycles per SM sub-partition, est.), a lane spends **~12 issue slots** on the rescale alone: subtract, FMA, FMA for its 4 outputs, plus unpacking the sub-block scales. The tensor cores wait on the instruction stream, not the reverse. Timing-only ablations (wrong results) show where it goes:

| Variant | GEMM speed vs bf16 `mma` |
|---|---|
| chosen kernel | **1.64×** |
| no scale unpacking | 1.87× |
| no min-term FMA | 1.92× |
| neither, nor the xsum loads | 2.10× |

Measured per FLUX shape, including the quantize pass: **1.45–1.72×** the bf16 kernel, 135–156% of cuBLAS bf16, ~25 bf16-equivalent TFLOP/s.

**The lesson: per-layer error did NOT predict the failure** (`Must`).

1. **The study.** Before writing the kernel, a fake-quant study measured each linear's error *on its own* (its output with per-token INT8 x vs exact x). The errors were fairly flat, 1–4% relative L2, worst in the text MLP (up to 7.2%) and the value projections. Attention-out ranked *least* hurt (1.1–1.2%).
2. **All-INT8 failed the walk gate** (spot 1 seam 1.25 > 1.1). Keeping the study's 5 worst families (A2) still failed, and so did adding the study's next-worst 23 layers.
3. **The ablation** over 8 keep-lists found what passes: A2 plus **every linear by which a double block writes into the residual stream**, the image MLP-out (`ff.net.2`) and both attention-outs (`to_out.0`, `to_add_out`). Drop either kind and spot 1 fails (1.22, 1.24). Keeping the first and last blocks whole added nothing.

```
 residual stream:  x ──(+)───────(+)───────(+)── ... ──> 57 blocks later
                       ^          ^          ^
                    attn-out    MLP-out    next block's writes   ← an error here is carried to the end
            inside attention / MLP: errors partly washed out by the next norm and nonlinearity
```

A plausible reading: an error written into the residual stream persists through every later block, while an error inside one attention or MLP is partly washed out. The isolated per-layer metric measured the second kind well and the first poorly. **Judge lossy changes on the real output, not layer by layer.**

**Gate methodology** (`Understand`).
- **A deterministic walk:** a fixed 6-spot `sharp` walk (haunted forest, seed 7, fixed headings). The same code gives bit-identical images, so a score moves only when the arithmetic changes.
- **A seam limit:** `walk_sheet.py --check` scores each spot's strongest full-height seam and fails above SEAM_LIMIT 1.1. The lossless paths set the noise floor (Task 5 `mma` 1.080, diffusers 1.082).
- **Evidence on committed code:** two fresh walks at commit a2bdb67 (clean tree, AC, no overrides), each recording kernel, keep-list and git sha. Both passed and were bit-identical: max seam **1.058**, below the lossless paths' own 1.08.
- **Caveat:** that is one scene. Tasks 6 and 8 re-gate it. Task 5c later moved the default sample (25.78 dB vs 5b) and the walk was not re-walked, so the 1.058 margin should be re-checked before relying on it. *(Re-checked in Task 6: the Task 5c default scores 1.15 at spot 1, above the fixed 1.1. So from Task 6 on, the gate is relative: no spot more than 0.03 above the previous task's walk. Ruling 2026-11-06.)*

**The PSNR trap.** The 512 test image under W4A8 is 25.22 dB from `mma`'s: the same path, ferns and light shafts, with some trunks in other places. Four denoising steps turn tiny rounding into a *different sample of the same prompt*, not a noisy copy. That is why a lossy stage is judged by the walk gate and the eye, not by PSNR against the reference.

**Results.** Step **1.15 s**, **15.37** bf16-equivalent TFLOP/s (80.4%), **75.0%** of token-facing FLOPs on INT8, max seam **1.058**, walk step 57.4 → **48.9 s**.

**In an interview.** "I moved FLUX's GEMMs to W4A8: per-token INT8 activations on the INT8 tensor cores, where k = 32 of `m16n8k32` is exactly one Q4_K sub-block, and the offset term uses precomputed activation sums. The GEMM got ~1.5× rather than 4× because the per-sub-block rescale makes it issue-bound. The real lesson was quality: a per-layer error study didn't predict which layers broke the image. An ablation on a deterministic end-to-end gate showed the residual-stream writes had to stay bf16."

<details><summary>Check yourself (Task 5b)</summary>

**Q1.** Why can the int-to-float trick use a float base and one subtract, and what limits it?

**A1.** A float of magnitude 2²³–2²⁴ has a mantissa step of exactly 1, so adding a small integer to its bit pattern gives that float plus the integer, exactly. Starting at 2²³ + 2²² leaves room for negative sums. It holds while |sum| < 2²²; ours is at most 60,960.

**Q2.** The study ranked attention-out as the least hurt layer. Why did keeping it bf16 still matter?

**A2.** The study measured each layer's error in isolation. Attention-out writes into the residual stream, which carries its error through every later block, so a small isolated error became a large end-to-end one. Only an end-to-end test (the walk gate) sees that.

</details>

**Tried and not chosen: the kernel** (`Understand`). GEMM speed against the bf16 `mma` kernel, from the same timing-only harness:

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **Chosen (v1):** 2 stages, scales unpacked per lane, min term as fp32 FMAs | **1.64×** (per-shape bench incl. quantize: 1.45–1.72×) | — | — |
| 3 pipeline stages | 1.52× | More shared memory, fewer resident blocks; the kernel is issue-bound, so deeper prefetch doesn't help | If the kernel became latency-bound instead of issue-bound |
| v2: min term as one fp16 `mma.m16n8k8` per super-block (correct results) | 1.44× | 18 more registers per thread **spilled** the 128×128 tile | A smaller tile, or a GPU with more registers per thread, where it wouldn't spill |
| Hardware int → float conversion (instead of the 0x4B400000 trick) | Not timed alone; it runs at 1/8 of the FMA rate here | It would have dominated the rescale | GPUs where conversion runs at full rate |
| No rescale work (timing-only, **wrong results**): no scale unpack / no min-term FMA / neither, nor xsum loads | 1.87× / 1.92× / 2.10× | Not a real option: it measures the prize | Shows what cooperative scale unpacking could win (follow-up; ceiling ~+15%) |

**Tried and not chosen: which layers stay bf16** (`Must`: the method). Each list was walked on the deterministic 6-spot gate; max seam over spots 0–5 must be ≤ 1.1. Lossless baselines: Task 5 `mma` 1.080, diffusers 1.082. "INT8 share" is the share of token-facing FLOPs on INT8.

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| All-INT8 (keep nothing) | 100% INT8; seam **1.25** at spot 1 (fail) | Failed the gate | A scene without borderline fog edges, or a looser gate; or with smarter activation scaling (SmoothQuant) |
| A2 = the study's 5 worst families | 86.7%; **1.16** (spot 1), 1.14 (spot 3): fail | Failed | — |
| A2 + the study's next-worst 23 single layers (S26) | 77.5%; **1.19** / 1.11: fail | Failed: **picking layers by per-layer error doesn't work** | — |
| A2 + whole blocks (double 0, 18; single 0, 37) (BL) | 80.8%; **1.16**: fail | The first and last blocks aren't the sensitive part | — |
| A2 + attention-outs, no MLP-out (A2+AO) | 83.9%; **1.22**: fail | Missing the MLP-out residual write | — |
| A2 + MLP-out + whole blocks, no attention-outs (A3-AO) | 72.8%; **1.24**: fail | Missing the attention-out residual writes | — |
| Attempt 3: A2 + MLP-out + attention-outs + whole blocks (Entry 1's guess) | 70.3%; **1.098: pass** (margin 0.002); step 1.17 s, walk 49.4 s, 24.91 dB vs mma | Passed, but FA (chosen) passes with a bigger margin (1.058) and more on INT8 (75.0%, step 1.15 s, walk 48.9 s): the whole blocks added nothing | If a later scene shows the FA list failing; it's the safer fallback |
| **Chosen: FA** = A2 + MLP-out + attention-outs | **75.0%; 1.058: pass**, twice, bit-identical, at committed a2bdb67 | — | Not proven minimal: the A2 patterns were never dropped one at a time |
| Choosing the keep-list from the per-layer fake-quant study alone | The study ranked attention-outs *least* hurt (1.1–1.2%); its lists all failed | Isolated per-layer error doesn't see errors accumulating through the residual stream | A cheap first filter (it did give A2, which every passing list needed) |
| Modulation linears on INT8 | Highest per-layer error in the study, 4–7% rel L2; 0.04% of the math | Big error for no speed: they're memory-bound GEMVs, where INT8 math buys nothing | Never on this model |

> **Remember this:** lower precision buys speed only if the kernel isn't issue-bound, and buys quality loss where errors accumulate (the residual stream), so check it on the real output, not layer by layer.

---

### Task 5c: the modulation GEMV

`Must` for the plain version, why 4-bit helps at M = 1, and re-measuring before optimizing. `Understand` for the kernel design. `Skip` the register counts and the two-rows-per-warp tuning.

> **Plain version.** Modulation (Entry 1 §1.6) is 27% of the weights (1.815 GB of Q4_K), but each linear multiplies **one row** per image. That makes it a **GEMV** (matrix × vector), with about one multiply-add per weight fetched. So it is **memory-bound**: time = bytes ÷ bandwidth. Here, unlike the big GEMMs, **4-bit weights do make it faster**: 3.6× fewer bytes than bf16 is 3.6× less time. The old path (the naive kernel) read those bytes in scattered pieces at 52 GB/s. The new kernel reads them in big coalesced chunks at **165 GB/s**, and modulation went **35.1 → 11.0 ms** per pass. The floor the bytes allow is 9.7 ms.

**Why M = 1 is different** (`Must`). Using the roofline from Entry 1, with VRAM now measured at **187 GB/s** (a 512 MB `torch.sum`; 192 on the spec sheet), the ridge is 19.1 T ÷ 187 G ≈ 102 FLOP/B:

```
 weight format   bytes/weight   AI at M = 1   AI at M = 1280
 bf16            2              1.0           1280      (compute-bound: 4-bit saves memory, not time)
 Q4_K            0.5625         3.6           4551
                                ^ far left of the ridge: every byte saved is time saved

 76 modulation linears per pass:  bf16 6.45 GB / 187 GB/s = 34.5 ms   Q4_K 1.815 GB / 187 = 9.7 ms (floor)
   before (naive, scattered):  35.1 ms   (51.7 GB/s: the 4-bit advantage lost to bad access patterns)
   after  (q4k_gemv):          10.97 ms  (165 GB/s = 86% of the spec, 88% of the measured 187)
```

Tensor cores can't help: an `mma` tile is at least 16 rows, so M = 1 is 15/16 padding and saves no bytes. Forced at M = 1, `mma` took 49.5 ms per pass and `w4a8` 31.8 ms.

**How the kernel works** (`Understand`).

```
 one warp = one weight row (one output); K in chunks of 1024 = 4 super-blocks = 576 contiguous bytes
   lanes 0-7  -> super-block 0, lanes 8-15 -> sb 1, ...      each lane: one 128-bit load of codes (32 weights)
                                                              + one 128-bit header load (8 lanes share it)
   decode in registers (exact gguf fp32), FMA with x from shared memory (XOR-swizzled), fp32
   the next chunk's bytes are requested before this chunk's math (loads in flight)

 end of row: 32 partial sums -> warp-shuffle reduction
   lane i += lane (i xor 16) -> xor 8 -> xor 4 -> xor 2 -> xor 1      5 steps, no shared memory
   -> every lane holds the total; one lane writes y[n] + bias
```

- **Coalesced 128-bit loads:** a warp's 32 loads cover 512 contiguous bytes, so every Q4_K byte crosses the bus once, in full transactions. Naive gave each thread its own 1,728-byte row, so neighbouring threads read addresses 1,728 bytes apart.
- **Enough bytes in flight:** 3 blocks (24 warps) per SM at M = 1, each with its next chunk already requested. That is what keeps the memory busy.
- **Swizzled x** removed 4-way bank conflicts: M = 1 went 154 → 173 GB/s on one linear.
- From M = 2 each warp takes two weight rows, so each x value read feeds both. At M = 4 (a walk pass) it reaches 114 GB/s: there it runs short of instruction slots, not memory.
- It stays **lossless**: bf16 x and an fp32 sum, so modulation never runs on INT8, even under the w4a8 default.

**The lesson: re-measure before optimizing** (`Must`). The plan expected "0.29 s → ~0". That number was modulation's share of the diffusers **dequant**, which Tasks 3–5 had already removed. By Task 5b modulation really cost 35 ms, so the whole task saved ~24 ms: **1.15 → 1.12 s, about 0.03 s**. The kernel is good (86% of bandwidth); the premise was stale. Entry 1 §1.6 made the same mistake by calling it a "cheap 0.29 s target". A five-minute measurement before planning would have sized it correctly.

**In an interview.** "At M = 1 a linear is a GEMV: one multiply-add per weight byte, far left of the roofline ridge, so time is just bytes over bandwidth. That is the one place where 4-bit weights buy speed, 3.6× over bf16. My GEMV used a warp per weight row with coalesced 128-bit loads and a warp-shuffle reduction, and reached 86% of DRAM bandwidth. I also learned to re-measure first: the plan's 0.29 s target had already been removed by an earlier kernel, and the real gain was 30 ms."

<details><summary>Check yourself (Task 5c)</summary>

**Q1.** Why is 4-bit storage a speedup for the GEMV but not for the big M = 1280 GEMMs?

**A1.** The GEMV has AI ~3.6 FLOP/B, far below the ridge (~102), so its time is bytes ÷ bandwidth, and 3.6× fewer bytes is 3.6× less time. The GEMMs sit far above the ridge, so their time is set by the math rate, and fewer weight bytes don't change it.

**Q2.** The naive kernel read the same 1.815 GB but took 35 ms. Why?

**A2.** Its access pattern: each thread walked its own weight row, so a warp's 32 simultaneous loads hit 32 addresses 1,728 bytes apart. Each became its own memory transaction, and the bus moved far more than the useful bytes per request. Coalescing gives the same bytes in full, contiguous transactions.

</details>

**Tried and not chosen** (`Understand`). The 76 modulation linears per pass at M = 1, 1.815 GB of Q4_K; the floor at the measured 187 GB/s is 9.7 ms:

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| The plan's premise: "M = 1 runs padded tensor-core tiles, ~0.29 s" | Wrong premise. M < 7 already ran naive, and the 0.29 s was dequant that Task 3 removed; the real cost was 35.1 ms | — (a planning error, kept here as a lesson) | — |
| Naive kernel (the previous routing) | 35.1 ms, 51.7 GB/s (M = 4: 38.8 ms) | Scattered reads: neighbouring threads 1,728 bytes apart | — |
| `mma` forced at M = 1 (padded 128-row tiles) | 49.5 ms | 15/16 of each `mma` tile is padding, and it saves no bytes | M ≥ 7 |
| `w4a8` forced at M = 1 | 31.8 ms (with its quantize) | Same padding; INT8 math doesn't help a memory-bound op, and it would be lossy | Never for modulation |
| **Chosen:** `q4k_gemv` | **10.97 ms, 165 GB/s** (M = 4: 15.9 ms) | — | M ≤ 8 (`GEMV_MAX_M`) |
| GEMV without the x swizzle | 154 GB/s at M = 1 (173 with it), 63 GB/s at M = 8 (79 with it), one linear | 4-way shared-memory bank conflicts on x | — |
| One weight row per warp at M ≥ 2 | 98 GB/s at M = 4, 62 at M = 8 (two rows per warp: 118, 79) | Each x value read from shared memory fed only one row; instruction slots ran out | M = 1, where it is used |
| Four weight rows per warp | "no better" than two (no figure recorded) | No gain for more registers | — |

> **Remember this:** at M = 1, bandwidth is the only number that matters, and fewer bytes means less time. But measure the cost first: this one was 35 ms, not 0.29 s.

---

### Task 6: glue fusion

`Must` for the plain version, why fusion saves bytes not FLOPs, Amdahl, profile first, bit-exactness as a gate and the bf16-equivalent caveat. `Understand` for Welford, reduction order and the layout search. `Skip` the launch-configuration numbers and the cat details.

> **Plain version.** Between the big multiplies, every block does small jobs: add the attention output back, normalize each token (LayerNorm), scale and shift it (modulate), normalize and rotate q and k (RMSNorm + RoPE). Each is a few adds per number, but PyTorch runs each as its own kernel, which **reads the whole tensor from VRAM and writes it back**. Task 6 does each chain in **one pass**: read once, keep the values in registers, write once. The maths is unchanged, and the output is **bit-identical** to diffusers. Step **1.12 → 0.93 s**, walk step **47.0 → 39.5 s**. The surprise: the real cost wasn't the norms the plan pointed at (68 ms). It was diffusers' RoPE in fp32 (147 ms).

**Memory-bound elementwise ops: fusion saves round trips, not FLOPs** (`Must`). One [1280 × 3072] bf16 activation is 7.9 MB. An op that does one multiply per value has AI ≈ 0.1–0.5 FLOP/B, far left of the ~102 FLOP/B ridge, so its time is bytes ÷ bandwidth:

```
 diffusers (each box reads the whole tensor from VRAM and writes it back)
   res, y, gate ─[ gate*y ]─[ res + . ]─[ LayerNorm ]─[ *(1+scale) ]─[ +shift ]─> x_mod
                    r/w        r/w          r/w           r/w          r/w        = 5 round trips

 fused residual_norm_modulate (one block of 128 threads per token row)
   res, y, gate, scale, shift ─[ v = res + gate*y ; LN stats ; LN(v)*(1+scale)+shift ; (w4a8: quantize) ]─> res_out, x_mod
                                   read once            values stay in registers            write once
```

Same arithmetic, ~5× fewer bytes. The three kernels run at **167–179 GB/s, 89–96% of the 187 GB/s read roof**: their time is now the bytes they *must* move. `qk_norm_rope` is **14.5×** faster than diffusers' chain (97 µs against 1,403 µs per call).

**Profile first: the second stale premise** (`Must`). The plan said "~0.4 s of glue" (Entry 1's estimate, from before Tasks 3–5c). A profile of one pass, attributing every GPU kernel to the op that launched it, said otherwise:

| Non-GEMM work, one pass (Task 5c code) | Before | After |
|---|---|---|
| **RoPE** (diffusers' `apply_rotary_emb`, fp32: ~6 passes per projection at twice the bytes, 114 times a pass) | **147 ms (13%)** | in `qk_norm_rope` |
| QK RMSNorm | 19 ms | in `qk_norm_rope` |
| Residual + LayerNorm + modulate + gates + cats (the plan's "glue"; a third of it `torch.cat`) | **68 ms** | all fused passes together: **35 ms** |
| Attention (flash SDPA) | 68 ms | 70 ms (not touched) |
| GELU | 21 ms | 22 ms (not fused) |
| **All non-GEMM GPU time** | **325 ms** | **128 ms** |

Same lesson as Task 5c: the plan's number was right once and stale by the time we got there. Fusing only what the plan named would have saved ~35 ms; profiling found the 147 ms.

**Amdahl: why glue matters more now** (`Must`). If a part takes fraction p of the time and you make it k× faster, the whole gets 1 / ((1 − p) + p/k) faster. The flip side: every task that sped up the GEMMs made the untouched glue a **bigger** share.

```
                  GPU time per pass    linears (GEMM-ish)    everything else
 Task 5c          1042 ms              717 ms  69%           325 ms  31%   ████████░░░ <- RoPE alone 13%
 Task 6            843 ms              715 ms  85%           128 ms  15%   ██████████░
```

Fusing ops that were never expensive in FLOPs bought 0.19 s, a 17% faster step, because by Task 5c they were almost a third of the time. Taking the remaining 128 ms to zero would give at most 843 / 715 = **1.18×** more. From here the GEMMs are the lever again.

**Welford, and why reduction order decides the last bits** (`Understand`). LayerNorm needs each row's mean and variance. **Welford's online algorithm** gets both in one pass, with good numerical stability:

```
 for each value x:   n += 1;  d = x - mean;  mean += d / n;  M2 += d * (x - mean)      variance = M2 / n
 combine two partial results a, b (how threads, then warps, merge their partials):
   n = na + nb;  d = mean_b - mean_a;  mean = mean_a + d*nb/n;  M2 = M2a + M2b + d*d*na*nb/n
```

**Floating-point addition isn't associative:** (a + b) + c can differ from a + (b + c) in the last bit, because each add rounds. In fp32, (10⁸ + 1) − 10⁸ = 0, but (10⁸ − 10⁸) + 1 = 1. So the *order* of the merges (which values each thread holds, the shuffle pattern, the tree over warps) decides the last bit of mean and rstd. Occasionally that flips a bf16 rounding of the output. To be bit-identical to torch, the kernel must merge in **torch's order**.

**The layout search** (`Understand`). torch's order isn't documented, so a probe ran Welford with every combination of load width (2, 4 or 8 values) and warps per row (1–16) against `torch.nn.functional.layer_norm` on 12.6 M outputs. **Only one of the 15 layouts matched: 4-value loads × 4 warps (128 threads), 0 mismatches.** The others mismatched on 104–214 outputs. That fixed the kernel's launch shape.

**Bit-exactness as a gate strategy** (`Must`). A lossless change should change *nothing*, so test it that way: `torch.equal` on x_mod, on q/k, on whole blocks and on the 512 picture. Then the walk gate reads **Δ = 0 on all six spots** (byte-identical PNGs), with no argument about noise. This matters most on the lossy W4A8 default. INT8 rounding amplifies one flipped bf16 bit into a different sample (as Task 5c's one-ulp modulation change did: 25.78 dB), and a different sample moves seam scores by more than the gate's ±0.03. "Almost identical" isn't good enough upstream of a lossy stage; "identical" is.

**The fused quantize (kept)** (`Understand`). Under W4A8, the norm's output feeds INT8 linears, each of which used to re-read x_mod to quantize it. The norm already holds x_mod in registers, so it emits the same s / xq / xsum (bit-identical to `quantize_act`) alongside it. Net **−10 ms** per pass; `quantize_act` fell 25.6 → 13.2 ms (228 → 38 launches).

**"99.4% of bf16 peak" isn't a contradiction** (`Must`). 19.00 TFLOP/s is the nominal 17.67 TFLOP ÷ 0.93 s, quoted against the bf16 roof. But 75% of the token FLOPs run on the INT8 tensor cores (Task 5b), which are faster than bf16, so this **bf16-equivalent** rate can pass 100%. It doesn't mean the GPU is at its limit: the pass still has ~80 ms where the GPU waits (0.925 s wall against ~845 ms of kernels), which is Task 7's target (CUDA graphs).

**Peak memory barely moved** (−1 to −2%): the peak is the streamed weights plus the largest GEMM's activations (the 12,288-wide MLP), not the norm intermediates. So fusion doesn't buy batch headroom, contrary to Entry 1's Technique 6 hope.

**In an interview.** "I fused FLUX's elementwise glue into three memory-bound kernels at 89–96% of DRAM bandwidth. Fusion can't cut FLOPs, only round trips. Profiling first showed the real cost was an fp32 RoPE chain at 13% of the step, not the norms the plan named. I made the kernels bit-identical to PyTorch, including matching its Welford reduction order, because downstream INT8 quantization turns a single last-bit difference into a visibly different image. That made the quality gate trivial: byte-identical output."

**Tried and not chosen** (`Understand`):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **Two-pass LayerNorm** (mean, then variance around it, fp32 in registers: the textbook way; the first version) | Same speed (94.6 against 96.4 µs). 2.8e-6 (~3e-6) of x_mod values off by one last bit; blocks max-rel 9.0e-3; picture 30.36 dB under mma but **25.17 dB under w4a8** (a different sample); walk spots 2, 4, 5 rose +0.04 / +0.08 / +0.21: **3 spots failed** the +0.03 gate | Not less accurate, just *different* from torch's bits, and W4A8 amplified the difference | Any pipeline without a lossy stage downstream, or where you own the reference (no need to match torch) |
| The other 14 Welford layouts (2/4/8-value loads × 1–16 warps, except 4 × 4) | 104–214 mismatching outputs of 12.6 M | Don't reproduce torch's reduction order | Same as two-pass: if matching torch didn't matter |
| Launch configs: rnm 64 or 256 threads, qk 4 or 16 warps (against 128 / 8 kept) | Within noise, e.g. 97.5 / **94.6** / 94.1 µs (M = 1024 + quantize); qk 96.7 / **96.6** / 96.0 | All at 89–96% of the memory roof, so the shape barely matters; 128 threads is also what torch's order needs | — |
| (a) GELU into the GEMM epilogue | GELU costs 22 ms (2.4% of the fused step) | Needs epilogues in both Q4_K GEMMs (mma and w4a8): GEMM work, not glue work | A later GEMM task; it also unlocks (b) |
| (b) Quantize `proj_out`'s input in its producers | 13.2 ms (1.4%) | Its row is [attention ‖ GELU(mlp)] from two producers; a per-token scale needs the whole row's max first | After (a) |
| (c) A block's last residual fused into the next block's first norm | `gated_residual` ~7.5 ms per pass; at most ~3–4 ms saved | Crosses diffusers' block loop and the group-offload hooks | When we own the whole block loop (no diffusers loop) |
| (d) Attention output written straight into `proj_out`'s buffer | 3.6 ms of copies | Flash SDPA allocates its own output; needs a custom attention call | With our own attention kernel |
| (e) The last cats (double-block text+image v, first single-block join) | 1.8 ms | Too small to justify | — |
| (f) RoPE folded into the q/k GEMM epilogue | `qk_norm_rope` already 97 µs per call (~11 ms per pass) at 176 GB/s | RMSNorm needs a whole head's sum of squares before RoPE; little left to save | A GEMM whose N-tile spans a whole head (128) |
| Attention, modulation GEMV, SiLU, embedders | 70 / 11.6 / 0.2 / 1.6 ms | Not glue (attention is already flash) or already at memory speed (Task 5c) | — |

One cost seen, not chased: with glue fused, the GEMMs ran ~1% slower (~10 ms of the 190 ms won). The likely cause is the denser schedule on a power- and heat-limited laptop GPU.

<details><summary>Check yourself (Task 6)</summary>

**Q1.** Fusing five elementwise ops doesn't remove a single FLOP. Why does it make them ~2.6× faster (14.5× for RoPE)?

**A1.** They are memory-bound (~1 FLOP per value, far left of the ~102 FLOP/B ridge), so their time is bytes ÷ bandwidth. Unfused, each op reads and writes the whole tensor; fused, the chain reads its inputs once and writes its outputs once. RoPE gained most because diffusers ran it as ~6 passes in fp32, twice the bytes per value.

**Q2.** The two-pass LayerNorm flipped only ~3e-6 of values by one last bit and was no less accurate. Why did it fail the gate under w4a8 but not under mma?

**A2.** Under mma the flip stays a last-bit difference (30.36 dB, a near-identical picture). Under w4a8 the next linears quantize x per token to INT8, and a last-bit change can flip an INT8 rounding. Four denoising steps grow that into a different sample (25.17 dB), and a different sample moves seam scores past the ±0.03 gate. Upstream of a lossy stage, only bit-identical is safe.

</details>

> **Remember this:** glue is memory-bound, so fuse to cut bytes, not FLOPs. Profile before you fuse (the cost was RoPE, not the norms), and make lossless changes bit-identical so the gate can prove Δ = 0.

---

### Task 7: fixed weight slots and CUDA graphs

`Must` for the plain version, profile first, why graphs tied, ring 3 vs 2, and why graphs are opt-in. `Understand` for graph capture constraints, the `_version` trap, output aliasing and the bench timing change. `Skip` the LRU and memory-pool details.

> **Plain version.** The plan was to remove *kernel launch overhead* with CUDA graphs: record each block's ~20 kernels once, replay them with one call. Profiling first showed launches weren't the problem. Gaps that look like launch overhead were only **0.34%** of a pass. The GPU was really idle **10.9%** of the time, waiting on **group offload's syncs**: diffusers re-allocated each block's weights on every load and synchronised around it. Graphs need fixed weight addresses anyway, so Task 7 built **three fixed-address weight "slots"** that blocks stream through. That alone removed the waits: **0.92 → 0.85 s** per pass. Graphs on top measured the **same** as plain eager (0.854 against 0.850 s), and they cost first-use time and memory. So the default is **eager on the slots**, with graphs opt-in (`VECWORLD_GRAPHS=1`). Every path is bit-identical; the gate walk is byte-identical (Δ = 0).

**Profile first: the third stale premise** (`Must`). Nsight Systems over three back-to-back passes (Task 6 build):

| Per pass (Task 6, group offload) | ms | Share |
|---|---|---|
| GPU span of a pass | 940 | |
| Kernels running | 838 | |
| **Idle** | **102** | **10.9%** |
| ...a ~0.87 ms sync after every one of the 57 blocks | ~50 | 5.3% |
| ...pass end (blocking offload of the non-block layers) + pass start (onload, nothing to overlap) | ~39 + ~28 | 7.1% (overlaps the row above) |
| ...gaps under 100 µs, **what launch overhead looks like** | **3.2** | **0.34%** |

1,220 kernels a pass, median gap 0.64 µs: the CPU was already far ahead of the GPU. Graphs fix launch overhead, which was a third of a percent. The plan's target was real in Entry 1's eager world (≈12k launches, est.) but stale after Tasks 3–6 cut the kernel count.

**What the slots changed** (`Must`):

```
 Task 6:  |##block##|..|##block##|..|##block##| ... |####|.....39 ms.....|...28 ms...|####
                     ^ ~0.87 ms sync per block           pass end           next pass start
 Task 7:  |##block##|##block##|##block##| ... |####|##block##|...       (no gap over 0.3 ms)

 main stream   [x_emb][blk 0 ][blk 1 ][blk 2 ][blk 3 ] ... [blk 56][out]
 side stream     [copy 2][copy 3 ][copy 4 ][copy 5 ] ... [copy 0][copy 1]   (next pass's first blocks)
 events        copy i done ──> blk i may start;   blk i done ──> slot i % 3 may take block i + 3
```

Each block's weights land in slot i % 3, a buffer at a **fixed address** sized to the largest block (191 MB). The copy for block i + 3 starts as soon as block i frees its slot, on a side stream ordered by CUDA events, so copies overlap compute (13.4 GB/s pinned: 14 ms for a double block). Idle per pass: **102 ms → 0.76 ms (0.09%)**.

**Why graphs then tied with eager** (`Must`). Once the syncs were gone, the CPU queued kernels a whole pass ahead of the GPU, so the queue never emptied and there was no idle time left for graphs to remove. A graph saves CPU time per launch; it only shows up as GPU time when the GPU is waiting for the CPU. Alternating A/B at batch 4 (5 pairs): graphs +0.8%, inside the order effect (whichever runs first in a pair is faster, from heat). nsys confirmed the same kernels, with the compute-bound ones ~1.6% slower in the later run: clocks, not launches.

**Ring 3, not 2: an odd number of blocks** (`Must`). With 2 slots, block i uses slot i % 2. **57 is odd**, so the last block (56) and the next pass's first block (0) both use slot 0. Block 0's 14 ms copy can't start until block 56 finishes, so every pass started with a **13.5 ms stall** (seen in nsys). With 3 slots, 3 divides 57, so block 0's copy runs during blocks 55–56. Measured, alternating 3 × 8 passes: **0.869 → 0.859 s**, idle 9.6 → 0.76 ms per pass, for 191 MB more memory. Lesson: ring buffers wrap, so check the wrap boundary as well as the steady state.

**How a graph works, and its constraints** (`Understand`). A CUDA graph records a sequence of launches (*capture*) and replays them with one call. Every kernel's grid and **every device address it touches are frozen at capture**. Hence:
1. **Static shapes:** one graph per (block, batch, image tokens, text tokens), with an LRU of 4 keys.
2. **Static weights:** group offload put each load at a new address, so a graph over it would read freed memory. The slots are a precondition.
3. **Static inputs and outputs:** new inputs are *copied into* the buffers the graph reads, and outputs appear in buffers it wrote.
4. **No CPU in the middle:** one graph per block, with the CPU between blocks queueing weight copies (Task 10 puts SSD reads there). A whole-pass graph would need all 6.7 GB resident on a 4 GB card.

**The inference-tensor `_version` trap (review I1)** (`Understand`). To skip re-copying an input that hasn't changed, the first version checked "same tensor object, and same `_version`". `_version` is PyTorch's per-tensor counter that bumps on every in-place write. But the painter runs under `torch.inference_mode`, and **inference tensors have no version counter**: `_version` is unavailable, so the check fell back to "same object". The reviewer reproduced the bug: a caller that writes the same buffer in place between passes (a preallocated timestep or rotary buffer, the natural graph-friendly design) got a replay on the **old values**. It looked plausible, was wrong, and raised no error. Fix: block 0 always copies; the skip holds only within one pass (blocks 1–56 seeing the very same `temb`/rotary objects). The same guard protects Task 6's attached INT8 quantization: `quantized_of` refuses it if `x._version` moved, and never trusts it on an inference tensor. **Identity doesn't prove "unchanged"; only a version counter can, and inference mode removes it.**

**Static output buffers alias** (`Understand`). With graphs, each key owns two "ping-pong" output buffers. Block i writes into buffer (i + 1) % 2, which is exactly block i + 1's static input, so nothing is copied between blocks. The flip side: a block **returns the same tensor objects every call**, overwritten by block i + 2 and by the next call of the same block. The transformer consumes each output in time, but anything that *keeps* a block output (a residual cache for block caching, a debug hook) must `clone()` it. Writing straight into the buffer (`gated_residual(..., out=)`) beat a generic copy at the end by 5 ms a pass.

**Why graphs are opt-in: the cost table** (`Must`). Ruling M5, after review:

| | Eager on slots (default) | Slots + graphs |
|---|---|---|
| Steady tile512 pass | 0.850 s | 0.854 s (tie) |
| Batch-4 pass, 5 alternating pairs | 3.538 s | 3.565 s (+0.8%, noise) |
| First pass of a new shape, batch 1 / 2 / 4 | 0.95 / 1.70 / 3.36 s | **1.98 / 3.70 / 7.60 s** (warm-up + capture + replay) |
| First panorama of a walk (spot 0) | 36.7 s | **40.6 s (+3.9 s** on the gate walks; the ruling cited +4.7 s from an earlier run) |
| Peak allocated, walk step | 1.77 GB | 1.92 GB (+0.2–0.6 GB across batch sizes, which Task 9's T5 needs) |
| GIL stress (2 spinning Python threads), batch-1 pass min / median / max | 19.6 / 28.9 / 64.9 s | 7.5 / 23.0 / 74.5 s |
| Other | — | a device sync + `empty_cache` per capture; an LRU eviction silently recaptures 57 graphs |

The speed-first goal gets nothing for those costs. Graphs do help when the CPU is busy (the GIL stress median), but both modes collapse there because the per-block sequencing is still Python. **Revisit after Task 10** batches that CPU work.

**The bench timing change** (`Understand`). The bench used to time a pass with a CPU clock at each `progress()` call. With no syncs, the CPU runs a pass ahead, so that clock measured *enqueueing*: the first graph pass read **0.01 s**. Each pass is now timed with **CUDA events on the compute stream** (recorded in the stream, so they measure GPU execution). A device sync per pass was rejected too: it also waits for the next pass's weight prefetch (~14 ms). **Asynchronous GPUs need GPU-side clocks.**

**Results.** Step **0.85 s** (same-session Task 6: 0.92), **20.79** bf16-equivalent TFLOP/s; walk step **38.3 s** (Task 6: 39.0; at 4 windows a pass the old waits were small next to 3.6 s of compute). Peak allocated rose (walk 1.30 → 1.77 GB eager: three 191 MB slots against group offload's ~two blocks), well under 3.6 GB. Per-pass spread didn't narrow (6–9 ms): on this laptop it is set by heat and clocks, not launch jitter.

**In an interview.** "I set out to add CUDA graphs, but an Nsight profile showed launch-sized gaps were 0.34% of the pass. The real 10.9% idle was the offload library's syncs. Replacing it with three fixed-address weight slots, event-ordered copies on a side stream and a ring of 3 because 57 blocks is odd, cut idle to 0.09% and the pass by 8%. With the GPU saturated, graphs tied with eager but cost capture latency and memory, so I shipped them opt-in. Review also caught a stale-replay bug: inference-mode tensors have no version counter, so object identity can't prove an input is unchanged."

**Tried and not chosen** (`Understand`):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **Graphs on by default** (the first version of this task) | Tie: 0.854 vs 0.850 s (tile512), 3.565 vs 3.538 s (batch 4, +0.8%, noise); first panorama +3.9 s; batch-4 capture pass 7.6 vs 3.4 s; walk peak 1.92 vs 1.77 GB | No speed gain once the GPU is saturated; costs first-use latency and memory | A CPU-bound host (the GIL stress median 23.0 vs 28.9 s), small kernels, or after Task 10 batches the per-block Python |
| **Graphs alone, keeping group offload** | Not buildable: group offload re-allocates weights at new addresses on every load | A graph over moving addresses reads freed memory | Never with that offloader; the slots were the precondition, and the win |
| **Ring of 2 slots** (the brief's default) | 0.867–0.871 vs 0.858–0.861 s (3 alternating rounds × 8 passes); a 13.5 ms stall at every pass start; idle 9.6 vs 0.76 ms | 57 blocks is odd: block 0 shares block 56's slot, so its copy can't overlap | An even block count, or memory too tight for a third 191 MB slot (still the class default) |
| **Copy outputs into the next block's input** (generic forward, no `out=`) | 0.863–0.866 vs 0.855–0.861 s | +5 ms a pass: 57 extra 8 MB copies | Unfused forwards, where it stays as the fallback |
| **A memory pool per graph key** | 2.84 vs 2.10 GB reserved with 4 keys (same 1.505 GB allocated) | +0.74 GB for nothing: nothing outlives a capture, so all graphs can share one pool | — |
| **Skip the static-input copy for the same object across passes** (first version) | Saves under 1 ms of 850 per pass; the reviewer reproduced a stale replay | Inference tensors have no `_version`, so identity can't prove "unchanged" | Outside inference mode, where `_version` works (still only a < 1 ms win) |
| **A whole-pass graph, or one graph per several blocks** | Estimated: 6.7 GB of weights against a 4 GB card; per-block idle already 0.09% | Needs all weights resident or many slots; no idle left; Task 10 needs the CPU between blocks for SSD reads | A model whose weights fit in VRAM |
| **Capture on the second use** instead of the first | First pass eager 0.87–0.92 s against a capture pass of 1.92–1.96 s: the same total, spread over two passes | Capturing on first use makes every later pass a replay | When the first pass's latency matters most and shapes rarely repeat |
| **Timing passes with a CPU clock** (the old bench) | First graph pass read 0.01 s, then 0.835–0.866 s | Measured enqueueing: the CPU runs a pass ahead | Never for async GPU work; use CUDA events |
| **Timing with a device sync per pass** | Reads ~10–14 ms high: it also waits for the next pass's prefetch | Measures more than the pass | When nothing else is in flight |
| **Shared ping-pong buffers across keys** (considered, not built) | Estimated ~236 MB saved (of ~393 MB static buffers for batches 1–4) | Memory is within budget (1.9 GB peak); a larger key would force recapturing every key | If Task 9/10 runs short of VRAM |

<details><summary>Check yourself (Task 7)</summary>

**Q1.** CUDA graphs removed almost all CPU launch work but didn't make the pass faster. Why not, and when would they?

**A1.** Launch cost only shows up as GPU time when the GPU waits for the CPU. After the slots removed the syncs, the CPU was a whole pass ahead and the GPU never waited (0.09% idle), so there was nothing to recover. Graphs help when the CPU is the bottleneck: tiny kernels, a busy or contended CPU (the GIL stress case), or host syncs in the loop.

**Q2.** Why did a ring of 2 slots stall 13.5 ms at every pass start, and why does 3 fix it?

**A2.** Block i uses slot i % ring. With 57 (odd) blocks and ring 2, block 56 and the next pass's block 0 both use slot 0, so block 0's 14 ms copy must wait for block 56 to finish. With ring 3 (which divides 57), block 0 uses the slot block 54 freed, so its copy overlaps blocks 55–56.

</details>

> **Remember this:** profile before you optimize: the "launch overhead" was 0.34%, and the real 10.9% was syncs. Fix the waits (fixed slots), and only add machinery (graphs) when it measurably pays.

---

### Task 8: first-block cache (built, measured, shipped off)

`Must` for the plain version, why a 4-step model has little to skip, the vacuous pass, and "speed is never bought with an untested quality loss". `Understand` for residual reuse, the probe, the chained-walk PSNR effect and the multi-scene gate. `Skip` the slot hold/release bookkeeping and the cache-key details.

> **Plain version.** Each FLUX block *adds* something to the running picture. If block 0's contribution barely changed since the last denoising step, the other 56 blocks' total contribution probably barely changed either, so reuse it and skip them. That's a first-block cache (TeaCache / First-Block-Cache). On FLUX-schnell it **does** buy speed: 1.32× on a tile, 1.32× on a panorama, 1.63× on a walk step at threshold 0.3. But schnell takes only 4 big steps, so a skip reuses a contribution from a quarter of the way back. On the new three-scene walk gate, every threshold that skipped anything **made seams worse** (up to +0.166). The only threshold that "passed" (0.2) **never skipped once**. So the cache **ships off**: we don't buy speed with a quality loss the gate hasn't approved. The code stays, tested, behind one config value.

**Residual reuse** (`Understand`). A block computes `h_{i+1} = h_i + f_i(h_i)`, so blocks 1–56 add up to a residual `R = h_56 − h_0`. If `h_0` barely moved since the last step, reuse the stored R:

```
 computed pass:  x_emb ─[blk 0]─probe─[blk 1]─ ... ─[blk 56]─> output        store R = h56 − h0
 skipped pass:   x_emb ─[blk 0]─probe─ h0 + R(stored) ─────────> output        56 blocks AND their weight copies skipped

 probe:  rel(t) = |r0(t) − r0(t−1)|₁ / |r0(t−1)|₁      r0 = block 0's own residual
         acc + rel < threshold → skip (acc += rel)      else compute (acc = 0, store new R)      step 0 never skips
```

A skipped pass costs block 0, the embedders, the output layers and one host sync: **21 ms** at batch 1 against 0.86 s. Skipping also skips the weight copies: `BlockSlots.hold` stops block 0 from prefetching block 3 into its slot, so a skip copies **0** bytes and the next full pass starts with blocks 0–2 already in place.

**Why a 4-step distilled model has little to skip** (`Must`). The probe's measured `rel`, step to step:

| | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| Panorama (sigmas 1, 0.75, 0.5, 0.25) | 0.83 | **0.23** | 0.29 | |
| Walk step (`sharp`, 5 sigmas) | 0.62 | **0.19** | 0.23 | 0.29 |

Thresholds published for 25–50-step models are **0.05–0.15**, and every step here is above that. Schnell is distilled to jump a quarter of the way from noise to image each step, so block 0's residual moves by **20–80%** between steps. Only the step right after the big first jump (0.19–0.23) is ever a candidate. Skipping it means reusing a residual from a quarter of the trajectory earlier. This is Entry 1 Technique 5's prediction (1.0–1.33× for a 4-step model), and the tile and panorama numbers (1.32×) land at its top.

**The sweep** (`Must`). Three scenes × 6 spots, against cache-off base walks. Pass = no spot's seam more than 0.03 above its base.

| Threshold | Skips (panorama / per walk step) | Worst seam rise | Min PSNR vs cache-off | Gate |
|---|---|---|---|---|
| 0.3 | 2 of 8 / 2–4 of 10 | **+0.150 / +0.131 / +0.153** (forest / desert / coast) | 21.70 dB | **fail**, all 3 scenes |
| 0.25 (added between) | 0–2 / 2 | **+0.166 / +0.074 / +0.082** | 22.05 dB | **fail**, all 3 scenes |
| 0.2 | **0 / 0 in all 18 jobs** | 0 (byte-identical) | inf | **"pass", vacuously** |
| 0.15, 0.1, 0.05 | pruned: every probe on these walks was ≥ 0.2, so they skip nothing either | — | — | not walked |

**A vacuous pass isn't a pass** (`Must`). 0.2 passed only because it never did anything on the gate's walks. Yet on the bench prompt a walk step probes **0.19** and *would* skip, a lossy path the gate never saw. Shipping 0.2 would buy no measured speed, add a ~1.5% probe cost per pass, and leave an ungated skip in the game. Ruling: **threshold 0.0** (off), over the brief's "largest passing threshold". The review asked the gate to flag this case in future (M4: a lossy setting that changes nothing should read "pass, nothing exercised").

**What the cache would buy at 0.3** (`Must`). Same session, measured. Labelled because it's **not shipped**:

| | Off | Threshold 0.3 | Speedup |
|---|---|---|---|
| tile512, s per pass (mean of 4; one pass skipped at **0.021 s**) | 0.859 | 0.650 | **1.32×** |
| Panorama: s per pass (mean of 8) / wall | 3.49 / 29.7 s | 2.65 / 22.7 s | **1.32× / 1.31×** |
| Walk step: s per pass (mean of 10) / wall | 3.78 / 39.2 s | 2.32 / 24.6 s | **1.63× / 1.59×** |
| Nominal TFLOP/s (tile512) | 20.55 | 27.19 (above the bf16 roof because work is *skipped*, not done faster) | |

That is a large win, and it still ships off. **Speed is never bought with a quality loss the gate hasn't approved.** The ruling's cost-if-wrong: 1.3–1.6× left on the table until a smarter probe or a panorama-only cache passes a gate.

**Why PSNR diverges on a chained walk after one skip** (`Understand`). Each walk step starts from the previous spot's picture. One skip changes spot 0 slightly (30.6–37.1 dB against cache-off: a softer, hazier panorama, same composition). Spot 1 then starts from a *different picture*, and four denoising steps grow the difference into different details: **22–29 dB from spot 1 on**. The forest's spot 1 loses the house on the right. So per-spot PSNR along a chained walk measures **trajectory divergence**, not quality: a single skip anywhere upstream drags every later spot below 30 dB. That's why the ruling made the **seam** score, a quality metric, the deciding measure, with PSNR reported but not decisive. Seams moved both ways (some −0.25, some +0.17), and the gate fails on the worsening.

**The new multi-scene relative gate** (`Understand`). Task 6 had already shown one scene is too noisy for a fixed limit (spot 1 sits at 1.15 > 1.1 even lossless). From Task 8 on, the gate is:
- three deterministic scenes ("haunted forest" seed 7, "red desert" 11, "stormy coast" 23), 6 spots each;
- base walks with the change off, at the task's BASE;
- pass = **no spot's seam more than 0.03 above the same spot of its base**, in every scene; per-spot PSNR is reported;
- `walk_sheet.py --scenes ... --relative BASE CAND`, reused as the gate for Tasks 9 and 10.

The review's follow-ups for it: check that base and candidate are the same scene and route (M3), and flag vacuous passes (M4).

**An untested future option: a panorama-only cache** (`Understand`). The panorama alone did look acceptable at 0.3: spot 0 scored 30.6 / 37.1 / 32.3 dB against cache-off, inside the 30 dB floor, for 1.31× on the first panorama the player waits for (29.7 → 22.7 s wall). It was **never gated on its own**. In the 0.3 walks the walk steps skipped too, and a changed panorama still seeds every later spot. Turning it on would need its own gate: cache the panorama only, and check that the walk that follows doesn't regress on seams in all three scenes.

**Results.** Default off, so the step is unchanged: **0.86 s**, 20.55 TFLOP/s in Task 8's session (Task 7's own session: 0.85 / 20.79). The probe alone, when on, costs about +1.5% per pass.

**In an interview.** "I built a first-block residual cache for FLUX-schnell. It measured 1.32× on a panorama and 1.63× on a walk step, and I shipped it off. Schnell takes four big steps, so block 0's residual moves 20–80% per step, far above the thresholds used for 50-step models. Every threshold that actually skipped worsened seams on a three-scene end-to-end gate, and the only passing threshold never skipped, which is a vacuous pass, not evidence. On a chained walk PSNR measures trajectory divergence, so I gated on a seam metric instead."

**Tried and not chosen** (`Understand`):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **Threshold 0.3** (largest in the brief's sweep) | 1.32× tile/panorama, 1.63× walk-step passes; worst seam +0.153; min PSNR 21.70 dB | Fails the seam gate in all 3 scenes | A model with many small steps (25–50), where adjacent steps barely differ |
| **Threshold 0.25** (added) | 0–2 panorama skips, 2 per walk step; worst +0.166 | One skip per batch per step is enough to change a chained walk | Same as 0.3 |
| **Threshold 0.2** | 0 skips in 18 jobs; byte-identical walks; would skip at 0.19 on the bench prompt | Vacuous pass: no measured gain, an ungated lossy path, +~1.5% probe cost | Never as a default; a gate that exercises it could change that |
| **0.15 / 0.1 / 0.05** | Not walked: every probe on these walks was ≥ 0.2, so they skip nothing | Provably identical to 0.2 (and to off) on the gate | Many-step models (where these are the usual values) |
| **A panorama-only cache** | Spot 0 at 0.3: 30.6–37.1 dB, 1.31× panorama wall; never gated alone | Untested: a changed panorama seeds the whole walk | If a gate shows the following walk doesn't regress; it speeds up the wait players notice most |
| **Cache decision inside `engine/graphs.py`** (the brief's place) | Graphs are off by default since Task 7 | It would only work on the opt-in path; the shared pass loop serves both (tested on both) | — |
| **Skipping the compute but keeping the slot prefetch** | A skip would still copy block 3 (14 ms), and the next pass would find the wrong block in slot 0 (another 14 ms stall) | `hold` / `release` makes a skip copy nothing (tested) | — |
| **Probe against the last *computed* step** (ParaAttention FBCache) instead of step to step | Not measured separately | The step-to-step sum is the stricter of the two (it bounds the change since the last compute); the gate fails at the first skip either way | A many-step model, where the looser probe skips more |
| **Per-window cache keys** | Not built | Nothing passes; exact (job, window-tuple) keys are never wrong, only miss more | Task 9 batches windows differently; needed if the cache is ever turned on |
| **Cache on the reference path (`VECWORLD_KERNEL=off`)** | — | Group offload's prefetch order assumes every block runs; skipping would desynchronise it | Never on that path |

<details><summary>Check yourself (Task 8)</summary>

**Q1.** Published first-block caches skip 30–50% of steps at thresholds of 0.05–0.15. Why does FLUX-schnell skip nothing below 0.2?

**A1.** Schnell is distilled to 4 steps, each moving a quarter of the way from noise to image, so block 0's residual changes by 20–80% between steps (measured rel 0.19–0.83). Thresholds tuned for 25–50 small steps sit below every step here. Only the step right after the first big jump comes close.

**Q2.** Threshold 0.2 passed the gate with zero seam change. Why wasn't it shipped?

**A2.** It passed because it never skipped, on any of the 18 gate jobs, so the gate tested nothing lossy. Yet it would skip on other prompts (the bench prompt probes 0.19). Shipping it means an ungated lossy path, no measured speedup and a probe cost on every pass.

</details>

> **Remember this:** a big speedup with an unapproved quality loss isn't a win, and a "pass" that exercised nothing isn't a pass. On a 4-step distilled model, adjacent steps are too different to reuse.

---

### Task 9: window batching

`Must` for the plain version, why batching ties when compute-bound, the cuBLAS-by-M rounding and `rowwise`, and why the reference is pinned to batch 1. `Understand` for the gate story, the gate hardening, activation memory and stop latency. `Skip` the `_Rows` wrapper internals and the graph-pool OOM details.

> **Plain version.** A panorama is 8 windows. Until now the painter put 4 through each transformer pass; now all 8 go together, so a panorama takes 4 passes instead of 8 and a walk step 5 instead of 10. **Speed didn't change:** a window costs ~0.9 s per step in a batch of 1 or of 8, because the engine is compute-bound. Batching only spreads costs paid *once per pass*: weight copies, the per-block Python and the modulation weight read. Those are small today, but large once weights stream from the SSD (Task 10). The surprise: batching wasn't lossless. One cuBLAS layer rounds differently depending on the batch size, and W4A8 turned that one-bit difference into a different picture (25.7 dB). A small fix, `rowwise`, makes every window **bit-identical to painting it alone**, for +0.16%.

**When you're compute-bound, batching is a speed tie** (`Must`). Measured sweep, one process per cell:

| Batch | Panorama s (passes) | s per window-step | Walk step s (passes) | Peak GB (one pass) |
|---|---|---|---|---|
| 1 | 30.3 (32) | 0.891 | 40.3 (40) | 1.225 |
| 2 | 30.6 (16) | 0.907 | 39.9 (20) | 1.342 |
| 4 (before) | 30.3 (8) | 0.897 | 39.4 (10) | 1.576 |
| 6 | 30.4 (8 = 6 + 2) | 0.902 | 39.5 (10) | 1.810 |
| **8 (chosen)** | **30.2 (4)** | **0.896** | **39.3 (5)** | **2.045** |

Each window's 1,280 tokens meet every weight anyway (17.67 TFLOP a window), so more windows per pass is more work in proportion. What batching saves is what's paid once per pass:

```
 batch 4:  [copy blk i+1 | GEMMs for 4 x 1280 rows]  x 57 blocks  x 2 passes a step
 batch 8:  [copy blk i+1 | GEMMs for 8 x 1280 rows]  x 57 blocks  x 1 pass  a step
           per pass: 6.66 GB of weight copies, 57 Python block calls, a 1.8 GB modulation weight read
```

Today those are hidden or small: a block's copy (~14 ms) hides under its compute (~15 ms per window). But the **per-block Python iterations halve** (panorama 456 → 228, walk step 570 → 285), and that's what Task 10 starts from. Once weights stream from the SSD at 2.0–2.96 GB/s, a pass costs 2.3–3.3 s of reads: that's **2.3–3.3 s per window-step at batch 1**, and 0.3–0.4 s at batch 8, under the ~0.9 s of compute. (This is Entry 1 Technique 3's point, now with a measured floor.) Batching also overlaps with Task 5: the reference's per-pass dequant was what batch 4 used to amortize (2.74 → 1.76 s per window), and the fused kernel had already removed it.

**cuBLAS picks its algorithm by M, so the batch size changed the picture** (`Must`). The brief's check (batched panorama within 35 dB of batch 1) **failed: 25.7 dB**, visibly another sample. Bisecting with forward hooks:
- the engine's own kernels (all token-facing GEMMs, the GEMV, fused glue, attention) give a row the **same bits at any M**: all 57 blocks were bit-identical at batch 1, 2 and 8;
- but `timestep_embedder.linear_1` and `text_embedder.linear_2` are diffusers' bf16 `nn.Linear`s, i.e. **cuBLAS, which picks its GEMM algorithm (tile shape, split-K, summation order) by problem size**. M here is the batch. At M = 2 against M = 1 they rounded **one bf16 ulp** (~2e-3) apart. `proj_out` (M = 1024 × batch) did the same.

That one ulp in the conditioning reaches every block's modulation. Under W4A8 it flips per-token INT8 roundings, and four Euler steps later the panorama is another sample. It's the same amplifier as Task 5c's one-ulp modulation change (25.78 dB) and Task 6's two-pass LayerNorm (25.17 dB).

**`rowwise`: per-window bit-exactness for +0.16%** (`Must`). `rowwise(transformer)` runs the five layers outside the blocks (`x_embedder`, `context_embedder`, `time_text_embed`, `norm_out.linear`, `proj_out`) **one window at a time**, splitting every argument whose first dimension is the batch, then concatenating. They're a few MFLOP a window, so a same-session interleaved A/B at batch 8 measured **0.9543 vs 0.9528 s per window-step (+0.16%)**, inside the run-to-run spread. Now every window of a batch of 8, 5 or 2 is `torch.equal` to itself alone (under w4a8 and mma, eager and graphs), and the batch-8 panorama matches batch 1 at **PSNR inf**. Ruling (b): `FLUX_BATCH` no longer changes pictures under the engine kernels, so it's purely a speed/latency knob.

**Why the old base "failed" the gate, and what the reference became** (`Understand`). Against Task 8's walks (batch 4, no `rowwise`), batch 8 + `rowwise` **failed 2 of 18 spots** (coast spot 5 +0.166, desert spot 4 +0.045) and improved 11. Even spot 0 differed (32–37 dB). The reason was that the *base* carried cuBLAS's M = 4 rounding in its conditioning; the candidate carries none. Against a base re-walked at batch 1: **18 of 18 spots byte-identical**. Ruling (a): the batch-invariant (batch-1) picture is the reference, and the old batch-4 pictures were one arbitrary rounding among several, the same class as Task 5c's. Later tasks gate against the Task 9 walks. Lesson: when a "lossless" change fails a gate, check whether the *base* depended on something arbitrary.

**The reference path itself had to be pinned to batch 1 (review I1)** (`Must`). `rowwise` can only fix layers outside the blocks. On the diffusers reference (`VECWORLD_KERNEL=off`), the blocks' own GEMMs are cuBLAS bf16 after a GGUF dequant, so they round by M as well. A reviewer probe measured a window in a batch of 2 or 8 at 2.8e-3 to 7.6e-3 relative from itself alone, never bit-equal. So `engine_batch()` returns **1 under `off`**, whatever `FLUX_BATCH` says. The reference always paints the one-window-a-pass picture, which every engine batch now matches. A canonical reference is worth more than reference speed, which nothing ships on.

**Gate hardening** (`Understand`). The Task 8 review's follow-ups, fixed first in their own commit:
- **Same-route check:** `--relative` refuses a base and candidate whose `timings.json` differ in theme, seed, headings, recipe or walked distance, or lack any of them. A forest-vs-desert comparison is refused instead of "measured".
- **VACUOUS verdict (exit 2):** a lossy setting under test (cache on, or `--expect-change`) that skipped nothing or left every spot byte-identical reads VACUOUS, not PASS. That was Task 8's 0.2 threshold.
- **No walks from a dirty tree:** `walk_sheet` refuses to walk uncommitted code unless `--allow-dirty`, which saves the diff beside the walk. Every walk records its git sha, batch and passes.

**Activation memory** (`Understand`). +0.117 GB per window over a ~1.11 GB floor (slots, non-block layers, inputs): batch 8 peaks at **2.05 GB**, well under 3.6 GB (room for ~20 windows; a panorama has 8). Entry 1 had estimated ~0.53 GB per window from the reference; fused glue and no dequant scratch cut that by ~4.5×. With opt-in graphs, four batch-8-ish keys peak at 2.23 GB. One environment lesson: without PyTorch's expandable-segments allocator, batch-8 graph pools fragmented and OOMed in the tests, so the tests now use the painter's allocator.

**Stop latency and progress** (`Understand`). `check_stop()` runs between passes, so a stop now waits up to one **7.2 s** pass (was 3.6 s). At batch 8 every pass boundary is a step boundary. Progress ticks halve (panorama 4, walk step 5), with the same average rate. Ruling (b)'s cost: slower stop response for no speed gain today. Task 10 re-measures batch 4 against 8 with streaming and drops back to 4 if streaming doesn't benefit.

**Results.** tile512 **0.87 s** (20.31 TFLOP/s; a single window, so batching can't move it), panorama **29.8 s** in 4 passes, walk step **39.3 s** in 5 passes, peak **2.05 GB**, every picture bit-identical to batch 1.

**In an interview.** "Batching panorama windows was a speed tie because the engine was compute-bound: each window's tokens meet every weight anyway. It halved the per-pass work, which matters once weights stream from an SSD at ~3 s a pass. The interesting part was that it wasn't lossless. cuBLAS chooses its GEMM algorithm by M, so two small bf16 conditioning layers rounded one ulp differently at batch 8, and downstream INT8 quantization turned that into a different image. Running those five tiny layers per window made every window bit-identical to batch 1 for 0.16%, and I pinned the reference path to batch 1 so the reference is canonical."

**Tried and not chosen** (`Understand`):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **Keep batch 4** | Panorama 30.2 s, walk step 39.4 s, peak 1.76 GB (same session) | Ties batch 8 on time with 2× the passes, copies and Python; it doubles Task 10's SSD floor per window-step | If Task 10 shows streaming doesn't benefit from 8: halves stop latency (3.6 s) |
| **Batch 6** | Panorama 30.4 s in 8 passes (6 + 2), walk step 39.5 s, peak 1.81 GB | Same pass count as 4 (ragged), no gain | Never for 8-window panoramas |
| **Batch 1 / 2** | Panorama 30.3 / 30.6 s, walk step 40.3 / 39.9 s; interleaved A/B 0.943 s per window at 1 vs 0.915 at 8 | Slightly slower, 4–8× the passes | Lowest stop latency and memory (1.2 GB) |
| **Batch 8 without `rowwise`** | Panorama **25.73 dB** against batch 1 | Not lossless: cuBLAS bf16 conditioning layers round by M, and W4A8 amplifies one ulp into a new sample | Under a lossless kernel with no INT8 stage downstream (the ulp would stay an ulp) |
| **`rowwise` on `time_text_embed` + `norm_out.linear` only** | All 57 blocks bit-exact, output still 2.1e-3 off at batch 2 (`proj_out`) | Incomplete; `proj_out` and, defensively, both embedders were added (equal today, but cuBLAS can switch algorithms by M) | — |
| **A 1e-2 relative-L2 tolerance** for batched vs single | Measured 3e-3 to 5.3e-2 on random inputs | Too loose to mean anything when real images moved 25.7 dB; replaced by bit equality | — |
| **Gate against Task 8's batch-4 walks** (as dispatched) | Fail, 2 of 18 spots (+0.166, +0.045); 11 improved; spot 0 already 32–37 dB | The base carried arbitrary M = 4 rounding; against batch 1, 18 of 18 byte-identical | — |
| **Restore Task 8's pictures** (batch 4, no `rowwise`) | Two lines of code | Would keep an arbitrary, batch-dependent picture as the reference | If the old pictures had to be reproduced exactly |
| **Fix I1 by narrowing the docs to "engine kernels only"** (the review's option 2) | Reference at batch 8: 2.8e-3 to 7.6e-3 relative per window, never bit-equal | A canonical reference beats reference speed | — |
| **Pin only the bench's reference to batch 1** | — | The game and walks can also run `off`; the rule belongs in `engine_batch`, which all three share | — |
| **`_Rows` holding a bound method** (first version) | A module → wrapper → module reference cycle, freed only by the cyclic GC | Replaced by a weakref and a module marker (also safe after a hook rewraps the forward) | — |
| **Graph keys without expandable segments** | Keys 8 and 5 fit (2.86 GB reserved); a third (7) OOM | The painter already uses expandable segments (2.37 GB for six keys); tests now match | — |

<details><summary>Check yourself (Task 9)</summary>

**Q1.** Batch 8 halves the passes, yet a panorama takes the same 30 s as at batch 1. Why, and when would batching pay?

**A1.** The engine is compute-bound: each window's tokens meet every weight, so 8 windows are 8× the GEMM work whatever the grouping. Batching only shares per-pass costs (weight copies, per-block Python, the modulation read), which are hidden or small today. It pays when a per-pass cost dominates: the old per-pass dequant (Task 1), or SSD streaming at 2.3–3.3 s a pass (Task 10).

**Q2.** All 57 blocks gave bit-identical rows at any batch size. How did batching still change the picture?

**A2.** Two small bf16 layers outside the blocks (the time/text embedders) ran on cuBLAS, which picks its algorithm by M (the batch), so they rounded one ulp differently. That conditioning feeds every block's modulation; W4A8 turned the ulp into different INT8 roundings, and four denoising steps grew it into another sample (25.7 dB). Running those layers per window (`rowwise`) removed it.

</details>

> **Remember this:** when compute-bound, batching doesn't add speed; it divides per-pass costs, so batch for the cost that's coming (SSD streaming). Library GEMMs can round differently by batch size, so make each sample independent of its batch if anything downstream amplifies bits.

---

### Task 10a: streaming FLUX blocks through a pinned ring

`Must` for the plain version, pinned vs pageable memory, why a busy Python thread slowed the GPU 25×, why the fix is "keep the GIL" rather than "release it", why eager can't be fixed but graphs can, and the two OOM root causes. `Understand` for the reader contract (tickets), the RNG guard, the ticker measurement, batch 4 vs 8 and the `card_limit` ordering. `Skip` the `SlotRing` event free-list, the `_info` weak-reference detail and the nsys gap listing.

> **Plain version.** Task 7 already copied each block's weights to the GPU while the previous block computed. Task 10a re-plumbs that so the *weights live in one big page-locked buffer* (a "pinned arena", 6.66 GB) and each block moves in **one** copy instead of ~20. The sequencing (wait, copy, record) moves into a small C++ object that **keeps the GIL**. **Speed in a quiet process didn't change** (tile512 0.86 s, panorama 29.3 s), because the copies were already hidden under compute; the pictures are bit-identical (18 of 18 gate-walk spots byte-identical to Task 9). What changed is two things for later. First, with graphs on, a pass holds its quiet speed (**0.86 s**) while two other Python threads spin, where it used to take 7.2 s. Second, there is now a seam, `source(i)`, where Task 10b's SSD reader plugs in. It also cost two out-of-memory hunts, and the lesson in both is that memory outside PyTorch's allocator is real.

**Pinned vs pageable memory** (`Must`). The GPU's copy engines read host memory by DMA, so they need physical pages the OS won't move or swap. Ordinary (pageable) memory makes no such promise, so `cudaMemcpyAsync` from it first stages through a pinned buffer *on the CPU* and returns only when that is done. It isn't asynchronous at all. Pinned memory is the promise, so the copy engine reads it directly while the SMs do other work, at PCIe speed (**13.4 GB/s** measured). Two events per slot order everything on two streams: a copy into slot *s* waits for "freed" (its last block ended), and a block waits for "copied".

```
 host arena (one buffer, cudaHostRegister'ed once, 6.66 GB)
 [ block 0 | block 1 | block 2 | ............................................ | block 56 ]
     |  block i laid out exactly as its slot: each tensor at its slot offset (256-aligned)
     v  ONE cudaMemcpyAsync per block, on the side stream
 device ring   [ slot 0 ]  [ slot 1 ]  [ slot 2 ]      block i in slot i % 3, 191 MB each, fixed addresses

 compute  | blk 0 | blk 1 | blk 2 | blk 3 | blk 4 | ...
 side     |c3|      |c4|    |c5|    |c6|    ...           c(i+3) waits "freed" of blk i, then copies;
                                                         each block waits "copied" of its own slot first
```

**Why a busy Python thread slowed the GPU 25×** (`Must`). Every PyTorch call from Python releases the GIL around its work and takes it back after. Alone that is free. With another thread wanting the GIL, taking it back means waiting until that thread gives it up, which a busy thread does only when the 5 ms switch interval runs out. So **every torch call costs up to a switch interval**. Measured per call (`gilprobe.py`, alone / under 2 spinning threads):

| call | alone | under contention |
|---|---|---|
| `x.view(-1)` | 0.6 µs | 3,457 µs |
| `x.add_(1)` | 3.9 µs | 4,977 µs |
| `dst.copy_(pinned_src, non_blocking=True)` | 3.2 µs | **44,295 µs** |
| `event.record()` / `stream.wait_event(e)` | 3.8 / 3.7 µs | 431 / 529 µs |
| an extension (pybind) call | 3.8 µs | **11 µs** |

Task 7's hooks made ~25 such calls per block (a `copy_` per tensor, a stream wait, two records): 57 blocks × 25 calls × milliseconds is the ~20 s a pass took. The GPU sat idle waiting for the CPU to enqueue the next block. A plain pybind call that *doesn't* release the GIL costs 11 µs even under contention, because it never gives the GIL up, so there is nothing to wait for. That is the whole design: **don't make the hand-off cheaper, don't have one.**

**The GIL stress table** (`Must`). Two spinning Python threads in the painter's process, batch-1 tile512 passes timed by CUDA events; min / median / max seconds, quiet = 0.85 s. One process per row.

| sequencing | eager (default) | graphs (opt-in) |
|---|---|---|
| BASE fcbfe73 (per-tensor `copy_` + torch events in Python) | 20.2 / 23.7 / 30.2 | 7.07 / 7.23 / 9.86 |
| arena + per-tensor `copy_` from Python | 22.9 / **64.2** / 166.8 | 7.15 / 8.25 / 12.0 |
| arena + one `copy_` per block from Python | 14.3 / 26.7 / 88.1 | 2.16 / 11.9 / 34.7 |
| arena + C++ `SlotRing` | 13.1 / 22.9 / 40.8 | 0.92 / 1.35 / 10.1 |
| **+ `graph_launch`** | (n/a: eager has no graphs) | **0.85 / 0.86 / 10.5** (5 runs, medians 0.885 / 3.76 / 0.858 / 0.862 / 0.860) |
| `sys.setswitchinterval(0.5 ms)`, SlotRing, eager | 10.5 / **45.3** / 58.4 | n/a |
| quiet (no spinning threads) | 0.853 (BASE 0.854) | 0.86 |

Reading it:
- **Graphs hold quiet speed once going** (median 7.2 → 0.86 s). A replayed block's Python is now the hooks and one launch, neither of which releases the GIL. `CUDAGraph.replay()` released it once a block, which left medians of 0.95–2.37 s over five runs. `graph_launch` calls `cudaGraphLaunch` on the graph's `raw_cuda_graph_exec()` itself: **0.86 s**.
- **Eager barely moves** (median 23.7 → 22.9 s). The eager block forward makes tens of PyTorch calls per block (reshapes, views, `.to`, `chunk`, the attention dispatch), each a hand-off. 20 s over 57 blocks at 5–10 ms a hand-off is ~40–70 hand-offs a block. Removing the hooks' ~25 leaves the forward's own. Under contention that cost belongs to eager execution itself.
- **The spread is wide** (0.9 s next to 10 s in the same run): who gets the GIL after a switch interval is up to the OS scheduler, so read medians. The slow graph passes come early in a run, from the per-pass Python outside the blocks (embedders, the `rowwise` layers, the Euler step), which still calls PyTorch.
- It is a stress case: the game's other threads mostly wait on I/O.

**A reader contract for Task 10b** (`Understand`). `source(i)` hands the streamer block i's bytes; the copy that reads them runs *later*, on the GPU. An SSD reader that reuses buffers must know when the copy has finished. The first version had two bugs, found by review (I1):
- **A race.** `copied(slot)` was answered from the slot's *previous* copy's event. In the window between `source(i)` returning and the copy being enqueued, a reader thread asking "is it done?" got True, overwrote the buffer, and the block computed on the next block's bytes. Silent, nondeterministic, no assertion catches it.
- **A lifetime bug.** The streamer kept only the latest source per slot. A reader handing fresh pinned buffers could see its buffer go back to torch's pinned pool while the DMA still read it (a raw `cudaMemcpyAsync` records no torch event).

The fix is per-copy **tickets**: `SlotRing.load` returns a number (1, 2, 3, ...) and records a per-ticket event, under the GIL, in the same call that records the copy. `done(ticket)` is exact however the slots have been reused since. `copy_done(i)` is **False from the moment `source(i)` is called**. Every source object is kept alive until its copy is done. Making the lifetime assertion fail also found that the per-object cache held every source strongly (up to 4 × 57, i.e. 228 page-locked block buffers for a fresh-buffer reader), so it now holds weak references. Cost: one event per copy.

**Why graph replays must not draw random numbers** (`Understand`). `graph_launch` skips `CUDAGraph.replay()`'s RNG prologue (the philox seed and offset update). A graph that drew random numbers would draw the same ones every replay, and nothing would fail except the output. FLUX's blocks draw none (`dropout_p = 0`), and review I2 made the code say so: the eager warm-up before each capture runs under a dispatch mode that raises, naming the block, on any op tagged `nondeterministic_seeded` (attention and dropout with p = 0 are exempt). A custom cuRAND kernel would not be seen. Ruling: guard it, don't drop `graph_launch`, because that fast replay is what makes graphs hold 0.86 s under contention.

**Does holding the GIL stall the server's other threads?** (`Understand`). Review I3: before 10a, blocking CUDA calls were torch ops, which release the GIL, so other threads ran. Now `graph_launch` and the ring hold it, and if a launch blocked on a full queue it could freeze everything for a block's GPU time. The ruling was to measure first and add mitigation only for a stall over 50 ms. A 1 ms ticker thread (sleep 1 ms, record how late it woke) ran in the painter's process during a real panorama (4 batch-8 passes + VAE, one process per row, AC). Lateness in ms:

| | eager: max / p99 / p99.9 / >10 ms / >50 ms | graphs: max / p99 / p99.9 / >10 ms / >50 ms | panorama s (eager / graphs) |
|---|---|---|---|
| BASE fcbfe73 | 42.1 / 0.10 / 23.8 / 55 / **0** | 0.39 / 0.064 / 0.09 / 0 / **0** | 29.8 / 30.2 |
| fe9e0a2 (10a) | 44.9 / 0.10 / 24.3 / 58 / **0** | 1.61 / 0.063 / 0.09 / 0 / **0** | 29.6 / 30.2 |
| 017b712 (fixes) | 45.3 / 0.10 / 24.2 / 65 / **0** | 2.49 / 0.064 / 0.09 / 0 / **0** | 30.1 / 30.2 |
| idle (no paint) | max 0.08–1.24 | | |

No wake-up passed 50 ms, so neither bounded run-ahead nor releasing the GIL around blocking calls was added. With graphs other threads see at most **2.5 ms**. Eager's 24–45 ms worst wake-ups are the same at BASE (42.1 ms): they come from eager execution, not from the streamer. One eager run made while the laptop slipped onto battery (61.5 ms, 42.9 s wall) was discarded and re-run on AC. Caveats: painter only, server stopped, one run per row.

**The ruling: make graphs the default next** (`Must`). The Task 7 ruling said "revisit graphs after Task 10", and the data now exist. Graphs tie eager when quiet (0.86 s, 38.6 s walk step), hold quiet speed under contention (0.86 s against 22.9 s eager), give other threads 2.5 ms worst against 45 ms, and the RNG guard is in. The cost is Task 7's capture: **+3.9 s on the first panorama**. The reviewer's first answer was "not yet" (the stress was synthetic, I3 unmeasured, the guard missing); the last two are now closed. Ruling: a small follow-up step (10a-g) before 10b flips the default. Cost if wrong: one environment default to flip back. Open until then: the stress is still synthetic.

**Batch 4 vs 8 under the streamer** (`Understand`). Task 9 left this open. Same session, a1d78dc:

| | batch 4 | batch 8 |
|---|---|---|
| panorama s (passes) / s per window-step | 29.9 (8) / 0.887 | **29.3 (4) / 0.861** |
| walk step s (passes) / s per window-step | 39.1 (10) / 0.942 | **38.6 (5) / 0.929** |
| stop latency (longest pass, pano / step) | **3.63 / 4.37 s** | 7.07 / 7.91 s |
| peak allocated (pano / step) | 1.76 / 1.77 GB | 2.05 / 2.05 GB |

Batch 8 is 1.4% (step) to 2.9% (panorama) faster per window-step, so it isn't a tie and **8 stays**. The price is a stop waiting up to 7.9 s instead of 4.4 s. Final call after 10b, where each pass also reads 6.66 GB from the SSD.

**Two out-of-memory hunts** (`Must`). Both came from running the full GPU suite on a 14 GB laptop with a 4 GB card. They look like leaks, and neither was.

1. **The GPU walk test ran out of device memory: the CUDA libraries live outside PyTorch's allocator.** `test_six_spot_walk_at_the_walk_limit` died in the suite, 5 MB short (asked 264 MiB, 259 MiB free), and passed alone. A per-test tracker showed PyTorch's allocated memory falling back to 0.009 GB after every module, so nothing was held. What grew was device memory **outside PyTorch's allocator**: CUDA libraries' code and handles in the process's context, **0.142 GB** when the module runs alone, **0.199 GB** after the first GPU module, **0.239 GB** by the walk test. It can't be given back without ending the process. The walk peaks at 3.32 GiB on a card with 3.68 GiB usable. The same numbers hold at fcbfe73, so it wasn't 10a. (The box's driver had also moved to 595.99.02 after the outage, which may have shifted the margin.) Fix: a `card_limit` marker, and `conftest` runs marked tests *first* in the session, before other modules load their libraries.
2. **The host ran out of memory: the bench loader held two copies of the model (13.2 → 7.75 GB).** With the walk test moved first, the suite was OOM-killed on the host (pytest at 13.8 GB anon) inside `test_glue.py`. A per-test RSS tracker showed that test alone peaks at **12.75–13.17 GB**, at fcbfe73 too. It loads the transformer through `bench_flux.load_transformer`, which is diffusers' loader: it copies each tensor out of a memory map of the whole file and **holds both**. The painter avoids this with `_read_gguf`. Fix: `load_transformer` reads like the painter does (that test's peak **13.17 → 7.75 GB**), and a module-scoped fixture collects and `malloc_trim`s after each test module.

Related and found earlier (`Must`): **`malloc_trim` in the arena pack.** Packing freed each GGUF tensor after copying it into the arena, but glibc kept the freed heap (each tensor is under its 32 MB mmap threshold), so a painter process held both copies: **9.10 GB** alone, OOM-killed at 13.8 GB in the suite. `malloc_trim(0)` after each block: **7.87 GB** peak, then 7.52 GB after setup, the same as Task 9. The test that guards it went red at 9.10 GB with the trim off. The cost is model load, **12.2–12.8 s → 15.1–16.1 s (+3 s)**, mostly the pack.

**The `card_limit` ordering, and what it cost** (`Understand`). Running marked tests first is an *ordering* fix, because the cause can't be removed in-process. It has a price: one flaky timing failure in a `card_limit` test can now OOM the rest of the GPU suite (N3). A failed test's traceback also pins its allocations. The walk's 3.29 GB stayed on the card, and the RSS test's painter subprocess couldn't get the ~1 GB it needs. So the RSS test moved into its own module (`test_stream_rss.py`) and calls `empty_cache()` before spawning. The next run (8410754) showed `test_three_spots_in_a_row_timing` (the same walk, also 3.32 GiB) needed the marker too; it got it (9884b2d).

**The tests that couldn't fail** (`Understand`). The review's I5: "pinned memory unchanged after a stop" was asserted by construction (`pinned_bytes` is the arena's size, a constant), and the mid-pass-stop test couldn't catch a missing `always_call`, because the damage lands on a pass that is discarded. The fixes measure real counters (torch's pinned-pool `active_bytes` and the driver's `host_pinned(arena)`), and a toy block that queues a matmul from its slot, then raises, checks the output against the block's own product. Each new assertion was then shown **red against deliberately broken code** (a source that leaks a pinned MB per copy: 419,430,400 ≠ 361,758,720 bytes; no `always_call`; the per-slot `copied`: True inside `source()` 5 times). Lesson: an assertion you haven't seen fail is a guess about what it covers. The original red run for `test_stream.py` was also shown only after the implementation (4 failed, 3 errors against a BASE worktree), reported honestly.

**Exception paths** (`Understand`). I4: `packed()` registered the 6.66 GB arena and then called `install()`. If `install()` raised (a `cudaMalloc` OOM for the slots), nothing owned the arena, so it was freed later while still registered: the "Copy is larger than memobj size" hazard that `test_flux_close_gpu` exists for. And `install()` repointed block tensors at slots *before* creating the stream and ring, so a failure left blocks viewing uninitialised memory with no hooks. Now everything is built first, the blocks are repointed last, and any failure rolls back. A follow-up (N2, folded into 10a-g): the arena can still be left registered if something *after* `packed()` raises.

**Bit-exactness gate** (`Must`). `walk_sheet.py --scenes forest,desert,coast`, then `--relative t9-b8 t10a`: verdict **PASS, worst delta +0.000, min PSNR inf**, 18 of 18 spots byte-identical (`cmp`) to Task 9's batch-8 walk. After the fix round the same walk gives Δ 0 at 9884b2d, with the full suite at **846 passed in 31:22**. (The walk's own exit is 1 for coast spot 5's seam 1.19 > 1.1, exactly as at Task 9: same picture.) `graph_launch` wasn't walked, since graphs are opt-in and the walk default is eager. `test_graph_matches_eager` and `test_stream_matches_resident` (graphs) hold its bit-equality.

**Copy and compute overlap** (`Understand`). nsys, 3 tile512 passes, eager. The change is 20× fewer copy calls, not a faster pipeline:

| | BASE fcbfe73 | 10a |
|---|---|---|
| H2D copies a pass | 1,140 | **57** |
| copy-engine busy a pass | 499.7 ms (13.3 GB/s) | 495.3 ms (13.5 GB/s) |
| copy time overlapped with a kernel | 98.82% | 98.84% |
| SM idle between kernels | 1.58 ms (0.19%) | 1.67 ms (0.20%) |

`test_stream_overlaps` (12 single blocks, one window, the least compute per copied byte): resident 162.1 ms, streamed 166.6 ms, **1.028×** against a 1.1× bar.

**Results.** Quiet process, AC, against Task 9: tile512 **0.86 s** (was 0.87; 20.55 TFLOP/s), panorama **29.3 s** (was 29.8), walk step **38.6 s** (was 39.3). The 1–2% is noise-sized, so call it a tie. Peak allocated unchanged (1.225 / 1.576 / 2.045 GB at batch 1 / 4 / 8). Page-locked host memory **6.66 GB in one range** (was ~1,100 registered ranges). Painter RSS 7.52 GB after setup, 8.54 GB in a walk step, of which the arena is 6.66 GB (what an SSD source can drop). The server at idle after startup: **7.53 GB VmRSS**, the baseline for 10b's under-12 GB target. Model load +3 s.

**What this sets up for 10b** (`Understand`). Copies already take **58%** of a batch-1 pass (495 of 854 ms at 13.4 GB/s). From an SSD at 2–3 GB/s a pass's 6.66 GB is **2.2–3.3 s**, longer than a batch-1 pass (0.85 s) and about half a batch-8 one (7.4 s). So 10b is I/O-bound at batch 1 and needs batch 8 plus read-ahead. `source(i)` is called when the copy is issued (ring − 1 blocks ahead), so the reader must read further ahead itself, and must not reuse a buffer before `copy_done`.

**Operations lesson** (`Understand`). The box went offline mid-suite. Wi-Fi dropped and NetworkManager stopped retrying, and a kernel update had arrived without its NVIDIA module. So after a reboot, check the network manager first, then that the driver module matches the running kernel. It wasn't the suite.

**In an interview.** "I moved FLUX's weights into one pinned host arena and streamed them through a three-slot device ring, one copy per block, with the stream and event sequencing in a C++ object that keeps the GIL. Quiet speed didn't change, because the copies were already hidden under compute, and I checked that with nsys: 98.8% of copy time overlapped a kernel. The point was two things downstream. One was a clean seam for an SSD reader, with a per-copy ticket so a reader can't reuse a buffer early. The other was GIL robustness: PyTorch calls release the GIL, so under a busy second thread each call can cost a 5 ms switch interval, and a pass took 23 s instead of 0.85. With graphs launched through the extension it stays at 0.86 s, and a 1 ms ticker showed the other threads never wait more than 2.5 ms. I measured that before deciding whether to add mitigation. The full GPU suite also taught me that memory outside PyTorch's allocator is real. CUDA's own libraries took 0.24 GB of a 3.68 GB card, and a test loader held two copies of the model."

**Tried and not chosen** (`Understand`):

| What it was | What we measured | Why we didn't pick it | When it might win instead |
|---|---|---|---|
| **C++ `SlotRing`** (chosen) | Eager 13.1 / 22.9 / 40.8 s; graphs 0.92 / 1.35 / 10.1 s (min / median / max); quiet 0.86 s | — | — |
| **Per-tensor `copy_` from Python on the arena** (Task 7's way) | Quiet 0.859 s; contention eager median **64.2 s**, graphs 8.25 s | ~20 GIL hand-offs per block | When nothing else runs in the process |
| **One `copy_` per block from Python** (packed arena, torch events) | Quiet 0.856 s; contention eager 26.7 s, graphs 11.9 s median (2.2–34.7 spread) | Still 4–5 hand-offs per block (copy, waits, records); the C++ ring has none | — |
| **`sys.setswitchinterval(0.5 ms)`** | Eager under contention median **45.3 s** (was 22.9) | Worse: more switches, and it is process-wide | Never for this; it adds switches where the goal is fewer hand-offs |
| **`CUDAGraph.replay()` with the C++ ring** | Graphs under contention medians 0.95–2.37 s (5 runs) | One hand-off per block left; `graph_launch` gives 0.86 s | If graph replays ever needed the RNG prologue |
| **`graph_launch`** (chosen; its own commit 470a036) | Graphs 0.85 / 0.86 / 10.5 s; ticker max 2.5 ms; `test_graph_launch_keeps_the_gil` 212 µs (replay(): 5,071 µs) | Deviation worth naming: it touches the graphs path, skips `replay()`'s RNG prologue, and is easy to drop. Guarded by the RNG check | A graph that draws random numbers |
| **Fall back to `replay()` for seeded graphs** (I2) | Not run | The ruling asked for a loud failure; a fallback would silently bring back a hand-off per block for the graphs that need care | — |
| **Check the RNG tag during the capture itself** | Not run | The warm-up runs the same ops eagerly just before; checking there keeps Python dispatch work out of the capture stream | — |
| **`torch.empty(pin_memory=True)` for the arena** | Not run: torch's pinned allocator rounds up to a power of two (6.66 → 8.6 GB) | 2 GB of locked RAM for nothing; `cudaHostRegister` on exact-size memory instead | A smaller or power-of-two buffer |
| **Enqueue several blocks' copies ahead** | Not run | A copy into a slot must wait for that slot's last block, whose "freed" event isn't recorded yet when enqueued ahead (`cudaStreamWaitEvent` takes the *last* record); would need `cuStreamWaitValue32` counters. The ring already makes each enqueue ~5 µs | Much faster compute per copy |
| **Whole per-block loop in C++** | Not run | The loop's Python is the block forward itself (eager) or one launch (graphs); the hooks were the only sequencing, now C++ | — |
| **Bound CPU run-ahead / release the GIL in `graph_launch`** (I3) | Ticker: no wake-up over 50 ms (graphs max 2.5 ms, eager 45.3 ms, same as BASE) | The ruling's 50 ms bar wasn't reached; a per-pass sync would add a stall for nothing | If a stall over 50 ms ever shows in the ticker (e.g. under 10b's reads) |
| **Batch 4 under the streamer** | Pano 29.9 vs 29.3 s, step 39.1 vs 38.6 s; stop latency 4.4 vs 7.9 s | 1.4–2.9% slower; the ruling keeps 8 unless 4 ties | Lowest stop latency; revisit after 10b |
| **Keep `copied(slot)` and add a lock** (I1) | Red: True inside `source()` | A slot's event says nothing about a copy not yet issued, or one since overwritten; tickets answer per copy | — |
| **`_info` cache holding sources strongly** (bounded 4 × 57) | The lifetime assertion couldn't fail; up to 228 block buffers alive for a fresh-buffer reader | Hid the lifetime contract and cost memory for 10b; now weak references | — |
| **No `malloc_trim` after each block** | 9.10 GB peak in a fresh process; suite OOM-killed at 13.8 GB | glibc keeps the freed heap (tensors under the 32 MB mmap threshold), so both copies stayed resident; trim gives 7.87 GB | musl or another allocator that returns memory itself (the trim then warns once and does nothing) |
| **`ru_maxrss` to measure the RSS test's subprocess** | It carried the parent's peak over fork + exec | Replaced by VmHWM, which is the process's own (includes file-backed pages, so an upper bound on anon) | — |
| **Six-spot walk in a fresh subprocess** (34319bc, reverted by 1466b4f) | Child died: `cuDevicePrimaryCtxRetain` CUDA_ERROR_OUT_OF_MEMORY | The suite's own CUDA context (0.24 GB) stays on the card, so there is no room for a second context | — |
| **`card_limit` ordering** (chosen) | Walk passes alone at peak 3.32 GiB; fails 5 MB short late in the suite | An ordering fix: the cause (0.24 GB of CUDA library memory) can't be removed in-process. Cost: one flaky `card_limit` test can OOM later modules | — |
| **`card_limit` first, no host-memory changes** | Suite host-OOM-killed at 17% (anon 13.8 GB) | The walk's leftover heap plus a 13.2 GB test; fixed at the source (loader and trim) instead of reshuffling more tests | — |
| **Keep diffusers' loader in `bench_flux.load_transformer`** | 12.75–13.17 GB peak for one test; 13.2 GB at fcbfe73 too | It holds the memory map and the copies at once; the painter's `_read_gguf` peaks at 7.75 GB | — |
| **Make graphs the default inside 10a** | Reviewer: "not yet" (stress synthetic, I3 unmeasured, RNG guard missing) | I3 and the RNG guard are now closed (the stress is still synthetic, the capture cost remains); ruling: a separate step (10a-g) before 10b, one default to flip back | — |
| **`host_pinned` check on BlockSlots' per-tensor sources** (M6) | Breaks Task 7's toy tests (pageable host copies) | Deferred; fold BlockSlots into BlockStreamer over a `HostArena` when 10b touches the seam (N1 is one more reason) | — |

Deferred minors: M2, M4 (cost of the 57 trims in the +3 s load not measured), M6, the rest of M7, N1 (the base `BlockSlots` path never retires ticket events, so one CUDA event per load accumulates), N4–N8 (a `close()` racing a reader raises `AttributeError`; tickets restart at 1 after a re-install; the `copy_done` docstring over-promises across threads; the pinned-stats assertion goes silent if its key is missing; CUDA error paths in `SlotRing::load`; `native_dropout(train=None)` read as eval). Folded into 10a-g: N2, N3 and N9 (commit `ticker.py`, which the explainer cites but the repo lacks).

<details><summary>Check yourself (Task 10a)</summary>

**Q1.** Streaming didn't change the speed of a quiet pass. Why build it, and what did the GIL stress show?

**A1.** The copies were already hidden under compute (98.8% of copy time overlapped a kernel), so nothing was left to gain when quiet. It was built for a seam (`source(i)`) for the SSD reader, and to cut per-block Python hand-offs. With two spinning threads, every torch call can cost a 5 ms switch interval, and a pass took 23 s eager and 7 s with graphs. A C++ ring plus `graph_launch`, which keep the GIL, put graphs back at 0.86 s. Eager stayed at 22.9 s, because the block forward's own torch calls remain.

**Q2.** Why did a test that passes alone OOM inside the suite, and why isn't it a leak?

**A2.** PyTorch's allocated memory fell back to 0.009 GB after each module. What grew was device memory outside PyTorch's allocator (CUDA libraries' code and handles in the context: 0.142 → 0.239 GB) on a card with 3.68 GiB usable and a 3.32 GiB peak. It can't be freed in-process, so the test runs first (`card_limit`).

**Q3.** A test peaked at 13.2 GB of host RAM and later at 7.75 GB without any model change. What was it?

**A3.** The bench loader was diffusers' loader, which copies each tensor out of a memory map and holds both. Reading as the painter does (`_read_gguf`) removed the double copy.

**Q4.** What race could a threaded SSD reader hit with a per-slot `copied(slot)`?

**A4.** Between `source(i)` returning and the copy being enqueued, the slot's event still belongs to the previous copy and reads as done. The reader reuses the buffer, and the block computes on the wrong bytes. Per-copy tickets close the window.

</details>

> **Remember this:** when you can't make a hand-off cheap, make it disappear. Pinned memory plus a C++ ring that keeps the GIL gave graphs quiet speed under contention, and a ticker showed it didn't starve other threads. Memory that PyTorch doesn't track (CUDA libraries, glibc's freed heap, a loader's double copy) is still memory, so measure the whole process.

---

### Patterns so far

`Must`

> 1. **Find which limit you're hitting** (roofline). The big GEMMs are compute-bound, so a faster number format helps (W4A8). The GEMV is memory-bound, so fewer bytes help (Q4_K at M = 1). Glue is memory-bound, so fusion helps (Task 6).
> 2. **Keep data in registers.** Decode Q4_K into the fragment the next instruction needs, convert int to float with a bit trick, and reduce with warp shuffles. Never write an intermediate to VRAM.
> 3. **Overlap loads with math.** `cp.async` double buffering, two blocks per SM, and the next chunk requested before this one's math.
> 4. **Check lossy changes against the real output, not layer by layer.** A per-layer error study missed the residual-stream layers; a deterministic end-to-end gate found them.
> 5. **Measure before trusting an estimate.** Task 5 beat its estimate (92% of cuBLAS, not 65–85%), Task 5b fell short (1.5×, not 2–4×), and Task 5c's 0.29 s premise was stale. Task 6's "0.4 s of glue" was 68 ms, and the real cost (RoPE, 147 ms) wasn't in the plan at all. Task 7's launch overhead was 0.34%; the real 10.9% was offload syncs. Re-measure the thing you're about to optimize, and keep the ablation builds.
> 6. **Make lossless changes bit-identical** (Task 6). Then the gate proves Δ = 0 instead of arguing about noise, which matters most upstream of a lossy stage like W4A8.
> 7. **Amdahl moves the target.** Each speedup makes the untouched parts a bigger share: glue went from small to 31% of GPU time by Task 5c, and after Task 6 the GEMMs are 85% again.
> 8. **Never buy speed with an unapproved quality loss, and don't trust a vacuous pass** (Task 8). The cache measured 1.32–1.63× and shipped off; the threshold that "passed" had skipped nothing.
> 9. **Make results independent of incidental choices** (Task 9). cuBLAS's algorithm depends on M, so batch size changed the picture until those layers ran per window. When a lossless change fails a gate, check whether the *base* depended on something arbitrary.
> 10. **Memory PyTorch doesn't track is still memory** (Task 10a). CUDA's libraries, glibc's freed heap and a loader's double copy each caused an OOM that looked like a leak. Measure the whole process, and when you can't make a hand-off cheap, remove it.

---

## Learner preferences

Recorded at the 2026-10-24 walkthrough. They apply to every future entry and walkthrough.

- Plain language first, numbers second. Open each idea with a one-line version before the detail.
- One concept per walkthrough.
- When the learner says "skip", reduce that part to a one-line takeaway, and keep the detail only under an "Understand later" label.
- Standing rule (2026-11-06): record **every** technique tested, including those not chosen, in a "Tried and not chosen" table with what we measured, why we didn't pick it and when it might win. Use measured numbers, and say so plainly when a number was never recorded.
