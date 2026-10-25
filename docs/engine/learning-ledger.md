# Engine learning ledger

A running study log for the FLUX engine work. Each entry explains one idea with this project's real numbers, so it can be revised before an interview. New entries go at the bottom; add a line to the index when you do.

**Conventions.** "Measured" numbers come from `bench/baseline.json` and `docs/engine/00-baseline.md` (on AC, 2026-10-20, git 895c75e) unless labelled otherwise. "(est.)" marks a back-of-envelope figure, with its assumptions stated next to it. "(battery)" marks a number from the 2026-10-20 spike, which ran throttled (35 W, PCIe Gen2). The spike report itself is no longer on disk; its numbers survive in the spec, `docs/superpowers/specs/2026-10-20-flux-engine-design.md` §2.

## Ledger index

| # | Date | Entry | Related plan tasks |
|---|---|---|---|
| 1 | 2026-10-24 | [Making FLUX faster: ops/sec on a 4 GB GPU](#entry-1-making-flux-faster-opssec-on-a-4-gb-gpu) | 3, 4, 5, 6, 7, 8 |
| 2 | 2026-10-25 | [Techniques from Tasks 5, 5b and 5c](#entry-2-techniques-from-tasks-5-5b-and-5c-tensor-cores-w4a8-the-modulation-gemv) | 5, 5b, 5c |

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
| **Task 5c: modulation on a Q4_K GEMV** | **1.12** | 15.78 (bf16-equivalent) | 82.6% | 47.6 |

From Task 5b on, "TFLOP/s" counts the same 17.67 TFLOP even though 75% of it runs as INT8, so it is **bf16-equivalent**: how fast a bf16 engine would need to be to match. It can pass the bf16 roof without breaking physics.

The step is 2.45× faster than the baseline (2.74 → 1.12 s), but the walk step only 1.29× (61.4 → 47.6 s). Why the walk gains less is the first lesson of Task 5.

**How the Entry 1 estimates did.**

| Estimate (Entry 1) | Predicted | Measured | Why it differed |
|---|---|---|---|
| Technique 1 (fused W4A16) step | ~1.9 s (1.6–2.2) | **1.40 s** | the kernel reached 92% of cuBLAS; I assumed 65–85% |
| Technique 2 (W4A8) GEMM speedup | 1.8–2.5× | **1.45–1.72×** | the kernel is issue-bound (Task 5b below) |
| Technique 2 step factor | ×1.4 | **×1.22** (1.40 → 1.15) | the slower GEMM, plus 25% of the FLOPs kept on bf16 for quality |
| Modulation "cheap target" | ~0.29 s | **~0.03 s** | the 0.29 s was dequant, which Task 3 already removed (Task 5c below) |

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
- **Caveat:** that is one scene. Tasks 6 and 8 re-gate it. Task 5c later moved the default sample (25.78 dB vs 5b) and the walk was not re-walked, so the 1.058 margin should be re-checked before relying on it.

**The PSNR trap.** The 512 test image under W4A8 is 25.22 dB from `mma`'s: the same path, ferns and light shafts, with some trunks in other places. Four denoising steps turn tiny rounding into a *different sample of the same prompt*, not a noisy copy. That is why a lossy stage is judged by the walk gate and the eye, not by PSNR against the reference.

**Results.** Step **1.15 s**, **15.37** bf16-equivalent TFLOP/s (80.4%), **75.0%** of token-facing FLOPs on INT8, max seam **1.058**, walk step 57.4 → **48.9 s**.

**In an interview.** "I moved FLUX's GEMMs to W4A8: per-token INT8 activations on the INT8 tensor cores, where k = 32 of `m16n8k32` is exactly one Q4_K sub-block, and the offset term uses precomputed activation sums. The GEMM got ~1.5× rather than 4× because the per-sub-block rescale makes it issue-bound. The real lesson was quality: a per-layer error study didn't predict which layers broke the image. An ablation on a deterministic end-to-end gate showed the residual-stream writes had to stay bf16."

<details><summary>Check yourself (Task 5b)</summary>

**Q1.** Why can the int-to-float trick use a float base and one subtract, and what limits it?

**A1.** A float of magnitude 2²³–2²⁴ has a mantissa step of exactly 1, so adding a small integer to its bit pattern gives that float plus the integer, exactly. Starting at 2²³ + 2²² leaves room for negative sums. It holds while |sum| < 2²²; ours is at most 60,960.

**Q2.** The study ranked attention-out as the least hurt layer. Why did keeping it bf16 still matter?

**A2.** The study measured each layer's error in isolation. Attention-out writes into the residual stream, which carries its error through every later block, so a small isolated error became a large end-to-end one. Only an end-to-end test (the walk gate) sees that.

</details>

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

> **Remember this:** at M = 1, bandwidth is the only number that matters, and fewer bytes means less time. But measure the cost first: this one was 35 ms, not 0.29 s.

---

### Patterns so far

`Must`

> 1. **Find which limit you're hitting** (roofline). The big GEMMs are compute-bound, so a faster number format helps (W4A8). The GEMV is memory-bound, so fewer bytes help (Q4_K at M = 1). Glue is memory-bound, so fusion helps (Task 6).
> 2. **Keep data in registers.** Decode Q4_K into the fragment the next instruction needs, convert int to float with a bit trick, and reduce with warp shuffles. Never write an intermediate to VRAM.
> 3. **Overlap loads with math.** `cp.async` double buffering, two blocks per SM, and the next chunk requested before this one's math.
> 4. **Check lossy changes against the real output, not layer by layer.** A per-layer error study missed the residual-stream layers; a deterministic end-to-end gate found them.
> 5. **Measure before trusting an estimate.** Task 5 beat its estimate (92% of cuBLAS, not 65–85%), Task 5b fell short (1.5×, not 2–4×), and Task 5c's 0.29 s premise was stale. Re-measure the thing you're about to optimize, and keep the ablation builds.

---

## Learner preferences

Recorded at the 2026-10-24 walkthrough. They apply to every future entry and walkthrough.

- Plain language first, numbers second. Open each idea with a one-line version before the detail.
- One concept per walkthrough.
- When the learner says "skip", reduce that part to a one-line takeaway, and keep the detail only under an "Understand later" label.
