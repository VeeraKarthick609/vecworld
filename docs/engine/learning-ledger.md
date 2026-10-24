# Engine learning ledger

A running study log for the FLUX engine work. Each entry explains one idea with this project's real numbers, so it can be revised before an interview. New entries go at the bottom; add a line to the index when you do.

**Conventions.** "Measured" numbers come from `bench/baseline.json` and `docs/engine/00-baseline.md` (on AC, 2026-10-20, git 895c75e) unless labelled otherwise. "(est.)" marks a back-of-envelope figure, with its assumptions stated next to it. "(battery)" marks a number from the 2026-10-20 spike, which ran throttled (35 W, PCIe Gen2). The spike report itself is no longer on disk; its numbers survive in the spec, `docs/superpowers/specs/2026-10-20-flux-engine-design.md` §2.

## Ledger index

| # | Date | Entry | Related plan tasks |
|---|---|---|---|
| 1 | 2026-10-24 | [Making FLUX faster: ops/sec on a 4 GB GPU](#entry-1-making-flux-faster-opssec-on-a-4-gb-gpu) | 3, 4, 5, 6, 7, 8 |

---

## Entry 1: Making FLUX faster: ops/sec on a 4 GB GPU

*2026-10-24. Study time: about 45 minutes. Prerequisites: matrix multiply, a rough idea of what a GPU thread and warp are.*

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

RTX 3050 Ti Laptop GPU (GA107, Ampere), in the laptop `mrjarvis`.

| Property | Value | Source |
|---|---|---|
| SMs (streaming multiprocessors, the GPU's "cores") | 20 | spec sheet |
| FP32 CUDA cores | 2560 (128 per SM) | spec sheet |
| Tensor cores (3rd gen) | 80 (4 per SM) | spec sheet |
| VRAM | 4 GB GDDR6, 128-bit bus | spec sheet |
| VRAM bandwidth | about 192 GB/s (est.: 128 bit × 12 Gbps ÷ 8; not measured here) | spec sheet, est. |
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

- A **FLOP** is one floating-point operation (an add or a multiply). It is a count of **work**.
- **FLOP/s** (FLOPS) is a **rate**: work per second. TFLOPS means 10¹² FLOP/s.
- A multiply-accumulate (`c += a*b`, an "FMA" or "MAC") counts as **2 FLOPs**. So multiplying an `M×K` matrix by a `K×N` matrix costs `2·M·K·N` FLOPs.

The core relation is time = work ÷ rate. If a FLUX step costs W FLOPs and we run at R FLOP/s, the step takes W/R seconds. You can make it faster in only two ways: do less work (smaller W: caching, skipping) or do it at a higher rate (bigger R: better kernels, lower precision, less waiting).

### 1.2 What peak throughput this GPU has

**Tensor cores** are units inside each SM that execute a small matrix multiply (for example 16×8×16) as one instruction. They are much faster than the ordinary FP32 "CUDA cores", but only for matrix shapes and specific data types.

| Data type / mode | Peak (est.) | How estimated |
|---|---|---|
| FP32 on CUDA cores (no tensor cores) | about 9.6 TFLOPS (est.) | 2560 cores × 2 FLOP × ~1.87 GHz |
| bf16 in, fp32 accumulate (tensor cores) | **19.1 TFLOPS measured** | baseline 4096² matmul |
| fp16 in, fp16 accumulate (tensor cores) | about 38 TFLOPS (est.) | GeForce Ampere runs fp16-accumulate at 2× the fp32-accumulate rate |
| INT8 in, int32 accumulate (tensor cores) | about 76 TOPS (est.) | GeForce Ampere INT8 rate is 4× the bf16/fp32-acc rate (ratio from NVIDIA's GA102 whitepaper; 3090: 71 vs 284 dense) |

A note on the 19.1. Ampere GeForce parts do 512 bf16 FLOP per SM per clock with fp32 accumulate (est., from the GA102 ratio). 19.1 TFLOPS ÷ (20 SMs × 512) ≈ 1.87 GHz, and real matmuls never hit 100% of the instruction rate. So 19.1 means the GPU was boosting at or above about 1.9 GHz on AC. That is plausible at 90 W, but it is at the top of the range. **Treat 19.1 as our practical ceiling**, and log the SM clock (`nvidia-smi --query-gpu=clocks.sm`) next time it is measured. On battery the same matmul gave 5.2 TFLOPS (battery), which is why only AC numbers count.

### 1.3 Memory bandwidth

**Memory bandwidth** is how many bytes per second can move between VRAM and the SMs: about 192 GB/s here (est.). Other links in our chain are much slower:

```
SSD --2.0-3.0 GB/s--> RAM (pinned) --13.5 GB/s--> VRAM --~192 GB/s (est.)--> SMs (registers, shared mem)
```

Every byte of weight a kernel needs must cross the VRAM→SM link at least once. For very large weight sets it must also cross the PCIe link (13.5 GB/s) and, from Task 8 on, the SSD link (2–3 GB/s).

### 1.4 Arithmetic intensity and the roofline

**Arithmetic intensity (AI)** = FLOPs done ÷ bytes moved from memory, in FLOP/byte. It says how much work you get per byte you fetch.

The **roofline model** says a kernel's best possible rate is:

```
attainable FLOP/s = min( peak FLOP/s ,  AI × bandwidth )
```

- If AI is low, you are limited by the sloped part (bandwidth). That is **memory-bound**: the SMs wait for data.
- If AI is high, you hit the flat part (peak compute). That is **compute-bound**: the data arrives faster than you can use it.
- The **ridge point** where they meet is peak ÷ bandwidth = 19.1 TFLOPS ÷ 192 GB/s ≈ **100 FLOP/byte (est.)**. A kernel needs more than about 100 FLOP per byte to use the tensor cores fully.

**AI of a GEMM** (General Matrix Multiply). An `M×K` activation times a `K×N` weight costs `2MKN` FLOPs. It moves about `K·N·b_w` bytes of weight plus `M·K·2 + M·N·2` bytes of bf16 activations in and out, where `b_w` is bytes per weight. When the weight dominates, AI ≈ `2M / b_w`. **The token count M sets the intensity.** Our cases:

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

Before optimising a kernel, find which side of the ridge it sits on:

- **Compute-bound** (big GEMMs, attention): use faster math units (tensor cores, lower precision like INT8) and keep them fed (tiling, double buffering).
- **Memory-bound** (dequant, norms, GEMVs): move fewer bytes. Fuse kernels so intermediate tensors never go to VRAM, use smaller formats, and read each byte once.

Making a memory-bound kernel do fewer FLOPs gains nothing. Making a compute-bound kernel read fewer bytes gains nothing either. That is why the roofline matters.

### 1.6 FLUX's FLOPs per step, from the real shapes

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

Today's 512 px step, measured, weights resident (2.59 s), plus 0.15 s offload overhead = 2.74 s:

| Part | Time | Share |
|---|---|---|
| GGUF Q4_K dequant | 1.08 s | 42% |
| Attention (SDPA) | 0.06 s | 2.5% |
| GEMM + glue | 1.44 s | 56% |
| of which GEMM (est.) | ~1.05 s | assumes cuBLAS at ~16 TFLOPS on M=1280 shapes and ~12 on the M=256 text GEMMs |
| of which glue (est.) | ~0.4 s (0.25–0.55) | the remainder |

### Technique 1: Fused Q4_K dequant + tensor-core GEMM (W4A16)

*W4A16 means 4-bit weights, 16-bit activations.* Plan Tasks 3 (naive), 4 (tiled), 5 (`mma`).

**The problem, in our numbers.** diffusers' GGUF path keeps weights in Q4_K on the GPU. On every linear call it unpacks the *whole* weight to a bf16 tensor in VRAM with a chain of PyTorch ops, then hands that to cuBLAS. That takes **1.08 s per step, 42% of the step**. The minimum traffic for that unpack is reading 6.66 GB of Q4_K and writing 23.7 GB of bf16 (11.83 B × 2 B): 30.3 GB ÷ 192 GB/s ≈ 0.16 s (est.). The measured 1.08 s is about 7× that, because each PyTorch op writes its intermediate result to VRAM and the next op reads it back. Then the GEMM reads the 23.7 GB again. Every step, for every weight, even though the weights never change.

**The Q4_K super-block.** Q4_K (llama.cpp's "k-quant") stores 256 weights in a 144-byte super-block, 4.5 bits per weight:

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

*W4A8 means 4-bit weights, 8-bit integer activations.* Not in the plan.

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

Plan Task 6. A panorama is painted as several overlapping windows (8 for the full pano). Batching runs several windows through each transformer pass together.

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

Not in the plan.

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

Not in the plan.

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

Not in the plan as a task.

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

**Decision 1: quality policy for speedups.**

- **A. Quality fixed.** Every speedup must produce the same picture, within the PSNR gates already in the plan. Do Techniques 1, 3, 4 and 6. Add 2 only if it passes a quality gate (same seed, PSNR against W4A16, plus your eye on contact sheets). Expected: ~1.4 s per window-step stacked, 1.6–2.0 s realistic (est.); ~1.0 s if W4A8 passes.
- **B. Speed first.** Accept measured, visible quality losses. Do all six, including block caching, with thresholds tuned on contact sheets. Expected: ~0.9 s per window-step stacked (est.), with a lower and less predictable quality floor.

What would decide it: whether FLUX's quality (the reason for phase 2, see spec §1) survives W4A8 and caching on your three reference journeys. A quick W4A8 PSNR study per layer, before writing the full kernel, would settle Technique 2 cheaply.

**Decision 2: the step recipe, pending from Task 2b.** The A/B contact sheets are yours to judge (`FLUX_STEP_RECIPE`, default `"sharp"`).

- **`sharp`**: graded release, focus 1.0 (the band ahead painted from scratch), sides 0.75, back 0.9, feather 0.25. It keeps the layout of where you came from. After the fix round its walk seam scores are ≤ 1.08 (limit 1.1); soft light-shaft veils remain (spot 3).
- **`fresh`**: a new panorama from the step's prompt and seed every step; only the prompt blend carries the walk on. It has no seams to manage, but no visual continuity either, and it is exempt from the continuity gate.

It matters for speed too: the recipe decides how many windows and denoising passes a walk step runs, and so whether walk steps are single-window (I/O-bound after Task 8, Technique 3 can't help) or batch-friendly. Before choosing, it is worth logging passes and windows per walk step for each recipe.

---

*Next entry suggestion: after Task 3's bench, record the naive kernel's numbers against these predictions and note where the estimates were wrong.*
