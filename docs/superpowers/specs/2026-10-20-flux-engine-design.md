# vecworld phase 2: FLUX engine design

Date: 2026-10-20 · Status: draft for review · Builds on Dream Roam (branch `roam`, e57ef32)

## 1. Intent

Dream Roam works, but its spots are painted by SD-Turbo, a small model. Phase 2 makes the scenes **much more beautiful** by painting every spot with **FLUX.1-schnell**, a 12B model. That model "shouldn't" fit on the laptop's 4 GB GPU. The **engineering that makes it fit and run fast is hand-written, not a library's offload switch**. The user wants to learn GPU/ML-systems work from it; it ties into their C++/CUDA prep for GPU roles.

The user's own decisions (2026-10-20):
- **Better scenes and the engineering, both.**
- **Quality over delay:** "quality >> delay". A wait at the edge of each spot is acceptable.
- **Write the core ourselves.** Claude writes it and **explains each piece** to the user, rather than the user writing it.
- **FLUX's own style is fine.** The impressionist-brushstroke look is not a requirement.

## 2. Evidence: the spike (2026-10-20, battery-throttled)

The full report is in the session scratchpad (`flux-spike/report.md`); its scripts are in `~/spikes/flux/` on gpu. The laptop was **on battery** (35 W GPU cap, PCIe Gen2 x8), so absolute speeds are lower bounds. The ratios are what to trust.

- **What loaded:**
  - `city96/FLUX.1-schnell-gguf` `flux1-schnell-Q4_K_S.gguf` (6.78 GB);
  - VAE, CLIP-L and tokenizers from the ungated mirror `unsloth/FLUX.1-schnell` (`black-forest-labs` is gated and there's no HF token);
  - T5-XXL from `city96/t5-v1_1-xxl-encoder-gguf` Q5_K_M. It dequantizes to bf16 on load: 48 s, 12.4 GB RAM peak, 27 s per prompt on the CPU.
  - All of this is already downloaded on the laptop.
- **Transformer shape:** 19 double blocks of 191 MB and 38 single blocks of 80 MB at Q4_K, 6.66 GB of blocks in total.
- **Speed with diffusers' group offload:**
  - 512×512, 4 steps: 69.6 s (17.1 s/step).
  - 1024×512: 92 s.
  - Peak VRAM 1.7 GB allocated (3.5 GB per nvidia-smi); peak RAM 9.6 GB.
  - 8-window panorama: about 557 s one window at a time, about 230–300 s batched.
- **Where the time goes:**
  - **Compute is the bottleneck, not data movement.** A step takes 18.7 s with everything resident and 17.1 s streamed, so copies are already hidden.
  - **About half of each step (≈8.3 s) is GGUF Q4_K dequantization**, which runs on every linear call, every step.
  - The GEMMs are about 30 TFLOP per step at 1280 tokens.
- **Hardware ceilings (battery):**
  - host→device: 3.4 GB/s at Gen2, 13.4 GB/s pinned at Gen4;
  - SSD O_DIRECT: 1.9 GB/s;
  - page cache: 6.5 GB/s;
  - bf16 matmul: 5.2 TFLOPS.
  - The 3050 Ti also has INT8 tensor cores.
- **Quality:**
  - FLUX follows the prompt far better: a genuinely haunted, moonlit forest that composes well across 1024×512.
  - Its style is smooth digital painting.
  - The 512 version drew a fake signature.

## 3. Goals

- **Every spot is painted by FLUX.1-schnell.** It runs at 4-bit, as a 2048×512 wrap-around panorama with the same image/depth/meta contract the viewer already uses. Depth, places, signposts, compass, moods, walking and walk-back all keep working.
- **A hand-written inference engine** runs FLUX's transformer (and T5, see §6) on the 4 GB GPU. In build order:
  1. **Fused 4-bit dequant + matrix-multiply CUDA kernel** (W4A16), reading GGUF Q4_K blocks directly and dequantizing in registers. It is built in stages, each measured: naive → tiled → tensor cores (`mma`). This is the main speedup.
  2. **Window batching:** every panorama window goes through each weight pass together, so dequant and copy costs are paid once per step, not once per window.
  3. **Layer streamer:** a pinned-RAM ring buffer, with the next block copied on a side CUDA stream while the current one computes. Hand-written, replacing diffusers' group offload.
  4. **io_uring + O_DIRECT weight loader:** blocks are read straight from the SSD into the pinned ring, bypassing the page cache. This is what lets FLUX and the game share 14 GB of RAM.
- **Every piece ships with an explanation for the user.** That means a short written explainer in the repo (what problem it solves, how it works, the numbers before and after), and Claude walks the user through it when it lands.
- **Every piece is measured against the spike's baseline** with the same harness: per-step time, peak VRAM and RAM, and correctness against the diffusers reference.
- **The waiting state becomes honest:** the fog shows real progress (step and window) while a spot paints.

## 4. Non-goals

- Keeping SD-Turbo in the game, or a "sketch first, refine later" hybrid. The user chose to wait for quality.
- The impressionist house style. FLUX's own style is fine; prompt wording is the builder's call.
- Training, LoRAs and fine-tuning.
- Other models (SD3.5, FLUX.1-dev). The engine should not preclude them, but phase 2 targets FLUX.1-schnell only.
- Multi-GPU, cloud or paid APIs.
- W4A8 (INT8 tensor cores). It's a possible later stretch, not phase 2.

## 5. Anti-goals

- **A FLUX image that's wrong.** Every kernel and engine path must match the diffusers reference numerically (within a tolerance the builder sets and justifies) and visually (same seed, same picture) before it replaces anything.
- **"Engineering" that's really a library call.** diffusers and transformers may load configs and tokenizers, and serve as the reference. The hot path (linear layers, streaming, loading) is ours.
- **Speedups that aren't measured.** Each piece has a before/after number from the same harness, on AC power.
- **A game that breaks while the engine is half-built.** The game keeps working at every commit.

## 6. Constraints

- **Hardware:** RTX 3050 Ti Laptop 4 GB (about 3.7 GB usable; target peak allocated **< 3.6 GB**), 14 GB RAM, 16 threads, NVMe (Intel 670p) with about 22 GB free. Ubuntu laptop `ssh gpu`, repo `~/dev/vecworld`.
- **Power:** all speed numbers are measured **plugged in**. A battery run is labelled as such and never used as a result.
- **RAM budget:** the game server, FLUX, T5, depth and the Qwen dreamer together stay **under about 12 GB RSS**.
  - Today the game holds about 8 GB, mostly Qwen2.5-1.5B in fp32 on the CPU. The FLUX spike peaked at 9.6 GB, and T5 at 12.4 GB.
  - The design must fit, with the SSD loader as the main lever. Running Qwen in bf16 and encoding T5 on the engine are allowed levers; the builder chooses and measures.
- **T5:** encoding a spot's prompt must not add tens of seconds. My call: T5-XXL runs on the same engine (4-bit, streamed, on the GPU), with a per-prompt-text embedding cache. If that proves impractical, the builder may choose another route, with a ledgered ruling and measurements.
- **The GPU is shared** through the existing single-worker GpuQueue. FLUX, depth and anything else run one job at a time. Several friends reaching edges at once wait in turn. That cost is accepted, not hidden.
- **Pre-dreaming** shrinks to the single direction the player faces (my call). Three speculative FLUX panoramas cost minutes of GPU time each.
- **Toolchain:**
  - C++/CUDA as a PyTorch extension (`torch.utils.cpp_extension`), built against the installed torch 2.14.1+cu130 (my call).
  - io_uring through liburing, either vendored or used through the raw syscalls. If a system package is needed (e.g. `liburing-dev`, the CUDA toolkit's nvcc), **the user runs the sudo command**.
- **Commits** carry no Claude attribution. Never log raw LLM output (only parsed phrases). Never restart systemd-logind on gpu.

## 7. How it fits together

- The game's painter stays behind its existing interface. A new FLUX panorama painter implements it: wrap-around windows, differential repaint of the region ahead with the background at partial strength, and VAE tiling.
  - Window size, stride and strengths for FLUX are re-tuned by the builder (FLUX latents are 16-channel and packed 2×2).
  - Visual checks are by eye on a 6-spot walk, as in Task 8b.
- **Build order is baseline first, then replace** (my call):
  1. Re-measure the spike on AC.
  2. Get FLUX painting in the game through diffusers + GGUF as a reference painter, with the RAM budget met.
  3. Swap in each engine piece in turn (kernel, batching, streamer, loader). Each is measured and visually checked against that reference, and each has its explainer.
  - This way the game gets FLUX early, and every engine piece has a ground truth to beat.
- The depth model stays Depth Anything V2 Small. Depth runs after the panorama, in the same queue.

### 7.1 Settled after review (Claude's calls)

- **Walk-step target, separate from the full panorama.** Most spots come from `Painter.step` (roll, dolly, repaint ahead, partial-strength background), not a fresh panorama. The FLUX step gets its own benchmark and a target of **about 90 s or less on AC**. The full panorama target (about 2 min or less) applies to the first spot. The builder may enlarge the walkable distance per spot and adjust the prefetch trigger so each wait buys more walking. Spot spacing, LIMIT and dolly range are open to re-tuning.
- **RAM before the SSD loader exists.** Until engine step 4, the **dreamer LLM (Qwen) may be switched off** and the fallback places table used, freeing about 6 GB, so the diffusers reference painter fits. Qwen in bf16 is also allowed. The about 12 GB limit is **strict from the end of engine step 4**, with Qwen back on.
- **T5 before the engine.** The baseline may use a slow T5 route (the spike's CPU GGUF, encoded once per prompt text and cached; cache misses accepted). It must still fit the RAM rule above; load and free it per use, or run it in a short-lived process. Once the kernel and streamer exist, **T5 runs on them as a consumer**: measured (target: encode in a few seconds), but no separate explainer.
- **When the kernel is done.** All three stages (naive → tiled → tensor-core `mma`) are required, because learning is half the point. If the speed targets are missed with every piece in, **phase 2 is still done**. The numbers are reported honestly, and fallbacks (fewer windows, lower resolution, W4A8) become a later decision for the user.
- **Preemption.** A FLUX job checks for stop and preemption **between denoising steps**, about 5–20 s apart, and reports progress at the same points. A player who turns or another player's demand job waits at most one step, not a whole panorama.
- **Step by step (the user's decision).** The build **pauses after each engine piece** (each kernel stage counts as a piece). Claude walks the user through the piece and its before/after numbers, and the next piece starts only after that walkthrough. The plan's milestones are cut at these checkpoints.
- **Toolchain detail.** nvcc on PATH is 12.4, but torch is cu130. Build the extension with `CUDA_HOME=/usr/local/cuda-13.4`.

## 8. Left to the builder

- Module layout (for example `vecworld/engine/` with `csrc/`).
- Kernel tile shapes and launch parameters, how deep the streamer's ring is, and io_uring queue depth and buffer alignment.
- How T5 is quantized and run, within the budget above.
- FLUX panorama window and stride parameters, strengths and prompt wording.
- Tolerances for the numerical checks.
- The progress protocol between server and viewer.
- The explainer format and where it lives.
- The test strategy, following the existing unit/gpu/ui markers, plus the kernel correctness tests.

## 9. Success looks like

- **Beauty:** the user walks a 6-spot journey and finds the FLUX spots clearly more beautiful than SD-Turbo's (judged by eye, side by side).
- **Speed:**
  - each engine piece shows a measured gain on AC over the diffusers reference;
  - the whole panorama time lands at or under about 2 minutes.
  - These are my estimates from the spike: with fused kernels and window batching, about 100 s per panorama on AC.
- **Memory:** the game, FLUX and the rest run together in 14 GB of RAM and 4 GB of VRAM without swapping.
- **Learning:** the user can explain each piece: why dequant dominated, how the fused kernel avoids it, how copy/compute overlap works, and why O_DIRECT + io_uring.
