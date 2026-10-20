# FLUX Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paint every Dream Roam spot with FLUX.1-schnell (4-bit), using a hand-written engine of a fused Q4_K CUDA kernel, window batching, a pinned layer streamer and an io_uring SSD loader. The user is walked through each piece before the next one starts.

**Architecture:**
- A `FluxPainter` implements the existing `Painter` protocol. It starts on diffusers + GGUF as the reference (Task 1).
- Engine pieces then replace its hot path one at a time, each checked against that reference for numbers and pictures.
- The engine lives in `src/vecworld/engine/`. CUDA/C++ sources are in `src/vecworld/engine/csrc/`, JIT-built with `torch.utils.cpp_extension.load`.

**Tech Stack:** Python 3.12, torch 2.14.1+cu130, CUDA 13.4 (`CUDA_HOME=/usr/local/cuda-13.4`), diffusers 0.41 (reference only), the `gguf` package (parsing and reference dequant), liburing or raw io_uring syscalls, FastAPI, three.js viewer.

**Spec:** `docs/superpowers/specs/2026-10-20-flux-engine-design.md`. The spike evidence is in the spec's §2, with scripts in `~/spikes/flux/` on gpu.

## Global Constraints

- GPU: RTX 3050 Ti 4 GB; peak allocated **< 3.6 GB**. RAM 14 GB; the whole server stays **under about 12 GB RSS** from the end of Task 8 (the dreamer may be off before then; spec §7.1).
- Speed numbers count **only on AC power**. The bench refuses to label a battery run as a result (`/sys/class/power_supply/*/online == 0` → `"power": "battery"`).
- Models (already downloaded on gpu):
  - `city96/FLUX.1-schnell-gguf` `flux1-schnell-Q4_K_S.gguf`;
  - `unsloth/FLUX.1-schnell` (VAE, CLIP-L, tokenizers, config);
  - `city96/t5-v1_1-xxl-encoder-gguf` `t5-v1_1-xxl-encoder-Q5_K_M.gguf`.
- Build the extension with `CUDA_HOME=/usr/local/cuda-13.4`, not the 12.4 nvcc on PATH. **The user runs any sudo command** (e.g. `apt install liburing-dev`).
- Every engine path must match the diffusers reference before it replaces it. Kernels are checked against `gguf` dequant + `torch.matmul`; images use the same seed against the reference, and are judged by eye.
- **Checkpoints:** Tasks 3, 4, 5, 6, 7 and 8 each end in a **user walkthrough**. The controller stops, explains the piece with its explainer and numbers, and waits for the user before dispatching the next task.
- Each engine task writes an explainer at `docs/engine/NN-<name>.md`: the problem, how it works (with a diagram in ASCII if useful), before/after numbers from `scripts/bench_flux.py`, and what to read next.
- A FLUX job calls `check_stop()` and reports progress **between denoising steps**.
- Commits carry no Claude attribution or Co-Authored-By. Never log raw LLM output. Never restart systemd-logind. Don't kill processes that aren't ours.
- The game keeps working at every commit. `VECWORLD_PAINTER=turbo` stays available until Task 9.

## Review Focus

1. **Tensors that aren't Q4_K.** Q4_K_S files mix in other GGUF types (and bf16 biases and norms), and some `in_features` aren't multiples of 256. The swap must leave those on the reference path, never mis-decode them. Pinned in Task 3: `test_swap_skips_non_q4k`.
2. **Matrix shapes that aren't tile multiples:** M = tokens × windows (e.g. 1280 + 512 text, or a batch of 3). Every kernel stage must be exact on ragged edges. Pinned in Tasks 3–5: `test_q4k_gemm_ragged[M=1,7,333,1792]`.
3. **A job stopped mid-step** (a player turns, a demand job arrives). The painter stops at the next step boundary, the streamer ring and pinned buffers are reused rather than leaked, and the next job is correct. Pinned in Task 7: `test_stream_stop_then_reuse`.
4. **Short or failed SSD reads, or a misaligned O_DIRECT offset or length.** These give a clear error, and never silently corrupt weights. Pinned in Task 8: `test_uring_short_read_raises`, `test_uring_unaligned_tail`.
5. **Out of memory during a FLUX job.** The existing OOM → 503 path still works, and the next request succeeds. Pinned in Task 2: `test_flux_oom_returns_503_then_recovers` (fake painter that raises `torch.cuda.OutOfMemoryError` once).

---

### Task 0: AC baseline and the bench harness

**Files:**
- Create: `scripts/bench_flux.py`, `bench/.gitkeep`, `docs/engine/00-baseline.md`
- Modify: `pyproject.toml` (add `gguf`)

**Interfaces:**
- Produces:
  - `scripts/bench_flux.py --painter {reference,engine} --what {tile512,pano,step,blocks,bw} --out bench/<name>.json`, which writes `{"power": "ac"|"battery", "git": sha, "results": {...}}`;
  - `bench/baseline.json`.

- [ ] **Step 1:** Port the spike's `flux_run.py`, `blocks.py` and `bw2.py` (from `~/spikes/flux/`) into `scripts/bench_flux.py` with the CLI above. Add the power check from Global Constraints.
- [ ] **Step 2:** On AC, run `tile512`, `pano` (8 windows, batched 2 and 4), `blocks` (per-block resident compute and dequant share) and `bw` (pinned/pageable H2D, O_DIRECT SSD). Expected: `"power": "ac"`, and numbers at or below the spike's battery figures (17.1 s/step at 512).
- [ ] **Step 3:** Write `docs/engine/00-baseline.md`: a table comparing the spike (battery) with AC, and the time breakdown (dequant / GEMM / attention / copy).
- [ ] **Step 4:** Commit: `Measure the FLUX baseline on AC`.

### Task 1: FluxPainter on the diffusers reference

**Files:**
- Create: `src/vecworld/flux/__init__.py`, `src/vecworld/flux/painter.py`, `src/vecworld/flux/text.py`, `tests/test_flux_painter.py`, `tests/test_flux_painter_gpu.py`
- Modify: `src/vecworld/config.py` (FLUX_* constants), `src/vecworld/__main__.py` (`VECWORLD_PAINTER`, `VECWORLD_DREAMER`)

**Interfaces:**
- Consumes: `Painter`, `Prompt` (`vecworld.dream.painter`), `check_stop` (`vecworld.dream.gpu`), `pano.windows`, `focus_mask` and `wrap_cols` (`vecworld.dream.pano`).
- Produces:
  - `FluxPainter(progress: Callable[[int, int], None] | None = None)`, implementing `panorama(base, seed)` and `step(prev, base, focus, focus_at, seed, bg_strength=None, focus_half=None)` and returning `PANO_W×PANO_H` RGB.
  - `FluxPainter.transformer`, the diffusers `FluxTransformer2DModel` that later tasks patch.
  - `T5Text(cache=64).encode(text: str) -> torch.Tensor` returns `[1, 256, 4096]` bf16 on the CPU. The model is loaded per call and freed after it.
  - Config: `FLUX_WIN`, `FLUX_STRIDE` (in packed latent tokens), `FLUX_STEPS = 4`, `FLUX_BG_STRENGTH`.

- [ ] **Step 1:** Write `tests/test_flux_painter.py` (CPU, fake transformer and VAE):
  - `test_windows_wrap_and_cover`: every packed latent column is covered, and windows wrap across the seam.
  - `test_progress_called_per_step`: `progress(k, total)` is called once per denoising step, with total = steps × window batches.
  - `test_check_stop_between_steps`: a `Stopped` raised by `check_stop` ends the job before the next step.
- [ ] **Step 2:** Run `uv run pytest tests/test_flux_painter.py -q`. Expected: FAIL (`vecworld.flux` does not exist).
- [ ] **Step 3:** Implement `FluxPainter` with wrap-around MultiDiffusion on FLUX's packed 2×2 latents, with the approach of `TurboPainter`:
  - windows averaged per step;
  - `step` runs a differential repaint ahead, with the background at `FLUX_BG_STRENGTH`;
  - VAE tiling with crossfade.
  - Transformer: GGUF via `GGUFQuantizationConfig`, with diffusers group offload for now.
  - Text: CLIP-L pooled on the GPU, and `T5Text` on the CPU.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Write `tests/test_flux_painter_gpu.py` (`@pytest.mark.gpu`): `test_flux_panorama_seamless` (the seam column difference stays under the mean column difference × 1.5), and `test_flux_step_keeps_background` (outside the focus, SSIM against `prev` > 0.6). Run them on AC. Expected: PASS.
- [ ] **Step 6:** In `__main__`, `VECWORLD_PAINTER=flux` (the default from now on) builds `FluxPainter`, and `VECWORLD_DREAMER=off` uses the fallback places with no Qwen. Measure RSS with the game running; it must not swap.
- [ ] **Step 7:** Commit: `Paint spots with FLUX on the diffusers reference`.

### Task 2: Game around a slow painter

**Files:**
- Modify: `src/vecworld/roam/service.py`, `src/vecworld/api/app.py`, `src/vecworld/config.py`, `web/app.js`, `web/style.css`
- Test: `tests/test_roam_service.py`, `tests/test_api.py`, `tests/test_ui.py`

**Interfaces:**
- Consumes: `FluxPainter(progress=...)`.
- Produces:
  - `RoamService.progress(sid) -> dict | None`, as `{"done": int, "total": int}`;
  - `GET /api/session/{sid}/progress` returning it, or `{}`;
  - config `PREDREAM_HEADINGS` (FLUX: `(0,)`; Turbo keeps `(0, -60, 60)`).

- [ ] **Step 1:** Write the failing tests:
  - `test_predream_only_facing_direction_for_slow_painter`;
  - `test_progress_reported_while_building`;
  - `test_flux_oom_returns_503_then_recovers` (Review Focus 5);
  - ui: `test_fog_shows_progress` (the fog text shows "painting… 3/8").
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement:
  - pre-dream headings from config;
  - a progress hook from the painter into the session;
  - the endpoint;
  - the viewer polling it every 1 s while in fog.
  - Re-tune spot spacing for slow spots: `WALK_LIMIT` and the dolly range may grow (spec §7.1). Record the new values and why in the commit body.
- [ ] **Step 4:** Run the full suite with the server stopped (`-m "gpu or not gpu or ui"`). Expected: green.
- [ ] **Step 5:** Real 6-spot walk on AC. Contact sheet against the SD-Turbo sheet (`roam-8b/t8b/sheet_final.png`); time per step and per first panorama; look at it.
- [ ] **Step 6:** Commit: `Play Dream Roam on FLUX: one-way pre-dream and painting progress`.

### Task 3: Q4_K kernel, stage 1 (naive) — CHECKPOINT

**Files:**
- Create:
  - `src/vecworld/engine/__init__.py`;
  - `engine/ext.py`;
  - `engine/q4k.py`;
  - `engine/swap.py`;
  - `engine/csrc/q4k.cu`, `engine/csrc/bindings.cpp`;
  - `tests/test_q4k.py` (gpu);
  - `docs/engine/01-q4k-naive.md`.

**Interfaces:**
- Produces:
  - `load_ext() -> module`, built once and cached, with `CUDA_HOME` set;
  - `read_gguf(path) -> dict[str, GGUFTensor]` (name, gguf type, shape, raw bytes);
  - `Q4KLinear(weight: GGUFTensor, bias: torch.Tensor | None, kernel: str)`, an `nn.Module` with `forward(x: bf16 [..., in]) -> bf16 [..., out]` and `kernel ∈ {"naive","tiled","mma"}`;
  - `swap_linears(model, gguf: dict, kernel: str) -> int`, which replaces only Q4_K linears and returns how many it replaced.

- [ ] **Step 1:** Tests:
  - `test_q4k_dequant_matches_gguf`: our device dequant equals `gguf.quants.dequantize` exactly, on the real FLUX tensors;
  - `test_q4k_gemm_matches_reference`: `max|ours − x @ deq(W)ᵀ| / max|ref| < 2e-2` (bf16);
  - `test_q4k_gemm_ragged[M=1,7,333,1792]`;
  - `test_swap_skips_non_q4k` (Review Focus 1);
  - `test_flux_image_matches_reference`: same seed at 512, PSNR against the reference ≥ 30 dB.
- [ ] **Step 2:** Run them. Expected: FAIL (no `vecworld.engine`).
- [ ] **Step 3:** Implement the naive kernel: one thread per output element, decoding the Q4_K super-block (scale, min, 6-bit sub-scales) in registers, then a dot product over `in`. Bind it, write `Q4KLinear`, and swap it into `FluxPainter.transformer` behind `VECWORLD_KERNEL=naive`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Run `bench_flux.py --painter engine --what tile512,blocks` on AC. Write the explainer: the Q4_K layout diagram, why dequant-per-call cost about 8 s, and the naive kernel's numbers (it may well be slower than the reference; say so honestly).
- [ ] **Step 6:** Commit: `Run FLUX linears on a hand-written Q4_K kernel (naive)`. **Stop: user walkthrough.**

### Task 4: Q4_K kernel, stage 2 (tiled) — CHECKPOINT

**Files:** Modify `engine/csrc/q4k.cu`, `engine/q4k.py`. Test: `tests/test_q4k.py` (parametrize `kernel`). Create `docs/engine/02-q4k-tiled.md`.

- [ ] **Step 1:** Parametrize the Task 3 kernel tests over `kernel ∈ {"naive","tiled"}`. Run them. Expected: the `tiled` cases FAIL.
- [ ] **Step 2:** Implement a tiled kernel: blocks of output tiles; `x` tiles and decoded weight tiles staged in shared memory; each weight byte decoded once per tile, not once per output; coalesced loads. Tile shape is the implementer's call, justified by measurement.
- [ ] **Step 3:** Run the tests. Expected: PASS. Then bench on AC.
- [ ] **Step 4:** Write the explainer: memory hierarchy, coalescing and reuse, numbers against the naive kernel and the reference. Commit: `Tile the Q4_K kernel through shared memory`. **Stop: user walkthrough.**

### Task 5: Q4_K kernel, stage 3 (tensor cores) — CHECKPOINT

**Files:** Modify `engine/csrc/q4k.cu`, `engine/q4k.py`. Test: `tests/test_q4k.py` (add `"mma"`). Create `docs/engine/03-q4k-mma.md`.

- [ ] **Step 1:** Add `"mma"` to the kernel parametrization. Run it. Expected: the `mma` cases FAIL.
- [ ] **Step 2:** Implement a bf16 `mma.sync` (m16n8k16) kernel: decode Q4_K to bf16 fragments in registers, accumulate in fp32, and double-buffer shared memory with `cp.async`. Make `mma` the default kernel.
- [ ] **Step 3:** Run the tests. Expected: PASS. Then bench on AC. The target from the spike: a step at 512 well below the reference, with dequant no longer dominant in `blocks`.
- [ ] **Step 4:** Write the explainer: warps, fragments, `mma`, why decoding in registers kills the 8 s, and the numbers for all three stages. Commit: `Feed tensor cores straight from Q4_K weights`. **Stop: user walkthrough.**

### Task 6: Window batching — CHECKPOINT

**Files:** Modify `src/vecworld/flux/painter.py`, `config.py` (`FLUX_BATCH`). Test: `tests/test_flux_painter.py`, `tests/test_flux_painter_gpu.py`. Create `docs/engine/04-batching.md`.

- [ ] **Step 1:** Tests:
  - `test_batched_equals_unbatched` (CPU fake: the same window outputs whatever the batch size);
  - gpu `test_flux_pano_batched_matches`: PSNR ≥ 35 dB against batch 1, same seed.
  Run them. Expected: FAIL (no `FLUX_BATCH`).
- [ ] **Step 2:** Implement: all active windows of a step go through each transformer pass as one batch of up to `FLUX_BATCH`, the largest that keeps peak allocated < 3.6 GB (measure it). Per-window text embeddings are stacked.
- [ ] **Step 3:** Run the tests. Expected: PASS. Bench `pano` and `step` on AC.
- [ ] **Step 4:** Write the explainer: why the weights are moved and decoded once per pass, the activation memory against batch size, and the numbers. Commit: `Batch every panorama window through each weight pass`. **Stop: user walkthrough.**

### Task 7: Layer streamer — CHECKPOINT

**Files:** Create `engine/stream.py`, `tests/test_stream.py` (gpu), `docs/engine/05-streamer.md`. Modify `flux/painter.py`, which drops diffusers group offload.

**Interfaces:**
- Produces: `BlockStreamer(blocks: list[nn.Module], source: Callable[[int], memoryview | torch.Tensor], ring: int = 2)`, with `install()` registering pre-forward hooks and `close()`. Block weights live in a pinned host arena. Copies to a ring of device slots run on a side stream, synchronised by CUDA events.

- [ ] **Step 1:** Tests:
  - `test_stream_matches_resident` (block outputs identical to resident weights);
  - `test_stream_overlaps` (step time ≤ 1.1 × the compute-only time on AC);
  - `test_stream_stop_then_reuse` (Review Focus 3: a `Stopped` mid-step, then a second job uses the same ring correctly, and pinned and device memory are unchanged afterwards).
  Run them. Expected: FAIL.
- [ ] **Step 2:** Implement the pinned arena (one `cudaHostAlloc` / `torch.empty(pin_memory=True)`) and a device ring of `ring` slots sized to the largest block. Block *i+1* is copied while block *i* computes. `Q4KLinear` views its weights into the current slot.
- [ ] **Step 3:** Run the tests. Expected: PASS. Bench on AC against group offload.
- [ ] **Step 4:** Write the explainer: pinned vs pageable memory, DMA, streams and events, a timeline diagram of copy and compute, and the numbers. Commit: `Stream FLUX blocks through a pinned ring`. **Stop: user walkthrough.**

### Task 8: io_uring + O_DIRECT loader — CHECKPOINT

**Files:** Create `engine/csrc/uring.cpp`, `engine/uring.py`, `tests/test_uring.py`, `docs/engine/06-io-uring.md`. Modify `engine/stream.py` (source = SSD), `__main__.py` (dreamer back on by default).

**Interfaces:**
- Produces: `DirectReader(path: str, queue_depth: int = 8)`, with `read_into(offset: int, nbytes: int, dst: torch.Tensor) -> None` (pinned, 4096-aligned) and `close()`; and `ssd_source(gguf_path, layout) -> Callable[[int], torch.Tensor]` for `BlockStreamer`. Block bytes are read from the GGUF file, and no weights stay resident in RAM.

- [ ] **Step 1:** Tests:
  - `test_uring_reads_exact_bytes` (matches `os.pread`);
  - `test_uring_unaligned_tail` (a block whose start or length isn't 4096-aligned reads correctly via an aligned superset);
  - `test_uring_short_read_raises` (a truncated temp file raises `IOError` naming the offset);
  - gpu `test_flux_from_ssd_matches` (image PSNR ≥ 40 dB against the RAM-sourced streamer).
  Run them. Expected: FAIL. If liburing headers are missing, stop and ask the user to run `sudo apt install liburing-dev`, or use the raw syscalls.
- [ ] **Step 2:** Implement submission and completion rings, O_DIRECT reads straight into the pinned arena slots, and prefetch of block *i+2* while *i+1* copies to the GPU.
- [ ] **Step 3:** Run the tests. Expected: PASS. On AC, with the full game running (Qwen on), measure the server's RSS (target < 12 GB, no swap) and the step time.
- [ ] **Step 4:** Write the explainer: page cache vs O_DIRECT, io_uring's two rings, alignment, the three-stage pipeline (SSD → pinned → GPU) and the numbers. Commit: `Read FLUX blocks from the SSD with io_uring`. **Stop: user walkthrough.**

### Task 9: T5 on the engine, cleanup and end to end

**Files:** Modify `src/vecworld/flux/text.py`, `__main__.py`, `README.md`, spec §2 (measured numbers). Delete `TurboPainter`'s default use (keep the class only if tests still need it). Create `docs/engine/07-t5-and-results.md`.

- [ ] **Step 1:** Test: gpu `test_t5_engine_matches_cpu` (embeddings cosine ≥ 0.999 against the Task 1 CPU path) and `test_t5_encode_fast` (≤ 5 s on AC). Run them. Expected: FAIL.
- [ ] **Step 2:** Run T5-XXL's linears through `Q4KLinear` (or the matching GGUF type, falling back to the reference for non-Q4_K) with `BlockStreamer` + `DirectReader`. The CPU path stays as a fallback behind a flag.
- [ ] **Step 3:** Run the full suite with the server stopped. Expected: green. Then e2e on AC: the 3 reference journeys (haunted forest + night, desert oasis, snowy mountains), 6 spots each. Measure first panorama, step, fog, peak VRAM/RAM and T5 time. Save contact sheets against SD-Turbo and look at them.
- [ ] **Step 4:** Write the results explainer (every piece's before/after in one table). Commit: `Run T5 on the engine; FLUX Dream Roam end to end`. Leave the server running on the final code.
