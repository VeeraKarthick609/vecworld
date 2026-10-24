# FLUX Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paint every Dream Roam spot with FLUX.1-schnell (4-bit), using a hand-written engine of a fused Q4_K CUDA kernel, window batching, a pinned layer streamer and an io_uring SSD loader. The user is walked through each piece before the next one starts.

**Goal B, speed first (2026-10-24, spec §7.2):** from Task 3 on, each engine task is judged by **step time and achieved TFLOP/s**. W4A8 on the INT8 tensor cores, glue fusion, CUDA graphs and block caching are in scope. Target: a 512 step from **~2.74 s to ~1 s before block caching**. Lossy speedups (W4A8, block caching) ship only if a `sharp` walk still passes `walk_sheet.py --check`.

**Architecture:**
- A `FluxPainter` implements the existing `Painter` protocol. It starts on diffusers + GGUF as the reference (Task 1).
- Engine pieces then replace its hot path one at a time, each checked against that reference for numbers and pictures (lossy pieces: against the walk gate, Global Constraints).
- The engine lives in `src/vecworld/engine/`. CUDA/C++ sources are in `src/vecworld/engine/csrc/`, JIT-built with `torch.utils.cpp_extension.load`.
- `VECWORLD_KERNEL` ∈ {`reference`, `naive`, `tiled`, `mma`, `w4a8`} picks the linear kernel; each later engine piece has an env switch (`VECWORLD_FUSE`, `VECWORLD_GRAPHS`, `FLUX_CACHE_THRESHOLD`) so the bench can A/B it.

**Tech Stack:** Python 3.12, torch 2.14.1+cu130, CUDA 13.4 (`CUDA_HOME=/usr/local/cuda-13.4`), diffusers 0.41 (reference only), the `gguf` package (parsing and reference dequant), liburing or raw io_uring syscalls, FastAPI, three.js viewer.

**Spec:** `docs/superpowers/specs/2026-10-20-flux-engine-design.md`; §7.2 (Goal B) supersedes the step target in §7.1. The spike evidence is in the spec's §2, with scripts in `~/spikes/flux/` on gpu. Step-time breakdown and technique estimates: `docs/engine/learning-ledger.md` (Part 2, §1.4, §1.6).

## Global Constraints

- GPU: RTX 3050 Ti 4 GB; peak allocated **< 3.6 GB**. RAM 14 GB; the whole server stays **under about 12 GB RSS** from the end of Task 10 (the dreamer may be off before then; spec §7.1).
- Speed numbers count **only on AC power**. The bench refuses to label a battery run as a result (`/sys/class/power_supply/*/online == 0` → `"power": "battery"`).
- Models (already downloaded on gpu):
  - `city96/FLUX.1-schnell-gguf` `flux1-schnell-Q4_K_S.gguf`;
  - `unsloth/FLUX.1-schnell` (VAE, CLIP-L, tokenizers, config);
  - `city96/t5-v1_1-xxl-encoder-gguf` `t5-v1_1-xxl-encoder-Q5_K_M.gguf`.
- Build the extension with `CUDA_HOME=/usr/local/cuda-13.4`, not the 12.4 nvcc on PATH. **The user runs any sudo command** (e.g. `apt install liburing-dev`).
- Every lossless engine path (Tasks 3, 4, 5, 5c, 6, 7, 9, 10) must match the diffusers reference before it replaces it. Kernels are checked against `gguf` dequant + `torch.matmul`; images use the same seed against the reference, and are judged by eye.
- **Speed metric (Goal B).** "Step s" = `tile512.s_per_pass_median` from `scripts/bench_flux.py --painter engine --what tile512` on AC: one 512×512 window, 1024 image + 256 text tokens, one transformer pass per denoising step. **Achieved TFLOP/s** = `flop_per_step(1024, 256)` ÷ step s, where `flop_per_step(n_img, n_txt) = 114 · 113,246,208 · N + 700,416 · N²`, N = n_img + n_txt (ledger §1.6: 17.67 TFLOP at N = 1280). It is the nominal count even when work runs in INT8 or is skipped, so label it **bf16-equivalent** from Task 5b on. Also report % of the measured 19.1 TFLOP/s bf16 peak. Baseline (`bench/baseline.json`): 2.74 s, 6.45 TFLOP/s, 34%. Each engine task also reports the walk step (`--what step`, `step.s`, a `sharp` walk step's wall time), not gated.
- **Bench row per engine task.** Each engine task writes `bench/tN-<name>.json` (tile512) and `bench/tN-<name>-step.json` (walk step), and appends one row to `docs/engine/speed.md`: task, flags, step s, achieved TFLOP/s, % of bf16 peak, walk step s, peak allocated GB, quality-gate result, git sha.
- **Quality gate (Goal B).** After Tasks 5b, 6 and 8, a 6-spot `sharp` walk must pass the seam check, and the user judges its contact sheet at the checkpoint:
  ```
  uv run python scripts/walk_sheet.py --recipe sharp --theme "haunted forest" --seed 7 --spots 6 \
      --headings 0,0,60,60,300,0 --walked 9 --out /tmp/walk/tN
  uv run python scripts/walk_sheet.py --check /tmp/walk/tN          # exit 0: no spot above SEAM_LIMIT 1.1
  uv run python scripts/walk_sheet.py --compare /tmp/walk/t5 /tmp/walk/tN --out /tmp/walk/tN-vs-t5.png
  ```
  W4A8 (5b) and block caching (8) are lossy: they are judged by this gate, not by a match to the reference.
- `FLUX_STEP_RECIPE = "sharp"` is final (user Decision 2, 2026-10-24).
- **Checkpoints:** Tasks 3, 4, 5, 5b, 5c, 6, 7, 8 and 9 each end in a **user walkthrough**. The controller stops, explains the piece with its explainer and numbers, and waits for the user before dispatching the next task. Task 10 is the final task; it ends with the results walkthrough, not a stop between its parts.
- Each engine task writes an explainer at `docs/engine/NN-<name>.md`: the problem, how it works (with a diagram in ASCII if useful), before/after numbers from `scripts/bench_flux.py` (step s and achieved TFLOP/s), and what to read next.
- A FLUX job calls `check_stop()` and reports progress **between denoising steps**.
- Commits carry no Claude attribution or Co-Authored-By. Never log raw LLM output. Never restart systemd-logind. Don't kill processes that aren't ours.
- The game keeps working at every commit. `VECWORLD_PAINTER=turbo` stays available until Task 10.

## Review Focus

1. **Tensors that aren't Q4_K.** Q4_K_S files mix in other GGUF types (and bf16 biases and norms), and some `in_features` aren't multiples of 256. The swap must leave those on the reference path, never mis-decode them. Pinned in Task 3: `test_swap_skips_non_q4k`.
2. **Matrix shapes that aren't tile multiples:** M = tokens × windows (e.g. 1280 + 512 text, or a batch of 3). Every kernel stage must be exact on ragged edges. Pinned in Tasks 3–5b: `test_q4k_gemm_ragged[M=1,7,333,1792]`, `test_w4a8_gemm_ragged[M=1,7,333,1792]`; Task 5c: `test_q4k_gemv_matches_reference[M=1,2,4,8]`.
3. **A job stopped mid-step** (a player turns, a demand job arrives). The painter stops at the next step boundary, the slots, streamer ring and pinned buffers are reused rather than leaked, and the next job is correct. Pinned in Task 7: `test_graph_stop_between_steps`; Task 10: `test_stream_stop_then_reuse`.
4. **Short or failed SSD reads, or a misaligned O_DIRECT offset or length.** These give a clear error, and never silently corrupt weights. Pinned in Task 10: `test_uring_short_read_raises`, `test_uring_unaligned_tail`.
5. **Out of memory during a FLUX job.** The existing OOM → 503 path still works, and the next request succeeds. Pinned in Task 2: `test_flux_oom_returns_503_then_recovers` (fake painter that raises `torch.cuda.OutOfMemoryError` once).
6. **Activation outliers in W4A8 (quality).** A few tokens or channels far larger than the rest (likely the post-GELU MLP-out input, attention-out, and the first and last blocks) waste INT8 levels for everything else in that row. Per-token scales contain a loud token; a loud *channel* needs the layer kept at W4A16 (`FLUX_A16_KEEP`). Check that the per-layer study drove the keep-list and that the walk gate, not PSNR alone, decided the default. Pinned in Task 5b: `test_w4a8_outlier_token`, `scripts/w4a8_study.py`, the `sharp` walk `--check`.
7. **Stale block cache (quality).** A cached residual reused across jobs, across window sets in a batch, after a stop, or at the first denoising step paints the wrong picture with no error. Check the cache key and every reset path. Pinned in Task 8: `test_cache_never_skips_first_step`, `test_cache_reset_per_job`, `test_cache_keyed_by_windows`, `test_cache_stop_clears`.
8. **CUDA-graph capture with changing shapes.** A ragged window batch (8 windows at batch 3 → 3, 3, 2), a different text length or a different resolution must use a graph captured for that shape (or run eagerly), never replay one whose static buffers or baked addresses belong to another shape. Pinned in Task 7: `test_graph_shape_change`; Task 9 re-runs it at the batch sizes it picks.

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

### Task 3: Fused Q4_K kernel, naive — correctness reference — CHECKPOINT

Targets the **1.08 s per step of GGUF dequant** (42% of the 2.74 s step). Expected: a correct, slow **reference** kernel that every later stage is tested against.

**Files:**
- Create:
  - `src/vecworld/engine/__init__.py`;
  - `engine/ext.py`;
  - `engine/q4k.py`;
  - `engine/swap.py`;
  - `engine/csrc/q4k.cu`, `engine/csrc/bindings.cpp`;
  - `tests/test_q4k.py` (gpu);
  - `docs/engine/01-q4k-naive.md`, `docs/engine/speed.md`.
- Modify:
  - `scripts/bench_flux.py` (`--painter engine`, `flop_per_step`, achieved TFLOP/s);
  - `tests/test_bench_flux.py`;
  - `src/vecworld/flux/painter.py` (swap behind `VECWORLD_KERNEL`);
  - `src/vecworld/config.py` (`FLUX_STEP_RECIPE` comment: sharp is the user's pick, final).

**Interfaces:**
- Produces:
  - `load_ext() -> module`, built once and cached, with `CUDA_HOME` set;
  - `read_gguf(path) -> dict[str, GGUFTensor]` (name, gguf type, shape, raw bytes);
  - `Q4KLinear(weight: GGUFTensor, bias: torch.Tensor | None, kernel: str)`, an `nn.Module` with `forward(x: bf16 [..., in]) -> bf16 [..., out]` and `kernel ∈ {"naive","tiled","mma","w4a8"}` (later tasks add the last three). The packed super-blocks and the bias are **registered buffers**, so diffusers group offload moves them until Task 7 replaces it;
  - `swap_linears(model, gguf: dict, kernel: str) -> int`, which replaces only Q4_K linears and returns how many it replaced;
  - `bench_flux.flop_per_step(n_img: int, n_txt: int) -> float` and, in every engine result, `flop_per_step`, `achieved_tflops`, `pct_bf16_peak` (peak = `blocks.bf16_matmul_tflops` in `bench/baseline.json`, 19.1).
  - `--painter engine` builds `FluxPainter` with the kernel from `VECWORLD_KERNEL` (ledger ruling: "engine" = FluxPainter with engine flags).

- [ ] **Step 1:** Tests:
  - `test_q4k_dequant_matches_gguf`: our device dequant equals `gguf.quants.dequantize` exactly, on the real FLUX tensors;
  - `test_q4k_gemm_matches_reference`: `max|ours − x @ deq(W)ᵀ| / max|ref| < 2e-2` (bf16);
  - `test_q4k_gemm_ragged[M=1,7,333,1792]`;
  - `test_swap_skips_non_q4k` (Review Focus 1);
  - `test_flux_image_matches_reference`: same seed at 512, PSNR against the reference ≥ 30 dB;
  - CPU, in `tests/test_bench_flux.py`: `test_flop_per_step_matches_ledger` (`flop_per_step(1024, 256) == 114·113_246_208·1280 + 700_416·1280²`, ≈ 17.67e12) and `test_engine_row_fields` (fake timings → `achieved_tflops`, `pct_bf16_peak`).
- [ ] **Step 2:** Run them. Expected: FAIL (no `vecworld.engine`, no `flop_per_step`).
- [ ] **Step 3:** Implement the naive kernel: one thread per output element, decoding the Q4_K super-block (scale, min, 6-bit sub-scales) in registers, then a dot product over `in`. Bind it, write `Q4KLinear`, and swap it into `FluxPainter.transformer` behind `VECWORLD_KERNEL=naive`. Add `--painter engine` and the TFLOP/s fields to the bench.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** On AC, `VECWORLD_KERNEL=naive`: `bench_flux.py --painter engine --what tile512 --out bench/t3-naive.json`, `--what step --out bench/t3-naive-step.json`, and `--what blocks`. Create `docs/engine/speed.md` with three rows: the baseline (2.74 s, 6.45 TFLOP/s, 34%), Task 1's reference painter (`bench/t1-tile512.json`, 2.79 s) and Task 3. Expected: correct; probably **slower than 2.74 s** (CUDA cores, no reuse, ≤ 9.6 TFLOP/s ceiling est.). Say so honestly; this kernel is the correctness reference, not a speedup.
- [ ] **Step 6:** Write the explainer: the Q4_K layout diagram, why per-call dequant costs 1.08 s per step (each PyTorch op round-trips VRAM, ledger Technique 1), and the naive kernel's step s and TFLOP/s.
- [ ] **Step 7:** Commit: `Run FLUX linears on a hand-written Q4_K kernel (naive)`. **Stop: user walkthrough.**

### Task 4: Fused Q4_K kernel, tiled — CHECKPOINT

Targets **dequant**. Expected: **dequant → ~0** (no bf16 weight is ever written to VRAM; `blocks` shows no separate dequant share).

**Files:** Modify `engine/csrc/q4k.cu`, `engine/q4k.py`. Test: `tests/test_q4k.py` (parametrize `kernel`). Create `docs/engine/02-q4k-tiled.md`. Bench row: `bench/t4-tiled*.json`, `docs/engine/speed.md`.

- [ ] **Step 1:** Parametrize the Task 3 kernel tests over `kernel ∈ {"naive","tiled"}`. Run them. Expected: the `tiled` cases FAIL.
- [ ] **Step 2:** Implement a tiled kernel: blocks of output tiles; `x` tiles and decoded weight tiles staged in shared memory; each weight byte decoded once per tile, not once per output; coalesced loads. Tile shape is the implementer's call, justified by measurement.
- [ ] **Step 3:** Run the tests. Expected: PASS. Then on AC, `VECWORLD_KERNEL=tiled`: tile512, step and blocks; add the bench row. Expected: dequant ~0; the GEMM itself still runs on CUDA cores, so step s may not beat the reference yet. Record both.
- [ ] **Step 4:** Write the explainer: memory hierarchy, coalescing and reuse, step s and TFLOP/s against the naive kernel and the reference. Commit: `Tile the Q4_K kernel through shared memory`. **Stop: user walkthrough.**

### Task 5: Fused Q4_K kernel, tensor-core bf16 `mma` (W4A16) — CHECKPOINT

Targets the **GEMM** (~1.05 s est., ledger Part 2). Expected: our W4A16 GEMM **~cuBLAS** on the FLUX shapes; step ≈ 1.6 s if it matches cuBLAS (1.8–2.2 s at 65–85% of cuBLAS, ledger Technique 1).

**Files:** Modify `engine/csrc/q4k.cu`, `engine/q4k.py`, `scripts/bench_flux.py` (`--what gemm`). Test: `tests/test_q4k.py` (add `"mma"`). Create `docs/engine/03-q4k-mma.md`. Bench row: `bench/t5-mma*.json`, `bench/t5-gemm.json`, `docs/engine/speed.md`.

**Interfaces:**
- Produces: `bench_flux.py --what gemm`: per FLUX shape, our TFLOP/s against `torch.matmul` on the pre-dequantized bf16 weight. Shapes (M, K, N): (1024, 3072, 9216), (1024, 3072, 3072), (1024, 3072, 12288), (1024, 12288, 3072), the same four at M = 256, (1280, 3072, 21504), (1280, 15360, 3072).

- [ ] **Step 1:** Add `"mma"` to the kernel parametrization. Run it. Expected: the `mma` cases FAIL.
- [ ] **Step 2:** Implement a bf16 `mma.sync` (m16n8k16) kernel: decode Q4_K to bf16 fragments in registers, accumulate in fp32, and double-buffer shared memory with `cp.async`. Offline repacking of the nibbles at load time is the implementer's call. Make `mma` the default kernel.
- [ ] **Step 3:** Run the tests. Expected: PASS. Then on AC: `--what gemm`, tile512, step, blocks; add the bench row. Expected: ≥ 85% of `torch.matmul`'s TFLOP/s on the M = 1024/1280 shapes (report the M = 256 shapes separately); dequant no longer dominant in `blocks`.
- [ ] **Step 4:** Write the explainer: warps, fragments, `mma`, why decoding in registers kills the 1.08 s, and step s and TFLOP/s for all three stages. Commit: `Feed tensor cores straight from Q4_K weights`. **Stop: user walkthrough.**

### Task 5b: W4A8 on the INT8 tensor cores, quality-gated — CHECKPOINT

Targets the **GEMM, ~1.05 s**. Expected: **~0.4–0.8 s**, if the walk gate passes. Achieved TFLOP/s is bf16-equivalent from here on.

**Files:**
- Create: `scripts/w4a8_study.py`, `tests/test_w4a8.py` (gpu), `docs/engine/04-w4a8.md`.
- Modify: `engine/csrc/q4k.cu`, `engine/csrc/bindings.cpp`, `engine/q4k.py`, `engine/swap.py`, `src/vecworld/config.py` (`FLUX_A16_KEEP`).
- Bench row: `bench/t5b-w4a8*.json`, `bench/t5b-study.json`, `docs/engine/speed.md`.

**Interfaces:**
- Produces:
  - `quantize_act(x: bf16 [M, K]) -> (xq: int8 [M, K], s: fp32 [M], xsum: int32 [M, K // 32])`: per-token `s = max|x_t| / 127`, `xq = round(x_t / s)`, and `xsum` = Σ xq over each 32-wide sub-block (for the `dmin·m_j` term);
  - `Q4KLinear(kernel="w4a8")`: INT8 `mma.sync.m16n8k32` (s8 × s8 → s32), one instruction's k = 32 = one Q4_K sub-block, codes q (0..15) used as they are; per sub-block, `acc_fp32 += (d·sc_j)·acc_int32 − (dmin·m_j)·xsum`, then × s per token in the epilogue (ledger Technique 2);
  - `swap_linears(model, gguf, kernel, keep_a16: Collection[str] = ())`: linears whose module name matches an `fnmatch` pattern in `keep_a16` stay on `mma`;
  - config `FLUX_A16_KEEP: tuple[str, ...]`, set from the study and the walk gate.

- [ ] **Step 1:** Study first, before the kernel (cheap): `scripts/w4a8_study.py` fake-quantizes each Q4_K linear's input per token in torch, on one real 512 pass and one `sharp` walk step. Per layer it records the relative output error against W4A16 and the activation max/median ratio per token and per channel → `bench/t5b-study.json`. Rank the layers. Expected: the loud layers are among MLP-out, attention-out and the first/last blocks (ledger); the study says which.
- [ ] **Step 2:** Tests:
  - `test_quantize_act_matches_torch` (`xq`, `s`, `xsum` equal a torch implementation exactly);
  - `test_w4a8_gemm_matches_fake_quant`: against fp32 `(xq·s) @ deq(W)ᵀ`, `max|ours − ref| / max|ref| < 1e-3` (the integer part is exact; only the fp32 rescale order differs);
  - `test_w4a8_gemm_ragged[M=1,7,333,1792]` (Review Focus 2);
  - `test_w4a8_outlier_token` (Review Focus 6): scaling one token ×1000 leaves every other token's output bit-identical;
  - `test_w4a8_keep_a16`: a `keep_a16` pattern leaves those linears on `mma`.
  Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `quantize_act` (one kernel) and the W4A8 GEMM with the same tiling, `cp.async` double buffering and ragged-edge handling as `mma`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Quality gate (Global Constraints) with `VECWORLD_KERNEL=w4a8`, `FLUX_A16_KEEP=()` first. If `--check` fails or the sheet looks worse than Task 5's walk, add the study's worst layers to `FLUX_A16_KEEP` and re-walk. Record each attempt (keep-list, seam scores, share of token-facing FLOPs on INT8) in the explainer. If no keep-list passes with at least half of the token-facing FLOPs on INT8, keep `mma` as the default and report the result.
- [ ] **Step 6:** On AC: `--what gemm` (now with a W4A8 column), tile512, step; add the bench row, with the PSNR of a same-seed 512 image against the `mma` image (recorded, not a gate). Expected: GEMM ~1.05 s → ~0.4–0.8 s. If the gate passed, make `w4a8` the default with the chosen `FLUX_A16_KEEP`.
- [ ] **Step 7:** Write the explainer: W4A16 vs W4A8, per-token scales, the min-term algebra, the study table, the gate result and the numbers. Commit: `Run FLUX GEMMs on the INT8 tensor cores (W4A8)`. **Stop: user walkthrough** (the user also judges the walk sheet).

### Task 5c: M=1 GEMV for modulation, reading Q4_K directly — CHECKPOINT

Targets **modulation, ~0.29 s** per step (3.23 B params, 27% of the weights, 0.04% of the math; ledger §1.6). Expected: **→ ~0**. The 0.29 s was modulation's dequant share; Task 4 already removed the dequant, so the gain measured here is what the GEMM kernels still spend on 1-row inputs padded to 16-row tiles. Report both.

**Files:** Modify `engine/csrc/q4k.cu`, `engine/csrc/bindings.cpp`, `engine/q4k.py`, `scripts/bench_flux.py` (`blocks` reports modulation time per step). Test: `tests/test_q4k.py`. Create `docs/engine/05-modulation-gemv.md`. Bench row: `bench/t5c-gemv*.json`, `docs/engine/speed.md`.

**Interfaces:**
- Produces: `q4k_gemv(x: bf16 [M, K], qweight, bias) -> bf16 [M, N]` for `M ≤ GEMV_MAX_M = 8` (modulation sees one row per image; a batch has ≤ 8 windows). `Q4KLinear.forward` routes to it whenever the flattened rows are ≤ 8, whatever `kernel` is. It is memory-bound: each Q4_K byte is read once, decoded in registers, and dotted with the row(s) held in shared memory.

- [ ] **Step 1:** Tests: `test_q4k_gemv_matches_reference[M=1,2,4,8]` (same tolerance as Task 3), `test_gemv_dispatch` (M ≤ 8 → GEMV, M = 9 → the configured GEMM kernel). Run them. Expected: FAIL.
- [ ] **Step 2:** Implement the GEMV: a warp per group of output columns, 128-bit loads of the packed super-blocks, fp32 accumulation, a warp-shuffle reduction.
- [ ] **Step 3:** Run the tests. Expected: PASS. On AC: `blocks` (modulation ms per step, and achieved GB/s against ~192 GB/s est.), tile512, step; add the bench row. Expected: modulation ≈ 1.8 GB of Q4_K per step at ≥ 60% of VRAM bandwidth ≈ 0.01–0.02 s.
- [ ] **Step 4:** Write the explainer: GEMV vs GEMM on the roofline (M = 1 is memory-bound, so 4-bit weights *do* speed it up), and the numbers. Commit: `Run FLUX modulation on a Q4_K GEMV`. **Stop: user walkthrough.**

### Task 6: Glue fusion: norm + modulate + residual in one pass — CHECKPOINT

Targets **glue, ~0.4 s** (est. 0.25–0.55; ledger Part 2). Expected: **→ ~0.1–0.15 s**. Quality-gated by the walk (Global Constraints), though it is lossless in principle.

**Files:** Create `engine/glue.py`, `engine/csrc/glue.cu`, `engine/blocks.py`, `tests/test_glue.py` (gpu), `docs/engine/06-glue-fusion.md`. Modify `engine/csrc/bindings.cpp`, `src/vecworld/flux/painter.py` (patch behind `VECWORLD_FUSE`, default on once the gate passes). Bench row: `bench/t6-fuse*.json`, `docs/engine/speed.md`.

**Interfaces:**
- Produces:
  - `residual_norm_modulate(res: bf16 [M, H], y: bf16 [M, H] | None, gate: bf16 [B, H] | None, shift: bf16 [B, H], scale: bf16 [B, H], eps: float = 1e-6) -> tuple[Tensor, Tensor]`: `res_out = res + gate·y` (skipped when `y` is None), then `x_mod = LayerNorm(res_out)·(1 + scale) + shift` (no affine, fp32 statistics, as the reference). One read of `res` and `y`, one write of each output. `B` rows are per image, broadcast over that image's tokens;
  - `patch_blocks(transformer) -> int`: replaces the FLUX double and single block forwards with ones that call it; returns how many blocks it patched.
  - When `w4a8` is the kernel, the same pass may also emit `quantize_act`'s outputs for the next linear (implementer's call, measured).

- [ ] **Step 1:** Tests:
  - `test_rnm_matches_eager[M=1,333,1280,5120]`: `max|ours − eager_fp32| ≤ 2⁻⁷·max|eager_fp32|` for both outputs;
  - `test_rnm_no_residual` (`y=None`);
  - `test_patched_block_matches_unpatched`: double and single block outputs, `max|Δ| / max|ref| < 2e-2`;
  - gpu `test_flux_fused_image_matches`: same seed at 512, PSNR ≥ 30 dB against unfused with the same kernel.
  Run them. Expected: FAIL.
- [ ] **Step 2:** Implement the kernel (one block per token row, warp reductions for mean and variance) and the patched block forwards.
- [ ] **Step 3:** Run the tests. Expected: PASS.
- [ ] **Step 4:** Quality gate (Global Constraints). On AC: tile512, step, and peak allocated at batch 1, 2 and 4 (fused glue keeps fewer intermediates live; Task 9 uses this). Add the bench row. Expected: glue ~0.4 s → ~0.1–0.15 s.
- [ ] **Step 5:** Write the explainer: memory-bound elementwise ops, why fusing saves VRAM round trips not FLOPs, Amdahl (glue's share grows as GEMMs shrink), and the numbers. Commit: `Fuse residual, norm and modulate into one pass`. **Stop: user walkthrough.**

### Task 7: CUDA graphs over the step — CHECKPOINT

Targets **launch overhead**. Expected: **smaller and steadier** (fewer GPU idle gaps; per-pass spread over 4 passes narrower), est. 1.0–1.1× on step s.

Graphs bake device addresses, but diffusers group offload re-allocates each block's weights on every onload. So this task replaces group offload with fixed-address device slots, and captures **one graph per block per shape**, with the CPU sequencing the block replays and the slot copies between them (ledger Technique 4). The slot copies stay outside the graphs, so Task 10's CPU-driven io_uring reads still fit.

**Files:** Create `engine/graphs.py`, `engine/slots.py`, `tests/test_graphs.py` (gpu), `docs/engine/07-cuda-graphs.md`. Modify `engine/q4k.py` (views into a slot), `src/vecworld/flux/painter.py` (drops diffusers group offload; graphs behind `VECWORLD_GRAPHS`, default on), `config.py` (`FLUX_GRAPH_MAX`). Bench row: `bench/t7-graphs*.json`, `docs/engine/speed.md`.

**Interfaces:**
- Produces:
  - `BlockSlots(blocks: list[nn.Module], host: Callable[[int], dict[str, torch.Tensor]], ring: int = 2)`: `ring` device slots of fixed address, each sized to the largest block. Before block *i* runs, its tensors (packed Q4_K and the bf16 ones) are copied from host memory into slot `i % ring` on a side stream, ordered by CUDA events; block *i+1*'s copy overlaps block *i*'s compute. The block's modules view into the slot. Task 10 grows this into `BlockStreamer`.
  - `GraphRunner(blocks, slots, max_graphs: int = FLUX_GRAPH_MAX)`: a CUDA graph per (block index, shape key), key = `(batch, n_img, n_txt)`. Static input and output buffers per key; captured on first use after one eager warm-up, from one shared memory pool; least-recently-used keys evicted beyond `max_graphs`; `VECWORLD_GRAPHS=0` runs eagerly.

- [ ] **Step 1:** Before writing code, on AC: `nsys profile` one tile512 step on the Task 6 build; record the GPU idle time between kernels in the explainer. If it is under about 2% of the step, say so: graphs then buy steadiness more than speed.
- [ ] **Step 2:** Tests:
  - `test_slots_match_offload`: block outputs with `BlockSlots` equal those with group offload, bit for bit;
  - `test_graph_matches_eager`: graph replay equals eager on the same slots, bit for bit;
  - `test_graph_shape_change` (Review Focus 8): key A (batch 1, 1024 + 256), then B (batch 3, 1024 + 256), then C (batch 1, 1024 + 77), then A again; each matches eager, and A's second run reuses its graph;
  - `test_graph_stop_between_steps` (Review Focus 3): a `Stopped` between steps, then a second job is correct, and allocated memory is unchanged afterwards;
  - peak allocated < 3.6 GB at batch 4 with graphs on.
  Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `BlockSlots` and `GraphRunner`.
- [ ] **Step 4:** Run the tests. Expected: PASS. On AC: tile512 (4 passes; record min, median and max per pass), step, and `nsys` again; add the bench row.
- [ ] **Step 5:** Write the explainer: what a launch costs, capture and replay, static shapes and addresses, why a graph per block (CPU between blocks for copies), the timelines before and after, and the numbers. Commit: `Replay FLUX blocks as CUDA graphs from fixed weight slots`. **Stop: user walkthrough.**

### Task 8: Block caching (TeaCache / First-Block-Cache style), quality-gated — CHECKPOINT

Targets **work per step: skips blocks 1..56** when the first block's residual barely changed since the last step. Expected: **~1.5×+** on step s (averaged over a job's passes), if the walk gate passes. Ledger Technique 5 estimates less for a 4-step distilled model (1.0–1.33×); measure, and report both.

**Files:** Create `engine/cache.py`, `tests/test_cache.py`, `docs/engine/08-block-cache.md`. Modify `engine/graphs.py` (decision between block 0's replay and the rest), `src/vecworld/flux/painter.py`, `config.py` (`FLUX_CACHE_THRESHOLD`, `0.0` = off). Bench row: `bench/t8-cache*.json`, `docs/engine/speed.md`.

**Interfaces:**
- Produces: `BlockCache(threshold: float)` with
  - `begin_job(job_id) -> None` (clears everything);
  - `probe(key, r0: Tensor, step: int) -> bool`: True means skip. `r0` is block 0's output residual; `key = (job_id, tuple(window ids in the batch))`. It accumulates `|r0(t) − r0(t−1)|₁ / |r0(t−1)|₁` across skipped steps and skips while the sum stays under `threshold`; never at step 0;
  - `store(key, residual: Tensor)` and `apply(key, x) -> Tensor` (`x + cached residual of blocks 1..56`);
  - `clear()` on `Stopped`.
  - The probe reads one scalar on the host between graph replays (Task 7's per-block graphs allow it).

- [ ] **Step 1:** Tests (CPU fakes, plus one gpu):
  - `test_cache_never_skips_first_step`;
  - `test_cache_threshold_zero_never_skips`: output identical to no cache;
  - `test_cache_reset_per_job` (Review Focus 7);
  - `test_cache_keyed_by_windows`: a batch with a different window set never uses another's residual;
  - `test_cache_stop_clears`: after a `Stopped`, the next job starts cold;
  - `test_cache_accumulates_change`: two small changes that sum above the threshold force a recompute;
  - gpu `test_cache_skips_on_flux`: at a high threshold, at least one pass of a 4-step 512 job skips.
  Run them. Expected: FAIL.
- [ ] **Step 2:** Implement `BlockCache` and wire it into the painter's pass loop (pano windows and walk steps).
- [ ] **Step 3:** Run the tests. Expected: PASS.
- [ ] **Step 4:** Threshold sweep on AC: `FLUX_CACHE_THRESHOLD` ∈ {0.05, 0.1, 0.15, 0.2, 0.3}. For each: skips per job (pano and walk step), step s, and the quality gate. Choose the largest threshold whose walk passes `--check` and that the user accepts by eye at the checkpoint; record the table.
- [ ] **Step 5:** Add the bench row at the chosen threshold (achieved TFLOP/s is nominal FLOPs ÷ time, so it can exceed the hardware rate; say so). Expected: ~1.5×+ on step s.
- [ ] **Step 6:** Write the explainer: residual reuse, the probe, why a 4-step distilled model has less to skip, staleness and the cache key, the sweep table. Commit: `Skip unchanged FLUX blocks with a first-block cache`. **Stop: user walkthrough** (the user picks the threshold from the sheets).

### Task 9: Window batching — CHECKPOINT

Targets **the panorama**: per-pass costs are paid once per batch. Expected: **fewer launches** and passes per panorama.

**Files:** Modify `src/vecworld/flux/painter.py`, `config.py` (`FLUX_BATCH`). Test: `tests/test_flux_painter.py`, `tests/test_flux_painter_gpu.py`, `tests/test_graphs.py`. Create `docs/engine/09-batching.md`. Bench row: `bench/t9-batch*.json` (plus `--what pano`), `docs/engine/speed.md`.

- [ ] **Step 1:** Tests:
  - `test_batched_equals_unbatched` (CPU fake: the same window outputs whatever the batch size);
  - gpu `test_flux_pano_batched_matches`: PSNR ≥ 35 dB against batch 1, same seed;
  - `test_graph_shape_change` re-run with the batch sizes this task picks, including the ragged last batch (Review Focus 8).
  Run them. Expected: FAIL (no `FLUX_BATCH`).
- [ ] **Step 2:** Implement: all active windows of a step go through each transformer pass as one batch of up to `FLUX_BATCH`, the largest that keeps peak allocated < 3.6 GB (measure it; Task 6's per-window activation numbers say where to start). Per-window text embeddings are stacked. Graph keys and cache keys follow the batch (Tasks 7, 8).
- [ ] **Step 3:** Run the tests. Expected: PASS. Bench `pano`, `tile512` and `step` on AC; add the bench row with s per window-step.
- [ ] **Step 4:** Write the explainer: why the weights are moved and decoded once per pass, why batching overlaps with the fused kernel (both attack the per-pass cost, ledger Technique 3), the activation memory against batch size, and the numbers. Commit: `Batch every panorama window through each weight pass`. **Stop: user walkthrough.**

### Task 10: Layer streamer, io_uring loader, T5 on the engine, cleanup and end to end

Targets **offload (0.15 s per step) and load**, and the RAM budget. Expected: these come **later**: they keep the server under 12 GB RSS rather than speed the step. Weights read from the SSD add an I/O floor (6.66 GB ÷ 2.0–2.96 GB/s per pass, ledger Technique 3), so batched passes matter here. This is the final task: its three parts each commit, and the user walkthrough is the results walkthrough at the end.

#### 10a: Layer streamer

**Files:** Create `engine/stream.py`, `tests/test_stream.py` (gpu), `docs/engine/10-streamer.md`. Modify `engine/slots.py`, `flux/painter.py`.

**Interfaces:**
- Produces: `BlockStreamer(blocks: list[nn.Module], source: Callable[[int], memoryview | torch.Tensor], ring: int = 2)`, with `install()` and `close()`, built on Task 7's `BlockSlots`. Block weights live in a pinned host arena. Copies to a ring of device slots run on a side stream, synchronised by CUDA events; block graphs replay against the slots as in Task 7.

- [ ] **Step 1:** Tests:
  - `test_stream_matches_resident` (block outputs identical to resident weights);
  - `test_stream_overlaps` (step time ≤ 1.1 × the compute-only time on AC);
  - `test_stream_stop_then_reuse` (Review Focus 3: a `Stopped` mid-step, then a second job uses the same ring correctly, and pinned and device memory are unchanged afterwards).
  Run them. Expected: FAIL.
- [ ] **Step 2:** Implement the pinned arena (one `cudaHostAlloc` / `torch.empty(pin_memory=True)`) and the device ring of `ring` slots sized to the largest block. Block *i+1* is copied while block *i* computes. `Q4KLinear` views its weights into the current slot.
- [ ] **Step 3:** Run the tests. Expected: PASS. Bench on AC against Task 9; add the bench row (`bench/t10a-stream*.json`).
- [ ] **Step 4:** Write the explainer: pinned vs pageable memory, DMA, streams and events, a timeline diagram of copy and compute, and the numbers. Commit: `Stream FLUX blocks through a pinned ring`.

#### 10b: io_uring + O_DIRECT loader

**Files:** Create `engine/csrc/uring.cpp`, `engine/uring.py`, `tests/test_uring.py`, `docs/engine/11-io-uring.md`. Modify `engine/stream.py` (source = SSD), `__main__.py` (dreamer back on by default).

**Interfaces:**
- Produces: `DirectReader(path: str, queue_depth: int = 8)`, with `read_into(offset: int, nbytes: int, dst: torch.Tensor) -> None` (pinned, 4096-aligned) and `close()`; and `ssd_source(gguf_path, layout) -> Callable[[int], torch.Tensor]` for `BlockStreamer`. Block bytes are read from the GGUF file, and no weights stay resident in RAM.

- [ ] **Step 1:** Tests:
  - `test_uring_reads_exact_bytes` (matches `os.pread`);
  - `test_uring_unaligned_tail` (a block whose start or length isn't 4096-aligned reads correctly via an aligned superset);
  - `test_uring_short_read_raises` (a truncated temp file raises `IOError` naming the offset);
  - gpu `test_flux_from_ssd_matches` (image PSNR ≥ 40 dB against the RAM-sourced streamer).
  Run them. Expected: FAIL. If liburing headers are missing, stop and ask the user to run `sudo apt install liburing-dev`, or use the raw syscalls.
- [ ] **Step 2:** Implement submission and completion rings, O_DIRECT reads straight into the pinned arena slots, and prefetch of block *i+2* while *i+1* copies to the GPU.
- [ ] **Step 3:** Run the tests. Expected: PASS. On AC, with the full game running (Qwen on), measure the server's RSS (target < 12 GB, no swap) and the step time; add the bench row (`bench/t10b-uring*.json`).
- [ ] **Step 4:** Write the explainer: page cache vs O_DIRECT, io_uring's two rings, alignment, the three-stage pipeline (SSD → pinned → GPU) and the numbers. Commit: `Read FLUX blocks from the SSD with io_uring`.

#### 10c: T5 on the engine, cleanup and end to end

**Files:** Modify `src/vecworld/flux/text.py`, `__main__.py`, `README.md`, spec §2 (measured numbers). Delete `TurboPainter`'s default use (keep the class only if tests still need it). Create `docs/engine/12-t5-and-results.md`.

- [ ] **Step 1:** Test: gpu `test_t5_engine_matches_cpu` (embeddings cosine ≥ 0.999 against the Task 1 CPU path) and `test_t5_encode_fast` (≤ 5 s on AC). Run them. Expected: FAIL.
- [ ] **Step 2:** Run T5-XXL's linears through `Q4KLinear` (or the matching GGUF type, falling back to the reference for non-Q4_K) with `BlockStreamer` + `DirectReader`. The CPU path stays as a fallback behind a flag.
- [ ] **Step 3:** Run the full suite with the server stopped. Expected: green. Then e2e on AC: the 3 reference journeys (haunted forest + night, desert oasis, snowy mountains), 6 spots each, plus the Global Constraints walk gate on the final configuration. Measure first panorama, step s, achieved TFLOP/s, walk step, fog, peak VRAM/RAM and T5 time. Save contact sheets against SD-Turbo and look at them.
- [ ] **Step 4:** Write the results explainer: `docs/engine/speed.md`'s rows in one table (every piece's step s and TFLOP/s before and after) and the gates. Commit: `Run T5 on the engine; FLUX Dream Roam end to end`. Leave the server running on the final code. **Results walkthrough with the user.**
