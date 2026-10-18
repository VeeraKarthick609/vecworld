# Dream Walk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the meaning-space game with Dream Walk. You type a theme, look around a seamless painted 360° panorama, walk in any direction, and discover labelled things, with mood chips and theme drift. Runs on the 4 GB laptop.

**Architecture:** A new `vecworld.dream` package:
- pure panorama maths: windows, wrap-around, masks, roll and dolly
- a `Painter` that runs wrap-around MultiDiffusion with differential diffusion on the GPU
- a `Dreamer` LLM that invents discoveries on the CPU
- a label locator (reusing CLIPSeg)
- a single-worker GPU queue with prefetch priority
- a `DreamService` that owns sessions

FastAPI exposes it, and a canvas viewer in `web/` draws a 90° window into the cylinder. The meaning-space modules are removed at the end.

**Tech Stack:** Python 3.12 (uv), PyTorch + CUDA, diffusers (stabilityai/sd-turbo), transformers (CLIPSeg, Qwen2.5-1.5B-Instruct), FastAPI, vanilla JS canvas, pytest + selenium (headless Firefox).

**Spec:** `docs/superpowers/specs/2026-10-18-dreamwalk-design.md`. The throwaway spike that proved the technique is `/tmp/vwspike/pano3.py` on the laptop. It's a reference for the maths, not code to copy blindly.

## Global Constraints

- Everything runs on the laptop (`ssh gpu`, repo `~/dev/vecworld`, branch `dreamwalk`). There are no paid or hosted model APIs, and the server runs directly on the host, not in Docker.
- RTX 3050 Ti 4 GB (about 3.7 GB usable). Peak allocated VRAM must stay **under 3.6 GB**. Only the painter lives on the GPU; the Dreamer LLM and CLIPSeg run on the CPU.
- Panorama size is 2048 × 512, cylindrical 360°, wrapping horizontally. The window is 64 × 64 latent (512 px) with stride 32. The VAE tiles are 64 latent wide with an 8-latent wrap margin.
- A step:
  1. Roll the panorama so the walk yaw is at the centre.
  2. Dolly in by 1.15.
  3. Run 4 denoising steps.
  4. The focus region is regenerated from noise. Everywhere else, the original noised latent is re-imposed until step `4 − int(4 × 0.6)`.
  5. Apply a 50% Lab palette pull toward the reference.
- The focus mask is full strength within ±0.08 of the width around `focus_at`, fading to 0 over the next 0.06.
- `STYLE = "impressionist oil painting, visible brushstrokes, dreamlike, soft volumetric light"`. The focus windows add `f", {discovery} ahead in the distance"`.
- Latency, single player: first panorama ≤ 5 s; a pre-generated walk starts its transition in under 0.5 s; an on-demand step ≤ about 6 s. With several players moving at once they queue fairly. There is no batching and no admission limit.
- About 1 in 3 steps has no discovery. Theme drift takes 3 steps (embedding weights 1/3, 2/3, 1). Pre-generation starts after 0.5 s without turning and is discarded if you turn more than 45° away.
- Labels only appear on discoveries CLIPSeg actually locates, using `HOTSPOT_MIN = 0.3`. Never show or log raw LLM text; only parsed discovery phrases leave `ideas.py`.
- Keep the existing error format `{"error": code, "message": str}` and the 503-on-out-of-memory behaviour. The UI keeps the previous view on any failure. Keep session TTL 3600 s and LRU 50.
- Test markers: `gpu` (real models), `ui` (headless Firefox). The default run excludes both. Commits carry **no Co-Authored-By or Claude attribution**. Push to `origin dreamwalk` after every commit.

## Review Focus

1. **Turning away after a prefetch, then pressing W:** you must walk the way you're facing now. A prefetch is only used within 45°. *(Task 6: `test_walk_ignores_prefetch_for_another_direction`)*
2. **Theme or mood change while a prefetch is ready or in flight:** the stale spot is never shown. *(Task 6: `test_mood_change_discards_prefetch`)*
3. **A discovery straddling the 360° seam** (u around 0.98 or 0.02): it's still located, and its label draws on both sides of the seam in the viewer. *(Task 4: `test_label_found_across_seam`; Task 8: `test_label_draws_across_seam`)*
4. **Dreamer output that's junk, empty, very long, or numbered and quoted:** only clean short phrases get through; otherwise use the fallback list. Never show it raw. *(Task 3: `test_parse_ideas_cleans_and_bounds`)*
5. **A session evicted, or a job hitting OOM, while its job is queued or running:** the worker thread survives and later jobs still run. *(Task 5: `test_queue_survives_failing_job`; Task 6: `test_evicted_session_prefetch_is_harmless`)*

---

### Task 1: Dream config + panorama maths

**Files:**
- Modify: `src/vecworld/config.py` (add the dream constants below; old constants are removed in Task 9)
- Create: `src/vecworld/dream/__init__.py`, `src/vecworld/dream/pano.py`
- Test: `tests/test_pano.py`

**Interfaces:**
- Produces, in config: `PANO_W = 2048`, `PANO_H = 512`, `WIN = 64`, `STRIDE = 32`, `VAE_TILE = 64`, `VAE_MARGIN = 8`, `DOLLY = 1.15`, `STEP_STEPS = 4`, `BG_STRENGTH = 0.6`, `FOCUS_HALF = 0.08`, `FOCUS_FEATHER = 0.06`, `KEEP_OFFSET = 0.17`, `PALETTE_STRENGTH = 0.5` (exists), `STYLE` (as in Global Constraints), `QUIET_EVERY = 3`, `DRIFT_STEPS = 3`, `PREFETCH_TOLERANCE = 45.0`, `MOODS = ("foggy", "night", "rain", "snow", "eerie", "warm")`, `MAX_THEME_CHARS = 120`.
- Produces in `dream/pano.py` (all pure, no torch):
  - `windows(h: int, w: int, win: int = WIN, stride: int = STRIDE) -> list[tuple[int, int]]`: (y, x) window origins. x steps by `stride` over the full wrap-around width. The last y row is clamped to `h − win`.
  - `wrap_cols(x0: int, n: int, w: int) -> list[int]`: `[(x0 + j) % w for j in range(n)]`
  - `focus_mask(w: int, at: float, half: float = FOCUS_HALF, feather: float = FOCUS_FEATHER) -> np.ndarray`: shape `(w,)`, float32 in [0, 1]. It's 1 within `half` of `at`, measured with **wrap-around distance**, and falls linearly to 0 across `feather`.
  - `roll_to(img: Image.Image, u: float) -> Image.Image`: horizontally rotate so panorama fraction `u` sits at the centre column.
  - `yaw_to_u(yaw: float) -> float`: `(0.5 + yaw / 360) % 1` (yaw 0 = centre column).
  - `dolly(img: Image.Image, z: float = DOLLY) -> Image.Image`: centre crop `1/z` in both axes, resized back to the same size.
  - `wrap_dist(a: float, b: float) -> float`: shortest distance on the unit circle.

- [ ] **Step 1: Write the failing tests**

```python
def test_windows_cover_every_column_twice_and_wrap():
    ws = windows(64, 256)
    assert {x for _, x in ws} == set(range(0, 256, 32)) and {y for y, _ in ws} == {0}
    cover = np.zeros(256, int)
    for _, x in ws:
        cover[wrap_cols(x, 64, 256)] += 1
    assert (cover == 2).all()

def test_windows_tall_rows_clamped():
    assert {y for y, _ in windows(96, 256)} == {0, 32}

def test_focus_mask_wraps_across_seam():
    m = focus_mask(1000, at=0.99)
    assert m[990] == 1 and m[30] == 1 and m[500] == 0
    assert 0 < m[int(1000 * (0.99 + 0.08 + 0.03)) % 1000] < 1

def test_roll_to_puts_u_at_centre():
    img = Image.new("RGB", (100, 10), "black"); img.putpixel((10, 5), (255, 0, 0))
    assert roll_to(img, 0.10).getpixel((50, 5)) == (255, 0, 0)

def test_yaw_to_u():
    assert yaw_to_u(0) == 0.5 and yaw_to_u(180) == 0.0 and abs(yaw_to_u(-90) - 0.25) < 1e-9

def test_dolly_keeps_size_and_enlarges_centre():
    img = Image.new("RGB", (200, 100), "black"); img.paste((255, 0, 0), (95, 45, 105, 55))
    out = dolly(img, 2.0)
    assert out.size == (200, 100) and out.getpixel((100, 50)) == (255, 0, 0) and out.getpixel((86, 50)) == (255, 0, 0)

def test_wrap_dist():
    assert abs(wrap_dist(0.98, 0.02) - 0.04) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**
Run: `uv run pytest tests/test_pano.py -q` → FAIL (ImportError: `vecworld.dream`).

- [ ] **Step 3: Implement config additions and `dream/pano.py`**

- [ ] **Step 4: Run tests**
Run: `uv run pytest -q -p no:warnings` → all pass.

- [ ] **Step 5: Commit and push** `"Add panorama maths for Dream Walk"`

---

### Task 2: Painter (wrap-around MultiDiffusion on SD-Turbo)

**Files:**
- Create: `src/vecworld/dream/painter.py`
- Test: `tests/test_painter_gpu.py` (gpu); add `FakePainter` to `tests/conftest.py`

**Interfaces:**
- Consumes: Task 1 `windows`, `wrap_cols`, `focus_mask`, config constants; `vecworld.scenes.palette.match_palette, palette_stats` (existing).
- Produces:
  - `Prompt = list[tuple[str, float]]` (weights sum to 1; one entry for a plain prompt)
  - `class Painter(Protocol)`:
    - `panorama(self, base: Prompt, seed: int) -> Image.Image` (PANO_W × PANO_H)
    - `step(self, prev: Image.Image, base: Prompt, focus: Prompt | None, focus_at: float | None, seed: int) -> Image.Image`. `prev` is already rolled and dollied. `focus_at=None` means no region regenerates: the whole frame is denoised at img2img strength `BG_STRENGTH`. Otherwise the `focus_mask(at=focus_at)` region regenerates from noise. Windows whose centre column has mask > 0 get the `focus` prompt if given, else `base`.
  - `class TurboPainter(Painter)`: `__init__(model_id="stabilityai/sd-turbo")`, fp16 on CUDA, default SDPA attention (no attention slicing; slicing ran out of memory at this size). The text embedding for a `Prompt` is the weight-sum of each text's `encode_prompt` embedding, cached per text. VAE encode and decode are tiled as in Global Constraints. UNet batches are 4 windows.
  - In `tests/conftest.py`: `class FakePainter` returning solid images of `(PANO_W // 16, PANO_H // 16)` px. It records `calls: list[tuple[str, dict]]` with kwargs; `fail_next: bool` raises `torch.cuda.OutOfMemoryError` once.

- [ ] **Step 1: Write the failing GPU tests**

```python
@pytest.mark.gpu
def test_panorama_is_seamless_and_fits(painter):
    img = painter.panorama([(f"a haunted forest, {STYLE}", 1.0)], seed=7)
    a = np.asarray(img, float)
    seam = np.abs(a[:, -1] - a[:, 0]).mean()
    inner = np.abs(a[:, 1023] - a[:, 1024]).mean()
    assert img.size == (2048, 512) and seam < 2.5 * inner + 2
    assert torch.cuda.max_memory_allocated() < 3.6 * 2**30

@pytest.mark.gpu
def test_step_times_and_keeps_background(painter):
    base = [(f"a haunted forest, {STYLE}", 1.0)]
    prev = dolly(painter.panorama(base, 7))
    t = time.time(); out = painter.step(prev, base, [(f"a haunted forest, an old lantern on a post ahead in the distance, {STYLE}", 1.0)], 0.5, 8)
    took = time.time() - t
    a, b = np.asarray(prev, float), np.asarray(out, float)
    side = np.abs(a[:, :400] - b[:, :400]).mean(); ahead = np.abs(a[:, 900:1150] - b[:, 900:1150]).mean()
    print(f"\nstep {took:.2f}s side-change {side:.1f} ahead-change {ahead:.1f}")
    assert took < 6.0 and ahead > 1.5 * side

@pytest.mark.gpu
def test_step_without_focus_changes_gently(painter):
    base = [(f"a haunted forest, {STYLE}", 1.0)]
    prev = dolly(painter.panorama(base, 7)); out = painter.step(prev, base, None, None, 9)
    assert np.abs(np.asarray(prev, float) - np.asarray(out, float)).mean() < 40
```

The `painter` fixture is module-scoped, built once, and has `torch.cuda.reset_peak_memory_stats()` called after warm-up.

- [ ] **Step 2: Run to verify they fail**
Stop the server first: `pkill -f "python3? -m [v]ecworld"`. Run: `uv run pytest -m gpu tests/test_painter_gpu.py -q -s` → FAIL (ImportError).

- [ ] **Step 3: Implement `dream/painter.py`**
Port the round-3 algorithm: per-window noise-prediction averaging over wrap-around columns, Euler steps via `sched.scale_model_input`/`sched.step`, and differential diffusion. Before step `k < STEP_STEPS − int(STEP_STEPS × BG_STRENGTH)`, re-impose `mask·latents + (1−mask)·add_noise(orig, noise, t)`, with the mask broadcast over latent columns. With `focus_at=None`, the mask is all zeros, which gives img2img at `BG_STRENGTH`. Add `"text, watermark, signature"` handling only if the GPU test images show it. SD-Turbo has no CFG, so a negative prompt has no effect; prefer cropping the bottom 3% of rows when encoding if text appears.

- [ ] **Step 4: Run tests**
Run: `uv run pytest -m gpu tests/test_painter_gpu.py -q -s`. All pass; record the step time. Save one panorama and one step to `/tmp/dw_*.png`, then look at them (scp to the Mac scratchpad and Read).

- [ ] **Step 5: Commit and push** `"Add wrap-around MultiDiffusion painter"` (body: measured times, peak VRAM)

---

### Task 3: Dreamer (discoveries from a small LLM)

**Files:**
- Create: `src/vecworld/dream/ideas.py`
- Test: `tests/test_ideas.py`, `tests/test_ideas_gpu.py` (marker `gpu`: heavy model, CPU)

**Interfaces:**
- Produces:
  - `FALLBACK_IDEAS: tuple[str, ...]`, 12 theme-agnostic phrases such as `"an old lantern on a post"`, `"a ruined stone archway"`, `"a still pond"`, `"a lone wooden bench"`.
  - `def parse_ideas(text: str, limit: int = 8) -> list[str]`: split on newlines and commas. Strip bullets, numbering, quotes and trailing punctuation, then lowercase. Keep phrases of 2–8 words made of letters, spaces, hyphens and apostrophes. Deduplicate, keep order, cap at `limit`.
  - `class Dreamer(Protocol): def ideas(self, theme: str, moods: list[str], n: int) -> list[str]`
  - `class QwenDreamer(Dreamer)`: `Qwen/Qwen2.5-1.5B-Instruct` on CPU fp32, greedy, `max_new_tokens=96`. The chat prompt asks for `n` short noun phrases for things one might come upon while wandering in "{theme}" ({moods}), one per line. It returns `parse_ideas(out, n)` padded from a shuffled `FALLBACK_IDEAS` if short. **It never logs the raw output.**
  - In `tests/conftest.py`: `class FakeDreamer` returning `["an old lantern on a post", "a ruined stone chapel", "a misty pond", ...]` and recording calls.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_ideas_cleans_and_bounds():
    raw = '1. "An old lantern on a post."\n- a ruined chapel,\n3) A MISTY POND\n\nok\n' + "word " * 40 + "\n<script>x</script>\na ruined chapel"
    assert parse_ideas(raw) == ["an old lantern on a post", "a ruined chapel", "a misty pond"]
    assert parse_ideas("") == [] and len(parse_ideas("\n".join(f"thing number {i}" for i in range(30)))) == 8

@pytest.mark.gpu
def test_qwen_dreamer_gives_themed_phrases():
    ideas = QwenDreamer().ideas("haunted forest", ["foggy"], 6)
    assert len(ideas) == 6 and all(2 <= len(i.split()) <= 8 for i in ideas)
```

- [ ] **Step 2: Run to verify they fail** → ImportError.
- [ ] **Step 3: Implement `dream/ideas.py`**
- [ ] **Step 4: Run tests**
Run: `uv run pytest -q -p no:warnings` and `uv run pytest -m gpu tests/test_ideas_gpu.py -q -s`. Pass; print the ideas and the generation time (CPU), and record them in the commit.
- [ ] **Step 5: Commit and push** `"Add LLM dreamer for discoveries"`

---

### Task 4: Discovery labels on the panorama

**Files:**
- Create: `src/vecworld/dream/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `vecworld.scenes.hotspots.locate` and `ClipSegFinder.heatmaps` (existing; threshold `HOTSPOT_MIN`, largest-blob boxes); Task 1 `wrap_cols`, `wrap_dist`.
- Produces:
  - `@dataclass(frozen=True) class Label: id: str; text: str; u: float; y: float; w: float; h: float`. `u` is the box centre as a panorama fraction in [0, 1); y, w and h are fractions of panorama height or width.
  - `def crop_at(img: Image.Image, u: float, size: int = 512) -> Image.Image`: a `size`-wide wrap-around crop centred on `u`, at full height.
  - `def to_label(text: str, box: Hotspot, u_centre: float, pano_w: int, crop_w: int = 512) -> Label`: maps a crop-relative box to panorama coordinates. `u` wraps into [0, 1). `id` is `secrets.token_urlsafe(4)`.
  - `def carry(labels: list[Label], walk_u: float, z: float = DOLLY) -> list[tuple[str, float]]`: where each kept label's text should now be. `u' = 0.5 + (wrapped offset from walk_u) × z`. Labels that land outside the 0.15–0.85 band are dropped.
  - `class LabelFinder`: `__init__(heatmaps=ClipSegFinder().heatmaps)`. `find(img: Image.Image, wanted: list[tuple[str, float]]) -> list[Label]` crops at each expected `u`, keeps only located ones, and returns them in input order.

- [ ] **Step 1: Write the failing tests** (fake heatmaps, built as numpy blobs at known crop positions)

```python
def test_to_label_maps_crop_box_to_panorama():
    l = to_label("chapel", Hotspot("chapel", 0.25, 0.4, 0.5, 0.2), u_centre=0.5, pano_w=2048)
    assert abs(l.u - 0.5) < 1e-6 and abs(l.w - 0.125) < 1e-6 and l.y == 0.4 and l.h == 0.2

def test_label_found_across_seam():
    def heat(img, texts):                      # blob at the crop's centre
        m = np.zeros((len(texts), 352, 352)); m[:, 150:200, 150:200] = 1; return m
    (l,) = LabelFinder(heat).find(Image.new("RGB", (2048, 512)), [("lantern", 0.99)])
    assert wrap_dist(l.u, 0.99) < 0.01

def test_unlocated_discovery_gets_no_label():
    assert LabelFinder(lambda img, t: np.zeros((len(t), 352, 352))).find(Image.new("RGB", (2048, 512)), [("x", 0.5)]) == []

def test_carry_moves_labels_away_from_centre_and_drops_offscreen():
    ls = [Label("a", "lantern", 0.55, .4, .05, .1), Label("b", "pond", 0.9, .4, .05, .1)]
    assert [(t, round(u, 4)) for t, u in carry(ls, walk_u=0.5)] == [("lantern", round(0.5 + 0.05 * 1.15, 4))]
```

- [ ] **Step 2: Run to verify they fail** → ImportError.
- [ ] **Step 3: Implement `dream/labels.py`**
- [ ] **Step 4: Run tests** → pass.
- [ ] **Step 5: Commit and push** `"Add discovery labels located on the panorama"`

---

### Task 5: GPU queue with prefetch priority

**Files:**
- Create: `src/vecworld/dream/gpu.py`
- Test: `tests/test_gpu_queue.py`

**Interfaces:**
- Produces: `class GpuQueue` with a single daemon worker thread.
  - `submit(fn: Callable[[], T], owner: str, prefetch: bool = False) -> concurrent.futures.Future[T]`. On-demand jobs always run before queued prefetch jobs. A new prefetch from the same owner **cancels** that owner's queued, not-yet-started prefetch. Owners are served round-robin within a priority.
  - `cancel_owner(owner: str) -> None`: cancels the owner's queued jobs.
  - Exceptions inside `fn` go to the future, and the worker keeps running.

- [ ] **Step 1: Write the failing tests** (jobs block on `threading.Event`s so ordering is deterministic)

```python
def test_on_demand_jumps_ahead_of_prefetch(): ...      # assert run order == ["busy", "demand", "pre"]
def test_new_prefetch_replaces_queued_one(): ...       # first prefetch future .cancelled() is True
def test_owners_take_turns(): ...                      # a1, a2, b1 queued → order a1, b1, a2
def test_queue_survives_failing_job():
    q = GpuQueue(); bad = q.submit(lambda: 1 / 0, "a"); ok = q.submit(lambda: 7, "a")
    assert isinstance(bad.exception(timeout=2), ZeroDivisionError) and ok.result(timeout=2) == 7
def test_cancel_owner(): ...
```

- [ ] **Step 2: Run to verify they fail** → ImportError.
- [ ] **Step 3: Implement `dream/gpu.py`**
A `threading.Condition` plus per-priority `OrderedDict[owner, deque]` for the round-robin.
- [ ] **Step 4: Run tests** → pass, and run them 20× in a row (`--count` is unavailable, so use a shell loop) to show they aren't flaky.
- [ ] **Step 5: Commit and push** `"Add GPU queue with prefetch priority"`

---

### Task 6: DreamService and sessions

**Files:**
- Create: `src/vecworld/dream/errors.py` (move `GameError`, `BadInput`, `NotStarted`, `Busy`, `GpuUnavailable`, `UnknownSession` here unchanged, and add `UnknownLabel(400, "unknown_label")`), `src/vecworld/dream/session.py`, `src/vecworld/dream/service.py`
- Test: `tests/test_dream_service.py` (FakePainter, FakeDreamer, a fake LabelFinder; `clock` injectable)

**Interfaces:**
- Consumes: Tasks 1–5; `match_palette`, `palette_stats`.
- Produces:
  - `@dataclass class View: view_id: str; image_path: Path; labels: list[Label]; theme: str; moods: list[str]; steps: int`
  - `@dataclass class DreamSession`. Fields: `id`, `seed`, `theme`, `moods`, `pano: Image | None`, `ref: Stats | None`, `labels`, `steps`, `ideas: deque[str]`, `drift: tuple[str, int] | None` (old theme and steps done), `version: int` (bumped on theme/mood change), `prefetch: tuple[float, int, Future] | None` (yaw, version, future), `busy`, `last_used`.
  - `class DreamService(painter, dreamer, labels: LabelFinder, scene_dir: Path, queue: GpuQueue | None = None, clock=time.monotonic)`:
    - `new_session() -> DreamSession` (TTL and LRU as before)
    - `start(sid, theme: str, moods: list[str]) -> View`: a fresh panorama with no discovery. It refills ideas in a background thread (not the GPU queue) and sets `ref`.
    - `walk(sid, yaw: float) -> View`: use the prefetch if `version` matches and the yaw is within `PREFETCH_TOLERANCE` (waiting for it if still running). Otherwise build on demand at priority.
    - `back(sid, yaw) -> View`: `walk(sid, yaw + 180)`
    - `approach(sid, label_id: str) -> View`: `walk` toward that label's yaw, with keep forced.
    - `set_theme(sid, theme) -> None`: starts a drift and bumps `version`.
    - `set_moods(sid, moods) -> None`: keeps only known `MOODS` and bumps `version`.
    - `prefetch(sid, yaw) -> None`: queues a prefetch build for that yaw, unless one for a yaw within tolerance and the current version already exists.
  - How a spot is built (`_build(s, yaw, keep: bool | None) -> tuple[Image, list[Label], dict]`, pure with respect to the session, so a prefetch never mutates anything until it's used):
    1. `u = yaw_to_u(yaw)`, `prev = dolly(roll_to(pano, u))`.
    2. **Keep:** if `keep` is true, or (when `keep` is None) any current label has `wrap_dist(label.u, u) ≤ FOCUS_HALF`, the focus moves to `0.5 ± KEEP_OFFSET`, alternating sides by step parity. Otherwise the focus is at 0.5.
    3. **Discovery:** none on steps where `(steps + 1) % QUIET_EVERY == 0` (focus still regenerates with `base`). Otherwise it's the next idea, or the next `FALLBACK_IDEAS` entry if the queue is empty.
    4. **Prompts:** `base = [(f"{theme}, {', '.join(moods)}, {STYLE}", 1.0)]`. During a drift it's weighted `[(old, 1 − k/3), (new, k/3)]`, with k the drift step 1..3. `focus` is the base text with `", {discovery} ahead in the distance"`.
    5. `img = painter.step(prev, base, focus, focus_at, seed)`, then `match_palette(img, ref, PALETTE_STRENGTH)`.
    6. **Labels:** `labels.find(img, carry(old_labels, u) + ([(discovery, focus_at)] if discovery else []))`.
  - **Using a built spot** (`_commit`) updates the session: pano, labels, steps+1, pops the idea, advances the drift. `ref` becomes this spot's palette when a drift completes or on the first spot after a mood change. It also saves the PNG and returns the View.
  - **Errors:** the same busy guard as before (409 busy on overlapping move requests; prefetch never takes the busy flag). Out-of-memory in the painter becomes `GpuUnavailable`, and the session is unchanged. Empty or whitespace theme gives `BadInput`; themes are truncated to `MAX_THEME_CHARS`. Moving before `start` gives `NotStarted`. A label id not in the current labels gives `UnknownLabel`.

- [ ] **Step 1: Write the failing tests** (names and key assertions)

```python
def test_start_makes_panorama_and_refills_ideas(svc): ...        # painter.calls[0][0] == "panorama"; dreamer called once
def test_walk_rolls_to_yaw_and_regenerates_ahead(svc): ...       # step kwargs focus_at == 0.5, focus contains "ahead in the distance"
def test_every_third_step_is_quiet(svc): ...                     # 3rd step: focus is None-discovery (focus text has no "ahead")
def test_walk_toward_label_keeps_it(svc): ...                    # label at u==yaw_to_u(yaw) → focus_at in (0.5 - KEEP_OFFSET, 0.5 + KEEP_OFFSET)
def test_approach_unknown_label(svc): ...                        # UnknownLabel
def test_back_walks_the_other_way(svc): ...                      # step called on roll_to(... yaw+180)
def test_walk_uses_matching_prefetch_without_new_paint(svc): ... # prefetch(yaw=10) then walk(yaw=30): painter step calls == 1
def test_walk_ignores_prefetch_for_another_direction(svc): ...   # prefetch(10), walk(100): 2 step calls, view from the 100° build
def test_mood_change_discards_prefetch(svc): ...                 # prefetch, set_moods(["night"]), walk: new build used; base text contains "night"
def test_theme_drift_weights(svc): ...                           # set_theme("frozen lake"): next 3 base weights 1/3, 2/3, 1
def test_mood_change_resets_palette_reference(svc): ...          # ref replaced after the first post-change spot
def test_oom_keeps_previous_view(svc): ...                       # painter.fail_next → GpuUnavailable; pano/labels/steps unchanged; busy False
def test_overlapping_moves_get_busy(svc): ...                    # a blocking painter + 2 threads → one 200, one Busy
def test_evicted_session_prefetch_is_harmless(svc_factory): ...  # prefetch queued, session evicted, queue still runs a later job
def test_empty_theme_rejected(svc): ...
def test_walk_before_start(svc): ...
```

- [ ] **Step 2: Run to verify they fail** → ImportError.
- [ ] **Step 3: Implement `dream/errors.py`, `dream/session.py`, `dream/service.py`**
- [ ] **Step 4: Run tests** → `uv run pytest -q -p no:warnings`: all pass (old game tests still pass until Task 9).
- [ ] **Step 5: Commit and push** `"Add Dream Walk service: start, walk, back, approach, drift, moods, prefetch"`

---

### Task 7: HTTP API for Dream Walk

**Files:**
- Modify: `src/vecworld/api/app.py` (replace the game routes with the dream routes; keep the error handler, thread pool and static mounts), `src/vecworld/__main__.py` (build `DreamService(TurboPainter(), QwenDreamer(), LabelFinder(), SCENE_DIR)`)
- Test: `tests/test_api.py` (rewrite for dream routes, using the fakes)

**Interfaces:**
- Produces:
  - `POST /api/session` → `{"session_id"}`
  - `POST /api/session/{sid}/start` body `{"theme": str, "moods": [str]}` → ViewOut
  - `POST .../walk` `{"yaw": float}` → ViewOut
  - `POST .../back` `{"yaw": float}` → ViewOut
  - `POST .../approach` `{"label": str}` → ViewOut
  - `POST .../theme` `{"theme": str}` → `{"ok": true}`
  - `POST .../moods` `{"moods": [str]}` → `{"ok": true}`
  - `POST .../prefetch` `{"yaw": float}` → 202 `{"ok": true}` (returns immediately)
  - ViewOut: `{"view_id", "image_url": "/scenes/{view_id}.png", "labels": [{"id","text","u","y","w","h"}], "theme", "moods", "steps"}`
  - Body validation errors use the same error format (`bad_input`, 422). `theme` has `max_length = 1000` (truncated to `MAX_THEME_CHARS` in the service). `yaw` must be finite.

- [ ] **Step 1: Write the failing tests**: a full flow (session → start → walk → back → approach a label → theme → moods), plus `prefetch` returning 202 quickly while a blocking painter runs, plus error codes (unknown session 404; walk before start 409; empty theme 422; NaN yaw 422; OOM 503 and the server survives).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests** → pass.
- [ ] **Step 5: Commit and push** `"Expose Dream Walk over HTTP"`

---

### Task 8: Viewer (look around, walk, labels, text bar, mood chips)

**Files:**
- Rewrite: `web/index.html`, `web/app.js`, `web/style.css` (keep the build 1–2 look: dark glass HUD, toast, spinner, always-visible label style)
- Test: `tests/test_ui.py` (rewrite; marker `ui`; fake-model server fixture as before, plus `window.vecworld` read-only state: `{yaw, view, labels}`)

**Interfaces:**
- Consumes: Task 7 routes.
- Produces (behaviour):
  - Canvas draws a 90° horizontal field of view into the current panorama at `yaw`, wrapping across the seam (spike viewer math).
  - Turning: drag (pointer events, including touch), A/D and ←/→ held at 150°/s, smooth at rAF rate.
  - **Labels** are drawn as HTML elements positioned from `(u, y, w, h)` relative to the current yaw. A label is visible when inside the 90° view, on both sides of the seam if needed. Clicking one calls `approach`.
  - **W/↑** → `walk(yaw)`. **S/↓** → `back(yaw)`, after which the view yaw becomes `yaw + 180`.
  - **Transition:** a dolly-in plus crossfade of 900 ms (back: dolly-out).
  - **Pre-generation:** after 500 ms without turning, call `prefetch(yaw)`; again on each settle more than 45° from the last prefetched yaw.
  - **Text bar:** the first submit calls `start(theme, moods)`; later submits call `theme(...)` and toast "the dream is drifting toward …".
  - **Mood chips:** six toggles from `MOODS` (served in `index.html`). A toggle calls `moods([...active])`. Chips work before start (sent with start).
  - **Phones:** on-screen ↑ and ↓ buttons (bottom right) that do W and S.
  - Keys are ignored while the text bar has focus or a move is in flight. Any failed request restores the previous view and labels and shows the error message in a toast.

- [ ] **Step 1: Write the failing UI tests**

```python
def test_start_shows_panorama_and_chips(server, browser): ...
def test_turning_changes_yaw_without_requests(server, browser): ...      # hold "d" 500 ms → yaw > 30; no /walk recorded
def test_settled_view_prefetches_once(server, browser): ...              # wait 800 ms → exactly one "prefetch" request
def test_w_walks_and_s_turns_around(server, browser): ...                # after S, window.vecworld.yaw ≈ old + 180 (mod 360)
def test_label_click_approaches(server, browser): ...
def test_label_draws_across_seam(server, browser): ...                   # label u=0.99, yaw=-178 → label element visible
def test_failed_walk_keeps_view(server, browser): ...
def test_text_bar_drifts_theme_after_start(server, browser): ...
def test_keys_ignored_while_typing(server, browser): ...
def test_phone_buttons_walk(server, browser): ...                        # 390×844 viewport, tap ↑ → /walk
```

- [ ] **Step 2: Run to verify they fail** (`uv run pytest -m ui -q`).
- [ ] **Step 3: Implement `web/`**
- [ ] **Step 4: Run tests** → `uv run pytest -m "not gpu" -q -p no:warnings` all pass; `node` isn't on the laptop, so the browser tests are the JS check.
- [ ] **Step 5: Commit and push** `"Add Dream Walk viewer"`

---

### Task 9: Remove the meaning-space game + end-to-end on the laptop

**Files:**
- Delete: `src/vecworld/space/`, `src/vecworld/game/`, `src/vecworld/scenes/generator.py`, `src/vecworld/scenes/prompt.py`, `scripts/build_vocab.py`, `scripts/embed_vocab.py`, `scripts/measure_scenes.py`, `data/vocab.txt`, and the tests for them: `test_index.py`, `test_vocab.py`, `test_embedder_gpu.py`, `test_service.py`, `test_prompt.py`, `test_generator_gpu.py`.
- Keep: `scenes/hotspots.py` (`locate` and `ClipSegFinder` are used by labels) and `scenes/palette.py`.
- Modify: `config.py` (drop the constants no longer referenced), `tests/conftest.py` (drop the meaning-space fakes), `pyproject.toml` (`uv remove wordfreq nltk` if nothing imports them), `README.md` (one short section: what it is and how to run it).

- [ ] **Step 1: Delete and clean up, then run** `grep -rn "vecworld.space\|vecworld.game\|wordfreq\|nltk" src tests` → no matches.
- [ ] **Step 2: Full suite with the server stopped:** `uv run pytest -q -p no:warnings -m "gpu or not gpu or ui"` → all pass.
- [ ] **Step 3: End-to-end:** start the server (`setsid nohup ~/.local/bin/uv run python -m vecworld > /tmp/vecworld.log 2>&1 < /dev/null &`) and, from the Mac, drive it with the puppeteer-core setup in the scratchpad (`/private/tmp/claude-501/-Users-veera/23f6a01b-7ea3-42a8-b8f5-815696036771/scratchpad/pptr`). Measure and record in the commit:
  - first panorama time (≤ 5 s)
  - on-demand step time (≤ 6 s)
  - time from W to transition start after a settled prefetch (< 0.5 s)
  - peak VRAM via `nvidia-smi` (< 4096 MiB total)
  - ideas generated for "haunted forest"
  - how many of 6 steps produced a located label

  Take screenshots of start, after 3 walks, a label close-up, and the phone layout, and look at them.
- [ ] **Step 4: Commit and push** `"Remove the meaning-space game; Dream Walk end to end"` (body: the measurements)

---

### Model note
If the SD-Turbo vs quantized SDXL-Turbo comparison (run before this plan executes; results go into spec §5) picks SDXL-Turbo, add `XLPainter(Painter)` in `dream/painter.py` in **Task 2** with the same interface:
- UNet int8 via optimum-quanto
- text encoders on CPU fp32, with the per-window `added_cond_kwargs` (pooled embeddings, `time_ids = [512, 512, 0, 0, 512, 512]`)
- VAE fp32 with 32-latent tiles

`__main__` picks the painter, and the same GPU tests apply.
