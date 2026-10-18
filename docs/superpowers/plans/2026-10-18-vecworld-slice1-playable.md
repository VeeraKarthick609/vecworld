# vecworld Slice 1 (Playable Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A player opens a browser page on the `gpu` laptop, types a word, lands in a generated scene of that region of a real LLM's meaning-space, clicks an object to step toward it, and sees the corner mind-map, end to end on real models.

**Architecture:** A Python package with three layers. `space` turns text into centred, normalised LLM vectors and does neighbour, step and projection maths. `scenes` turns concepts into SD-Turbo images and finds where each concept sits in the image. `game` + `api` hold per-player sessions behind FastAPI, with a vanilla-JS page in front. Models sit behind small protocols so tests use fakes, and slice 3 can move them behind SmartBatch without touching game logic.

**Tech Stack:** Python 3.12 (via uv), PyTorch + CUDA, transformers (Qwen2.5-0.5B), diffusers (stabilityai/sd-turbo), CLIPSeg (CIDAS/clipseg-rd64-refined), FastAPI + uvicorn, numpy, pytest, vanilla HTML/JS/canvas.

**Spec:** `docs/superpowers/specs/2026-10-18-vecworld-design.md`

### Where this slice sits

The spec is split into four plans, each shippable on its own:
1. **This plan: playable core.** Jump, step, scenes, hotspots, mind-map. Transitions are the spec's fallback "zoom toward the clicked object" animation, not real morphs.
2. **Continuity:** real latent morphs, background fade-out over steps, and tuning of the step cap and image-to-image strength.
3. **Serving and caching:** SmartBatch, prefetch sliding window, GPU budget, RAM tier, shared jump scenes on a grid, public tunnel, 5-player load test.
4. **Storage library:** native `io_uring` + `O_DIRECT` slot store (file first, raw partition later).

### Decisions this plan makes (the spec left them to the builder)

- **Vectors:** Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`). Mean-pool hidden states of layer `LAYER = 16` (of 24) over the text's tokens. Subtract the vocabulary mean and L2-normalise, because raw LLM hidden states are highly anisotropic. Runs on **CPU**, which keeps VRAM for SD-Turbo.
- **Vocabulary:** about 3,000 picturable English nouns, built once from `wordfreq` top words intersected with WordNet nouns. Saved as `data/vocab.txt`, with embeddings in `data/vocab.npy`.
- **Distance and step cap:** angle between unit vectors. A step moves along the great circle (slerp) toward the clicked concept by `min(angle, MAX_STEP_ANGLE)`, with `MAX_STEP_ANGLE = 0.35` rad (tunable). This is the spec's "5 points maximum in a single click".
- **Projection:** PCA to 2D, fitted on the vocabulary.
- **Scenes:** SD-Turbo at 512×512 on the GPU, fp16. A jump uses txt2img with 1 step. A step uses img2img on a zoom-crop of the previous image centred on the clicked hotspot, with `STEP_STRENGTH = 0.5` and 2 steps. That keeps the clicked object centred and carries the old scene forward. Seed is fixed per session.
- **Clickable objects:** the pinned centre concept plus the top `N_NEIGHBOURS = 3` neighbours of the landed position. That's 3–4 total, per the spec's GPU budget.
- **Hotspots:** CLIPSeg on **CPU** gives a heatmap per concept. The hotspot is the bounding box of pixels above 0.5 × that concept's max. If the max is below `HOTSPOT_MIN = 0.2`, the concept gets a fallback slot so it's still clickable.
- **Style:** `STYLE = "dreamlike painterly landscape, soft volumetric light, oil painting, cohesive palette"`.
- **No SmartBatch yet.** Models load in-process behind `Embedder`, `SceneGenerator` and `HotspotFinder` protocols, and one asyncio lock serialises GPU work. Slice 3 swaps the implementations.

## Global Constraints

- Everything runs on the `gpu` laptop (`ssh gpu`, repo `~/dev/vecworld`). No paid or hosted model APIs.
- RTX 3050 Ti, 4 GB VRAM. Only SD-Turbo goes on the GPU in this slice. Embedder and CLIPSeg run on CPU.
- Python 3.12 managed by `uv`. The system Python is 3.14, so don't rely on it.
- Run directly on the host, not in Docker.
- Only *clickable* objects must be real current neighbours or the pinned clicked concept. Background carry-over is fine.
- The clicked concept is pinned at the centre of the next scene even if the step cap stops the player short of it.
- No scores, accounts, or shared multiplayer state. Each session is independent.
- Unit tests must not load real models (use fakes). Real-model tests are marked `@pytest.mark.gpu`.
- Commits: author `Veera <veerakarthick609@gmail.com>` (already set in repo config), and end each message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Junk jump text** (empty, only spaces, emoji, 5,000 characters): empty or whitespace gives 422. Anything else is truncated to `MAX_TEXT_CHARS = 200` and still lands somewhere. Owned by Task 5.
2. **Step before any jump**, or clicking a concept that isn't a current hotspot: 409 or 400 with a clear message, never a 500. Owned by Task 5.
3. **Double-click / overlapping requests for one session:** the second request gets 409 `busy` while the first runs. State is never half-updated. Owned by Task 5.
4. **CLIPSeg can't find a concept in the image:** the concept still gets a clickable fallback hotspot, and hotspots never fall outside the 0–1 range. Owned by Task 4.
5. **GPU out of memory mid-generation:** the request returns 503, the session keeps its previous scene, and the server keeps serving. Owned by Task 6.

---

### Task 1: Project scaffold + meaning-space maths

**Files:**
- Create: `pyproject.toml`, `vecworld/__init__.py`, `vecworld/config.py`, `vecworld/space/__init__.py`, `vecworld/space/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Produces:
  - `config.py` constants: `LAYER=16`, `MAX_STEP_ANGLE=0.35`, `N_NEIGHBOURS=3`, `STEP_STRENGTH=0.5`, `HOTSPOT_MIN=0.2`, `MAX_TEXT_CHARS=200`, `STYLE` (as above), `DATA_DIR=Path("data")`, `SCENE_DIR=Path("var/scenes")`
  - `class MeaningSpace(words: list[str], vectors: np.ndarray)`: `vectors` is `(V, D)` float32, already unit-norm
    - `.neighbours(p: np.ndarray, k: int, exclude: set[str] = frozenset()) -> list[tuple[str, float]]`: top-k by cosine, descending
    - `.vector(word: str) -> np.ndarray` (KeyError if unknown)
    - `.step_toward(p: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray`: slerp, unit-norm result
    - `.project(p: np.ndarray) -> tuple[float, float]`: PCA-2D, fitted in `__init__`, scaled so vocab coords lie in [-1, 1]
  - `angle(a, b) -> float` (radians, clipped dot)

- [ ] **Step 1: Install uv and create the project**

Run on gpu: `curl -LsSf https://astral.sh/uv/install.sh | sh && cd ~/dev/vecworld && uv init --package --python 3.12 --name vecworld . && uv add numpy && uv add --dev pytest`
Expected: `uv run python -V` prints `Python 3.12.x`.
Add to `pyproject.toml`: `[tool.pytest.ini_options] markers = ["gpu: needs real models/GPU"]` and `addopts = "-m 'not gpu'"`. Append `data/*.npy` and `var/` to `.gitignore`.

- [ ] **Step 2: Write the failing tests**

```python
def unit(v): v = np.asarray(v, np.float32); return v / np.linalg.norm(v)

def make_space():
    words = ["ocean", "sea", "ship", "desert", "sand"]
    vecs = np.stack([unit([1,0,0]), unit([.95,.1,0]), unit([.8,.6,0]), unit([0,0,1]), unit([0,.1,.95])])
    return MeaningSpace(words, vecs)

def test_neighbours_sorted_and_excludes():
    s = make_space()
    got = s.neighbours(s.vector("ocean"), k=2, exclude={"ocean"})
    assert [w for w, _ in got] == ["sea", "ship"]
    assert got[0][1] >= got[1][1]

def test_step_is_capped():
    s = make_space()
    p = s.step_toward(s.vector("ocean"), s.vector("desert"), max_angle=0.35)
    assert abs(angle(s.vector("ocean"), p) - 0.35) < 1e-4
    assert abs(np.linalg.norm(p) - 1) < 1e-5

def test_step_reaches_close_target():
    s = make_space()
    p = s.step_toward(s.vector("ocean"), s.vector("sea"), max_angle=0.35)
    assert angle(p, s.vector("sea")) < 1e-4

def test_step_toward_self_is_noop():
    s = make_space()
    p = s.step_toward(s.vector("ocean"), s.vector("ocean"), 0.35)
    assert angle(p, s.vector("ocean")) < 1e-5

def test_project_in_range():
    s = make_space()
    for w in s.words:
        x, y = s.project(s.vector(w))
        assert -1.0001 <= x <= 1.0001 and -1.0001 <= y <= 1.0001
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_index.py -v`
Expected: FAIL with `ImportError` / `cannot import name 'MeaningSpace'`.

- [ ] **Step 4: Implement `config.py` and `space/index.py`**

PCA via `np.linalg.svd` on vocab vectors minus their mean. Keep the mean, the top-2 components, and the max-abs scale. Slerp with `sin` formula. When the angle is below 1e-6, return the target.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add meaning-space maths" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: LLM embedder + vocabulary

**Files:**
- Create: `vecworld/space/embedder.py`, `vecworld/space/vocab.py`, `scripts/build_vocab.py`, `scripts/embed_vocab.py`
- Test: `tests/test_vocab.py`, `tests/test_embedder_gpu.py`
- Output (committed): `data/vocab.txt`. Not committed (gitignored): `data/vocab.npy`, `data/vocab_mean.npy`

**Interfaces:**
- Consumes: `MeaningSpace`, `config.LAYER`, `config.MAX_TEXT_CHARS`
- Produces:
  - `class Embedder(Protocol): def embed(self, texts: list[str]) -> np.ndarray` returns raw `(N, D)` float32, not centred
  - `class LLMEmbedder(Embedder)`: `__init__(model_id="Qwen/Qwen2.5-0.5B", layer=LAYER, device="cpu")`. Truncates each text to `MAX_TEXT_CHARS` before tokenising, mean-pools layer `layer` over non-padding tokens, batches by 64.
  - `def centre(raw: np.ndarray, mean: np.ndarray) -> np.ndarray`: subtract the mean, L2-normalise rows
  - `def load_space(data_dir: Path) -> tuple[MeaningSpace, np.ndarray]` returns the space plus `vocab_mean`
  - `def build_vocab(n: int = 3000) -> list[str]`: lowercase, alphabetic, 3–15 chars, has a WordNet noun synset, ordered by wordfreq frequency, deduplicated

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vocab.py
def test_build_vocab_shape():
    v = build_vocab(500)
    assert len(v) == 500 and len(set(v)) == 500
    assert all(w.isalpha() and w.islower() and 3 <= len(w) <= 15 for w in v)
    assert "ocean" in build_vocab(3000)

def test_centre_unit_rows():
    raw = np.random.rand(10, 8).astype(np.float32) + 5
    out = centre(raw, raw.mean(0))
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)

# tests/test_embedder_gpu.py
@pytest.mark.gpu
def test_llm_neighbourhood_is_sensible():
    space, mean = load_space(DATA_DIR)
    e = LLMEmbedder()
    ocean = centre(e.embed(["ocean"]), mean)[0]
    words = [w for w, _ in space.neighbours(ocean, k=25)]
    assert {"sea", "water", "waves", "beach", "coast", "marine"} & set(words)
    assert angle(ocean, space.vector("sea")) < angle(ocean, space.vector("keyboard"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_vocab.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `vocab.py`, `embedder.py` and both scripts**

Add deps: `uv add torch transformers wordfreq nltk` (torch from the CUDA index uv picks by default on Linux). `build_vocab.py` runs `nltk.download("wordnet")` if missing and writes `data/vocab.txt`. `embed_vocab.py` embeds the vocab with `LLMEmbedder`, saves `vocab_mean.npy` and the centred `vocab.npy`, and prints the top-10 neighbours for `ocean`, `storm`, `loneliness` and `ship` so a human can sanity-check them.

- [ ] **Step 4: Build the data and run all tests**

Run: `uv run python scripts/build_vocab.py && uv run python scripts/embed_vocab.py && uv run pytest -v && uv run pytest -m gpu -v`
Expected: unit tests pass, `test_llm_neighbourhood_is_sensible` passes, and the printed neighbour lists look related. If the gpu test fails, try `LAYER` in {8, 12, 20} and keep the best. Record the chosen layer and the four neighbour lists in the commit message.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add Qwen embedder and vocabulary (layer N, neighbours: ...)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Scene generator

**Files:**
- Create: `vecworld/scenes/__init__.py`, `vecworld/scenes/prompt.py`, `vecworld/scenes/generator.py`
- Test: `tests/test_prompt.py`, `tests/test_generator_gpu.py`

**Interfaces:**
- Consumes: `config.STYLE`, `config.STEP_STRENGTH`
- Produces:
  - `def scene_prompt(centre: str, others: list[str]) -> str` returns `f"{STYLE}, a {centre} in the centre, with {', '.join(others)} around it"` (when `others` is empty, drop the `with …` clause)
  - `def zoom_crop(img: Image.Image, cx: float, cy: float, frac: float = 0.6) -> Image.Image`: square crop of side `frac`×size centred on (cx, cy) in 0–1 coords, clamped inside the image, resized back to the original size
  - `class SceneGenerator(Protocol)`:
    - `def fresh(self, prompt: str, seed: int) -> Image.Image`
    - `def evolve(self, prev: Image.Image, prompt: str, seed: int) -> Image.Image`
  - `class TurboGenerator(SceneGenerator)`: `stabilityai/sd-turbo`, fp16, `cuda`, 512×512, `guidance_scale=0.0`. `fresh` uses 1 step. `evolve` uses `AutoPipelineForImage2Image.from_pipe` (shared weights), `strength=STEP_STRENGTH`, `num_inference_steps=2`. Call `enable_attention_slicing()` on both.

- [ ] **Step 1: Write the failing tests**

```python
def test_scene_prompt():
    p = scene_prompt("ship", ["waves", "gull"])
    assert p.startswith(STYLE) and "a ship in the centre" in p and "waves, gull" in p
    assert "with" not in scene_prompt("ship", [])

def test_zoom_crop_clamps_and_keeps_size():
    img = Image.new("RGB", (512, 512))
    assert zoom_crop(img, 0.98, 0.02).size == (512, 512)   # corner click must not crash

@pytest.mark.gpu
def test_turbo_fresh_and_evolve_fit_in_vram():
    g = TurboGenerator()
    a = g.fresh(scene_prompt("ocean", ["ship", "storm"]), seed=1)
    b = g.evolve(zoom_crop(a, .5, .5), scene_prompt("ship", ["harbor"]), seed=1)
    assert a.size == b.size == (512, 512)
    assert torch.cuda.max_memory_allocated() < 3.6 * 2**30
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `prompt.py` and `generator.py`**

`uv add diffusers accelerate pillow`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v && uv run pytest -m gpu tests/test_generator_gpu.py -v -s`
Expected: all pass. Print the wall time of `fresh` and `evolve` (after one warm-up) and record both in the commit message. The spec's 1–2 s figure is an estimate, so this measures it.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add SD-Turbo scene generator (fresh Xs, evolve Ys)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Hotspot finder

**Files:**
- Create: `vecworld/scenes/hotspots.py`
- Test: `tests/test_hotspots.py`, `tests/test_hotspots_gpu.py`

**Interfaces:**
- Consumes: `config.HOTSPOT_MIN`
- Produces:
  - `@dataclass(frozen=True) class Hotspot: concept: str; x: float; y: float; w: float; h: float`, top-left plus size, all in 0–1
  - `def boxes_from_heatmaps(concepts: list[str], maps: np.ndarray) -> list[Hotspot]`: `maps` is `(N, H, W)` in 0–1. Box = bounding box of pixels ≥ 0.5×that map's max. If the max is below `HOTSPOT_MIN`, use a fallback slot. Fallback slots are fixed, non-overlapping boxes `FALLBACK = [(0.4,0.35,0.2,0.3), (0.08,0.55,0.2,0.3), (0.72,0.55,0.2,0.3), (0.4,0.05,0.2,0.25)]`, taken in order and skipping any already used. Output order equals input order.
  - `class HotspotFinder(Protocol): def find(self, img: Image.Image, concepts: list[str]) -> list[Hotspot]`
  - `class ClipSegFinder(HotspotFinder)`: `CIDAS/clipseg-rd64-refined` on CPU, sigmoid logits, one batch for all concepts, returns `boxes_from_heatmaps(...)`

- [ ] **Step 1: Write the failing tests**

```python
def test_box_from_blob():
    m = np.zeros((1, 100, 100)); m[0, 20:40, 60:80] = 1.0
    (h,) = boxes_from_heatmaps(["ship"], m)
    assert h.concept == "ship"
    assert (round(h.x, 2), round(h.y, 2), round(h.w, 2), round(h.h, 2)) == (0.6, 0.2, 0.2, 0.2)

def test_weak_map_gets_fallback_inside_bounds():
    m = np.full((2, 50, 50), 0.05)
    hs = boxes_from_heatmaps(["ghost", "void"], m)
    assert [h.concept for h in hs] == ["ghost", "void"]
    assert (hs[0].x, hs[0].y) != (hs[1].x, hs[1].y)
    for h in hs:
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 and h.y + h.h <= 1
```

Also a `@pytest.mark.gpu` test: generate `fresh(scene_prompt("ship", ["sun"]))`, call `ClipSegFinder().find(img, ["ship", "sun"])`, assert 2 hotspots in bounds, and print the elapsed time.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_hotspots.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `hotspots.py`**

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v && uv run pytest -m gpu tests/test_hotspots_gpu.py -v -s`
Expected: all pass. Record the CLIPSeg CPU time. If it's over 1.5 s for 4 concepts, note it for slice 3.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add CLIPSeg hotspot finder" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Game service + sessions

**Files:**
- Create: `vecworld/game/__init__.py`, `vecworld/game/session.py`, `vecworld/game/service.py`
- Test: `tests/test_service.py` (fakes only)

**Interfaces:**
- Consumes: `MeaningSpace`, `Embedder`, `centre`, `SceneGenerator`, `scene_prompt`, `zoom_crop`, `HotspotFinder`, `Hotspot`, config constants
- Produces:
  - `@dataclass class Session: id: str; seed: int; position: np.ndarray | None = None; image: Image.Image | None = None; hotspots: list[Hotspot] = field(default_factory=list); trail: list[tuple[float, float]] = field(default_factory=list); busy: bool = False`
  - `@dataclass class Scene: scene_id: str; image_path: Path; hotspots: list[Hotspot]; me: tuple[float, float]; trail: list[tuple[float, float]]; neighbours: list[tuple[str, float, float]]; reach_radius: float`. `neighbours` holds (word, x, y) for the top 30 neighbours, for the mind-map.
  - Errors: `class GameError(Exception)` with subclasses `BadInput` (→422), `NotStarted` (→409), `Busy` (→409), `UnknownConcept` (→400), `GpuUnavailable` (→503)
  - `class GameService(space, vocab_mean, embedder, generator, finder, scene_dir: Path)`:
    - `def new_session(self) -> Session`: random id (`secrets.token_urlsafe(8)`), random seed
    - `def get(self, sid: str) -> Session` raises `KeyError`
    - `def jump(self, sid: str, text: str) -> Scene`
    - `def step(self, sid: str, concept: str) -> Scene`
- Behaviour to implement:
  - **jump:** strip the text. Empty → `BadInput`. Truncate to `MAX_TEXT_CHARS`. Position = `centre(embedder.embed([text]), vocab_mean)[0]`. Centre concept = the nearest vocab word. Others = the next `N_NEIGHBOURS` excluding the centre. `generator.fresh(scene_prompt(...), session.seed)`.
  - **step:** no position yet → `NotStarted`. Concept not in the current hotspots → `UnknownConcept`. New position = `space.step_toward(position, space.vector(concept), MAX_STEP_ANGLE)`. The centre concept is the **clicked concept (pinned)**. Others = the top `N_NEIGHBOURS` of the new position excluding the clicked concept. Image = `generator.evolve(zoom_crop(prev, hotspot centre x, y), prompt, seed)`.
  - **Both:** hotspots = `finder.find(img, [centre] + others)`. Save the image to `scene_dir/{scene_id}.png`. Append `space.project(position)` to the trail. `reach_radius` = mean 2D distance from `me` to `project(step_toward(position, v, MAX_STEP_ANGLE))` over the current hotspot concepts.
  - **Busy guard:** if `session.busy`, raise `Busy`. Otherwise set it, then clear it in `finally`. Mutate the session only after the image and hotspots are ready, so a failure leaves the previous scene intact.
  - **OOM:** `torch.cuda.OutOfMemoryError` from the generator → `torch.cuda.empty_cache()` and raise `GpuUnavailable`.

- [ ] **Step 1: Write the failing tests**

Fakes: `FakeEmbedder` maps text to fixed vectors from the Task 1 toy space (unknown text → a fixed vector near "ocean"). `FakeGenerator` returns a solid-colour 64×64 image and records calls. `FakeFinder` returns evenly spaced hotspots in input order. `OomGenerator.fresh` raises `torch.cuda.OutOfMemoryError`.

```python
def test_jump_lands_and_lists_clickables(svc):
    s = svc.new_session(); sc = svc.jump(s.id, "ocean")
    assert [h.concept for h in sc.hotspots][0] == "ocean"
    assert 2 <= len(sc.hotspots) <= 1 + N_NEIGHBOURS
    assert sc.image_path.exists() and len(sc.trail) == 1

def test_step_pins_clicked_concept_even_when_capped(svc):
    s = svc.new_session(); svc.jump(s.id, "ocean")
    target = [h.concept for h in svc.get(s.id).hotspots][-1]
    sc = svc.step(s.id, target)
    assert sc.hotspots[0].concept == target
    assert svc.generator.calls[-1][0] == "evolve" and len(sc.trail) == 2

@pytest.mark.parametrize("text", ["", "   "])
def test_empty_jump_rejected(svc, text):
    with pytest.raises(BadInput): svc.jump(svc.new_session().id, text)

def test_huge_and_emoji_jump_still_lands(svc):
    s = svc.new_session()
    assert svc.jump(s.id, "🌊" * 5000).hotspots
    assert len(svc.embedder.last_texts[0]) <= MAX_TEXT_CHARS

def test_step_before_jump(svc):
    with pytest.raises(NotStarted): svc.step(svc.new_session().id, "ship")

def test_step_unknown_concept(svc):
    s = svc.new_session(); svc.jump(s.id, "ocean")
    with pytest.raises(UnknownConcept): svc.step(s.id, "keyboard")

def test_busy_session_rejected(svc):
    s = svc.new_session(); s.busy = True
    with pytest.raises(Busy): svc.jump(s.id, "ocean")

def test_oom_keeps_previous_scene(svc_factory):
    svc = svc_factory(generator=FakeGenerator()); s = svc.new_session(); svc.jump(s.id, "ocean")
    before = (svc.get(s.id).position.copy(), list(svc.get(s.id).trail))
    svc.generator = OomGenerator()
    with pytest.raises(GpuUnavailable): svc.jump(s.id, "desert")
    assert np.allclose(svc.get(s.id).position, before[0]) and svc.get(s.id).trail == before[1]
    assert svc.get(s.id).busy is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_service.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `session.py` and `service.py`**

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v`
Expected: all unit tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add game service with jump and capped step" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: HTTP API

**Files:**
- Create: `vecworld/api/__init__.py`, `vecworld/api/app.py`, `vecworld/__main__.py`
- Test: `tests/test_api.py` (FastAPI `TestClient` + Task 5 fakes)

**Interfaces:**
- Consumes: `GameService`, `GameError` subclasses, `Scene`
- Produces:
  - `def create_app(service: GameService, web_dir: Path) -> FastAPI`
  - `POST /api/session` → `{"session_id": str}`
  - `POST /api/session/{sid}/jump` body `{"text": str}` → SceneOut
  - `POST /api/session/{sid}/step` body `{"concept": str}` → SceneOut
  - SceneOut JSON: `{"scene_id", "image_url": "/scenes/{scene_id}.png", "hotspots": [{"concept","x","y","w","h"}], "mindmap": {"me": [x,y], "trail": [[x,y]...], "neighbours": [{"concept","x","y"}], "reach_radius": float}}`
  - Error JSON: `{"error": "<snake_case code>", "message": str}`, with status from the `GameError` mapping in Task 5. Unknown sid → 404 `unknown_session`.
  - Static: `/scenes/*` from `SCENE_DIR`, `/` from `web_dir`
  - `python -m vecworld` builds the real service (`load_space`, `LLMEmbedder`, `TurboGenerator`, `ClipSegFinder`) and serves on `0.0.0.0:8080`
  - Service calls run in a single-worker `ThreadPoolExecutor` via `run_in_executor`, so GPU work is serialised and the event loop stays free

- [ ] **Step 1: Write the failing tests**

```python
def test_full_flow(client):
    sid = client.post("/api/session").json()["session_id"]
    r = client.post(f"/api/session/{sid}/jump", json={"text": "ocean"}); assert r.status_code == 200
    body = r.json(); assert body["image_url"].startswith("/scenes/") and body["mindmap"]["me"]
    assert client.get(body["image_url"]).status_code == 200
    c = body["hotspots"][-1]["concept"]
    assert client.post(f"/api/session/{sid}/step", json={"concept": c}).status_code == 200

@pytest.mark.parametrize("path,payload,status,code", [
    ("jump", {"text": " "}, 422, "bad_input"),
    ("step", {"concept": "ship"}, 409, "not_started"),
])
def test_errors(client, path, payload, status, code):
    sid = client.post("/api/session").json()["session_id"]
    r = client.post(f"/api/session/{sid}/{path}", json=payload)
    assert (r.status_code, r.json()["error"]) == (status, code)

def test_unknown_session(client):
    assert client.post("/api/session/nope/jump", json={"text": "x"}).status_code == 404

def test_oom_is_503_and_server_survives(client_oom):
    sid = client_oom.post("/api/session").json()["session_id"]
    assert client_oom.post(f"/api/session/{sid}/jump", json={"text": "ocean"}).status_code == 503
    assert client_oom.post("/api/session").status_code == 200
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `app.py` and `__main__.py`**

`uv add fastapi uvicorn httpx`. Register one exception handler for `GameError` that maps the subclass to its status and code.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Add HTTP API" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Play screen (layout B) + end-to-end run

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/style.css`
- Test: manual end-to-end on gpu (below), plus `tests/test_api.py::test_index_served`

**Interfaces:**
- Consumes: the Task 6 HTTP API and JSON shapes
- Produces: the page described in spec §5.2, layout B:
  - The scene image fills the viewport (`object-fit: cover`). Absolutely positioned hotspot divs use the 0–1 boxes with a dashed outline on hover and a word label.
  - A bottom typing bar: Enter → jump.
  - A corner mind-map `<canvas>` (about 200×160): neighbours as small dots with no labels, the trail as a line, the player as a highlighted dot, and a dashed ring of `reach_radius`. Map coords are [-1, 1] centred on `me` with a zoom so the ring is about 25% of the canvas width.
  - **Step transition (fallback from spec §6.3):** on click, animate a CSS zoom (`transform: scale(1.6)` with `transform-origin` at the hotspot centre, 600 ms) on the current image while the request runs, then crossfade (300 ms) to the new image. A jump does a plain fade.
  - First load shows only the typing bar, with the placeholder text "type a word to enter the model's mind…". A spinner is shown while a request runs. Error JSON shows its `message` in a small toast. Input is disabled while a request runs, which is the client side of the busy guard.

- [ ] **Step 1: Write the failing test**

```python
def test_index_served(client):
    r = client.get("/"); assert r.status_code == 200 and "<canvas" in r.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api.py::test_index_served -v`
Expected: FAIL (404).

- [ ] **Step 3: Implement `web/` files**

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 5: End-to-end on the real models**

Run on gpu: `uv run python -m vecworld`, then from the Mac open `http://mrjarvis:8080`.
Check by hand and write the results in the commit message:
- Jumping to "ocean" shows a scene within the spec's **5 s** target (record the time).
- 3–4 hotspots appear, each labelled with a real word, and clicking one zooms, then shows a new scene with that object at the centre. Record the click-to-new-scene time against the spec's **2 s** target. Missing it is expected in this slice (no prefetch until slice 3), but record it.
- The mind-map dot moves, the trail grows, and the ring is visible.
- `nvidia-smi` peak memory stays under 4096 MiB.
- Two browser tabs playing at once both work. One may wait, and neither errors.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add play screen; e2e: jump Xs, step Ys, peak VRAM Z MiB" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
