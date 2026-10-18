# vecworld Build 2 (Walking) Implementation Plan

**Goal:** What you see is what you can click, abstract words get real scenes, you
can walk with WASD/arrows without looping, and colours hold steady over long walks.

**Spec:** `docs/superpowers/specs/2026-10-18-vecworld-design.md` and
`docs/superpowers/specs/2026-10-18-vecworld-build2-addendum.md`, which has the
calls and the measured baseline.

**Branch:** `slice2-walking` from `slice1-playable`. Commit and push after each task.
There are no attribution trailers in commit messages (the user's preference).

## Global constraints

- Everything from slice 1 stays green: error codes 422/409/400/404/503, the
  busy guard, OOM keeping the previous scene, UI restore on failed moves,
  session TTL/LRU, VRAM < 4 GB.
- Targets: step ≈ 1.5 s (≤ 2 s), jump ≤ 5 s, back instant (no GPU).
- Unit tests use fakes. Real-model checks are `@pytest.mark.gpu`, browser
  checks are `@pytest.mark.ui`.
- Visual changes get checked by generating images and looking at them, not by
  a metric alone.

## Config added (`config.py`)

`N_PROMPT_NEIGHBOURS = 5`, `N_CANDIDATES = 6`, `N_NEIGHBOURS = 3` (offered),
`MAX_HOTSPOT_AREA = 0.5`, `MAX_HOTSPOT_IOU = 0.6`, `CENTRE_BOX = (0.35, 0.35, 0.3, 0.3)`,
`RECENT = 4`, `HISTORY = 10`, `PALETTE_STRENGTH` (tuned in Task 5).
`STYLE = "dreamlike painterly scene, soft volumetric light, oil painting, cohesive palette"`.

---

### Task 1: Only offer located hotspots (+ fast CLIPSeg)

**Files:** `scenes/hotspots.py`, `game/service.py`, `config.py`, `tests/test_hotspots.py`,
`tests/test_service.py`, `tests/conftest.py`, `tests/test_hotspots_gpu.py`

- `locate(concepts, maps) -> list[Hotspot | None]`: connected blob around the peak, or None below `HOTSPOT_MIN`.
- `choose(found, k) -> list[Hotspot]`: `found[0]` is the centre and always comes back, as its box or `CENTRE_BOX` when missing or bigger than `MAX_HOTSPOT_AREA`. Then up to k neighbours, in order, that are located, ≤ area cap, and ≤ IoU cap against everything already chosen.
- `ClipSegFinder.find` returns `locate(...)` from one shared vision pass.
- The service asks about centre + `N_CANDIDATES`, prompts with `N_PROMPT_NEIGHBOURS`, and offers `choose(found, N_NEIGHBOURS)`.

Tests (fail first):
- [ ] the box is the peak's blob, not the bounding box of two far-apart blobs
- [ ] a weak map gives None (no fallback slot)
- [ ] `choose`: the centre is kept with `CENTRE_BOX` when missing or huge; neighbours that are missing, huge, or near-duplicates (IoU > 0.6) are dropped; at most k; meaning order kept
- [ ] service: a fake finder that can't locate some neighbours means those concepts are never offered; the centre is always offered first
- [ ] gpu: the shared vision pass matches the per-prompt batched CLIPSeg output (max diff < 1e-4), and 7 prompts take < 0.6 s

### Task 2: Concrete vs abstract framing

**Files:** `space/vocab.py` (`is_concrete`), `scenes/prompt.py`, `config.py`, `game/service.py`, `tests/test_prompt.py`, `tests/test_vocab.py`

- `is_concrete(word)`: is the first noun synset under `physical_entity.n.01`? It's cached.
- `scene_prompt(centre, others, concrete)` uses the two templates from the addendum. The subject comes first, and the style goes last without "landscape".
- `GameService(..., concrete=is_concrete)` is injectable.

Tests:
- [ ] guitar, elephant, kitchen, volcano, library, bicycle are concrete; anger, friendship, loneliness are abstract
- [ ] concrete prompt reads "a scene of a guitar in the centre, with … around it" and abstract reads "a symbolic scene evoking anger"; "landscape" doesn't appear; STYLE is at the end
- [ ] service: jumping to an abstract concept uses the symbolic prompt (via the injected `concrete`)
- [ ] gpu/visual: regenerate the 8 words and look at them. anger and friendship must not be generic landscapes.

### Task 3: No cluster looping

**Files:** `game/session.py` (`recent`), `game/service.py`, `tests/test_service.py`

Tests:
- [ ] after stepping A→B, A isn't offered as a neighbour of B for the next `RECENT` moves
- [ ] a jump resets recent to just its centre
- [ ] the toy-space walk that used to ping-pong (ocean ↔ sea) now stops offering the previous centre

### Task 4: Heading, forward and back (service + API)

**Files:** `space/index.py` (`heading`, `walk`), `game/session.py` (heading, history, scene, centre), `game/service.py` (`forward`, `back`, `NoHistory`), `api/app.py`, `tests/test_index.py`, `tests/test_service.py`, `tests/test_api.py`

- `MeaningSpace.heading(p_from, p_to) -> np.ndarray | None`: the unit tangent at p_to pointing away from p_from.
- `MeaningSpace.walk(p, h, angle) -> (p2, h2)`: moves along the great circle, and h2 is the transported tangent.
- `forward(sid)`: uses the heading, or with no heading the direction toward the nearest non-recent concept, then moves `MAX_STEP_ANGLE`. The new centre is the nearest non-recent concept. Neighbours exclude the centre and the recent ones. img2img runs on the centre zoom-crop.
- `back(sid)`: pops a snapshot and restores it. With nothing to pop it raises `NoHistory` (409 `no_history`). No generator call.
- The scene gains `centre`, and the API outputs it.

Tests:
- [ ] walk keeps unit norm, moves exactly `angle`, and two walks in a row equal one walk of 2×angle (same great circle)
- [ ] heading is None for identical points, and stepping p→q then walking from q along heading(p,q) moves further from p
- [ ] forward after a step continues away from where you came from, and the position moves exactly `MAX_STEP_ANGLE`
- [ ] forward right after a jump heads toward the nearest non-recent concept
- [ ] forward before any jump gives 409 not_started; busy gives 409; OOM gives 503 and keeps the state
- [ ] back restores position, hotspots, heading, recent, trail and centre exactly, and calls the generator 0 times
- [ ] back with no history gives 409 no_history, history is bounded at `HISTORY`, and after back a step uses the restored image as its source
- [ ] API: `/forward` and `/back` return the scene JSON with `centre`, and the error codes still hold

### Task 5: Colour drift fix

**Files:** `scenes/palette.py` (`palette_stats`, `match_palette`), `game/session.py`, `game/service.py`, `config.py`, `tests/test_palette.py`, `tests/test_generator_gpu.py`

- `palette_stats(img) -> (mean[3], std[3])` in CIELAB, using PIL ImageCms.
- `match_palette(img, stats, strength) -> Image`: Reinhard transfer blended by strength.
- The jump stores the stats, and every step or forward output is matched before CLIPSeg and the save.

Tests:
- [ ] strength 0 is the identity; strength 1 moves the LAB mean and std to the reference (within tolerance); the size and mode are kept
- [ ] service: the step output passed to the finder has been matched (a fake checks the colour moved toward the jump scene)
- [ ] gpu: over a 6-step walk, the LAB-mean drift from the first scene is < 0.5 × the drift without correction, and `test_six_steps_neither_dissolve_nor_fry` still passes
- [ ] visual: a 7-step walk contact sheet, with and without, checked by eye

### Task 6: Play screen: labels, keys

**Files:** `web/app.js`, `web/style.css`, `web/index.html` (hint line), `tests/test_ui.py`

Tests (headless Firefox):
- [ ] labels are visible without hover (computed opacity > 0.9)
- [ ] after a jump, W calls `/forward`, S calls `/back`, A or D step toward the left or right hotspot (checked by the request made and the new centre)
- [ ] keys are ignored while the jump box is focused and before the first jump
- [ ] a key during an in-flight move sends no second request
- [ ] a failed forward restores the scene and hotspots; the existing failed-jump, failed-step and small-box-not-covered tests still pass

### Task 7: Measure, verify, restart

- `scripts/measure_scenes.py`: 8-word jumps (fallback-rate, timings) + a 7-step walk (loops, drift) + a contact sheet.
- Run the whole suite `-m "gpu or not gpu"` with the server stopped, restart the server on this branch, and check it answers 200 at http://mrjarvis:8080.
