# Dream Roam

Type a place ("haunted forest"), pick a mood or two, and roam it in third person. The world is a chain of
360° painted spots with depth: SD-Turbo paints each spot as a seamless panorama, Depth Anything V2 Small
gives it 360° depth, and the browser (three.js) walks a small robot around it with real parallax. Walk
out of a spot in any direction and the next one is dreamed that way, grown from what lay ahead.

Nearby places (from Qwen2.5-1.5B-Instruct on the CPU, with a built-in fallback table until it answers)
lie at compass directions. Signposts and the compass point to them; walk toward one and the scenery
drifts into it until you arrive, and the arrival gets new neighbours. Typing a new place mid-roam drifts
the world toward it.

Runs on one 4 GB GPU (RTX 3050 Ti). The first run downloads SD-Turbo, Depth Anything V2 Small and
Qwen2.5-1.5B-Instruct from Hugging Face (a few GB).

```sh
uv sync
uv run python -m vecworld        # http://localhost:8080/
```

Controls: W/A/S/D or arrows move relative to the camera, Shift runs; drag (or click to capture the
pointer, Esc releases) to look around. On a phone, use the on-screen stick and drag to look.

Measured on the laptop (2026-10-19, three reference journeys of 8 spots plus a running one):
- a new world appears 2.8–5.7 s after Enter (the first spot builds in 2.6–3.6 s);
- each further spot builds in 5.0–5.6 s (median 5.3 s), against at most ~5.7 s of walking from one spot's edge to the next;
- so a short "dreaming the way ahead" fog shows 9–11 times a minute while walking (mostly 0.1–2.5 s,
  some 4–8 s when the way had not been pre-dreamed), about 15–20 s of fog per minute walking and 33 s
  running;
- peak GPU memory 3.76 GB of 4 GB; the places LLM takes 9–22 s per answer on the CPU.

Tests: `uv run pytest -q` (fast, fakes only). Add `-m gpu` for real models (stop the server first so
the GPU is free) and `-m ui` for the browser tests, which need Firefox and geckodriver
(`/snap/bin/geckodriver`). Everything: `uv run pytest -q -m "gpu or not gpu or ui"`.
`scripts/dev_fake_server.py` serves the game on fake models, for looking at the viewer without the GPU.
