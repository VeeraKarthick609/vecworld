# Dream Roam

Type a place ("haunted forest"), pick a mood or two, and roam it in third person. The world is a chain of
360° painted spots with depth: FLUX.1-schnell (4-bit) paints each spot as a seamless panorama on a hand-written
engine (`src/vecworld/engine/`, explained piece by piece in `docs/engine/`), Depth Anything V2 Small gives it 360°
depth, and the browser (three.js) walks a small robot around it with real parallax. Walk
out of a spot in any direction and the next one is dreamed that way, grown from what lay ahead.

Nearby places (from Qwen2.5-1.5B-Instruct on the CPU, with a built-in fallback table until it answers)
lie at compass directions. Signposts and the compass point to them; walk toward one and the scenery
drifts into it until you arrive, and the arrival gets new neighbours. Typing a new place mid-roam drifts
the world toward it.

Runs on one 4 GB GPU (RTX 3050 Ti) and 14 GB of RAM. FLUX's weights (`city96/FLUX.1-schnell-gguf` Q4_K_S, the
`unsloth/FLUX.1-schnell` VAE, CLIP and tokenizers) and T5-XXL (`city96/t5-v1_1-xxl-encoder-gguf` Q5_K_M) are read
from the SSD block by block every pass, so they never sit in RAM; the first run downloads them, Depth Anything V2
Small and Qwen2.5-1.5B-Instruct from Hugging Face (~14 GB). The engine's CUDA extension builds on first use with
the CUDA 13.4 toolkit (`/usr/local/cuda-13.4`). `VECWORLD_PAINTER=turbo` brings back the earlier SD-Turbo painter
(fast, less beautiful) for comparison; `VECWORLD_KERNEL=off` paints FLUX on the diffusers reference path.

```sh
uv sync
uv run python -m vecworld        # http://localhost:8080/
```

Controls: W/A/S/D or arrows move relative to the camera, Shift runs; drag (or click to capture the
pointer, Esc releases) to look around. On a phone, use the on-screen stick and drag to look.

Measured on the laptop (2026-11-09, FLUX on the engine, dreamer on, three reference journeys of 6 spots;
`docs/engine/12-t5-and-results.md`):
- a new world appears ~34 s after Enter (the panorama: 4 passes of 8 windows, ~33 s);
- each further spot takes 39-46 s to build (median 42 s), so after ~8 s of walking to the edge the
  "dreaming the way ahead" fog shows for 32-39 s a step, with the painting's progress;
- the whole server stays under 9.8 GB of RAM in play without swapping, the GPU under 2.6 GB;
- the places LLM takes 11-29 s per answer on the CPU, beside the painter.
- SD-Turbo (`VECWORLD_PAINTER=turbo`) builds a spot in ~4 s, with fogs of 0-8 s.

Tests: `uv run pytest -q` (fast, fakes only). Add `-m gpu` for real models (stop the server first so
the GPU is free) and `-m ui` for the browser tests, which need Firefox and geckodriver
(`/snap/bin/geckodriver`). Everything: `uv run pytest -q -m "gpu or not gpu or ui"`.
`scripts/dev_fake_server.py` serves the game on fake models, for looking at the viewer without the GPU.
