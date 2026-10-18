# Dream Walk

Type a place ("haunted forest") and pick a mood, and SD-Turbo paints a seamless 360° panorama you can
look around in. Each walk step dollies forward and repaints the view ahead, and a small local LLM
(Qwen) dreams up things to discover there. CLIPSeg finds them in the painting and labels them, and
clicking a label walks you toward it. Change the theme or a mood and the dream drifts there over a
few steps.

Runs on one 4 GB GPU (RTX 3050 Ti); the painter, ideas model and CLIPSeg load at startup.

```sh
uv sync
uv run python -m vecworld        # http://localhost:8080/
```

Controls: drag or A/D to look, W to walk on, S to turn back, click a discovery to approach it.
On a phone, use the arrow pad.

Tests: `uv run pytest -q` (fast, fakes only). Add `-m gpu` for real models and `-m ui` for headless
Firefox; stop the server first so the GPU is free.
