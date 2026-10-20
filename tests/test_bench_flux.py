"""scripts/bench_flux.py's timed paths on the fake painter (CPU, no models)."""
import importlib.util
from pathlib import Path

import torch
from test_flux_painter import FakeTransformer, fake_painter

from vecworld.config import FLUX_STEPS, FLUX_WIN

_spec = importlib.util.spec_from_file_location("bench_flux", Path(__file__).parents[1] / "scripts/bench_flux.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_tile512_runs_one_window_per_step():
    tr = FakeTransformer()
    p = fake_painter(tr)
    out = bench.run_tile512(p, p._embed([("a forest", 1.0)]), FLUX_STEPS)
    assert out.shape == (1, 64, FLUX_WIN, FLUX_WIN) and torch.isfinite(out).all()
    assert [c.shape[0] for c in tr.calls] == [1] * FLUX_STEPS           # one 512x512 window, once per step
