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


def test_flop_per_step_matches_ledger():
    """Ledger section 1.6: 57 blocks x 2 x P weights per token plus attention, 17.67 TFLOP at N = 1024 + 256."""
    f = bench.flop_per_step(1024, 256)
    assert f == 114 * 113_246_208 * 1280 + 700_416 * 1280 ** 2
    assert abs(f - 17.67e12) < 0.01e12


def test_engine_row_fields():
    """An engine result gets the FLOP count, achieved TFLOP/s and % of the measured bf16 peak from its pass time."""
    row = bench.engine_fields({"s_per_pass_median": 2.74, "passes": 4}, n_img=1024, n_txt=256, peak_tflops=19.1)
    assert row["s_per_pass_median"] == 2.74 and row["passes"] == 4                  # timings kept
    assert row["flop_per_step"] == bench.flop_per_step(1024, 256)
    assert row["achieved_tflops"] == 6.45                                            # the baseline's numbers
    assert row["pct_bf16_peak"] == 33.8
    assert bench.bf16_peak() == 19.11                                                # bench/baseline.json blocks


def test_gemm_bench_covers_the_flux_linear_shapes():
    """--what gemm times the brief's FLUX shapes (M, K, N): the double blocks' image (M = 1024) and text (M = 256)
    streams, and the single blocks' joined stream (M = 1280)."""
    four = [(3072, 9216), (3072, 3072), (3072, 12288), (12288, 3072)]
    assert bench.GEMM_SHAPES == ([(1024, k, n) for k, n in four] + [(256, k, n) for k, n in four]
                                 + [(1280, 3072, 21504), (1280, 15360, 3072)])
    assert bench.WHATS["gemm"] is bench.do_gemm
