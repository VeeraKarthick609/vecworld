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
    assert p.last_cache == {"passes": FLUX_STEPS, "skips": 0}


def test_tile512_is_one_cache_job():
    """With the block cache on, tile512 runs as one cache job (keyed per step) and records its skips; every step but
    the first may skip, and the job's entries are dropped after it."""
    from test_cache import FakeFlux, _painter

    p = _painter(100.0, FakeFlux())
    p._embed = lambda prompt: (torch.zeros(1, 4, 8), torch.zeros(1, 8))
    bench.run_tile512(p, p._embed(None), FLUX_STEPS)
    assert p.last_cache == {"passes": FLUX_STEPS, "skips": FLUX_STEPS - 1} and len(p.cache) == 0


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


def test_load_transformer_resolves_the_kernel_as_the_painter_does(monkeypatch):
    """bench_flux.load_transformer uses vecworld.engine.engine_kernel: unset VECWORLD_KERNEL swaps in w4a8 (as the
    painter does, with config.FLUX_A16_KEEP on mma), "off" leaves diffusers' linears, a name picks that kernel (the
    keep-list applies to w4a8 only)."""
    import diffusers

    import vecworld.engine.swap as swap_mod

    swapped = []
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", lambda *a, **k: "transformer")
    monkeypatch.setattr(swap_mod, "gguf_linears", lambda tr: {})
    monkeypatch.setattr(swap_mod, "swap_linears",
                        lambda tr, gg, kernel, keep_a16=(): swapped.append((kernel, tuple(keep_a16))))
    from vecworld.config import FLUX_A16_KEEP

    keep = tuple(FLUX_A16_KEEP)
    for value, want in ((None, [("w4a8", keep)]), ("off", []), ("reference", []), ("tiled", [("tiled", ())]),
                        ("mma", [("mma", ())]), ("w4a8", [("w4a8", keep)])):
        swapped.clear()
        if value is None:
            monkeypatch.delenv("VECWORLD_KERNEL", raising=False)
        else:
            monkeypatch.setenv("VECWORLD_KERNEL", value)
        assert bench.load_transformer()[0] == "transformer"
        assert swapped == want, value


def test_modulation_linears_are_the_adaln_ones():
    """blocks times the modulation linears (Task 5c): the Q4KLinear under each block's AdaLN norm (double blocks'
    norm1 and norm1_context, single blocks' norm, the output norm), one row per image. Token-facing Q4_K linears and
    linears left on diffusers are not modulation."""
    import numpy as np
    from gguf import GGMLQuantizationType

    from vecworld.engine.q4k import GGUFTensor, Q4KLinear

    def q4k(out=64, inp=256):
        return Q4KLinear(GGUFTensor("w", GGMLQuantizationType.Q4_K, (out, inp), np.zeros((out, inp // 256 * 144),
                                                                                           np.uint8)), None, "w4a8")

    tr = torch.nn.Module()
    tr.transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    blk = tr.transformer_blocks[0]
    blk.norm1, blk.norm1_context, blk.ff = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
    blk.norm1.linear, blk.norm1_context.linear, blk.ff.proj = q4k(), q4k(), q4k()
    tr.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    tr.single_transformer_blocks[0].norm = torch.nn.Module()
    tr.single_transformer_blocks[0].norm.linear = q4k()
    tr.single_transformer_blocks[0].proj_out = q4k()
    tr.norm_out = torch.nn.Module()
    tr.norm_out.linear = torch.nn.Linear(256, 64)                     # not Q4_K in the file: diffusers' own
    names = [n for n, _ in bench.modulation_linears(tr)]
    assert names == ["transformer_blocks.0.norm1.linear", "transformer_blocks.0.norm1_context.linear",
                     "single_transformer_blocks.0.norm.linear"]
