"""FluxPainter's MultiDiffusion loop on fakes (CPU, no models): windows, progress and stopping."""
import math
import threading
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vecworld.config import FLUX_BATCH, FLUX_STEPS, FLUX_STRIDE, FLUX_WIN, PANO_H, PANO_W
from vecworld.dream import gpu
from vecworld.dream.gpu import Stopped
from vecworld.flux.painter import FluxPainter

GH, GW = PANO_H // 16, PANO_W // 16          # the packed token grid
N_WIN = GW // FLUX_STRIDE


class FakeTransformer:
    """Predicts zero velocity; records every call's packed tokens. on_call(n) runs after the n-th call."""

    def __init__(self, on_call=None):
        self.calls, self.on_call = [], on_call

    def __call__(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids,
                 guidance=None, return_dict=False):
        assert hidden_states.shape[1:] == (FLUX_WIN * FLUX_WIN, 64)
        assert encoder_hidden_states.shape[0] == pooled_projections.shape[0] == hidden_states.shape[0]
        assert img_ids.shape == (FLUX_WIN * FLUX_WIN, 3)
        self.calls.append(hidden_states.detach().clone())
        if self.on_call:
            self.on_call(len(self.calls))
        return (torch.zeros_like(hidden_states),)


class FakeVae:
    """Decode: the first 3 latent channels nearest-upsampled; encode: 8x8 means (16 channels)."""
    config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, z):
        return SimpleNamespace(sample=z[:, :3].repeat_interleave(8, -2).repeat_interleave(8, -1))

    def encode(self, x):
        lat = torch.nn.functional.avg_pool2d(x[:, :1], 8).expand(-1, 16, -1, -1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=lat))


def fake_painter(transformer=None, progress=None):
    p = FluxPainter.__new__(FluxPainter)
    p._setup(transformer or FakeTransformer(), FakeVae(),
             lambda text: (torch.zeros(1, 4, 8, dtype=torch.bfloat16), torch.zeros(1, 8, dtype=torch.bfloat16)),
             progress, dev="cpu", dtype=torch.float32)
    return p


def test_windows_wrap_and_cover():
    tr = FakeTransformer()
    p = fake_painter(tr)
    grid = torch.zeros(1, 64, GH, GW)
    grid[:, 0] = torch.arange(GW, dtype=torch.float32)          # channel 0 holds each token's column
    emb = p._embed([("x", 1.0)])
    p._denoise(grid, p._sigmas(1), [emb] * N_WIN)
    seen, wrapped = set(), False
    for call in tr.calls:
        for tile in call:
            cols = tile[:, 0].view(FLUX_WIN, FLUX_WIN)[0].round().long().tolist()   # one row of the window
            seen.update(cols)
            wrapped |= cols[0] > cols[-1]                         # runs past the right edge back to column 0
            assert all((b - a) % GW == 1 for a, b in zip(cols, cols[1:]))
    assert seen == set(range(GW)), "every packed column is covered"
    assert wrapped, "a window wraps across the seam"
    assert sum(len(c) for c in tr.calls) == N_WIN


def test_progress_called_per_step():
    got = []
    p = fake_painter(progress=lambda k, total: got.append((k, total)))
    img = p.panorama([("a forest", 0.5), ("a lake", 0.5)], seed=3)
    assert img.size == (PANO_W, PANO_H)
    total = FLUX_STEPS * math.ceil(N_WIN / FLUX_BATCH)
    assert got == [(k, total) for k in range(1, total + 1)]


def test_progress_counts_only_active_windows_in_a_step():
    got = []
    tr = FakeTransformer()
    p = fake_painter(tr, progress=lambda k, total: got.append((k, total)))
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    p.step(prev, [("a forest", 1.0)], None, 0.5, seed=4)
    total = got[-1][1]
    assert got == [(k, total) for k in range(1, total + 1)]
    assert len(tr.calls) == total
    assert sum(len(c) for c in tr.calls) < FLUX_STEPS * N_WIN          # background windows skipped early


def test_check_stop_between_steps():
    stop = threading.Event()
    tr = FakeTransformer(on_call=lambda n: stop.set())        # asked to stop during the first pass
    p = fake_painter(tr)
    gpu._job.stop = stop
    try:
        with pytest.raises(Stopped):
            p.panorama([("a forest", 1.0)], seed=1)
    finally:
        gpu._job.stop = None
    assert len(tr.calls) == 1


def test_prompt_blend_is_a_weighted_sum():
    p = FluxPainter.__new__(FluxPainter)
    vals = {"a": 1.0, "b": 3.0}
    p._setup(FakeTransformer(), FakeVae(),
             lambda t: (torch.full((1, 4, 8), vals[t], dtype=torch.bfloat16),
                        torch.full((1, 8), 2 * vals[t], dtype=torch.bfloat16)),
             None, dev="cpu", dtype=torch.float32)
    t5, pooled = p._embed([("a", 0.25), ("b", 0.75)])
    assert torch.allclose(t5.float(), torch.full((1, 4, 8), 2.5))
    assert torch.allclose(pooled.float(), torch.full((1, 8), 5.0))


def test_pack_roundtrip():
    p = fake_painter()
    lat = torch.randn(1, 16, PANO_H // 8, PANO_W // 8)
    g = p._pack(lat)
    assert g.shape == (1, 64, GH, GW)
    assert torch.equal(p._unpack(g), lat)
    # token (i, j) channel c*4 + dy*2 + dx is latent (c, 2i + dy, 2j + dx), as diffusers packs FLUX latents
    assert g[0, 5 * 4 + 1 * 2 + 0, 3, 7] == lat[0, 5, 7, 14]


def test_step_strength_zero_keeps_the_background_exactly():
    import numpy as np

    p = fake_painter()
    prev = Image.new("RGB", (PANO_W, PANO_H), (128, 128, 128))
    out = np.asarray(p.step(prev, [("a forest", 1.0)], None, 0.5, seed=5, bg_strength=0.0), np.float64)
    ref = np.asarray(p._decode(p._encode(prev)), np.float64)
    assert np.abs(out[:, :PANO_W // 4] - ref[:, :PANO_W // 4]).max() <= 1


def test_step_schedule_puts_the_background_sigma_on_it():
    p = fake_painter()
    assert p._step_sigmas(0.25) == ([1.0, 0.75, 0.5, 0.25, 0.0], 3)          # on schnell's grid: unchanged
    assert p._step_sigmas(0.15) == ([1.0, 0.75, 0.5, 0.25, 0.15, 0.0], 4)    # off it: added
    assert p._step_sigmas(0.0) == ([1.0, 0.75, 0.5, 0.25, 0.0], FLUX_STEPS)  # never released


def test_read_gguf_matches_diffusers_loader(tmp_path):
    import gguf
    import numpy as np
    from diffusers.models.model_loading_utils import load_gguf_checkpoint

    from vecworld.flux.painter import _read_gguf

    path = str(tmp_path / "t.gguf")
    w = gguf.GGUFWriter(path, "flux")
    rng = np.random.default_rng(0)
    w.add_tensor("a.weight", rng.standard_normal((4, 8)).astype(np.float32))
    w.add_tensor("b.weight", rng.standard_normal((3, 16)).astype(np.float16))
    q = gguf.quants.quantize(rng.standard_normal((2, 256)).astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("c.weight", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    mine, ref = _read_gguf(path), load_gguf_checkpoint(path)
    assert mine.keys() == ref.keys()
    for k in ref:
        assert type(mine[k]) is type(ref[k]) and mine[k].dtype == ref[k].dtype and torch.equal(mine[k], ref[k]), k
        assert getattr(mine[k], "quant_type", None) == getattr(ref[k], "quant_type", None)
