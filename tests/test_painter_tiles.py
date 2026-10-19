"""The tiled VAE pass on a stub VAE (CPU, no models): tile boundaries must not show."""
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vecworld.config import PANO_H, PANO_W, VAE_TILE
from vecworld.dream.painter import TurboPainter


class StubVae:
    """Decode: nearest-upsampled latent + a per-call offset; encode: 8x8 mean + a per-call offset.

    Each call stands for one tile whose GroupNorm statistics differ, so a hard tile cut shows as a step.
    """

    def __init__(self, step=0.1):
        self.step, self.n = step, 0

    def decode(self, z):
        self.n += 1
        px = z[:, :3].repeat_interleave(8, -2).repeat_interleave(8, -1)
        return SimpleNamespace(sample=px + self.step * self.n)

    def encode(self, x):
        self.n += 1
        lat = torch.nn.functional.avg_pool2d(x[:, :1], 8).expand(-1, 4, -1, -1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=lat + self.step * self.n))


def stub_painter():
    p = TurboPainter.__new__(TurboPainter)
    p._torch, p.vae, p._sf = torch, StubVae(), 1.0
    p._dev, p._dtype = "cuda" if torch.cuda.is_available() else "cpu", torch.float32
    return p


def col_steps(a: np.ndarray) -> np.ndarray:
    """|column x+1 - column x| (wrapping), averaged over rows and channels."""
    a = a.astype(np.float64)
    return np.abs(np.roll(a, -1, axis=1) - a).reshape(a.shape[0], a.shape[1], -1).mean(axis=(0, 2))


@pytest.mark.parametrize("shift", [0, 5, 37])
def test_decode_tiles_crossfade(shift):
    p = stub_painter()
    lat = torch.zeros((1, 4, PANO_H // 8, PANO_W // 8), device=p._dev)
    img = p._decode(lat, shift=shift)
    assert img.size == (PANO_W, PANO_H)
    steps = col_steps(np.asarray(img))
    assert steps.max() <= 2, f"step {steps.max():.1f} at column {steps.argmax()}"


@pytest.mark.parametrize("shift", [0, 5, 37])
def test_encode_tiles_crossfade(shift):
    p = stub_painter()
    lat = p._encode(Image.new("RGB", (PANO_W, PANO_H), (128, 128, 128)), shift=shift).cpu().numpy()[0]
    assert lat.shape == (4, PANO_H // 8, PANO_W // 8)
    steps = col_steps(np.moveaxis(lat, 0, -1))
    assert steps.max() < 0.05, f"step {steps.max():.3f} at latent column {steps.argmax()}"


def test_tile_grid_moves_with_the_seed():
    from vecworld.dream.pano import tile_origins

    w = PANO_W // 8
    a, b = tile_origins(w, 0), tile_origins(w, 1234 % VAE_TILE)
    assert len(a) == math.ceil(w / 48) and set(a).isdisjoint(b)


def test_text_embedding_cache_is_a_bounded_lru():
    from vecworld.dream.painter import EMB_CACHE

    p = stub_painter()
    encoded = []
    p._pipe = SimpleNamespace(encode_prompt=lambda t, *a: (encoded.append(t) or torch.zeros(1),))
    p._embs = getattr(p, "_embs", None) or __import__("collections").OrderedDict()
    p._text("keep me")
    for i in range(EMB_CACHE + 10):
        p._text(f"prompt {i}")
        p._text("keep me")                     # a hit keeps it fresh
    assert len(p._embs) == EMB_CACHE and "keep me" in p._embs and "prompt 0" not in p._embs
    assert encoded.count("keep me") == 1


# --- Task 8b: pacing (a stoppable paint, and no UNet passes whose output is thrown away) ----------------------

class StubSched:
    """Euler-like stand-in: add_noise mixes by t, step moves against eps."""

    def scale_model_input(self, x, t):
        return x

    def add_noise(self, orig, noise, t):
        return orig + noise * (t.float().reshape(-1)[0] / 1000)

    def step(self, eps, t, lat):
        return SimpleNamespace(prev_sample=lat - 0.1 * eps)


class StubUnet:
    """A local, deterministic 'noise prediction'; counts the windows it is asked for."""

    def __init__(self):
        self.windows = 0

    def __call__(self, tiles, t, encoder_hidden_states):
        self.windows += tiles.shape[0]
        return SimpleNamespace(sample=0.5 * tiles + encoder_hidden_states.mean(dim=(1, 2))[:, None, None, None])


def denoise_painter():
    p = stub_painter()
    p.unet, p.sched = StubUnet(), StubSched()
    return p


def _denoise_inputs(seed=0):
    from vecworld.dream.pano import focus_mask

    g = torch.Generator().manual_seed(seed)
    h, w = PANO_H // 8, PANO_W // 8
    lat = torch.randn((1, 4, h, w), generator=g)
    orig, noise = torch.randn((1, 4, h, w), generator=g), torch.randn((1, 4, h, w), generator=g)
    mask = torch.from_numpy(focus_mask(w, 0.5))[None, None, None, :]
    ts = torch.tensor([999, 749, 499, 249])
    return lat, ts, orig, noise, mask


def test_denoise_skips_background_windows_whose_output_is_reset(monkeypatch):
    from vecworld.dream import painter as painter_mod
    from vecworld.dream.pano import windows

    lat, ts, orig, noise, mask = _denoise_inputs()
    n = len(windows(*lat.shape[-2:]))
    embs = [torch.full((1, 2, 3), float(i)) for i in range(n)]
    p = denoise_painter()
    out = p._denoise(lat, ts, embs, keep=(orig, noise, mask, 2))
    fast = p.unet.windows
    monkeypatch.setattr(painter_mod, "_active", lambda wins, cols, k, k_bg: list(range(len(wins))))
    ref = denoise_painter()
    want = ref._denoise(lat, ts, embs, keep=(orig, noise, mask, 2))
    assert ref.unet.windows == 4 * n
    assert fast < ref.unet.windows                 # steps 0 and 1: only the windows touching the focus band
    assert torch.allclose(out, want, atol=1e-6)


def test_paint_stages_check_in_with_the_gpu_queue():
    from vecworld.dream import gpu

    lat, ts, orig, noise, mask = _denoise_inputs()
    p = denoise_painter()
    stop = threading.Event()
    stop.set()
    gpu._job.stop = stop
    try:
        with pytest.raises(gpu.Stopped):
            p._denoise(lat, ts, [torch.zeros((1, 2, 3))] * 16, keep=(orig, noise, mask, 2))
        with pytest.raises(gpu.Stopped):
            p._decode(torch.zeros((1, 4, PANO_H // 8, PANO_W // 8), device=p._dev))
        with pytest.raises(gpu.Stopped):
            p._encode(Image.new("RGB", (PANO_W, PANO_H)))
    finally:
        gpu._job.stop = None
    assert p.unet.windows == 0
