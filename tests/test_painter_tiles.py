"""The tiled VAE pass on a stub VAE (CPU, no models): tile boundaries must not show."""
import math
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
