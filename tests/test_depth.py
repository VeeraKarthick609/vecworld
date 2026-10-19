import io

import numpy as np
from PIL import Image

from vecworld.roam.depth import depth_png, stitch

W, H, S, STEP = 2048, 64, 512, 256


def _truth():
    rng = np.random.default_rng(3)
    x = np.arange(W)
    d = np.zeros((H, W))
    for k in range(1, 9):
        d += rng.uniform(0.3, 1) / k * np.sin(2 * np.pi * k * x / W + rng.uniform(0, 6.28))[None, :]
    d += np.linspace(0, 1, H)[:, None]
    return (d - d.min()) / (d.max() - d.min())


def _slices(D, flat=None):
    out = []
    for i, x0 in enumerate(range(0, W, STEP)):
        cols = [(x0 + j) % W for j in range(S)]
        a, b = 0.5 + 0.37 * i, 0.3 * (i % 3) - 0.4
        s = (D[:, cols] - b) / a
        out.append((cols, np.full_like(s, 0.7) if i == flat else s))
    return out


def test_stitch_recovers_scale_and_shift():
    D = _truth()
    out = stitch(_slices(D), W)
    assert out.shape == (H, W) and out.dtype == np.float32
    assert np.corrcoef(out.ravel(), D.ravel())[0, 1] > 0.99


def test_stitch_wrap_seam_is_smooth():
    out = stitch(_slices(_truth()), W)
    seam = np.abs(out[:, 0] - out[:, -1]).mean()
    assert seam <= 2 * np.median(np.abs(np.diff(out, axis=1)).mean(axis=0))


def test_flat_slice_does_not_break_alignment():
    D = _truth()
    for flat in (0, 3, 7):
        out = stitch(_slices(D, flat=flat), W)
        assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
        assert np.corrcoef(out.ravel(), D.ravel())[0, 1] > 0.9


def test_all_flat_is_finite():
    sl = [(c, np.zeros_like(d)) for c, d in _slices(_truth())]
    out = stitch(sl, W)
    assert np.isfinite(out).all()


def test_depth_png_roundtrip():
    d = _truth()
    img = Image.open(io.BytesIO(depth_png(d)))
    assert img.mode == "L" and np.asarray(img).shape == d.shape
