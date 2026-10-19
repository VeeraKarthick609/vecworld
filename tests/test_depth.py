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


def _slices(D, flat=(), noisy=(), a_of=None, b_of=None):
    a_of = a_of or (lambda i: 0.5 + 0.37 * i)
    b_of = b_of or (lambda i: 0.3 * (i % 3) - 0.4)
    rng = np.random.default_rng(9)
    out = []
    for i, x0 in enumerate(range(0, W, STEP)):
        cols = [(x0 + j) % W for j in range(S)]
        s = (D[:, cols] - b_of(i)) / a_of(i)
        if i in flat:
            s = np.full_like(s, 0.7)
        if i in noisy:
            s = 0.7 + 0.01 * rng.standard_normal(s.shape)
        out.append((cols, s))
    return out


def _steps(out):
    return np.abs(np.diff(out, axis=1)).mean(axis=0)


def _check(out, D, corr=0.9, jump=8, seam=3):   # the truth itself steps 4.3x its median; 8 still catches seams (old code: 18-1449x)
    assert out.shape == (H, W) and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    st = _steps(out)
    med = np.median(st)
    assert st.max() <= jump * med, f"max jump {st.max():.4f} vs median {med:.4f}"
    assert np.abs(out[:, 0] - out[:, -1]).mean() <= seam * med
    assert np.corrcoef(out.ravel(), D.ravel())[0, 1] > corr


def test_stitch_recovers_scale_and_shift():
    D = _truth()
    out = stitch(_slices(D), W)
    assert np.corrcoef(out.ravel(), D.ravel())[0, 1] > 0.99
    _check(out, D, corr=0.99, jump=5)


def test_stitch_wrap_seam_is_smooth():
    out = stitch(_slices(_truth()), W)
    seam = np.abs(out[:, 0] - out[:, -1]).mean()
    assert seam <= 2 * np.median(_steps(out))


def test_flat_slice_does_not_break_alignment():
    D = _truth()
    for flat in (0, 3, 7):
        _check(stitch(_slices(D, flat=(flat,)), W), D)


def test_consecutive_and_edge_flat_runs():
    D = _truth()
    for flat in [(3, 4), (3, 4, 5), (0, 1), (6, 7), (7, 0), (1, 3, 5), (2, 3)]:
        _check(stitch(_slices(D, flat=flat), W), D, corr=0.9)


def test_noisy_flat_slice():
    D = _truth()
    for noisy in [(3,), (3, 4), (7,)]:
        _check(stitch(_slices(D, noisy=noisy), W), D)


def test_drift_exercises_loop_closure():
    D = _truth()
    # per-slice scale/shift drift: raw slices are not exactly affine-consistent with their neighbours
    out = stitch(_slices(D, a_of=lambda i: 1.0 + 0.04 * i * i, b_of=lambda i: 0.05 * i * (i % 2 + 1)), W)
    _check(out, D, corr=0.95, jump=5, seam=3)


def test_all_flat_is_finite():
    sl = [(c, np.zeros_like(d)) for c, d in _slices(_truth())]
    assert np.isfinite(stitch(sl, W)).all()


def test_depth_png_roundtrip():
    d = _truth()
    img = Image.open(io.BytesIO(depth_png(d)))
    assert img.mode == "L" and np.asarray(img).shape == d.shape
