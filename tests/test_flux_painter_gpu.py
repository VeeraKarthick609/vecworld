"""FluxPainter on the real models (run on AC, game server stopped)."""
import time

import numpy as np
import pytest

from vecworld.config import FLUX_STYLE, FOCUS_FEATHER
from vecworld.dream.pano import dolly

BASE = [(f"a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns, "
         f"{FLUX_STYLE}", 1.0)]


@pytest.fixture(scope="module")
def painter():
    from vecworld.flux.painter import FluxPainter

    return FluxPainter()


@pytest.fixture(scope="module")
def pano(painter):
    import torch

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    img = painter.panorama(BASE, seed=21)
    print(f"\npanorama {time.time() - t:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return img, torch.cuda.max_memory_allocated()


def _box(a, r=3):
    """Mean over a (2r+1)^2 window (edges replicated)."""
    p = np.pad(a, r, mode="edge").cumsum(0).cumsum(1)
    p = np.pad(p, ((1, 0), (1, 0)))
    n = 2 * r + 1
    return (p[n:, n:] - p[:-n, n:] - p[n:, :-n] + p[:-n, :-n]) / n ** 2


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM of two greyscale images (0-255), 7x7 box windows."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _box(a), _box(b)
    va, vb, cov = _box(a * a) - ma ** 2, _box(b * b) - mb ** 2, _box(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def grey(img) -> np.ndarray:
    return np.asarray(img.convert("L"), np.float64)


@pytest.mark.gpu
def test_flux_panorama_seamless(pano):
    img, peak = pano
    img.save("/tmp/flux_pano.png")
    a = np.asarray(img, np.float64)
    seam = np.abs(a[:, -1] - a[:, 0]).mean()
    mean_col = np.abs(np.diff(a, axis=1)).mean()
    print(f"seam {seam:.2f} mean column difference {mean_col:.2f}")
    assert img.size == (2048, 512)
    assert seam < 1.5 * mean_col
    assert peak < 3.6e9


@pytest.mark.gpu
def test_flux_step_keeps_background(painter, pano):
    import torch

    from vecworld.roam.spot import focus_half

    z = 1.6
    prev = dolly(pano[0], z)
    half = focus_half(z)                                  # as next_spot calls it: the band the dolly magnified
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = painter.step(prev, BASE, None, 0.5, seed=22, focus_half=half)
    took = time.time() - t
    prev.save("/tmp/flux_prev.png")
    out.save("/tmp/flux_step.png")
    w = prev.width
    reach = half + FOCUS_FEATHER                          # the mask is 0 beyond this from the centre
    lo, hi = round((0.5 - reach) * w) - 64, round((0.5 + reach) * w) + 64
    a, b = grey(prev), grey(out)
    outside = np.concatenate([np.arange(0, lo), np.arange(hi, w)])
    s_out = ssim(a[:, outside], b[:, outside])
    s_in = ssim(a[:, lo:hi], b[:, lo:hi])
    print(f"\nstep {took:.1f}s ssim outside {s_out:.3f} inside {s_in:.3f} "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    assert s_out > 0.6
    assert s_in < s_out                                   # the band ahead was repainted
    assert torch.cuda.max_memory_allocated() < 3.6e9
