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


Z = 1.6


@pytest.fixture(scope="module")
def stepped(painter, pano):
    """A walk step as next_spot makes it: the panorama dollied by Z, the band it magnified repainted."""
    import torch

    from vecworld.roam.spot import focus_half

    prev = dolly(pano[0], Z)
    half = focus_half(Z)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = painter.step(prev, BASE, None, 0.5, seed=22, focus_half=half)
    took = time.time() - t
    prev.save("/tmp/flux_prev.png")
    out.save("/tmp/flux_step.png")
    print(f"\nstep {took:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return prev, out, half, torch.cuda.max_memory_allocated()


def wrap_dist_cols(w: int, at: float = 0.5) -> np.ndarray:
    d = np.abs((np.arange(w) + 0.5) / w - at) % 1.0
    return np.minimum(d, 1 - d)


def mirror_score(img, z: float = Z, back: float = 0.4, k: int = 24) -> float:
    """How much the rows above dolly()'s top mirror line look like their reflection below it, in the columns at
    least `back` behind the camera: ~0.7 on a dollied view, ~0 on a painted panorama. High-passed first (minus a
    9x9 mean), since smooth sky and haze correlate with anything and detail only with its mirror image."""
    a = grey(img)
    h, w = a.shape
    a = a - _box(a, 4)
    u = (np.arange(w) + 0.5) / w
    mag = 1 / (1 - (1 - 1 / z) * np.cos(2 * np.pi * (u - 0.5)))
    ym = np.round(h / 2 - mag * h / 2).astype(int)                 # the output row dolly() sampled source row 0 into
    xs, ys = [], []
    for c in np.where(wrap_dist_cols(w) >= back)[0]:
        n = min(k, ym[c])
        xs.append(a[ym[c] - n:ym[c], c][::-1])
        ys.append(a[ym[c]:ym[c] + n, c])
    return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])


@pytest.mark.gpu
def test_flux_step_keeps_background(stepped):
    """Outside the band ahead and outside the band behind the camera (where the dolly shrank the view and the
    background is repainted harder), the step keeps the scene."""
    from vecworld.config import FLUX_BACK_FROM

    prev, out, half, peak = stepped
    w = prev.width
    d = wrap_dist_cols(w)
    reach = half + FOCUS_FEATHER + 64 / w                 # the focus mask is 0 beyond half + feather
    side = np.where((d > reach) & (d < FLUX_BACK_FROM))[0]
    ahead = np.where(d < half)[0]
    a, b = grey(prev), grey(out)
    s_side, s_ahead = ssim(a[:, side], b[:, side]), ssim(a[:, ahead], b[:, ahead])
    print(f"ssim sides {s_side:.3f} ({len(side)} columns) ahead {s_ahead:.3f}")
    assert s_side > 0.6
    assert s_ahead < s_side                               # the band ahead was repainted
    assert peak < 3.6e9


@pytest.mark.gpu
def test_flux_step_repaints_the_mirrored_rows_behind(stepped):
    """dolly() fills the top and bottom rows behind the camera with a mirror image of the rows next to them
    (upside-down tree tops); the step must paint over them."""
    prev, out, _, _ = stepped
    before, after = mirror_score(prev), mirror_score(out)
    print(f"mirror score behind: dollied {before:.3f} step {after:.3f}")
    assert before > 0.5                                   # the measure sees the mirror
    assert after < 0.2
