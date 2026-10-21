"""FluxPainter on the real models (run on AC, game server stopped)."""
import time

import numpy as np
import pytest

from vecworld.config import FLUX_STEP_RECIPE, FLUX_STEP_RECIPES, FLUX_STYLE
from vecworld.dream.pano import dolly

BASE = [(f"a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns, "
         f"{FLUX_STYLE}", 1.0)]


@pytest.fixture(scope="module")
def painter():
    from vecworld.flux.painter import FluxPainter

    p = FluxPainter()
    yield p
    p.close()     # unregister its page-locked host memory before it is freed (see test_flux_close_gpu)


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
CONTINUOUS = [n for n, r in FLUX_STEP_RECIPES.items() if r["release"] != "fresh"]
# Continuity bar per recipe: SSIM between the dollied view and the step over the columns it keeps most (released
# within 0.1 of its lowest strength). Each is ~70% of what the recipe measured on 2026-10-21 (in the comments), so a
# change that repaints the kept view much harder fails. A band repainted from scratch scores ~0.08 here, so sharp's
# bar says little more than "keeps more than a fresh paint": at Z 1.6 its 0.25 feather and 0.75 sides release every
# column at 0.89 or more. "fresh" paints a new panorama and is exempt (test_flux_fresh_step_is_exempt_from_continuity).
SSIM_KEPT = {"t1": 0.49,         # 0.699 at strength 0.15
             "repaint": 0.28,    # 0.396 at 0.5
             "carry": 0.25,      # 0.363 at 0.5-0.6
             "renew": 0.23,      # 0.331 at 0.5-0.6
             "sharp": 0.12}      # 0.177 at 0.89-0.99


@pytest.fixture(scope="module", params=CONTINUOUS)
def stepped(request, painter, pano):
    """A walk step as next_spot makes it, by each continuous recipe: the panorama dollied by Z, then stepped."""
    import torch

    from vecworld.roam.spot import focus_half

    prev = dolly(pano[0], Z)
    half = focus_half(Z)
    painter.recipe = request.param
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        out = painter.step(prev, BASE, None, 0.5, seed=22, focus_half=half)
    finally:
        painter.recipe = FLUX_STEP_RECIPE
    took = time.time() - t
    out.save(f"/tmp/flux_step_{request.param}.png")
    print(f"\n{request.param} step {took:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return request.param, prev, out, half, torch.cuda.max_memory_allocated()


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
    """Each continuous recipe keeps the scene where it releases the view latest (the columns within 0.1 of its
    lowest release strength), and keeps more there than ahead, where it repaints."""
    from vecworld.dream.pano import focus_mask
    from vecworld.flux.painter import release_strengths

    name, prev, out, half, peak = stepped
    rec = FLUX_STEP_RECIPES[name]
    w = prev.width
    m = focus_mask(w, 0.5, half, rec["feather"])
    r = release_strengths(m, 0.5, rec["side"], max(rec["side"], rec["back"]), rec["focus"])
    kept = np.where(r <= r.min() + 0.1)[0]
    ahead = np.where(wrap_dist_cols(w) < half)[0]
    a, b = grey(prev), grey(out)
    s_kept, s_ahead = ssim(a[:, kept], b[:, kept]), ssim(a[:, ahead], b[:, ahead])
    print(f"{name}: ssim kept {s_kept:.3f} ({len(kept)} columns at strength <= {r.min() + 0.1:.2f}) "
          f"ahead {s_ahead:.3f}")
    assert len(kept) >= 64
    assert s_kept > SSIM_KEPT[name]                       # KeyError: a new continuous recipe needs its bar
    assert s_ahead < s_kept                               # the band ahead was repainted
    assert peak < 3.6e9


def test_flux_fresh_step_is_exempt_from_continuity():
    """"fresh" is exempt from the continuity gate by design: its step is a new panorama (tested on CPU)."""
    assert FLUX_STEP_RECIPES["fresh"]["release"] == "fresh" and "fresh" not in CONTINUOUS


@pytest.mark.gpu
def test_flux_step_repaints_the_mirrored_rows_behind(stepped):
    """dolly() fills the top and bottom rows behind the camera with a mirror image of the rows next to them
    (upside-down tree tops); the step must paint over them."""
    name, prev, out, _, _ = stepped
    before, after = mirror_score(prev), mirror_score(out)
    print(f"{name}: mirror score behind: dollied {before:.3f} step {after:.3f}")
    assert before > 0.5                                   # the measure sees the mirror
    assert after < 0.2
