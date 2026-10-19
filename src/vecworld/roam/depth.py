"""360-degree depth: overlapping Depth Anything slices stitched into one cylindrical inverse-depth map."""
import io
from typing import Protocol

import numpy as np
from PIL import Image

from vecworld.config import DEPTH_SLICE, DEPTH_STEP

_EPS = 1e-6
_MIN_SCALE, _MAX_SCALE = 0.1, 10.0   # a fitted scale outside this is a degenerate fit, not a real rescale


def _fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """(a, b) so that a*src + b ~ dst. Least squares; shift-only when src is nearly flat or the fit is
    degenerate (non-finite, inverted, wildly scaled); identity when even the shift is not finite."""
    src, dst = src.ravel().astype(np.float64), dst.ravel().astype(np.float64)
    ok = np.isfinite(src) & np.isfinite(dst)
    src, dst = src[ok], dst[ok]
    if src.size == 0:
        return 1.0, 0.0
    shift = float(dst.mean() - src.mean())
    if not np.isfinite(shift):
        return 1.0, 0.0
    sv = src.var()
    if sv < _EPS * max(1.0, float(np.abs(src).max()) ** 2):
        return 1.0, shift
    a = float(((src - src.mean()) * (dst - dst.mean())).mean() / sv)
    if not np.isfinite(a) or not _MIN_SCALE <= a <= _MAX_SCALE:
        return 1.0, shift
    b = float(dst.mean() - a * src.mean())
    return (a, b) if np.isfinite(b) else (1.0, shift)


def _is_flat(x: np.ndarray) -> bool:
    return float(np.var(x)) < _EPS * max(1.0, float(np.abs(x).max()) ** 2)


_FLAT_RATIO = 0.15      # a slice whose std is below this fraction of the median slice std is sky/fog: no structure to align on
_EDGE = 8              # columns either side of a gap compared when joining chains with no overlap
_MIN_R2 = 0.5           # an overlap fit explaining less than this is not trusted to link two slices


def _overlap(cols_a, cols_b):
    """Positions within each slice of the columns both cover (in a's order)."""
    ia = {c: i for i, c in enumerate(cols_a)}
    pairs = [(ia[c], j) for j, c in enumerate(cols_b) if c in ia]
    if not pairs:
        return np.empty(0, int), np.empty(0, int)
    pa, pb = zip(*pairs)
    return np.array(pa), np.array(pb)


def _linked(a_raw, b_raw, pa, pb) -> bool:
    """Two real slices are linked when their overlap fits an affine map well (R^2 of the correlation)."""
    if not len(pa):
        return False
    x, y = a_raw[:, pa].ravel(), b_raw[:, pb].ravel()
    if _is_flat(x) or _is_flat(y):
        return False
    r = np.corrcoef(x, y)[0, 1]
    return bool(np.isfinite(r) and r * r >= _MIN_R2)


def stitch(slices: list[tuple[list[int], np.ndarray]], width: int) -> np.ndarray:
    """Slices are (column indices, HxS relative inverse depth), each with its own unknown scale and shift.
    Real slices are chain-aligned (least squares) around the circle; a flat slice (sky, fog) carries no
    scale information, so it is dropped and the chain breaks there: the slices either side are joined by
    matching medians, and the columns only the flat slice covered are interpolated between the aligned
    neighbours. The loop-closure error (additive only) is spread over the real slices when the circle has no
    break. Blend with a triangular weight, normalise by the 1st/99th percentile.
    Returns HxW float32, 0..1, 1 = near."""
    n, H = len(slices), slices[0][1].shape[0]
    ds = [np.nan_to_num(np.asarray(d, np.float64), nan=0.0, posinf=0.0, neginf=0.0) for _, d in slices]
    stds = np.array([d.std() for d in ds])
    ref = float(np.median(stds))
    real = [i for i in range(n) if not _is_flat(ds[i]) and stds[i] >= _FLAT_RATIO * ref]
    if not real:
        return np.full((H, width), 0.5, np.float32)

    def link(i):        # is slice i linked to slice i+1 (mod n)?
        j = (i + 1) % n
        if i not in real or j not in real or n < 2:
            return False
        pa, pb = _overlap(slices[i][0], slices[j][0])
        return _linked(ds[i], ds[j], pa, pb)

    links = [link(i) for i in range(n)]
    breaks = [i for i in range(n) if not links[i]]
    start = (breaks[0] + 1) % n if breaks else 0       # begin just after a break so closure never crosses one
    order = [i for i in (start + k for k in range(n)) if i % n in real]
    order = [i % n for i in order]

    aligned: dict[int, np.ndarray] = {order[0]: ds[order[0]]}
    for p, j in zip(order, order[1:]):
        pa, pb = _overlap(slices[p][0], slices[j][0])
        if links[p] and j == (p + 1) % n:
            a, b = _fit(ds[j][:, pb], aligned[p][:, pa])
        else:                                          # a break: join the chains at the boundary
            if len(pa):
                src, dst = ds[j][:, pb], aligned[p][:, pa]
            else:                                      # no shared columns: compare the two edges facing the gap
                src, dst = ds[j][:, :_EDGE], aligned[p][:, -_EDGE:]
            # the surface is continuous across the join, so the rows of the edge strips give a scale too;
            # _fit falls back to a median shift when that is degenerate
            a, b = _fit(src, dst)
            if not len(pa):                            # make the two edges meet exactly, row by row on average
                b += float(np.median(dst[:, -1] - (a * src[:, 0] + b)))
        aligned[j] = a * ds[j] + b
    if not breaks and len(order) > 1:
        # closure is additive only: it removes the accumulated shift, not scale drift
        pa, pb = _overlap(slices[order[-1]][0], slices[order[0]][0])
        if len(pa):
            err = float(np.median(aligned[order[-1]][:, pa] - aligned[order[0]][:, pb]))
            if np.isfinite(err):
                for k, i in enumerate(order):
                    aligned[i] = aligned[i] - err * k / len(order)

    acc, wsum = np.zeros((H, width)), np.zeros((H, width))
    for i in order:
        cols = np.asarray(slices[i][0])
        m = len(cols)
        ramp = np.minimum(np.arange(m), m - 1 - np.arange(m)).astype(np.float64) + 1
        np.add.at(acc, (slice(None), cols), aligned[i] * ramp)
        np.add.at(wsum, (slice(None), cols), np.broadcast_to(ramp, aligned[i].shape))
    covered = wsum[0] > 0
    full = np.zeros((H, width))
    full[:, covered] = acc[:, covered] / wsum[:, covered]
    if not covered.all():                              # columns only flat slices covered: interpolate round the circle
        xs, miss = np.flatnonzero(covered), np.flatnonzero(~covered)
        for r in range(H):
            full[r, miss] = np.interp(miss, xs, full[r, xs], period=width)
    lo, hi = np.percentile(full, 1), np.percentile(full, 99)
    if not np.isfinite(lo + hi) or hi - lo < _EPS:
        return np.full((H, width), 0.5, np.float32)
    return np.clip((full - lo) / (hi - lo), 0, 1).astype(np.float32)


class DepthEstimator(Protocol):
    def depth360(self, pano: Image.Image) -> np.ndarray: ...


class AnythingDepth:
    """Depth Anything V2 Small over overlapping slices of the cylinder, stitched into one map."""

    MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.MODEL).to(device).eval()

    def depth360(self, pano: Image.Image) -> np.ndarray:
        import torch

        W, H = pano.size
        arr = np.asarray(pano.convert("RGB"))
        cols = [[(x0 + j) % W for j in range(DEPTH_SLICE)] for x0 in range(0, W, DEPTH_STEP)]
        inputs = self.processor(images=[Image.fromarray(arr[:, c]) for c in cols], return_tensors="pt")
        with torch.inference_mode():
            pred = self.model(pixel_values=inputs["pixel_values"].to(self.device)).predicted_depth
            pred = torch.nn.functional.interpolate(pred[:, None].float(), size=(H, DEPTH_SLICE),
                                                   mode="bicubic", align_corners=False)[:, 0]
        pred = pred.cpu().numpy()
        return stitch(list(zip(cols, pred)), W)


def depth_png(d: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.clip(d, 0, 1) * 255).round().astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()
