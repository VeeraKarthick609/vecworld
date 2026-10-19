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


def _overlap(cols_a, cols_b):
    """Positions within each slice of the columns both cover (in a's order)."""
    ia = {c: i for i, c in enumerate(cols_a)}
    pairs = [(ia[c], j) for j, c in enumerate(cols_b) if c in ia]
    if not pairs:
        return np.empty(0, int), np.empty(0, int)
    pa, pb = zip(*pairs)
    return np.array(pa), np.array(pb)


def stitch(slices: list[tuple[list[int], np.ndarray]], width: int) -> np.ndarray:
    """Slices are (column indices, HxS relative inverse depth), each with its own unknown scale and shift.
    Chain-align each to the previous on their overlap, spread the loop-closure error around the circle,
    blend with a triangular weight, and normalise by the 1st/99th percentile. Returns HxW float32, 0..1, 1 = near."""
    H = slices[0][1].shape[0]
    ds = [np.nan_to_num(np.asarray(d, np.float64), nan=0.0, posinf=0.0, neginf=0.0) for _, d in slices]
    flat = [_is_flat(d) for d in ds]
    aligned = [ds[0]]
    last_good = (1.0, 0.0)          # transform of the latest slice that was fitted against real structure
    for i in range(1, len(slices)):
        pa, pb = _overlap(slices[i - 1][0], slices[i][0])
        if not len(pa):
            a, b = 1.0, 0.0
        elif flat[i - 1] or _is_flat(aligned[-1][:, pa]):
            a, b = last_good        # the previous slice says nothing about scale: carry the last good fit over
        else:
            a, b = _fit(ds[i][:, pb], aligned[-1][:, pa])
            if not flat[i]:
                last_good = (a, b)
        aligned.append(a * ds[i] + b)
    if len(slices) > 1 and not (flat[0] or flat[-1]):
        pa, pb = _overlap(slices[-1][0], slices[0][0])
        if len(pa):
            err = float(np.median(aligned[-1][:, pa] - aligned[0][:, pb]))
            if np.isfinite(err):
                aligned = [d - err * i / len(aligned) for i, d in enumerate(aligned)]
    acc, wsum = np.zeros((H, width)), np.zeros((H, width))
    for (cols, _), d, fl in zip(slices, aligned, flat):
        n = len(cols)
        ramp = np.minimum(np.arange(n), n - 1 - np.arange(n)).astype(np.float64) + 1
        if fl:
            ramp = ramp * 1e-3      # a featureless slice only fills columns nothing else covers
        np.add.at(acc, (slice(None), np.asarray(cols)), d * ramp)
        np.add.at(wsum, (slice(None), np.asarray(cols)), np.broadcast_to(ramp, d.shape))
    full = acc / np.maximum(wsum, _EPS)
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
