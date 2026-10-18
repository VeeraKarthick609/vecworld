"""Pure panorama maths for the wrap-around Dream Walk canvas (no torch)."""
import numpy as np
from PIL import Image

from vecworld.config import DOLLY, FOCUS_FEATHER, FOCUS_HALF, STRIDE, WIN


def windows(h: int, w: int, win: int = WIN, stride: int = STRIDE) -> list[tuple[int, int]]:
    """(y, x) window origins; x wraps around the full width, the last y row is clamped to h - win."""
    ys = list(range(0, h - win, stride)) + [h - win]
    return [(y, x) for y in ys for x in range(0, w, stride)]


def wrap_cols(x0: int, n: int, w: int) -> list[int]:
    return [(x0 + j) % w for j in range(n)]


def wrap_dist(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def focus_mask(w: int, at: float, half: float = FOCUS_HALF, feather: float = FOCUS_FEATHER) -> np.ndarray:
    u = np.arange(w, dtype=np.float64) / w
    d = np.abs(u - at) % 1.0
    d = np.minimum(d, 1.0 - d)
    return np.clip(1.0 - (d - half) / feather, 0.0, 1.0).astype(np.float32)


def roll_to(img: Image.Image, u: float) -> Image.Image:
    """Rotate horizontally so panorama fraction u sits at the centre column."""
    w = img.width
    shift = w // 2 - round(u * w)
    return Image.fromarray(np.roll(np.asarray(img), shift, axis=1))


def yaw_to_u(yaw: float) -> float:
    return (0.5 + yaw / 360) % 1


def dolly(img: Image.Image, z: float = DOLLY) -> Image.Image:
    """Centre crop 1/z in both axes, resized back to the original size."""
    w, h = img.size
    cw, ch = w / z, h / z
    x0, y0 = (w - cw) / 2, (h - ch) / 2
    return img.resize((w, h), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))
