"""Pure panorama maths for the wrap-around 360-degree panorama (no torch)."""
import math

import numpy as np
from PIL import Image

from vecworld.config import DOLLY, FOCUS_FEATHER, FOCUS_HALF, STRIDE, VAE_STRIDE, VAE_TILE, WIN


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


def tile_origins(w: int, shift: int, tile: int = VAE_TILE, stride: int = VAE_STRIDE) -> list[int]:
    """Start columns of VAE tiles evenly spaced (at most stride apart) round a w-wide cylinder, moved by shift."""
    n = math.ceil(w / stride)
    assert tile > w / n, "tiles must overlap"
    return [(shift + round(i * w / n)) % w for i in range(n)]


def ramp(n: int, fade: int) -> np.ndarray:
    """Weights across one tile: rising linearly over its first `fade` columns, falling over its last."""
    j = np.arange(n) + 0.5
    return np.clip(np.minimum(j, n - j) / fade, 0.0, 1.0).astype(np.float32)


def dolly_src(u, z: float = DOLLY):
    """Where output fraction u samples the input in a dolly: magnified z ahead (u = 0.5), shrunk behind, no cut."""
    a = 1 - 1 / z
    return u - a / (2 * np.pi) * np.sin(2 * np.pi * (u - 0.5))


def dolly_u(u_src, z: float = DOLLY):
    """Where input fraction u_src lands after a dolly (the inverse of dolly_src)."""
    a, u = 1 - 1 / z, np.asarray(u_src, np.float64)
    for _ in range(6):   # Newton; the map is monotonic and nearly linear
        u = u - (dolly_src(u, z) - u_src) / (1 - a * np.cos(2 * np.pi * (u - 0.5)))
    return u


def _catmull(p0, p1, p2, p3, t):
    return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))


def dolly(img: Image.Image, z: float = DOLLY) -> Image.Image:
    """Step forward: a periodic warp of the cylinder, ×z ahead, ×1 to the sides, ×~1/z behind, with no cut.

    Each column is zoomed vertically by the same factor as horizontally, so nothing is stretched (a
    uniform ×z vertical zoom stretched what is behind you 1.3× per step into vertical bars); behind you
    the rows above and below are mirrored in. A plain 1/z centre crop instead cut out the columns behind
    you and joined the two sides it left with a hard vertical line.
    """
    w, h = img.size
    a = np.asarray(img.convert("RGB"), np.float32)
    u = (np.arange(w) + 0.5) / w
    # horizontally: every row samples the same source columns (wrapping)
    x = dolly_src(u, z) * w - 0.5
    i = np.floor(x).astype(int)
    t = (x - i).astype(np.float32)[None, :, None]
    a = _catmull(*(a[:, (i + k) % w] for k in (-1, 0, 1, 2)), t)
    # vertically: each column zoomed about the horizon by its own horizontal zoom (rows mirrored at the ends)
    mag = 1 / (1 - (1 - 1 / z) * np.cos(2 * np.pi * (u - 0.5)))
    y = h / 2 + (np.arange(h)[:, None] + 0.5 - h / 2) / mag[None, :] - 0.5
    j = np.floor(y).astype(int)
    t = (y - j).astype(np.float32)[..., None]
    cols = np.arange(w)[None, :]

    def row(k):
        r = np.abs(j + k)
        return a[np.where(r > h - 1, 2 * (h - 1) - r, r), cols]

    a = _catmull(row(-1), row(0), row(1), row(2), t)
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))
