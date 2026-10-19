"""Reinhard colour transfer in CIELAB, done in float so the correction itself never adds banding."""
import numpy as np
from PIL import Image

Stats = tuple[np.ndarray, np.ndarray]   # per-channel (L, a, b) mean and std

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_M_INV = np.linalg.inv(_M)
_WHITE = _M.sum(axis=1)                 # D65 white point (X, Y, Z)
# float32 working copies: a 2048 x 512 panorama converts in a third of the float64 time, with no visible difference
_M32, _M_INV32, _WHITE32 = (x.astype(np.float32) for x in (_M, _M_INV, _WHITE))
_E = 216 / 24389
_K = 24389 / 27
_c = np.arange(256) / 255.0
_LINEAR = np.where(_c <= 0.04045, _c / 12.92, ((_c + 0.055) / 1.055) ** 2.4).astype(np.float32)   # sRGB byte -> linear
STATS_PIXELS = 1 << 16      # palette_stats reads about this many pixels of a larger image


def _to_lab(img) -> np.ndarray:
    """CIELAB of an image (or an (H, W, 3) uint8 array)."""
    lin = _LINEAR[np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img, np.uint8)]
    xyz = lin @ _M32.T / _WHITE32
    f = np.where(xyz > _E, np.cbrt(xyz), (_K * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _to_rgb(lab: np.ndarray) -> Image.Image:
    lab = lab.astype(np.float32, copy=False)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    f3 = f * f * f
    xyz = np.where(f3 > _E, f3, (116 * f - 16) / _K) * _WHITE32
    lin = np.clip(xyz @ _M_INV32.T, 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.round(np.clip(c, 0, 1) * 255).astype(np.uint8), "RGB")


def palette_stats(img: Image.Image) -> Stats:
    """LAB mean and spread. A large image is read every few pixels (no filtering, which would smooth the spread)."""
    a = np.asarray(img.convert("RGB"))
    k = max(1, int(np.sqrt(a.shape[0] * a.shape[1] / STATS_PIXELS)))
    lab = _to_lab(a[::k, ::k]).reshape(-1, 3)
    return lab.mean(0), lab.std(0)


def match_palette(img: Image.Image, ref: Stats, strength: float) -> Image.Image:
    """Move img's LAB mean and spread `strength` of the way to ref's (0 = unchanged, 1 = ref's palette)."""
    if strength <= 0:
        return img
    lab = _to_lab(img)
    mean, std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
    want_mean = mean + strength * (ref[0] - mean)
    want_std = std + strength * (ref[1] - std)
    return _to_rgb((lab - mean) * (want_std / np.maximum(std, 1e-6)) + want_mean)
