"""Reinhard colour transfer in CIELAB, done in float so the correction itself never adds banding."""
import numpy as np
from PIL import Image

Stats = tuple[np.ndarray, np.ndarray]   # per-channel (L, a, b) mean and std

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_M_INV = np.linalg.inv(_M)
_WHITE = _M.sum(axis=1)                 # D65 white point (X, Y, Z)
_E = 216 / 24389
_K = 24389 / 27


def _to_lab(img: Image.Image) -> np.ndarray:
    c = np.asarray(img.convert("RGB"), np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > _E, np.cbrt(xyz), (_K * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _to_rgb(lab: np.ndarray) -> Image.Image:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f ** 3 > _E, f ** 3, (116 * f - 16) / _K) * _WHITE
    lin = np.clip(xyz @ _M_INV.T, 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return Image.fromarray(np.round(np.clip(c, 0, 1) * 255).astype(np.uint8), "RGB")


def palette_stats(img: Image.Image) -> Stats:
    lab = _to_lab(img).reshape(-1, 3)
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
