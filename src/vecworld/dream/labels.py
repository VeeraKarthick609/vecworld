"""Clickable discovery labels: locate what was painted ahead with CLIPSeg, keep only what is really there."""
import secrets
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vecworld.config import DOLLY
from vecworld.dream.pano import dolly_u, wrap_cols
from vecworld.scenes.hotspots import Hotspot, locate

BAND = (0.15, 0.85)


@dataclass(frozen=True)
class Label:
    id: str
    text: str
    u: float  # box centre, panorama fraction in [0, 1)
    y: float
    w: float
    h: float


def crop_at(img: Image.Image, u: float, size: int = 512) -> Image.Image:
    """A size-wide wrap-around crop centred on panorama fraction u, at full height."""
    w = img.width
    x0 = (round(u * w) - size // 2) % w
    return Image.fromarray(np.asarray(img)[:, wrap_cols(x0, size, w)])


def to_label(text: str, box: Hotspot, u_centre: float, pano_w: int, crop_w: int = 512) -> Label:
    u = (u_centre + (box.x + box.w / 2 - 0.5) * crop_w / pano_w) % 1.0
    return Label(secrets.token_urlsafe(4), text, u, box.y, box.w * crop_w / pano_w, box.h)


def carry(labels: list[Label], walk_u: float, z: float = DOLLY) -> list[tuple[str, float]]:
    """Where each label's text lands after a step toward walk_u (the dolly's warp); those outside the band are dropped."""
    out = []
    for l in labels:
        u = float(dolly_u((l.u - walk_u + 0.5) % 1.0, z))
        if BAND[0] <= u <= BAND[1]:
            out.append((l.text, u))
    return out


class LabelFinder:
    def __init__(self, heatmaps=None):
        if heatmaps is None:
            from vecworld.scenes.hotspots import ClipSegFinder
            heatmaps = ClipSegFinder().heatmaps
        self._heatmaps = heatmaps

    def find(self, img: Image.Image, wanted: list[tuple[str, float]]) -> list[Label]:
        out = []
        for text, u in wanted:
            (box,) = locate([text], self._heatmaps(crop_at(img, u), [text]))
            if box is not None:
                out.append(to_label(text, box, u, img.width))
        return out
