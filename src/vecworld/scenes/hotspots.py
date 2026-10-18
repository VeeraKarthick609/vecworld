from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from vecworld.config import HOTSPOT_MIN

# Fixed, non-overlapping boxes for concepts CLIPSeg can't locate, so they stay clickable.
FALLBACK = [(0.4, 0.35, 0.2, 0.3), (0.08, 0.55, 0.2, 0.3), (0.72, 0.55, 0.2, 0.3), (0.4, 0.05, 0.2, 0.25)]


@dataclass(frozen=True)
class Hotspot:
    concept: str
    x: float
    y: float
    w: float
    h: float


def boxes_from_heatmaps(concepts: list[str], maps: np.ndarray) -> list[Hotspot]:
    found: dict[int, Hotspot] = {}
    for i, (concept, m) in enumerate(zip(concepts, maps)):
        peak = float(m.max())
        if peak < HOTSPOT_MIN:
            continue
        ys, xs = np.nonzero(m >= 0.5 * peak)
        H, W = m.shape
        found[i] = Hotspot(
            concept,
            float(xs.min() / W), float(ys.min() / H),
            float((xs.max() - xs.min() + 1) / W), float((ys.max() - ys.min() + 1) / H),
        )

    free = list(FALLBACK)
    out = []
    for i, concept in enumerate(concepts):
        if i in found:
            out.append(found[i])
        else:
            slot = free.pop(0) if free else FALLBACK[i % len(FALLBACK)]
            out.append(Hotspot(concept, *slot))
    return out


class HotspotFinder(Protocol):
    def find(self, img: Image.Image, concepts: list[str]) -> list[Hotspot]: ...


class ClipSegFinder:
    """Text-prompted segmentation on the CPU: one heatmap per concept."""

    def __init__(self, model_id: str = "CIDAS/clipseg-rd64-refined"):
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).eval()

    def find(self, img: Image.Image, concepts: list[str]) -> list[Hotspot]:
        inputs = self.processor(text=concepts, images=[img] * len(concepts), padding=True, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits
        maps = self._torch.sigmoid(logits).reshape(len(concepts), *logits.shape[-2:]).numpy()
        return boxes_from_heatmaps(concepts, maps)
