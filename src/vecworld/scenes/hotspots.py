from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from vecworld.config import CENTRE_BOX, HOTSPOT_MIN, MAX_HOTSPOT_AREA, MAX_HOTSPOT_IOU


@dataclass(frozen=True)
class Hotspot:
    concept: str
    x: float
    y: float
    w: float
    h: float


def _peak_blob(m: np.ndarray, level: float) -> np.ndarray:
    """Pixels >= level that are 4-connected to the heatmap's peak (grown by repeated dilation)."""
    mask = m >= level
    blob = np.zeros_like(mask)
    blob[np.unravel_index(int(m.argmax()), m.shape)] = True
    while True:
        grown = blob.copy()
        grown[1:] |= blob[:-1]
        grown[:-1] |= blob[1:]
        grown[:, 1:] |= blob[:, :-1]
        grown[:, :-1] |= blob[:, 1:]
        grown &= mask
        if np.array_equal(grown, blob):
            return blob
        blob = grown


def locate(concepts: list[str], maps: np.ndarray) -> list[Hotspot | None]:
    """Box around each concept's strongest blob, or None when CLIPSeg didn't really find it."""
    out: list[Hotspot | None] = []
    for concept, m in zip(concepts, maps):
        peak = float(m.max())
        if peak < HOTSPOT_MIN:
            out.append(None)
            continue
        ys, xs = np.nonzero(_peak_blob(m, 0.5 * peak))
        H, W = m.shape
        out.append(Hotspot(
            concept,
            float(xs.min() / W), float(ys.min() / H),
            float((xs.max() - xs.min() + 1) / W), float((ys.max() - ys.min() + 1) / H),
        ))
    return out


def _iou(a: Hotspot, b: Hotspot) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    return inter / (a.w * a.h + b.w * b.h - inter)


def choose(centre: str, found: list[Hotspot | None], k: int) -> list[Hotspot]:
    """The pinned centre (always), then up to k neighbours that are really, distinctly in the picture.

    found[0] is the centre's located box (or None); the rest are candidate neighbours in meaning order.
    """
    c = found[0]
    if c is None or c.w * c.h > MAX_HOTSPOT_AREA:
        c = Hotspot(centre, *CENTRE_BOX)
    out = [c]
    for h in found[1:]:
        if len(out) == k + 1:
            break
        if h is None or h.w * h.h > MAX_HOTSPOT_AREA:
            continue
        if any(_iou(h, o) > MAX_HOTSPOT_IOU for o in out):
            continue
        out.append(h)
    return out


class HotspotFinder(Protocol):
    def find(self, img: Image.Image, concepts: list[str]) -> list[Hotspot | None]: ...


class ClipSegFinder:
    """Text-prompted segmentation on the CPU: one heatmap per concept, one vision pass per image."""

    def __init__(self, model_id: str = "CIDAS/clipseg-rd64-refined"):
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self._torch = torch
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id).eval()

    def heatmaps(self, img: Image.Image, concepts: list[str]) -> np.ndarray:
        # The stock forward runs the vision tower once per prompt on the same image (~7x the work);
        # run it once and share its activations with every prompt's decoder pass.
        m = self.model
        pixels = self.processor(images=[img], return_tensors="pt").pixel_values
        text = self.processor.tokenizer(concepts, padding=True, return_tensors="pt")
        with self._torch.inference_mode():
            vision = m.clip.get_image_features(pixel_values=pixels, output_hidden_states=True)
            acts = [vision.hidden_states[i + 1].expand(len(concepts), -1, -1) for i in m.extract_layers]
            cond = m.clip.get_text_features(text.input_ids, attention_mask=text.attention_mask).pooler_output
            logits = m.decoder(acts, cond).logits
        return self._torch.sigmoid(logits).reshape(len(concepts), *logits.shape[-2:]).numpy()

    def find(self, img: Image.Image, concepts: list[str]) -> list[Hotspot | None]:
        return locate(concepts, self.heatmaps(img, concepts))
