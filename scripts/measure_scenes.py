"""Measure scenes on the real models: hotspots on 8 jump words, then a walk; writes annotated PNGs.

    uv run python scripts/measure_scenes.py OUT_DIR [--walk lighthouse] [--moves click,forward,...]

Stop the game server first (it holds the GPU). Moves: click (first offered neighbour), left, right,
forward, back. Prints one line per scene and writes OUT_DIR/report.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vecworld.config import CENTRE_BOX, DATA_DIR
from vecworld.game.service import GameError, GameService
from vecworld.scenes.generator import TurboGenerator
from vecworld.scenes.hotspots import ClipSegFinder
from vecworld.space.embedder import LLMEmbedder
from vecworld.space.vocab import load_space

WORDS = ["guitar", "elephant", "kitchen", "anger", "volcano", "library", "bicycle", "friendship"]


class PeakFinder(ClipSegFinder):
    """Remembers each concept's heatmap peak from the last call."""

    peaks: dict[str, float] = {}

    def find(self, img, concepts):
        maps = self.heatmaps(img, concepts)
        self.peaks = {c: round(float(m.max()), 2) for c, m in zip(concepts, maps)}
        from vecworld.scenes.hotspots import locate
        return locate(concepts, maps)


def annotate(sc, path: Path) -> None:
    im = Image.open(sc.image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    for i, h in enumerate(sc.hotspots):
        default = i == 0 and (h.x, h.y, h.w, h.h) == CENTRE_BOX
        box = [h.x * W, h.y * H, (h.x + h.w) * W, (h.y + h.h) * H]
        d.rectangle(box, outline="red" if default else ("cyan" if i == 0 else "yellow"), width=3)
        d.text((box[0] + 4, box[1] + 4), h.concept + (" (centre box)" if default else ""), fill="white")
    im.save(path)


def lab_mean(path) -> np.ndarray:
    from PIL import ImageCms
    to_lab = ImageCms.buildTransform(ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")
    lab = np.asarray(ImageCms.applyTransform(Image.open(path).convert("RGB"), to_lab), np.float32)
    return lab.reshape(-1, 3).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--walk", default="lighthouse")
    ap.add_argument("--moves", default="click,click,click,click,click,click,click")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-jumps", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    space, mean = load_space(DATA_DIR)
    finder = PeakFinder()
    svc = GameService(space, mean, LLMEmbedder(), TurboGenerator(), finder, a.out / "scenes")
    s = svc.new_session()
    s.seed = a.seed
    svc.jump(s.id, "warm up")
    torch.cuda.reset_peak_memory_stats()
    rep = {"jumps": [], "walk": []}

    for w in [] if a.skip_jumps else WORDS:
        t = time.time()
        sc = svc.jump(s.id, w)
        dt = time.time() - t
        hs = sc.hotspots
        centre_default = (hs[0].x, hs[0].y, hs[0].w, hs[0].h) == CENTRE_BOX
        rep["jumps"].append({"word": w, "t": round(dt, 2), "offered": [h.concept for h in hs],
                             "centre_box": centre_default, "peaks": finder.peaks})
        annotate(sc, a.out / f"jump_{w}.png")
        print(f"jump {w:<11} {dt:.2f}s offered={[h.concept for h in hs]} centre_box={centre_default} "
              f"peaks={finder.peaks}", flush=True)

    sc = svc.jump(s.id, a.walk)
    annotate(sc, a.out / "walk_0.png")
    first = lab_mean(sc.image_path)
    print(f"walk start {a.walk}: {[h.concept for h in sc.hotspots]}", flush=True)
    for i, move in enumerate(a.moves.split(","), 1):
        t = time.time()
        try:
            if move == "click":
                sc = svc.step(s.id, (sc.hotspots[1:] or sc.hotspots)[0].concept)
            elif move in ("left", "right"):
                side = [h for h in sc.hotspots[1:] if (h.x + h.w / 2 < 0.5) == (move == "left")]
                if not side:
                    print(f"  {i} {move}: nothing there")
                    continue
                pick = min if move == "left" else max
                sc = svc.step(s.id, pick(side, key=lambda h: h.x + h.w / 2).concept)
            else:
                sc = getattr(svc, move)(s.id)
        except GameError as e:
            print(f"  {i} {move}: {e.code}")
            continue
        dt = time.time() - t
        drift = float(np.linalg.norm(lab_mean(sc.image_path) - first))
        rep["walk"].append({"move": move, "centre": sc.hotspots[0].concept, "t": round(dt, 2),
                            "offered": [h.concept for h in sc.hotspots], "lab_drift": round(drift, 1)})
        annotate(sc, a.out / f"walk_{i}.png")
        print(f"  {i} {move:<7} {dt:.2f}s -> {[h.concept for h in sc.hotspots]} lab drift {drift:.1f}", flush=True)

    rep["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    rep["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    print("peak VRAM allocated", rep["peak_vram_mib"], "MiB, reserved", rep["peak_reserved_mib"], "MiB")
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
