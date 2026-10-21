"""A deterministic offline Dream Roam walk on the real painter, depth and palette, with a contact sheet.

  uv run python scripts/walk_sheet.py --recipe repaint --theme "haunted forest" --seed 7 --spots 6 \
      --headings 0,0,60,60,300,0 --walked 9 --out /tmp/walk/repaint
  uv run python scripts/walk_sheet.py --compare /tmp/walk/t1 /tmp/walk/repaint --out /tmp/walk/compare.png

A walk calls first_spot / next_spot exactly as RoamService.start and RoamService._build do (the same prompts,
fallback neighbours, journey drift, colour reference and seeds), with no browser, server or GPU queue. headings[i]
is the world bearing spot i is reached by (headings[0] is the first spot's, 0). It writes spot{i}.png, prev{i}.png
(the dollied view each step started from), sheet.png and timings.json into --out.

Each spot gets a seam score (seam_profile: the strongest full-height step in local detail between the columns
either side of one column). A walk whose spot scores above --max-seam exits 1; --check re-scores walks already
written. Calibration (2026-10-21, haunted-forest walk): painted panoramas and natural fog edges scored up to 1.08;
the walk-step seams 1.13-1.79 (Task 1's, and "sharp" at feather 0.15: spots 2, 4, 5 at 1.36, 1.38, 1.13). The
margin is thin: it catches hard full-height edges, not soft haze bands.

--compare puts several walks side by side: OUT (one column per walk, half size, for an overview) and, at native
size, OUT_spot{i}.png per spot (the walks' spot i stacked, full 2048 px width) to judge by eye.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vecworld.roam.places import Journey, assign_bearings, fallback_neighbours  # noqa: E402
from vecworld.roam.service import MAX_NEIGHBOURS, OPEN_GROUND, RoamService, _blend, _copy  # noqa: E402
from vecworld.roam.spot import first_spot, next_spot  # noqa: E402
from vecworld.scenes.palette import palette_stats  # noqa: E402

SHEET_W = 1024          # each panorama is drawn this wide on the overview sheets
SEAM_LIMIT = 1.1        # a spot scoring above this has a seam (see the module docstring)
SEAM_W, SEAM_GAP, SEAM_ROWS = 32, 8, 16
LABEL_H = 18


class _Recorder:
    """Wraps a painter: keeps each step's input view and times each painter call."""

    def __init__(self, painter):
        self.painter, self.prevs, self.paint_s = painter, [], []

    def __getattr__(self, name):
        return getattr(self.painter, name)

    def panorama(self, base, seed):
        t = time.perf_counter()
        img = self.painter.panorama(base, seed)
        self.paint_s.append(time.perf_counter() - t)
        return img

    def step(self, prev, *a, **k):
        self.prevs.append(prev)
        t = time.perf_counter()
        img = self.painter.step(prev, *a, **k)
        self.paint_s.append(time.perf_counter() - t)
        return img


def walk(painter, depth, theme: str, seed: int, headings: list[float], walked: float | None) -> dict:
    """The walk: spots, prompts and timings. Mirrors RoamService.start (no moods) then _build/_commit per step."""
    rec = _Recorder(painter)
    text = lambda name, scenery: RoamService._text(SimpleNamespace(painter=painter), name, scenery, ())  # noqa: E731
    prompt = [(text(theme, OPEN_GROUND), 1.0)]
    t = time.perf_counter()
    spot = first_spot(rec, depth, prompt, seed)
    spots, prompts, times = [spot], [prompt], [time.perf_counter() - t]
    journey = Journey(theme, assign_bearings(fallback_neighbours(theme)[:MAX_NEIGHBOURS], None, random.Random(seed)))
    stats = palette_stats(spot.image)
    ref = stats
    for steps, heading in enumerate(headings[1:]):
        j = _copy(journey)
        ds = j.step(heading)
        w = 1.0 if ds.arrived else (ds.weights[1][1] if len(ds.weights) > 1 else 0.0)
        if ds.arrived:
            names = [n for n in fallback_neighbours(ds.arrived) if n not in (j.place, ds.arrived)]
            j.arrive(ds.arrived, names[:MAX_NEIGHBOURS - 1], random.Random(f"{seed}:{steps}"))
        base = [(text(name, ds.scenery), wt) for name, wt in ds.weights]
        r = _blend(ref, stats, w) if w > 0 else ref
        t = time.perf_counter()
        spot = next_spot(rec, depth, spot, heading, base, None, (seed + steps + 1) % 2**31, r, walked=walked)
        times.append(time.perf_counter() - t)
        stats, journey = palette_stats(spot.image), j
        if ds.arrived:
            ref = stats
        spots.append(spot)
        prompts.append(base)
    return {"spots": spots, "prompts": prompts, "spot_s": times, "paint_s": rec.paint_s, "prevs": rec.prevs}


def seam_profile(img: Image.Image) -> np.ndarray:
    """Per column x: how far the local detail (log of the high-pass magnitude, averaged over each of SEAM_ROWS
    row blocks) differs between the SEAM_W columns left of x and right of it (SEAM_GAP apart), as the median over
    the row blocks. A seam where a repainted band meets a kept one runs the whole height with sharp detail on one
    side and haze on the other; a tree trunk or a path edge changes the colour more than the detail, or not all
    the way up (a featureless object wider than SEAM_W running the full height would count). Wraps round the
    cylinder."""
    g = np.asarray(img.convert("L"), np.float32)
    h, w = g.shape
    p = np.pad(g, 2, mode="wrap")
    e = np.abs(g - sum(p[dy:dy + h, dx:dx + w] for dy in range(5) for dx in range(5)) / 25)
    rows = min(SEAM_ROWS, h)
    e = np.log(e[: h // rows * rows].reshape(rows, h // rows, w).mean(1) + 0.5)
    k = SEAM_W + SEAM_GAP
    c = np.cumsum(np.pad(np.concatenate([e[:, -k:], e, e[:, :k]], 1), ((0, 0), (1, 0))), 1)
    x = np.arange(w) + k
    left = (c[:, x - SEAM_GAP] - c[:, x - k]) / SEAM_W
    right = (c[:, x + k] - c[:, x + SEAM_GAP]) / SEAM_W
    return np.median(np.abs(left - right), axis=0)


def seam(img: Image.Image) -> tuple[float, int]:
    """(score, column) of the image's strongest seam."""
    prof = seam_profile(img)
    return round(float(prof.max()), 3), int(prof.argmax())


def sheet(images: list[Image.Image], labels: list[str], width: int = SHEET_W) -> Image.Image:
    """One panorama per row, each under its label."""
    rows = [im.resize((width, round(im.height * width / im.width)), Image.LANCZOS) for im in images]
    out = Image.new("RGB", (width, sum(r.height + LABEL_H for r in rows)), "white")
    d, y = ImageDraw.Draw(out), 0
    for r, lab in zip(rows, labels):
        d.text((4, y + 3), lab, fill="black")
        out.paste(r, (0, y + LABEL_H))
        y += r.height + LABEL_H
    return out


def compare(dirs: list[Path], width: int = SHEET_W, gap: int = 12) -> Image.Image:
    """Walks side by side: one column per walk directory (its spot images in order, under its recipe name)."""
    cols = []
    for d in dirs:
        info = json.loads((d / "timings.json").read_text())
        ims = [Image.open(d / f"spot{i}.png") for i in range(len(info["spot_s"]))]
        labs = [f"{info['recipe']}  spot {i}  heading {info['headings'][i]:g}  "
                f"{'pano' if i == 0 else 'step'} {info['paint_s'][i]:.1f}s" for i in range(len(ims))]
        cols.append(sheet(ims, labs, width))
    h = max(c.height for c in cols) + LABEL_H
    out = Image.new("RGB", (len(cols) * width + (len(cols) - 1) * gap, h), "white")
    d = ImageDraw.Draw(out)
    for i, (c, dd) in enumerate(zip(cols, dirs)):
        x = i * (width + gap)
        d.text((x + 4, 3), dd.name.upper(), fill="black")
        out.paste(c, (x, LABEL_H))
    return out


def compare_native(dirs: list[Path], out: Path) -> list[Path]:
    """One native-size sheet per spot: every walk's spot i, full width, stacked (out_spot{i}.png)."""
    infos = [json.loads((d / "timings.json").read_text()) for d in dirs]
    paths = []
    for i in range(min(len(x["spot_s"]) for x in infos)):
        ims, labs = [], []
        for d, info in zip(dirs, infos):
            im = Image.open(d / f"spot{i}.png")
            ims.append(im)
            labs.append(f"{info['recipe']}  spot {i}  heading {info['headings'][i]:g}  "
                        f"{'pano' if i == 0 else 'step'} {info['paint_s'][i]:.1f}s"
                        + (f"  seam {info['seam'][i]:.2f} @ {info['seam_at'][i]}" if "seam" in info else ""))
        p = out.with_name(f"{out.stem}_spot{i}.png")
        sheet(ims, labs, width=ims[0].width).save(p)
        paths.append(p)
    return paths


def check(dirs: list[Path], limit: float = SEAM_LIMIT) -> list[str]:
    """Re-score the spots of walks already written; returns the failures ("dir/spotN: score at column")."""
    bad = []
    for d in dirs:
        for f in sorted(d.glob("spot*.png"), key=lambda p: int(p.stem[4:])):
            score, at = seam(Image.open(f))
            print(f"{d.name}/{f.name}: seam {score:.2f} at {at}")
            if score > limit:
                bad.append(f"{d.name}/{f.stem}: {score:.2f} at column {at}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", help="FLUX step recipe (config.FLUX_STEP_RECIPES); default: the configured one")
    ap.add_argument("--theme", default="haunted forest")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--headings", default="0,0,60,60,300,0")
    ap.add_argument("--walked", type=float, default=9.0)
    ap.add_argument("--compare", nargs="+", type=Path, help="walk directories to put side by side")
    ap.add_argument("--check", nargs="+", type=Path, help="re-score the seams of walks already written")
    ap.add_argument("--max-seam", type=float, default=SEAM_LIMIT, help="fail (exit 1) on a spot scoring above it")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.check:
        bad = check(a.check, a.max_seam)
        if bad:
            sys.exit("seams above %.2f: %s" % (a.max_seam, "; ".join(bad)))
        return
    if a.out is None:
        ap.error("--out is required")
    if a.compare:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        compare(a.compare).save(a.out)
        compare_native(a.compare, a.out)
        return
    headings = [float(x) for x in a.headings.split(",")][:a.spots]
    if len(headings) < a.spots:
        ap.error(f"--headings gives {len(headings)} bearings for {a.spots} spots")
    from vecworld.flux.painter import FluxPainter
    from vecworld.roam.depth import AnythingDepth

    painter, depth = FluxPainter(), AnythingDepth()
    try:
        info = run(painter, depth, a.recipe, a.theme, a.seed, headings, a.walked, a.out)
    finally:
        painter.close()
    bad = [f"spot{i}: {sc:.2f} at column {at}" for i, (sc, at) in enumerate(zip(info["seam"], info["seam_at"]))
           if sc > a.max_seam]
    if bad:
        sys.exit("seams above %.2f: %s" % (a.max_seam, "; ".join(bad)))


def run(painter, depth, recipe, theme, seed, headings, walked, out: Path) -> dict:
    """One walk into `out` (spot/prev images, sheet.png, timings.json); returns the timings record."""
    if recipe is not None:
        painter.recipe = recipe
    recipe = getattr(painter, "recipe", recipe) or "fake"
    out.mkdir(parents=True, exist_ok=True)
    w = walk(painter, depth, theme, seed, headings, walked)
    for i, s in enumerate(w["spots"]):
        s.image.save(out / f"spot{i}.png")
    for i, p in enumerate(w["prevs"], 1):
        p.save(out / f"prev{i}.png")
    info = {"recipe": recipe, "theme": theme, "seed": seed, "headings": headings, "walked": walked,
            "spot_s": [round(x, 2) for x in w["spot_s"]], "paint_s": [round(x, 2) for x in w["paint_s"]],
            "distance": [round(s.distance, 3) for s in w["spots"]], "scale": [round(s.scale, 3) for s in w["spots"]],
            "prompts": [[[t, round(x, 3)] for t, x in p] for p in w["prompts"]]}
    seams = [seam(s.image) for s in w["spots"]]
    info["seam"], info["seam_at"] = [x[0] for x in seams], [x[1] for x in seams]
    (out / "timings.json").write_text(json.dumps(info, indent=1))
    labels = [f"{recipe}  spot {i}  heading {h:g}  {'pano' if i == 0 else 'step'} {info['paint_s'][i]:.1f}s  "
              f"seam {info['seam'][i]:.2f} @ {info['seam_at'][i]}" for i, h in enumerate(headings)]
    sheet([s.image for s in w["spots"]], labels).save(out / "sheet.png")
    return info


if __name__ == "__main__":
    main()
