"""A deterministic offline Dream Roam walk on the real painter, depth and palette, with a contact sheet.

  uv run python scripts/walk_sheet.py --recipe repaint --theme "haunted forest" --seed 7 --spots 6 \
      --headings 0,0,60,60,300,0 --walked 9 --out /tmp/walk/repaint
  uv run python scripts/walk_sheet.py --compare /tmp/walk/t1 /tmp/walk/repaint --out /tmp/walk/compare.png

A walk calls first_spot / next_spot exactly as RoamService.start and RoamService._build do (the same prompts,
fallback neighbours, journey drift, colour reference and seeds), with no browser, server or GPU queue. headings[i]
is the world bearing spot i is reached by (headings[0] is the first spot's, 0). It writes spot{i}.png, prev{i}.png
(the dollied view each step started from), sheet.png and timings.json into --out. --compare puts several walks'
spots side by side, one column per walk, into one image.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vecworld.roam.places import Journey, assign_bearings, fallback_neighbours  # noqa: E402
from vecworld.roam.service import MAX_NEIGHBOURS, OPEN_GROUND, RoamService, _blend, _copy  # noqa: E402
from vecworld.roam.spot import first_spot, next_spot  # noqa: E402
from vecworld.scenes.palette import palette_stats  # noqa: E402

SHEET_W = 1024          # each panorama is drawn this wide on a sheet
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", help="FLUX step recipe (config.FLUX_STEP_RECIPES); default: the configured one")
    ap.add_argument("--theme", default="haunted forest")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--headings", default="0,0,60,60,300,0")
    ap.add_argument("--walked", type=float, default=9.0)
    ap.add_argument("--compare", nargs="+", type=Path, help="walk directories to put side by side")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.compare:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        compare(a.compare).save(a.out)
        return
    headings = [float(x) for x in a.headings.split(",")][:a.spots]
    if len(headings) < a.spots:
        ap.error(f"--headings gives {len(headings)} bearings for {a.spots} spots")
    from vecworld.flux.painter import FluxPainter
    from vecworld.roam.depth import AnythingDepth

    painter, depth = FluxPainter(), AnythingDepth()
    try:
        run(painter, depth, a.recipe, a.theme, a.seed, headings, a.walked, a.out)
    finally:
        painter.close()


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
    (out / "timings.json").write_text(json.dumps(info, indent=1))
    labels = [f"{recipe}  spot {i}  heading {h:g}  {'pano' if i == 0 else 'step'} {info['paint_s'][i]:.1f}s"
              for i, h in enumerate(headings)]
    sheet([s.image for s in w["spots"]], labels).save(out / "sheet.png")
    return info


if __name__ == "__main__":
    main()
