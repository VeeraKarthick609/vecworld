"""A deterministic offline Dream Roam walk on the real painter, depth and palette, with a contact sheet.

  uv run python scripts/walk_sheet.py --recipe repaint --theme "haunted forest" --seed 7 --spots 6 \
      --headings 0,0,60,60,300,0 --walked 9 --out /tmp/walk/repaint
  uv run python scripts/walk_sheet.py --compare /tmp/walk/t1 /tmp/walk/repaint --out /tmp/walk/compare.png

A walk calls first_spot / next_spot exactly as RoamService.start and RoamService._build do (the same prompts,
fallback neighbours, journey drift, colour reference and seeds), with no browser, server or GPU queue. headings[i]
is the world bearing spot i is reached by (headings[0] is the first spot's, 0). It writes spot{i}.png, prev{i}.png
(the dollied view each step started from), sheet.png and timings.json into --out; timings.json also records the
engine kernel, the linears kept on mma (FLUX_A16_KEEP under w4a8), whether the glue was fused (VECWORLD_FUSE),
whether the blocks replayed as CUDA graphs (VECWORLD_GRAPHS), the windows per pass (painter.batch, VECWORLD_BATCH;
1 for a painter that doesn't batch windows) and the git sha the walk ran. A walk refuses a tree with uncommitted
changes (its sha wouldn't name the code) unless --allow-dirty, which keeps the diff beside it (git-dirty.diff).

Each spot gets a seam score (seam_profile: the strongest full-height step in local detail between the columns
either side of one column). A walk whose spot scores above --max-seam exits 1; --check re-scores walks already
written. Calibration (2026-10-21, haunted-forest walk): painted panoramas and natural fog edges scored up to 1.08;
the walk-step seams 1.13-1.79 (Task 1's, and "sharp" at feather 0.15: spots 2, 4, 5 at 1.36, 1.38, 1.13). The
margin is thin: it catches hard full-height edges, not soft haze bands.

--compare puts several walks side by side: OUT (one column per walk, half size, for an overview) and, at native
size, OUT_spot{i}.png per spot (the walks' spot i stacked, full 2048 px width) to judge by eye.

The multi-scene relative gate (Task 8; for every lossy engine change from then on):

  uv run python scripts/walk_sheet.py --scenes forest,desert,coast --out ~/walk-archive/t8-base
  uv run python scripts/walk_sheet.py --relative ~/walk-archive/t8-base ~/walk-archive/t8-c0.1 \
      --scenes forest,desert,coast

--scenes walks each named scene (SCENES: theme and seed) in one process, into OUT-<scene>. --relative BASE CAND
compares every spot of CAND-<scene> with the same spot of BASE-<scene> (without --scenes: BASE and CAND are walk
directories): the seam delta (CAND - BASE; above --max-delta, 0.03, fails, exit 1) and the PSNR of CAND's image
against BASE's (inf: identical). timings.json also records the block cache threshold (FLUX_CACHE_THRESHOLD).

The gate refuses (exit 1) a pair that did not walk the same scene and route (theme, seed, headings, recipe, walked,
from both timings.json): their per-spot deltas would compare unrelated images. Its verdict is PASS, FAIL or VACUOUS
(exit 2): a candidate that claims a lossy setting (a block cache threshold > 0, or --expect-change for any other
lossy knob) but exercised nothing (no skipped pass, or every spot byte-identical to its base) proves nothing. A
lossless candidate that comes out identical is a PASS.
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
MAX_DELTA = 0.03        # --relative: a spot may score at most this much above the same spot of its base walk
ROUTE = ("theme", "seed", "headings", "recipe", "walked")   # --relative: what BASE and CAND must share
# The gate's scenes (name: (theme, seed)): a dark forest, a bright desert and a sea-and-sky coast.
SCENES = {"forest": ("haunted forest", 7), "desert": ("red desert", 11), "coast": ("stormy coast", 23)}


class _Recorder:
    """Wraps a painter: keeps each step's input view, times each painter call and keeps the painter's block cache
    counts for it (painter.last_cache: passes, skips; None without a cache)."""

    def __init__(self, painter):
        self.painter, self.prevs, self.paint_s, self.cache = painter, [], [], []

    def __getattr__(self, name):
        return getattr(self.painter, name)

    def panorama(self, base, seed):
        t = time.perf_counter()
        img = self.painter.panorama(base, seed)
        self.paint_s.append(time.perf_counter() - t)
        self.cache.append(getattr(self.painter, "last_cache", None))
        return img

    def step(self, prev, *a, **k):
        self.prevs.append(prev)
        t = time.perf_counter()
        img = self.painter.step(prev, *a, **k)
        self.paint_s.append(time.perf_counter() - t)
        self.cache.append(getattr(self.painter, "last_cache", None))
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
    return {"spots": spots, "prompts": prompts, "spot_s": times, "paint_s": rec.paint_s, "prevs": rec.prevs,
            "cache": rec.cache}


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


def psnr(a: Image.Image, b: Image.Image) -> float:
    """PSNR of a against b in dB, over RGB 0..255 (inf: identical)."""
    x = np.asarray(a.convert("RGB"), np.float64)
    y = np.asarray(b.convert("RGB"), np.float64)
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else round(10 * np.log10(255.0 ** 2 / mse), 2)


def _pairs(base: Path, cand: Path, scenes: list[str] | None) -> list[tuple[str, Path, Path]]:
    if not scenes:
        return [(cand.name, base, cand)]
    return [(s, base.with_name(f"{base.name}-{s}"), cand.with_name(f"{cand.name}-{s}")) for s in scenes]


def _info(d: Path) -> dict:
    f = d / "timings.json"
    if not f.exists():
        raise SystemExit(f"{d}: no timings.json, so the gate can't check it walked the same scene and route as its pair")
    return json.loads(f.read_text())


def same_route(b: Path, c: Path) -> None:
    """Exit (1) unless walks b and c share their scene and route (ROUTE, from their timings.json)."""
    ib, ic = _info(b), _info(c)
    missing = [k for k in ROUTE if k not in ib or k not in ic]
    if missing:
        raise SystemExit(f"{b.name} / {c.name}: timings.json lacks {', '.join(missing)}, so the gate can't tell they "
                         f"walked the same route")
    diff = [k for k in ROUTE if ib.get(k) != ic.get(k)]
    if diff:
        raise SystemExit(f"{b.name} and {c.name} walked different scenes or routes ("
                         + "; ".join(f"{k} {ib.get(k)!r} vs {ic.get(k)!r}" for k in diff)
                         + "): their spots can't be compared")


def relative(base: Path, cand: Path, scenes: list[str] | None, max_delta: float = MAX_DELTA):
    """The relative gate: (rows, failures). One row per scene and spot: seams of base and candidate, their delta and
    the candidate's PSNR against the base image. A failure is a spot whose seam rose by more than max_delta. Exits if a
    pair did not walk the same scene and route (same_route)."""
    rows, bad = [], []
    for scene, b, c in _pairs(base, cand, scenes):
        same_route(b, c)
        n = len(list(b.glob("spot*.png")))
        if n == 0 or len(list(c.glob("spot*.png"))) != n:
            raise SystemExit(f"{b} and {c} must hold the same number of spots")
        for i in range(n):
            ib, ic = Image.open(b / f"spot{i}.png"), Image.open(c / f"spot{i}.png")
            sb, sc = seam(ib)[0], seam(ic)[0]
            r = {"scene": scene, "spot": i, "base": sb, "seam": sc, "delta": round(sc - sb, 3), "psnr": psnr(ic, ib)}
            rows.append(r)
            if r["delta"] > max_delta:
                bad.append(f"{scene}/spot{i}: seam {sc:.3f} vs base {sb:.3f} (+{r['delta']:.3f})")
    return rows, bad


def _cache_counts(infos: list[dict]) -> tuple[int, int]:
    """(passes, skipped passes) of the block cache over the walks' painter calls (timings.json "cache")."""
    calls = [c for i in infos for c in i.get("cache") or [] if c]
    return sum(c.get("passes", 0) for c in calls), sum(c.get("skips", 0) for c in calls)


def verdict(cand: Path, scenes, rows, bad, expect_change: bool = False) -> tuple[str, str]:
    """(PASS / FAIL / VACUOUS, why). VACUOUS: the candidate claims a lossy setting (its block cache threshold > 0, or
    expect_change) but exercised nothing: no skipped pass, or every spot byte-identical to its base."""
    infos = [_info(c) for _, _, c in _pairs(cand, cand, scenes)]
    if bad:
        return "FAIL", f"{len(bad)} spot(s) worse than the base by more than the allowed delta"
    cached = any((i.get("cache_threshold") or 0) > 0 for i in infos)
    _, skips = _cache_counts(infos)
    if cached and skips == 0:
        return "VACUOUS", "the block cache is on (threshold > 0) but skipped no pass"
    if (cached or expect_change) and all(r["psnr"] == float("inf") for r in rows):
        return "VACUOUS", "a lossy setting was under test but every spot is byte-identical to its base"
    return "PASS", ""


def _print_relative(base: Path, cand: Path, scenes, rows) -> None:
    for scene, b, c in _pairs(base, cand, scenes):
        ib, ic = (json.loads((d / "timings.json").read_text()) if (d / "timings.json").exists() else {} for d in (b, c))
        passes, skips = _cache_counts([ic])
        bpasses, _ = _cache_counts([ib])
        print(f"{scene}: base {b.name} (git {ib.get('git')}, kernel {ib.get('kernel')}, batch {ib.get('batch')}, cache "
              f"{ib.get('cache_threshold')}: passes {bpasses}) vs {c.name} (git {ic.get('git')}, kernel "
              f"{ic.get('kernel')}, batch {ic.get('batch')}, cache {ic.get('cache_threshold')}: passes {passes}, "
              f"skips {skips})")
        if ib.get("kernel") != ic.get("kernel") or ib.get("keep_a16") != ic.get("keep_a16"):
            print("  warning: the walks ran different kernels or keep-lists")
        for r in (r for r in rows if r["scene"] == scene):
            print(f"  spot{r['spot']}: seam {r['seam']:.3f} base {r['base']:.3f} delta {r['delta']:+.3f} "
                  f"psnr {r['psnr']:.2f} dB")
    worst = max(r["delta"] for r in rows)
    print(f"worst delta {worst:+.3f}, min psnr {min(r['psnr'] for r in rows):.2f} dB")


def run_scenes(painter, depth, recipe, scenes: list[str], headings, walked, out: Path) -> list[Path]:
    """Walk each scene of SCENES (its theme and seed) into OUT-<scene>; returns the directories."""
    dirs = []
    for name in scenes:
        theme, seed = SCENES[name]
        d = out.with_name(f"{out.name}-{name}")
        run(painter, depth, recipe, theme, seed, headings, walked, d, scene=name)
        dirs.append(d)
    return dirs


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
    ap.add_argument("--scenes", help=f"comma-separated scene names to walk (into OUT-<scene>) or gate; known: "
                                     f"{', '.join(SCENES)}")
    ap.add_argument("--relative", nargs=2, type=Path, metavar=("BASE", "CAND"),
                    help="gate CAND against BASE spot by spot (seam delta, PSNR); with --scenes, both are prefixes")
    ap.add_argument("--max-delta", type=float, default=MAX_DELTA, help="--relative: the largest seam rise allowed")
    ap.add_argument("--expect-change", action="store_true",
                    help="--relative: CAND runs a lossy setting, so byte-identical spots are VACUOUS, not PASS")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="walk uncommitted code (the diff is kept as OUT/git-dirty.diff, the sha recorded -dirty)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    scenes = [x.strip() for x in a.scenes.split(",") if x.strip()] if a.scenes else None
    unknown = [x for x in scenes or () if x not in SCENES]
    if unknown:
        ap.error(f"unknown scenes {unknown}; known: {', '.join(SCENES)}")
    if a.relative:
        rows, bad = relative(*a.relative, scenes, a.max_delta)
        _print_relative(*a.relative, scenes, rows)
        v, why = verdict(a.relative[1], scenes, rows, bad, a.expect_change)
        _, skips = _cache_counts([_info(c) for _, _, c in _pairs(*a.relative, scenes)])
        print(f"verdict {v} (candidate skips {skips})" + (f": {why}" if why else ""))
        if v == "FAIL":
            sys.exit("seams worse than the base by more than %.3f: %s" % (a.max_delta, "; ".join(bad)))
        if v == "VACUOUS":
            print(f"VACUOUS, not a pass: {why}", file=sys.stderr)
            sys.exit(2)
        return
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
    if _git().endswith("-dirty"):
        if not a.allow_dirty:
            sys.exit("the tree has uncommitted changes, so the walk's recorded sha wouldn't name its code: commit, "
                     "or pass --allow-dirty (keeps the diff beside the walk)")
        for d in ([a.out.with_name(f"{a.out.name}-{x}") for x in scenes] if scenes else [a.out]):
            save_dirty_diff(d)
    from vecworld.flux.painter import FluxPainter
    from vecworld.roam.depth import AnythingDepth

    painter, depth = FluxPainter(), AnythingDepth()
    try:
        if scenes:
            dirs = run_scenes(painter, depth, a.recipe, scenes, headings, a.walked, a.out)
            infos = [json.loads((d / "timings.json").read_text()) for d in dirs]
        else:
            infos = [run(painter, depth, a.recipe, a.theme, a.seed, headings, a.walked, a.out)]
    finally:
        painter.close()
    bad = [f"{info.get('scene') or ''} spot{i}: {sc:.2f} at column {at}" for info in infos
           for i, (sc, at) in enumerate(zip(info["seam"], info["seam_at"])) if sc > a.max_seam]
    if bad:
        sys.exit("seams above %.2f: %s" % (a.max_seam, "; ".join(bad)))


def _git_diff() -> str:
    import subprocess

    return subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout


def save_dirty_diff(d: Path) -> None:
    """Keep the uncommitted diff a --allow-dirty walk ran beside it."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "git-dirty.diff").write_text(_git_diff())


def _git() -> str:
    """HEAD's short sha, "-dirty" with uncommitted tracked changes: which code a walk ran."""
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def run(painter, depth, recipe, theme, seed, headings, walked, out: Path, scene: str | None = None) -> dict:
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
    info = {"recipe": recipe, "scene": scene, "theme": theme, "seed": seed, "headings": headings, "walked": walked,
            "kernel": getattr(painter, "kernel", None), "keep_a16": list(getattr(painter, "keep_a16", ())),
            "fused": getattr(painter, "fused", None), "graphs": getattr(painter, "graphs", None),
            "cache_threshold": getattr(painter, "cache_threshold", 0.0), "cache": w["cache"],
            "batch": int(getattr(painter, "batch", 1)), "git": _git(),
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
