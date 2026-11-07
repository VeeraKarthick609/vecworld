"""scripts/walk_sheet.py on the fake painter and depth (CPU): a walk writes N spots, a sheet and its timings."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FakeDepth, FakePainter

SPEC = importlib.util.spec_from_file_location("walk_sheet", Path(__file__).parent.parent / "scripts" / "walk_sheet.py")
walk_sheet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(walk_sheet)


def test_walk_writes_each_spot_a_sheet_and_timings(tmp_path):
    painter = FakePainter()
    info = walk_sheet.run(painter, FakeDepth(), "t1", "haunted forest", 7, [0, 0, 60, 60, 300, 0], 9.0, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("spot*.png")) == [f"spot{i}.png" for i in range(6)]
    assert len(list(tmp_path.glob("prev*.png"))) == 5
    sheet = Image.open(tmp_path / "sheet.png")
    assert sheet.width == walk_sheet.SHEET_W and sheet.height > 6 * walk_sheet.LABEL_H
    saved = json.loads((tmp_path / "timings.json").read_text())
    assert saved == info and saved["recipe"] == "t1" and len(saved["spot_s"]) == len(saved["paint_s"]) == 6
    # as RoamService: the first spot is painted from scratch, then one step per heading, seeds counting up
    kinds = [c[0] for c in painter.calls]
    assert kinds == ["panorama"] + ["step"] * 5
    assert [c[1]["seed"] for c in painter.calls] == [7, 8, 9, 10, 11, 12]
    assert painter.calls[0][1]["base"][0][0].startswith("haunted forest, open ground in the foreground")
    assert len(painter.calls[2][1]["base"]) == 2                  # drifting toward a neighbour: a prompt blend


def test_compare_puts_walks_side_by_side(tmp_path):
    for name in ("a", "b"):
        walk_sheet.run(FakePainter(), FakeDepth(), name, "forest", 1, [0, 0, 60], 9.0, tmp_path / name)
    walk_sheet.main(["--compare", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "c.png")])
    im = Image.open(tmp_path / "c.png")
    assert im.width > 2 * walk_sheet.SHEET_W


def _texture(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(128, 40, (h, w)).clip(0, 255)


def _img(a):
    return Image.fromarray(np.repeat(a[..., None], 3, 2).astype(np.uint8))


def test_seam_score_sees_detail_meet_haze_but_not_a_trunk():
    h, w = 512, 2048
    tex = _texture(h, w)
    assert walk_sheet.seam(_img(tex))[0] < 0.5                      # detail everywhere
    trunk = tex.copy()
    trunk[:, 1000:1040] *= 0.6                                     # a darker full-height trunk with its own bark
    assert walk_sheet.seam(_img(trunk))[0] < walk_sheet.SEAM_LIMIT
    veil = tex.copy()
    veil[:, 1200:1500] = 200                                       # a featureless band: haze against detail
    score, at = walk_sheet.seam(_img(veil))
    assert score > walk_sheet.SEAM_LIMIT and min(abs(at - 1200), abs(at - 1500)) <= 12


def test_check_fails_a_walk_with_a_seam(tmp_path, capsys):
    for name, a in (("clean", _texture(64, 256)), ("seamed", _texture(64, 256))):
        if name == "seamed":
            a[:, 100:180] = 200
        d = tmp_path / name
        d.mkdir()
        _img(a).save(d / "spot0.png")
    assert walk_sheet.check([tmp_path / "clean"]) == []
    bad = walk_sheet.check([tmp_path / "seamed"])
    assert len(bad) == 1 and bad[0].startswith("seamed/spot0")
    with pytest.raises(SystemExit):
        walk_sheet.main(["--check", str(tmp_path / "seamed")])


def test_a_walk_records_each_spots_seam_and_native_compare_sheets(tmp_path):
    for name in ("a", "b"):
        info = walk_sheet.run(FakePainter(), FakeDepth(), name, "forest", 1, [0, 0, 60], 9.0, tmp_path / name)
        assert len(info["seam"]) == len(info["seam_at"]) == 3
    walk_sheet.main(["--compare", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "c.png")])
    spot = Image.open(tmp_path / "a" / "spot1.png")
    for i in range(3):
        native = Image.open(tmp_path / f"c_spot{i}.png")
        assert native.width == spot.width                            # native size, not scaled
        assert native.height == 2 * (spot.height + walk_sheet.LABEL_H)


# --- the multi-scene relative gate (Task 8) -------------------------------------------------------------------------

def test_a_walk_records_the_cache_threshold(tmp_path):
    """timings.json records the block cache threshold the walk ran with (0.0 for a painter without one: no cache) and
    the cache's passes and skips per painter call (painter.last_cache)."""
    info = walk_sheet.run(FakePainter(), FakeDepth(), "t1", "forest", 1, [0, 0], 9.0, tmp_path / "a")
    assert info["cache_threshold"] == 0.0 and info["cache"] == [None, None]
    p = FakePainter()
    p.cache_threshold, p.last_cache = 0.15, {"passes": 8, "skips": 3}
    info = walk_sheet.run(p, FakeDepth(), "t1", "forest", 1, [0, 0], 9.0, tmp_path / "b")
    assert info["cache_threshold"] == 0.15 and info["cache"] == [{"passes": 8, "skips": 3}] * 2


def test_scenes_are_three_different_themes():
    assert len(walk_sheet.SCENES) >= 3
    themes = {theme for theme, _ in walk_sheet.SCENES.values()}
    assert len(themes) == len(walk_sheet.SCENES)


def test_run_scenes_walks_each_scene_into_its_own_dir(tmp_path):
    painter = FakePainter()
    dirs = walk_sheet.run_scenes(painter, FakeDepth(), None, ["forest", "desert"], [0, 0, 60], 9.0, tmp_path / "w")
    assert [d.name for d in dirs] == ["w-forest", "w-desert"]
    for d, name in zip(dirs, ("forest", "desert")):
        info = json.loads((d / "timings.json").read_text())
        assert (info["theme"], info["seed"]) == walk_sheet.SCENES[name]
        assert info["scene"] == name
    assert [c[1]["seed"] for c in painter.calls if c[0] == "panorama"] == [walk_sheet.SCENES[n][1]
                                                                           for n in ("forest", "desert")]
    with pytest.raises(SystemExit):
        walk_sheet.main(["--scenes", "nowhere", "--out", str(tmp_path / "x")])


def _walk_dir(d, arrays, **over):
    """A walk directory as run() writes it; its scene (theme, seed) from the directory's -<scene> suffix."""
    d.mkdir(parents=True)
    for i, a in enumerate(arrays):
        _img(a).save(d / f"spot{i}.png")
    scene = d.name.rsplit("-", 1)[-1]
    theme, seed = walk_sheet.SCENES.get(scene, ("haunted forest", 7))
    info = {"recipe": "sharp", "scene": scene, "theme": theme, "seed": seed, "headings": [0.0, 0.0, 60.0][:len(arrays)],
            "walked": 9.0, "seam": [walk_sheet.seam(_img(a))[0] for a in arrays], "kernel": "w4a8", "keep_a16": [],
            "git": "abc", "cache_threshold": 0.0, "cache": [None] * len(arrays), **over}
    (d / "timings.json").write_text(json.dumps(info))


def test_relative_gate_compares_each_spot_with_its_base(tmp_path):
    """--relative: per scene and spot, the seam delta against the base walk's same spot (fail above max_delta) and the
    PSNR against the base image (identical: inf)."""
    clean = [_texture(64, 256, s) for s in range(3)]
    seamed = [a.copy() for a in clean]
    seamed[2][:, 100:180] = 200                                      # spot 2 grows a hard seam
    noisy = [np.clip(a + np.random.default_rng(1).normal(0, 2, a.shape), 0, 255) for a in clean]
    for scene in ("forest", "desert"):
        _walk_dir(tmp_path / f"base-{scene}", clean)
        _walk_dir(tmp_path / f"same-{scene}", clean)
        _walk_dir(tmp_path / f"noisy-{scene}", noisy)
    _walk_dir(tmp_path / "bad-forest", clean)
    _walk_dir(tmp_path / "bad-desert", seamed)
    scenes = ["forest", "desert"]
    rows, bad = walk_sheet.relative(tmp_path / "base", tmp_path / "same", scenes)
    assert bad == [] and len(rows) == 6
    assert all(r["delta"] == 0 and r["psnr"] == float("inf") for r in rows)
    rows, bad = walk_sheet.relative(tmp_path / "base", tmp_path / "noisy", scenes)
    assert all(30 < r["psnr"] < 60 for r in rows)
    rows, bad = walk_sheet.relative(tmp_path / "base", tmp_path / "bad", scenes)
    assert len(bad) == 1 and "desert" in bad[0] and "spot2" in bad[0]
    with pytest.raises(SystemExit):
        walk_sheet.main(["--relative", str(tmp_path / "base"), str(tmp_path / "bad"), "--scenes", "forest,desert"])
    walk_sheet.main(["--relative", str(tmp_path / "base"), str(tmp_path / "same"), "--scenes", "forest,desert"])
    # without --scenes the two paths are walk directories themselves
    rows, bad = walk_sheet.relative(tmp_path / "base-desert", tmp_path / "bad-desert", None)
    assert len(bad) == 1


def test_relative_refuses_walks_of_another_scene_or_route(tmp_path):
    """Task 8 review M3: BASE and CAND must have walked the same scene and route (theme, seed, headings, recipe,
    walked, by their timings.json); otherwise the per-spot deltas compare unrelated images, and the gate fails loudly
    instead of scoring them."""
    clean = [_texture(64, 256, s) for s in range(3)]
    _walk_dir(tmp_path / "base-forest", clean)
    _walk_dir(tmp_path / "base-desert", clean)
    _walk_dir(tmp_path / "ok-forest", clean)
    for name, over in (("seed", {"seed": 8}), ("headings", {"headings": [0.0, 0.0, 300.0]}),
                       ("recipe", {"recipe": "t1"}), ("walked", {"walked": 6.0}), ("theme", {"theme": "red desert"})):
        _walk_dir(tmp_path / f"{name}x-forest", clean, **over)
        with pytest.raises(SystemExit, match=name):
            walk_sheet.relative(tmp_path / f"{name}x", tmp_path / "base", ["forest"])
    with pytest.raises(SystemExit, match="theme|seed"):                  # a typo'd prefix that happens to exist
        walk_sheet.relative(tmp_path / "base-forest", tmp_path / "base-desert", None)
    (tmp_path / "ok-forest" / "timings.json").unlink()
    with pytest.raises(SystemExit, match="timings.json"):
        walk_sheet.relative(tmp_path / "base", tmp_path / "ok", ["forest"])
    with pytest.raises(SystemExit):
        walk_sheet.main(["--relative", str(tmp_path / "base"), str(tmp_path / "seedx"), "--scenes", "forest"])


def test_relative_gate_flags_a_vacuous_pass(tmp_path, capsys):
    """Task 8 review M4: a candidate that claims a lossy setting (a cache threshold > 0, or --expect-change) but
    exercised nothing (no skipped pass, or every spot byte-identical to its base) is VACUOUS, not PASS (exit 2). A
    lossless candidate that comes out identical is a PASS."""
    clean = [_texture(64, 256, s) for s in range(2)]
    noisy = [np.clip(a + np.random.default_rng(1).normal(0, 2, a.shape), 0, 255) for a in clean]
    lossy = {"cache_threshold": 0.2}
    _walk_dir(tmp_path / "base-forest", clean)
    _walk_dir(tmp_path / "same-forest", clean)                                     # lossless, identical: PASS
    _walk_dir(tmp_path / "noskip-forest", noisy, **lossy, cache=[{"passes": 8, "skips": 0}] * 2)
    _walk_dir(tmp_path / "ident-forest", clean, **lossy, cache=[{"passes": 8, "skips": 3}] * 2)
    _walk_dir(tmp_path / "real-forest", noisy, **lossy, cache=[{"passes": 8, "skips": 3}] * 2)
    s = ["forest"]

    def verdict(cand, **k):
        rows, bad = walk_sheet.relative(tmp_path / "base", tmp_path / cand, s)
        return walk_sheet.verdict(tmp_path / cand, s, rows, bad, **k)[0]

    assert verdict("same") == "PASS"
    assert verdict("same", expect_change=True) == "VACUOUS"
    assert verdict("noskip") == "VACUOUS"
    assert verdict("ident") == "VACUOUS"
    assert verdict("real") == "PASS"
    with pytest.raises(SystemExit) as e:
        walk_sheet.main(["--relative", str(tmp_path / "base"), str(tmp_path / "noskip"), "--scenes", "forest"])
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "VACUOUS" in out and "skips 0" in out
    walk_sheet.main(["--relative", str(tmp_path / "base"), str(tmp_path / "real"), "--scenes", "forest"])
    out = capsys.readouterr().out
    assert "PASS" in out and "skips 6" in out
