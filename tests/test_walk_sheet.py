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
