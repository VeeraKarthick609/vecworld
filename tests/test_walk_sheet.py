"""scripts/walk_sheet.py on the fake painter and depth (CPU): a walk writes N spots, a sheet and its timings."""
import importlib.util
import json
from pathlib import Path

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
