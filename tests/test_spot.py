import numpy as np
import pytest
from PIL import Image

from conftest import FakeDepth, FakePainter
from vecworld.config import DOLLY, FAR, NEAR, PALETTE_STRENGTH
from vecworld.dream.pano import dolly, roll_to, yaw_to_u
from vecworld.roam.spot import Spot, align_scale, dolly_distance, first_spot, next_spot
from vecworld.scenes.palette import match_palette, palette_stats

W, H = 128, 32


def _depth_for(r):
    """The 0..1 inverse depth whose viewer radius is r."""
    return (1 / np.asarray(r, np.float64) - 1 / FAR) / (1 / NEAR - 1 / FAR)


def _gradient(w=W, h=H):
    a = np.zeros((h, w, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    a[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return Image.fromarray(a)


def _spot(r, heading=0.0, scale=1.0):
    d = _depth_for(r) if np.ndim(r) == 2 else np.full((H, W), _depth_for(r))
    return Spot(id="p", image=_gradient(), depth=d.astype(np.float32), heading=heading, scale=scale)


def test_first_spot_heading_zero_scale_one():
    painter = FakePainter()
    s = first_spot(painter, FakeDepth(), [("a forest", 1.0)], seed=5)
    assert s.heading == 0 and s.scale == 1
    assert painter.calls == [("panorama", {"base": [("a forest", 1.0)], "seed": 5})]
    assert s.depth.shape == (s.image.height, s.image.width)
    assert s.id


def test_next_spot_rolls_to_relative_heading():
    painter = FakePainter()
    parent = _spot(10.0, heading=300.0)
    child = next_spot(painter, FakeDepth(), parent, exit_heading=100.0, base=[("b", 1.0)],
                      focus=[("f", 1.0)], seed=9, ref=None, bg_strength=0.7)
    (name, kw), = painter.calls
    assert name == "step"
    # 100 - 300 = -200, normalised to +160
    want = dolly(roll_to(parent.image, yaw_to_u(160.0)))
    assert np.array_equal(np.asarray(kw["prev"]), np.asarray(want))
    assert kw["focus_at"] == 0.5 and kw["focus"] == [("f", 1.0)] and kw["base"] == [("b", 1.0)]
    assert kw["seed"] == 9 and kw["bg_strength"] == 0.7
    assert child.heading == pytest.approx(100.0)
    assert child.depth.shape == (child.image.height, child.image.width)
    assert child.id != parent.id


@pytest.mark.parametrize("parent_h, exit_h, rel", [(0, 180, 180), (90, -90, 180), (350, 10, 20), (10, 350, -20)])
def test_relative_heading_is_normalised(parent_h, exit_h, rel):
    painter = FakePainter()
    parent = _spot(10.0, heading=parent_h)
    child = next_spot(painter, FakeDepth(), parent, exit_h, [("b", 1.0)], None, 1, None)
    want = dolly(roll_to(parent.image, yaw_to_u(rel)))
    assert np.array_equal(np.asarray(painter.calls[0][1]["prev"]), np.asarray(want))
    assert 0 <= child.heading < 360 and child.heading == pytest.approx(exit_h % 360)


def test_next_spot_pulls_palette_toward_ref():
    painter = FakePainter()
    parent = _spot(10.0)
    ref = palette_stats(Image.new("RGB", (8, 8), "orange"))
    child = next_spot(painter, FakeDepth(), parent, 0.0, [("b", 1.0)], None, 1, ref)
    want = match_palette(painter._img("teal"), ref, PALETTE_STRENGTH)
    assert np.array_equal(np.asarray(child.image), np.asarray(want))
    plain = next_spot(FakePainter(), FakeDepth(), parent, 0.0, [("b", 1.0)], None, 1, None)
    assert np.array_equal(np.asarray(plain.image), np.asarray(painter._img("teal")))


@pytest.mark.parametrize("k", [0.6, 1.0, 1.5])
@pytest.mark.parametrize("parent_scale", [1.0, 2.0])
def test_align_scale_matches_medians(k, parent_scale):
    # Parent sees radius 12 straight ahead (heading 0). After the dolly that content is z times closer,
    # so a child at the same scale would see it at 12/z; this child's depth reads k times that.
    parent = _spot(12.0, scale=parent_scale)
    child = _spot(k * 12.0 / DOLLY)
    assert align_scale(parent, child, 0.0) == pytest.approx(parent_scale / k, rel=1e-3)


def test_align_scale_uses_parent_view_at_rel_heading():
    r = np.full((H, W), 30.0)
    u = (np.arange(W) + 0.5) / W
    near = np.abs(u - yaw_to_u(90.0)) < 0.1      # parent sees radius 8 to the east, 30 elsewhere
    r[:, near] = 8.0
    parent = _spot(r)
    child = _spot(8.0 / DOLLY)
    assert align_scale(parent, child, 90.0) == pytest.approx(1.0, rel=1e-3)
    assert align_scale(parent, child, -90.0) == pytest.approx(30.0 / 8.0, rel=1e-3)


def test_dolly_distance():
    assert dolly_distance(1.0, radius=10.0) == 0
    assert dolly_distance(DOLLY, radius=10.0) == pytest.approx(10.0 * (1 - 1 / DOLLY))
    assert dolly_distance() > 0


def test_next_spot_records_distance_in_world_units():
    parent = _spot(12.0, scale=2.0)
    child = next_spot(FakePainter(), FakeDepth(), parent, 0.0, [("b", 1.0)], None, 1, None)
    # forward median radius of the parent in world units is 2 * 12
    assert child.distance == pytest.approx(dolly_distance(DOLLY, radius=24.0), rel=1e-3)
    assert np.isfinite(child.scale) and child.scale > 0


def test_align_scale_clamps_a_wild_rescale():
    parent = _spot(NEAR)
    child = _spot(FAR)               # child sees everything at FAR where the parent saw NEAR: a bad depth read
    assert align_scale(parent, child, 0.0) == pytest.approx(0.25)
    assert align_scale(_spot(FAR), _spot(NEAR), 0.0) == pytest.approx(4.0)
