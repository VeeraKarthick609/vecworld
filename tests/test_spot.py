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


def _z(parent, rel):
    """The dolly a pre-dream uses toward rel: walked to the parent's walkable radius that way (uniform radius parents
    only: R is then the same for every z)."""
    from vecworld.roam.spot import limit_at, walk_limits, walk_z

    return walk_z(float(np.median(1 / (parent.depth * (1 / NEAR - 1 / FAR) + 1 / FAR))),
                  limit_at(walk_limits(parent.depth), rel))


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
    # 100 - 300 = -200, normalised to +160; dollied by the walk to the parent's limit that way (8b: was DOLLY)
    want = dolly(roll_to(parent.image, yaw_to_u(160.0)), _z(parent, 160.0))
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
    want = dolly(roll_to(parent.image, yaw_to_u(rel)), _z(parent, rel))
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
    # so a child in the parent's frame would see it at 12/z; this child's depth reads k times that.
    # The scale is the parent-to-child ratio only: the parent's own scale does not enter it.
    parent = _spot(12.0, scale=parent_scale)
    child = _spot(k * 12.0 / DOLLY)
    assert align_scale(parent, child, 0.0) == pytest.approx(1 / k, rel=1e-3)


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


def test_next_spot_records_distance_in_parent_frame():
    parent = _spot(12.0, scale=2.0)
    child = next_spot(FakePainter(), FakeDepth(), parent, 0.0, [("b", 1.0)], None, 1, None, walked=3.0)
    # the walk is in the parent's own frame (forward median radius 12), whatever the parent's scale
    assert child.distance == pytest.approx(3.0, rel=1e-3)
    assert np.isfinite(child.scale) and child.scale > 0


class NanDepth:
    def depth360(self, pano):
        d = FakeDepth().depth360(pano)
        d[:, :] = np.nan
        return d


def test_nan_depth_is_sanitised_and_guarded():
    parent = _spot(12.0)
    child = next_spot(FakePainter(), NanDepth(), parent, 30.0, [("b", 1.0)], None, 1, None)
    assert np.isfinite(child.depth).all() and child.depth.min() >= 0 and child.depth.max() <= 1
    assert np.isfinite(child.scale) and child.scale > 0
    assert np.isfinite(child.distance) and child.distance > 0
    # a parent whose stored depth is NaN (nothing to align on) still gives finite scale and distance
    bad = Spot(id="b", image=_gradient(), depth=np.full((H, W), np.nan, np.float32), heading=0.0, scale=1.0)
    grand = next_spot(FakePainter(), FakeDepth(), bad, 0.0, [("b", 1.0)], None, 1, None)
    assert np.isfinite(grand.scale) and grand.scale > 0
    assert np.isfinite(grand.distance) and grand.distance > 0


def test_exit_heading_sign_puts_that_bearing_ahead():
    # Independent of roll_to/yaw_to_u: a red marker column 60 degrees clockwise of the parent's forward
    # (u = 0.5 + 60/360) must be dead ahead (the centre column) of what the painter repaints for exit +60.
    w, h = 360, 64
    a = np.zeros((h, w, 3), np.uint8)
    x = int((0.5 + 60 / 360) * w)               # column 240
    a[:, x, 0] = 255
    parent = Spot(id="p", image=Image.fromarray(a), depth=np.full((h, w), 0.5, np.float32), heading=0.0, scale=1.0)
    painter = FakePainter()
    next_spot(painter, FakeDepth(), parent, 60.0, [("b", 1.0)], None, 1, None)
    prev = np.asarray(painter.calls[0][1]["prev"]).astype(float)
    peak = int(np.argmax(prev[h // 2, :, 0]))
    assert abs(peak - w // 2) <= 2
    # and it is not the mirror image: exit -60 puts it far from the centre
    painter = FakePainter()
    next_spot(painter, FakeDepth(), parent, -60.0, [("b", 1.0)], None, 1, None)
    prev = np.asarray(painter.calls[0][1]["prev"]).astype(float)
    assert abs(int(np.argmax(prev[h // 2, :, 0])) - w // 2) > 60


def test_align_scale_clamps_a_wild_rescale():
    parent = _spot(NEAR)
    child = _spot(FAR)               # child sees everything at FAR where the parent saw NEAR: a bad depth read
    assert align_scale(parent, child, 0.0) == pytest.approx(0.25)
    assert align_scale(_spot(FAR), _spot(NEAR), 0.0) == pytest.approx(4.0)


# --- walking farther: the dolly matched to the distance walked ------------------------------------

def test_walk_z_is_the_dolly_that_walks_w_into_radius_r():
    from vecworld.config import WALK_Z_MAX, WALK_Z_MIN
    from vecworld.roam.spot import walk_z

    assert walk_z(10.0, 5.0) == pytest.approx(2.0)            # what was 10 ahead is now 5 away: ×2
    assert walk_z(12.0, 4.0) == pytest.approx(1.5)
    assert dolly_distance(walk_z(12.0, 4.0), radius=12.0) == pytest.approx(4.0)
    assert walk_z(10.0, 0.0) == WALK_Z_MIN                    # clamped below...
    assert walk_z(10.0, 9.0) == WALK_Z_MAX                    # ...and above
    assert walk_z(10.0, 10.0) == WALK_Z_MAX and walk_z(10.0, 50.0) == WALK_Z_MAX
    assert walk_z(float("nan"), 3.0) == WALK_Z_MIN


def test_next_spot_dollies_by_the_walked_distance():
    painter = FakePainter()
    parent = _spot(12.0)
    child = next_spot(painter, FakeDepth(), parent, 0.0, [("b", 1.0)], None, 1, None, walked=4.0)
    want = dolly(roll_to(parent.image, yaw_to_u(0.0)), 1.5)   # z = 12 / (12 - 4)
    assert np.array_equal(np.asarray(painter.calls[0][1]["prev"]), np.asarray(want))
    assert child.distance == pytest.approx(4.0, rel=1e-3)     # the walked distance, in the parent's frame


def test_next_spot_walked_too_far_is_clamped_and_says_so():
    from vecworld.config import WALK_Z_MAX

    painter = FakePainter()
    parent = _spot(12.0)
    child = next_spot(painter, FakeDepth(), parent, 0.0, [("b", 1.0)], None, 1, None, walked=11.0)
    want = dolly(roll_to(parent.image, yaw_to_u(0.0)), WALK_Z_MAX)
    assert np.array_equal(np.asarray(painter.calls[0][1]["prev"]), np.asarray(want))
    # the distance is what the clamped dolly represents, not what was walked
    assert child.distance == pytest.approx(dolly_distance(WALK_Z_MAX, radius=12.0), rel=1e-3)


def test_next_spot_scale_matches_the_variable_dolly():
    # The child sees what the parent saw at 12 now 12/z away (z = 1.5 for 4 walked); this one reads it k× that.
    parent = _spot(12.0)
    child = next_spot(FakePainter(), _Flat(0.8 * 12.0 / 1.5), parent, 0.0, [("b", 1.0)], None, 1, None, walked=4.0)
    assert child.scale == pytest.approx(1 / 0.8, rel=1e-3)


class _Flat:
    """Depth model: every pixel at radius r."""

    def __init__(self, r):
        self.r = r

    def depth360(self, pano):
        w, h = pano.size
        return np.full((h, w), _depth_for(self.r), np.float32)


@pytest.mark.parametrize("z", [1.15, 2.0, 3.0])
def test_align_scale_for_any_z(z):
    parent = _spot(12.0)
    assert align_scale(parent, _spot(12.0 / z), 0.0, z) == pytest.approx(1.0, rel=1e-3)


def _ground_wall(wall_cols=None, wall_r=6.0, far=30.0, w=W * 4, h=H * 4, eye=3.0):
    """Radii of a flat ground `eye` below the camera, open sky-high scenery at `far`, and (optionally) a wall at
    wall_r over the columns wall_cols (a boolean mask over u) from the ground up to well above the horizon."""
    from vecworld.roam.spot import row_angles

    a = row_angles(h)[:, None]
    r = np.where(a < 0, np.clip(-eye / np.minimum(a, -1e-6), NEAR, FAR), far) * np.ones((1, w))
    if wall_cols is not None:
        r[:, wall_cols] = np.where(a > -eye / wall_r, wall_r, r[:, wall_cols])
    return r


def test_walk_limits_open_ground_walks_the_full_limit():
    from vecworld.config import WALK_LIMIT, LIMIT_BINS
    from vecworld.roam.spot import walk_limits

    lim = walk_limits(_depth_for(_ground_wall()))
    assert lim.shape == (LIMIT_BINS,)
    assert np.allclose(lim, WALK_LIMIT)


def test_walk_limits_stop_short_of_a_wall_in_that_direction():
    from vecworld.config import MIN_WALK, WALK_LIMIT, WALL_FRACTION
    from vecworld.roam.spot import limit_at, walk_limits

    w = W * 4
    u = (np.arange(w) + 0.5) / w
    east = np.abs(u - yaw_to_u(90.0)) < 0.06                 # a wall 6 away, about 40 degrees wide, to the east
    r = _ground_wall(east, wall_r=6.0, w=w)
    spot = Spot(id="p", image=_gradient(w, H * 4), depth=_depth_for(r).astype(np.float32), heading=0.0, scale=1.0)
    lim = walk_limits(spot.depth)
    assert MIN_WALK <= limit_at(lim, 90.0) <= WALL_FRACTION * 6.0 + 1e-6
    assert limit_at(lim, 0.0) == pytest.approx(WALK_LIMIT)
    assert limit_at(lim, -90.0) == pytest.approx(WALK_LIMIT)


def test_walk_limits_keep_the_dolly_representable():
    # Scenery only 6 away all round: no z within WALK_Z_MAX walks further than 6·(1 − 1/WALK_Z_MAX).
    from vecworld.config import WALK_Z_MAX
    from vecworld.roam.spot import walk_limits

    lim = walk_limits(_depth_for(np.full((H * 4, W * 4), 6.0)))
    assert np.all(lim <= 6.0 * (1 - 1 / WALK_Z_MAX) + 1e-6)


def test_limit_at_interpolates_round_the_circle():
    from vecworld.roam.spot import limit_at

    lim = np.zeros(36)
    lim[0], lim[1], lim[35] = 4.0, 6.0, 2.0                 # bins every 10 degrees, 0 = forward
    assert limit_at(lim, 0.0) == pytest.approx(4.0)
    assert limit_at(lim, 5.0) == pytest.approx(5.0)
    assert limit_at(lim, -5.0) == pytest.approx(3.0)
    assert limit_at(lim, 355.0) == pytest.approx(3.0)


def test_spots_carry_their_walk_limits():
    from vecworld.roam.spot import walk_limits

    first = first_spot(FakePainter(), FakeDepth(), [("a", 1.0)], seed=1)
    assert np.allclose(first.limits, walk_limits(first.depth))
    child = next_spot(FakePainter(), FakeDepth(), first, 30.0, [("b", 1.0)], None, 1, None)
    assert np.allclose(child.limits, walk_limits(child.depth))


def test_next_spot_without_walked_expects_the_exit_at_the_parents_limit():
    from vecworld.roam.spot import limit_at, walk_limits

    parent = _spot(12.0)
    child = next_spot(FakePainter(), FakeDepth(), parent, 40.0, [("b", 1.0)], None, 1, None)
    assert child.distance == pytest.approx(limit_at(walk_limits(parent.depth), 40.0), rel=1e-3)


def test_next_spot_stops_before_depth_when_the_queue_asks():
    import threading

    from vecworld.dream import gpu

    class Depth:
        calls = 0

        def depth360(self, pano):
            Depth.calls += 1
            return FakeDepth().depth360(pano)

    stop = threading.Event()

    class StoppingPainter(FakePainter):
        def step(self, *a, **kw):
            stop.set()                               # asked to stop while painting
            return super().step(*a, **kw)

    gpu._job.stop = stop
    try:
        with pytest.raises(gpu.Stopped):
            next_spot(StoppingPainter(), Depth(), _spot(12.0), 0.0, [("b", 1.0)], None, 1, None)
    finally:
        gpu._job.stop = None
    assert Depth.calls == 0


def test_focus_band_widens_with_the_dolly():
    from vecworld.config import FOCUS_HALF, FOCUS_HALF_MAX
    from vecworld.roam.spot import focus_half

    assert focus_half(1.15) == FOCUS_HALF                       # a small step regenerates the usual band
    assert FOCUS_HALF < focus_half(1.6) < focus_half(2.0) < focus_half(2.5) <= FOCUS_HALF_MAX
    painter = FakePainter()
    next_spot(painter, FakeDepth(), _spot(12.0), 0.0, [("b", 1.0)], None, 1, None, walked=6.0)   # z = 2
    assert painter.calls[0][1]["focus_half"] == pytest.approx(focus_half(2.0))
