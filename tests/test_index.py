import numpy as np

from conftest import make_space
from vecworld.space.index import angle


def test_neighbours_sorted_and_excludes():
    s = make_space()
    got = s.neighbours(s.vector("ocean"), k=2, exclude={"ocean"})
    assert [w for w, _ in got] == ["sea", "ship"]
    assert got[0][1] >= got[1][1]


def test_step_is_capped():
    s = make_space()
    p = s.step_toward(s.vector("ocean"), s.vector("desert"), max_angle=0.35)
    assert abs(angle(s.vector("ocean"), p) - 0.35) < 1e-4
    assert abs(np.linalg.norm(p) - 1) < 1e-5


def test_step_reaches_close_target():
    s = make_space()
    p = s.step_toward(s.vector("ocean"), s.vector("sea"), max_angle=0.35)
    assert angle(p, s.vector("sea")) < 1e-4


def test_step_toward_self_is_noop():
    s = make_space()
    p = s.step_toward(s.vector("ocean"), s.vector("ocean"), 0.35)
    assert angle(p, s.vector("ocean")) < 1e-5


def test_project_in_range():
    s = make_space()
    for w in s.words:
        x, y = s.project(s.vector(w))
        assert -1.0001 <= x <= 1.0001 and -1.0001 <= y <= 1.0001


def test_walk_moves_exactly_along_one_great_circle():
    s = make_space()
    p = s.vector("ocean")
    h = s.heading(s.vector("sea"), p)
    p1, h1 = s.walk(p, h, 0.2)
    p2, _ = s.walk(p1, h1, 0.2)
    q, _ = s.walk(p, h, 0.4)
    assert abs(np.linalg.norm(p1) - 1) < 1e-5 and abs(np.linalg.norm(h1) - 1) < 1e-5
    assert abs(angle(p, p1) - 0.2) < 1e-4
    assert angle(p2, q) < 1e-4
    assert abs(float(np.dot(p1, h1))) < 1e-5          # heading stays tangent


def test_heading_points_away_from_where_you_came_from():
    s = make_space()
    a, b = s.vector("ocean"), s.vector("ship")
    h = s.heading(a, b)
    assert abs(float(np.dot(h, b))) < 1e-5 and abs(np.linalg.norm(h) - 1) < 1e-5
    further, _ = s.walk(b, h, 0.1)
    assert angle(a, further) > angle(a, b) + 0.09


def test_no_heading_between_identical_points():
    s = make_space()
    assert s.heading(s.vector("ocean"), s.vector("ocean")) is None
