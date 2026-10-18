import numpy as np

from vecworld.config import CENTRE_BOX, N_NEIGHBOURS
from vecworld.scenes.hotspots import Hotspot, choose, locate


def test_box_from_blob():
    m = np.zeros((1, 100, 100))
    m[0, 20:40, 60:80] = 1.0
    (h,) = locate(["ship"], m)
    assert h.concept == "ship"
    assert (round(h.x, 2), round(h.y, 2), round(h.w, 2), round(h.h, 2)) == (0.6, 0.2, 0.2, 0.2)


def test_box_fields_are_plain_floats():
    m = np.zeros((1, 10, 10))
    m[0, 2:5, 2:5] = 1.0
    (h,) = locate(["ship"], m)
    assert all(type(v) is float for v in (h.x, h.y, h.w, h.h))


def test_box_is_the_peak_blob_not_every_bright_pixel():
    m = np.zeros((1, 100, 100))
    m[0, 10:20, 10:20] = 1.0     # the object
    m[0, 80:90, 80:90] = 0.8     # a far-away echo above half the peak
    (h,) = locate(["ship"], m)
    assert (round(h.x, 2), round(h.y, 2), round(h.w, 2), round(h.h, 2)) == (0.1, 0.1, 0.1, 0.1)


def test_weak_map_is_not_located():
    assert locate(["ghost", "void"], np.full((2, 50, 50), 0.05)) == [None, None]


def box(c, x, y, w, h):
    return Hotspot(c, x, y, w, h)


def test_centre_always_kept_and_gets_centre_box_when_missing_or_huge():
    for centre in (None, box("kitchen", 0, 0, 1, 0.98)):
        (h,) = choose("kitchen", [centre], N_NEIGHBOURS)
        assert h.concept == "kitchen" and (h.x, h.y, h.w, h.h) == CENTRE_BOX


def test_choose_keeps_located_centre_box():
    c = box("ship", 0.4, 0.4, 0.2, 0.2)
    assert choose("ship", [c], 3) == [c]


def test_choose_drops_missing_huge_and_duplicate_neighbours_in_meaning_order():
    found = [
        box("bicycle", 0.3, 0.5, 0.4, 0.3),
        None,                                   # tiger: not in the picture
        box("bathroom", 0.0, 0.0, 1.0, 0.97),   # the whole room, not an object
        box("bike", 0.31, 0.5, 0.4, 0.3),       # same box as the bicycle
        box("tree", 0.0, 0.1, 0.2, 0.4),
        box("lamp", 0.8, 0.1, 0.1, 0.2),
        box("dog", 0.7, 0.7, 0.2, 0.2),
        box("cat", 0.1, 0.8, 0.1, 0.1),
    ]
    hs = choose("bicycle", found, 3)
    assert [h.concept for h in hs] == ["bicycle", "tree", "lamp", "dog"]


def test_nested_small_box_is_allowed():
    big, small = box("tree", 0.0, 0.0, 0.6, 0.6), box("bird", 0.2, 0.2, 0.1, 0.1)
    assert [h.concept for h in choose("forest", [box("forest", 0.4, 0.4, 0.2, 0.2), big, small], 3)] == ["forest", "tree", "bird"]


def test_all_boxes_in_bounds():
    rng = np.random.default_rng(0)
    m = rng.random((7, 40, 40))
    for h in choose("c0", locate([f"c{i}" for i in range(7)], m), 3):
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 + 1e-9 and h.y + h.h <= 1 + 1e-9
