import numpy as np

from vecworld.scenes.hotspots import boxes_from_heatmaps


def test_box_from_blob():
    m = np.zeros((1, 100, 100))
    m[0, 20:40, 60:80] = 1.0
    (h,) = boxes_from_heatmaps(["ship"], m)
    assert h.concept == "ship"
    assert (round(h.x, 2), round(h.y, 2), round(h.w, 2), round(h.h, 2)) == (0.6, 0.2, 0.2, 0.2)


def test_box_fields_are_plain_floats():
    m = np.zeros((1, 10, 10))
    m[0, 2:5, 2:5] = 1.0
    (h,) = boxes_from_heatmaps(["ship"], m)
    assert all(type(v) is float for v in (h.x, h.y, h.w, h.h))


def test_weak_map_gets_fallback_inside_bounds():
    m = np.full((2, 50, 50), 0.05)
    hs = boxes_from_heatmaps(["ghost", "void"], m)
    assert [h.concept for h in hs] == ["ghost", "void"]
    assert (hs[0].x, hs[0].y) != (hs[1].x, hs[1].y)
    for h in hs:
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 and h.y + h.h <= 1


def test_more_weak_concepts_than_fallback_slots_still_in_bounds():
    hs = boxes_from_heatmaps([f"c{i}" for i in range(6)], np.zeros((6, 20, 20)))
    assert len(hs) == 6
    for h in hs:
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 and h.y + h.h <= 1
