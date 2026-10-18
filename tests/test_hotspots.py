import numpy as np

from vecworld.scenes.hotspots import locate


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


def test_all_boxes_in_bounds():
    rng = np.random.default_rng(0)
    m = rng.random((7, 40, 40))
    for h in locate([f"c{i}" for i in range(7)], m):
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 + 1e-9 and h.y + h.h <= 1 + 1e-9
