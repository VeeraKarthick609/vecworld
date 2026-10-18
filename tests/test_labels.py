import numpy as np
from PIL import Image

from vecworld.dream.labels import Label, LabelFinder, carry, crop_at, to_label
from vecworld.dream.pano import wrap_dist
from vecworld.scenes.hotspots import Hotspot


def test_to_label_maps_crop_box_to_panorama():
    l = to_label("chapel", Hotspot("chapel", 0.25, 0.4, 0.5, 0.2), u_centre=0.5, pano_w=2048)
    assert abs(l.u - 0.5) < 1e-6 and abs(l.w - 0.125) < 1e-6 and l.y == 0.4 and l.h == 0.2


def test_to_label_wraps_u_into_unit_interval():
    l = to_label("x", Hotspot("x", 0.9, 0.1, 0.1, 0.1), u_centre=0.99, pano_w=2048)
    assert 0.0 <= l.u < 1.0 and wrap_dist(l.u, 0.99 + 0.45 * 0.25) < 1e-6


def test_crop_at_wraps_around_seam():
    img = Image.new("RGB", (2048, 512))
    img.putpixel((0, 0), (255, 0, 0))
    c = crop_at(img, 0.999)
    assert c.size == (512, 512) and c.getpixel((256 + 2, 0)) == (255, 0, 0)


def test_label_found_across_seam():
    def heat(img, texts):  # blob at the crop's centre
        m = np.zeros((len(texts), 352, 352)); m[:, 150:200, 150:200] = 1; return m
    (l,) = LabelFinder(heat).find(Image.new("RGB", (2048, 512)), [("lantern", 0.99)])
    assert wrap_dist(l.u, 0.99) < 0.01


def test_unlocated_discovery_gets_no_label():
    assert LabelFinder(lambda img, t: np.zeros((len(t), 352, 352))).find(Image.new("RGB", (2048, 512)), [("x", 0.5)]) == []


def test_find_keeps_input_order_and_skips_misses():
    def heat(img, texts):
        m = np.zeros((len(texts), 352, 352))
        if texts[0] != "miss":
            m[:, 150:200, 150:200] = 1
        return m
    ls = LabelFinder(heat).find(Image.new("RGB", (2048, 512)), [("a", 0.2), ("miss", 0.5), ("b", 0.7)])
    assert [l.text for l in ls] == ["a", "b"]


def test_carry_moves_labels_away_from_centre_and_drops_offscreen():
    ls = [Label("a", "lantern", 0.55, .4, .05, .1), Label("b", "pond", 0.9, .4, .05, .1)]
    assert [(t, round(u, 4)) for t, u in carry(ls, walk_u=0.5)] == [("lantern", round(0.5 + 0.05 * 1.15, 4))]
