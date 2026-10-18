import numpy as np
from PIL import Image

from vecworld.dream.pano import dolly, focus_mask, roll_to, windows, wrap_cols, wrap_dist, yaw_to_u


def test_windows_cover_every_column_twice_and_wrap():
    ws = windows(64, 256)
    assert {x for _, x in ws} == set(range(0, 256, 32)) and {y for y, _ in ws} == {0}
    cover = np.zeros(256, int)
    for _, x in ws:
        cover[wrap_cols(x, 64, 256)] += 1
    assert (cover == 2).all()


def test_windows_tall_rows_clamped():
    assert {y for y, _ in windows(96, 256)} == {0, 32}


def test_focus_mask_wraps_across_seam():
    m = focus_mask(1000, at=0.99)
    assert m[990] == 1 and m[30] == 1 and m[500] == 0
    assert 0 < m[int(1000 * (0.99 + 0.08 + 0.03)) % 1000] < 1


def test_roll_to_puts_u_at_centre():
    img = Image.new("RGB", (100, 10), "black"); img.putpixel((10, 5), (255, 0, 0))
    assert roll_to(img, 0.10).getpixel((50, 5)) == (255, 0, 0)


def test_yaw_to_u():
    assert yaw_to_u(0) == 0.5 and yaw_to_u(180) == 0.0 and abs(yaw_to_u(-90) - 0.25) < 1e-9


def test_dolly_keeps_size_and_enlarges_centre():
    img = Image.new("RGB", (200, 100), "black"); img.paste((255, 0, 0), (95, 45, 105, 55))
    out = dolly(img, 2.0)
    assert out.size == (200, 100) and out.getpixel((100, 50)) == (255, 0, 0) and out.getpixel((92, 50)) == (255, 0, 0) and out.getpixel((80, 50)) == (0, 0, 0)


def test_wrap_dist():
    assert abs(wrap_dist(0.98, 0.02) - 0.04) < 1e-9
