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


def periodic(w=512, h=64, seed=0):
    """A smooth image that wraps around horizontally (no seam of its own)."""
    rng = np.random.default_rng(seed)
    x = np.arange(w) / w
    a = np.zeros((h, w, 3))
    for k in (1, 3, 7, 13):
        a += rng.uniform(10, 25) * np.sin(2 * np.pi * (k * x[None, :, None] + rng.uniform(0, 1, (h, 1, 3))))
    return Image.fromarray(np.clip(128 + a, 0, 255).astype(np.uint8))


def test_dolly_leaves_no_seam_behind_you():
    out = np.asarray(dolly(periodic(), 1.15), float)
    seam = np.abs(out[:, 0] - out[:, -1]).mean()
    inner = np.mean([np.abs(out[:, x] - out[:, x - 1]).mean() for x in (64, 128, 384, 448)])
    assert seam < 1.5 * inner + 1, (seam, inner)


def test_dolly_magnifies_ahead_and_shrinks_behind():
    img = Image.new("RGB", (400, 40), "black")
    img.paste((255, 0, 0), (190, 0, 210, 40))           # 20 px ahead
    img.paste((0, 255, 0), (0, 0, 10, 40)); img.paste((0, 255, 0), (390, 0, 400, 40))   # 20 px behind
    out = np.asarray(dolly(img, 1.15))
    red = (out[20, :, 0] > 128).sum()
    green = (out[20, :, 1] > 128).sum()
    assert 21 <= red <= 25 and 15 <= green <= 19, (red, green)


def test_dolly_keeps_shapes_behind_you():
    """Repeated steps must not stretch what is behind you into vertical bars."""
    img = Image.new("RGB", (512, 128), "black")
    img.paste((255, 255, 255), (0, 54, 10, 74)); img.paste((255, 255, 255), (502, 54, 512, 74))   # 20 x 20 behind
    for _ in range(8):
        img = dolly(img, 1.15)
    m = np.asarray(img)[..., 0] > 128
    cols, rows = np.nonzero(m.any(0))[0], np.nonzero(m.any(1))[0]
    w = min(len(cols), 512 - len(cols)) if len(cols) > 256 else len(cols)
    assert 0.7 < len(rows) / w < 1.4, (len(rows), w)


def test_wrap_dist():
    assert abs(wrap_dist(0.98, 0.02) - 0.04) < 1e-9
