import numpy as np
from PIL import Image

from vecworld.scenes.palette import match_palette, palette_stats


def noisy(rgb, seed, spread=40, size=(64, 48)):
    rng = np.random.default_rng(seed)
    a = np.clip(np.array(rgb, float) + rng.normal(0, spread, (size[1], size[0], 3)), 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def test_stats_are_cielab():
    mean, std = palette_stats(Image.new("RGB", (8, 8), (255, 0, 0)))
    assert np.allclose(mean, [53.24, 80.09, 67.20], atol=0.5) and np.allclose(std, 0, atol=1e-3)
    mean, _ = palette_stats(Image.new("RGB", (8, 8), (128, 128, 128)))
    assert abs(mean[1]) < 0.5 and abs(mean[2]) < 0.5


def test_strength_zero_is_identity():
    img = noisy((200, 120, 60), 0)
    out = match_palette(img, palette_stats(noisy((40, 60, 160), 1)), 0.0)
    assert out.tobytes() == img.tobytes()


def test_full_strength_takes_on_the_reference_palette():
    img, ref = noisy((200, 120, 60), 0), noisy((90, 110, 150), 1, spread=25)
    out = match_palette(img, palette_stats(ref), 1.0)
    assert out.size == img.size and out.mode == "RGB"
    got, want = palette_stats(out), palette_stats(ref)
    assert np.allclose(got[0], want[0], atol=1.5) and np.allclose(got[1], want[1], atol=1.5)


def test_half_strength_goes_halfway():
    img, ref = noisy((200, 120, 60), 0), noisy((90, 110, 150), 1, spread=25)
    before, want = palette_stats(img)[0], palette_stats(ref)[0]
    got = palette_stats(match_palette(img, palette_stats(ref), 0.5))[0]
    assert np.allclose(got, (before + want) / 2, atol=1.5)
