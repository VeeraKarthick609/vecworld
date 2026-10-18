import time

import pytest

from vecworld.scenes.prompt import scene_prompt, zoom_crop


@pytest.mark.gpu
def test_turbo_fresh_and_evolve_fit_in_vram():
    import torch

    from vecworld.scenes.generator import TurboGenerator

    g = TurboGenerator()
    g.fresh("warm up", seed=0)
    t = time.time()
    a = g.fresh(scene_prompt("ocean", ["ship", "storm"], True), seed=1)
    t_fresh = time.time() - t
    t = time.time()
    b = g.evolve(zoom_crop(a, .5, .5), scene_prompt("ship", ["harbor"], True), seed=1)
    t_evolve = time.time() - t
    print(f"\nfresh {t_fresh:.2f}s  evolve {t_evolve:.2f}s  peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    a.save("/tmp/vw_fresh.png")
    b.save("/tmp/vw_evolve.png")
    assert a.size == b.size == (512, 512)
    assert torch.cuda.max_memory_allocated() < 3.6 * 2**30


def _sharpness(im):
    import numpy as np

    a = np.asarray(im.convert("L"), float)
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


@pytest.mark.gpu
def test_six_steps_neither_dissolve_nor_fry():
    """The game's real step path: img2img, then the palette pull toward the jump scene."""
    from vecworld.config import PALETTE_STRENGTH, ZOOM_FRAC
    from vecworld.scenes.generator import TurboGenerator
    from vecworld.scenes.palette import match_palette, palette_stats

    g = TurboGenerator()
    img = g.fresh(scene_prompt("ocean", ["sea", "river", "sky"], True), seed=1)
    ref = palette_stats(img)
    start = _sharpness(img)
    clicks = [(.3, .6), (.7, .4), (.5, .7), (.25, .35), (.75, .65), (.4, .3)]
    concepts = ["moonlight", "blizzard", "moonlight", "blackout", "twilight", "dusk"]
    for (cx, cy), c in zip(clicks, concepts):
        img = match_palette(g.evolve(zoom_crop(img, cx, cy, ZOOM_FRAC), scene_prompt(c, ["sunset"]), seed=1),
                            ref, PALETTE_STRENGTH)
    ratio = _sharpness(img) / start
    print(f"\nsharpness after 6 steps: {ratio:.2f}x of the first scene")
    assert 0.4 < ratio < 2.5


def _lab_drift_over_walk(g, strength):
    import numpy as np

    from vecworld.config import ZOOM_FRAC
    from vecworld.scenes.palette import match_palette, palette_stats

    img = g.fresh(scene_prompt("ocean", ["sea", "river", "sky"], True), seed=1)
    ref = palette_stats(img)
    clicks = [(.3, .6), (.7, .4), (.5, .7), (.25, .35), (.75, .65), (.4, .3)]
    concepts = ["moonlight", "blizzard", "moonlight", "blackout", "twilight", "dusk"]
    for (cx, cy), c in zip(clicks, concepts):
        img = match_palette(g.evolve(zoom_crop(img, cx, cy, ZOOM_FRAC), scene_prompt(c, ["sunset"]), seed=1),
                            ref, strength)
    return float(np.linalg.norm(palette_stats(img)[0] - ref[0])), float(np.abs(palette_stats(img)[1] - ref[1]).max())


@pytest.mark.gpu
def test_palette_correction_stops_colour_drift():
    from vecworld.config import PALETTE_STRENGTH
    from vecworld.scenes.generator import TurboGenerator

    g = TurboGenerator()
    raw, raw_spread = _lab_drift_over_walk(g, 0.0)
    fixed, fixed_spread = _lab_drift_over_walk(g, PALETTE_STRENGTH)
    print(f"\nLAB drift after 6 steps: {raw:.1f} raw -> {fixed:.1f} corrected; "
          f"worst std change {raw_spread:.1f} -> {fixed_spread:.1f}")
    assert fixed < 0.5 * raw and fixed_spread < 0.6 * raw_spread
