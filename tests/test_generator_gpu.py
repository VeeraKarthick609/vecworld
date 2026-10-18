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
    a = g.fresh(scene_prompt("ocean", ["ship", "storm"]), seed=1)
    t_fresh = time.time() - t
    t = time.time()
    b = g.evolve(zoom_crop(a, .5, .5), scene_prompt("ship", ["harbor"]), seed=1)
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
    from vecworld.config import ZOOM_FRAC
    from vecworld.scenes.generator import TurboGenerator

    g = TurboGenerator()
    img = g.fresh(scene_prompt("ocean", ["sea", "river", "sky"]), seed=1)
    start = _sharpness(img)
    clicks = [(.3, .6), (.7, .4), (.5, .7), (.25, .35), (.75, .65), (.4, .3)]
    concepts = ["moonlight", "blizzard", "moonlight", "blackout", "twilight", "dusk"]
    for (cx, cy), c in zip(clicks, concepts):
        img = g.evolve(zoom_crop(img, cx, cy, ZOOM_FRAC), scene_prompt(c, ["sunset"]), seed=1)
    ratio = _sharpness(img) / start
    print(f"\nsharpness after 6 steps: {ratio:.2f}x of the first scene")
    assert 0.4 < ratio < 2.5
