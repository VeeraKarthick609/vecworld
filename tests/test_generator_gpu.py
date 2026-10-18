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
