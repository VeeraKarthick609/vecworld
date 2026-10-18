import time

import numpy as np
import pytest

from vecworld.config import STYLE
from vecworld.dream.pano import dolly


@pytest.fixture(scope="module")
def painter():
    import torch

    from vecworld.dream.painter import TurboPainter

    p = TurboPainter()
    p.step(dolly(p.panorama([("warm up", 1.0)], seed=0)), [("warm up", 1.0)], None, 0.5, 0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return p


@pytest.mark.gpu
def test_panorama_is_seamless_and_fits(painter):
    import torch

    t = time.time()
    img = painter.panorama([(f"a haunted forest, {STYLE}", 1.0)], seed=7)
    took = time.time() - t
    img.save("/tmp/dw_pano.png")
    a = np.asarray(img, float)
    seam = np.abs(a[:, -1] - a[:, 0]).mean()
    inner = np.abs(a[:, 1023] - a[:, 1024]).mean()
    print(f"\npanorama {took:.2f}s seam {seam:.1f} inner {inner:.1f} "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    assert img.size == (2048, 512) and seam < 2.5 * inner + 2
    assert torch.cuda.max_memory_allocated() < 3.6 * 2**30


@pytest.mark.gpu
def test_step_times_and_keeps_background(painter):
    import torch

    base = [(f"a haunted forest, {STYLE}", 1.0)]
    prev = dolly(painter.panorama(base, 7))
    t = time.time()
    out = painter.step(prev, base, [(f"a haunted forest, an old lantern on a post ahead in the distance, {STYLE}", 1.0)], 0.5, 8)
    took = time.time() - t
    prev.save("/tmp/dw_prev.png")
    out.save("/tmp/dw_step.png")
    a, b = np.asarray(prev, float), np.asarray(out, float)
    side = np.abs(a[:, :400] - b[:, :400]).mean()
    ahead = np.abs(a[:, 900:1150] - b[:, 900:1150]).mean()
    print(f"\nstep {took:.2f}s side-change {side:.1f} ahead-change {ahead:.1f} "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    assert took < 6.0 and ahead > 1.5 * side


@pytest.mark.gpu
def test_step_without_focus_changes_gently(painter):
    base = [(f"a haunted forest, {STYLE}", 1.0)]
    prev = dolly(painter.panorama(base, 7))
    out = painter.step(prev, base, None, None, 9)
    out.save("/tmp/dw_quiet.png")
    diff = np.abs(np.asarray(prev, float) - np.asarray(out, float)).mean()
    print(f"\nwhole-frame img2img change {diff:.1f}")
    assert diff < 40
