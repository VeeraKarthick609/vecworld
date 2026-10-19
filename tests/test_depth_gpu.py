import time

import numpy as np
import pytest

from vecworld.config import STYLE


@pytest.mark.gpu
def test_anything_depth_on_painted_pano_and_vram():
    import torch

    from vecworld.dream.painter import TurboPainter
    from vecworld.roam.depth import AnythingDepth, depth_png

    torch.cuda.reset_peak_memory_stats()
    painter = TurboPainter()
    pano = painter.panorama([(f"a dirt path through a sunlit old forest clearing, big trees, ferns, {STYLE}", 1.0)], seed=21)
    depth = AnythingDepth("cuda")
    depth.depth360(pano)                       # warm-up
    torch.cuda.synchronize()
    t = time.time()
    d = depth.depth360(pano)
    torch.cuda.synchronize()
    took = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2**30
    seam = float(np.abs(d[:, 0] - d[:, -1]).mean())
    typical = float(np.abs(np.diff(d, axis=1)).mean())
    print(f"\ndepth {took:.2f}s  seam {seam:.4f} typical {typical:.4f}  peak VRAM {peak:.2f} GiB")
    pano.save("/tmp/roam_pano.jpg", quality=92)
    open("/tmp/roam_depth.png", "wb").write(depth_png(d))
    assert d.shape == (pano.height, pano.width) and np.isfinite(d).all()
    assert took < 1.0
    assert seam <= 2 * typical
    assert peak < 3.6
