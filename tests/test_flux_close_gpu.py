"""FluxPainter.close() gives back the host memory it page-locked (run on AC, game server stopped).

FluxPainter registers the transformer's CPU tensors with the driver (cudaHostRegister) where they lie. Freed while
still registered, those pages come back from the allocator as ordinary buffers that the driver still takes for a
pinned range of the old size, and a pageable copy from one fails ("CUDA error: invalid argument ... Copy is larger
than memobj size"), as SD-Turbo's encode did in GPU tests run after the FLUX ones.
"""
import gc

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_closed_painter_leaves_host_copies_working():
    import torch

    from vecworld.flux.painter import FluxPainter

    p = FluxPainter()
    assert p._pinned
    p.close()
    p.close()                                       # safe twice
    assert p._pinned == []
    del p
    gc.collect()
    # Turbo-style pageable copies (fresh numpy buffers of the sizes the freed weights had, and a panorama)
    for n in [512 * 2048 * 3] * 8 + [80 << 20, 191 << 20, 80 << 20] * 4:
        a = np.ones(n, np.uint8)
        x = torch.from_numpy(a).to("cuda")
        assert int(x[-1]) == 1
        del a, x
    torch.cuda.synchronize()
