import json
import time
from pathlib import Path

import numpy as np
import pytest

from vecworld.config import STYLE

OUT = Path("/tmp/roam_t2")


def _save(dirs, i, spot):
    from vecworld.roam.depth import depth_png

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        spot.image.save(d / f"spot_{i}.jpg", quality=92)
        (d / f"depth_{i}.png").write_bytes(depth_png(spot.depth))
        (d / f"meta_{i}.json").write_text(json.dumps(
            {"heading": spot.heading, "scale": spot.scale, "distance": spot.distance}))


@pytest.mark.gpu
def test_three_spots_in_a_row_timing(tmp_path):
    import torch

    from vecworld.dream.painter import TurboPainter
    from vecworld.roam.depth import AnythingDepth
    from vecworld.roam.spot import first_spot, next_spot
    from vecworld.scenes.palette import palette_stats

    painter, depth = TurboPainter(), AnythingDepth("cuda")
    base = [(f"a dirt path through a sunlit old forest clearing, open ground in front, big trees, ferns, {STYLE}", 1.0)]
    focus = [(f"a dirt path through a sunlit old forest clearing, a mossy stone well ahead, {STYLE}", 1.0)]
    first_spot(painter, depth, base, seed=1)     # warm-up (model load, cudnn autotune)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t = time.time()
    spots = [first_spot(painter, depth, base, seed=21)]
    torch.cuda.synchronize()
    times = [time.time() - t]
    ref = palette_stats(spots[0].image)
    for i, exit_heading in enumerate((0.0, 60.0)):
        t = time.time()
        spots.append(next_spot(painter, depth, spots[-1], exit_heading, base, focus, seed=22 + i, ref=ref))
        torch.cuda.synchronize()
        times.append(time.time() - t)
    peak = torch.cuda.max_memory_allocated() / 2**30
    for i, s in enumerate(spots):
        _save([tmp_path, OUT], i, s)
    print("\nspot times " + " ".join(f"{x:.2f}s" for x in times) + f"  peak VRAM {peak:.2f} GiB")
    print("meta " + " | ".join(f"h={s.heading:.0f} s={s.scale:.3f} d={s.distance:.2f}" for s in spots))
    for s in spots:
        assert s.image.size == (2048, 512) and s.depth.shape == (512, 2048) and np.isfinite(s.depth).all()
        assert np.isfinite(s.scale) and s.scale > 0
    assert all(x < 6.5 for x in times[1:])
    assert peak < 3.6
