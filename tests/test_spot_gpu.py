import json
import time
from pathlib import Path

import numpy as np
import pytest

from vecworld.config import STYLE

OUT = Path("/tmp/roam_t8b")      # the walk at the 8b spacing (the spots before it: /tmp/roam_t2_before8b)


def _sheet(path, spots):
    from PIL import Image, ImageDraw

    w, h, pad = 1024, 256, 18
    sheet = Image.new("RGB", (w, (h + pad) * len(spots)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, s in enumerate(spots):
        y = i * (h + pad)
        draw.text((4, y + 3), f"spot {i}  heading {s.heading:.0f}  distance {s.distance:.2f}  scale {s.scale:.2f}",
                  fill="black")
        sheet.paste(s.image.resize((w, h)), (0, y + pad))
    sheet.save(path)


def _save(dirs, i, spot):
    from vecworld.roam.depth import depth_png

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        spot.image.save(d / f"spot_{i}.jpg", quality=92)
        (d / f"depth_{i}.png").write_bytes(depth_png(spot.depth))
        (d / f"meta_{i}.json").write_text(json.dumps(
            {"heading": spot.heading, "scale": spot.scale, "distance": spot.distance,
             "limits": [round(float(x), 4) for x in spot.limits]}))


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
        _save([tmp_path], i, s)
    print("\nspot times " + " ".join(f"{x:.2f}s" for x in times) + f"  peak VRAM {peak:.2f} GiB")
    print("meta " + " | ".join(f"h={s.heading:.0f} s={s.scale:.3f} d={s.distance:.2f}" for s in spots))
    for s in spots:
        assert s.image.size == (2048, 512) and s.depth.shape == (512, 2048) and np.isfinite(s.depth).all()
        assert np.isfinite(s.scale) and s.scale > 0
    assert all(x < 6.5 for x in times[1:])
    assert peak < 3.6


@pytest.mark.gpu
def test_six_spot_walk_at_the_walk_limit():
    """Heading 0, 0, 60, 60, -60, 0, each spot built for an exit at its parent's walk limit (as a pre-dream is):
    the distances are what was walked, and the frames go to /tmp/roam_t8b (sheet.png) to be looked at."""
    import torch

    from vecworld.config import WALK_Z_MAX
    from vecworld.dream.painter import TurboPainter
    from vecworld.roam.depth import AnythingDepth
    from vecworld.roam.spot import _dolly_for, first_spot, limit_at, next_spot, wrap180
    from vecworld.scenes.palette import palette_stats

    painter, depth = TurboPainter(), AnythingDepth("cuda")
    base = [(f"a dirt path through a sunlit old forest clearing, open ground in front, big trees, ferns, {STYLE}", 1.0)]
    first_spot(painter, depth, base, seed=1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    spots = [first_spot(painter, depth, base, seed=21)]
    ref = palette_stats(spots[0].image)
    times, lines = [], []
    for i, h in enumerate((0.0, 0.0, 60.0, 60.0, -60.0, 0.0)):
        parent = spots[-1]
        rel = wrap180(h - parent.heading)
        walked = limit_at(parent.limits, rel)
        z, r = _dolly_for(parent, rel, walked)
        t = time.time()
        spots.append(next_spot(painter, depth, parent, h, base, None, seed=22 + i, ref=ref))
        torch.cuda.synchronize()
        times.append(time.time() - t)
        s = spots[-1]
        lines.append(f"h={h:.0f} walked={walked:.2f} R={r:.1f} z={z:.2f} d={s.distance:.2f} s={s.scale:.2f} "
                     f"{times[-1]:.2f}s limits mean {s.limits.mean():.2f} min {s.limits.min():.2f}")
        if z < WALK_Z_MAX - 1e-3:
            assert s.distance == pytest.approx(walked, rel=0.02)
    peak = torch.cuda.max_memory_allocated() / 2**30
    for i, s in enumerate(spots):
        _save([OUT], i, s)
    _sheet(OUT / "sheet.png", spots)
    print("\n" + "\n".join(lines) + f"\npeak VRAM {peak:.2f} GiB")
    assert all(np.isfinite(s.scale) and s.scale > 0 for s in spots)
    assert all(x < 6.5 for x in times)
    assert peak < 3.6
