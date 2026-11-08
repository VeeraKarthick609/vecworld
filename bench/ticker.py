"""Task 10a fix round, I3 (committed at 10a-g so the explainer's numbers can be reproduced; run it with the game server
stopped, from the repo: `VECWORLD_GRAPHS=1 uv run python bench/ticker.py` (graphs), then without VECWORLD_GRAPHS (eager, the default)):
 how late does a 1 ms ticker thread wake while the painter paints? One process per run.
A painter panorama (4 batch-8 passes + VAE) is painted once to warm up (graphs capture there), then once with the
ticker running; the ticker sleeps 1 ms in a loop and records how late each wake-up was (a stall of the GIL holder
shows up as one long wake-up). Prints RESULT json."""
import json, os, sys, threading, time
import numpy as np
import torch
from vecworld.config import FLUX_STYLE
from vecworld.flux.painter import FluxPainter

p = FluxPainter()
prompt = [("a dirt path through a sunlit old forest clearing, " + FLUX_STYLE, 1.0)]
p.panorama(prompt, seed=1)
torch.cuda.synchronize()


def tick(stop, out):
    while not stop.is_set():
        t = time.perf_counter()
        time.sleep(0.001)
        out.append(time.perf_counter() - t - 0.001)


def measure(job):
    stop, lat = threading.Event(), []
    th = threading.Thread(target=tick, args=(stop, lat), daemon=True)
    th.start()
    t = time.perf_counter()
    job()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    stop.set(); th.join()
    a = np.array(lat) * 1e3
    return dict(wall_s=round(wall, 2), ticks=len(a), max_ms=round(float(a.max()), 2),
                p99_ms=round(float(np.percentile(a, 99)), 3), p999_ms=round(float(np.percentile(a, 99.9)), 2),
                over_10ms=int((a > 10).sum()), over_50ms=int((a > 50).sum()))


out = dict(graphs=p.graphs, env=os.environ.get("VECWORLD_GRAPHS", ""), git=os.popen("git rev-parse --short HEAD").read().strip(),
           src=os.environ.get("PYTHONPATH", "tree"))
out["idle"] = measure(lambda: time.sleep(5))
out["panorama"] = measure(lambda: p.panorama(prompt, seed=2))
print("RESULT", json.dumps(out), flush=True)
p.close()
