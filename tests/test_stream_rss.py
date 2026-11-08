"""FluxPainter packs the GGUF it read into the block arena without holding a second copy (Task 10a; gpu, game server
stopped). In a module of its own: it runs a second painter process, which on this 14 GB laptop must not overlap a
module that holds a transformer (test_stream's fixture)."""
import gc

import pytest

pytestmark = pytest.mark.gpu

# VmHWM, this process' own peak: getrusage's ru_maxrss carries the parent's peak over fork + exec (pytest's, after
# the modules before it loaded a transformer from a memory-mapped file).
PAINTER_RSS = """
import sys
import vecworld.engine.stream as stream
if sys.argv[1] == "notrim":
    stream._trim = lambda: None
from vecworld.flux.painter import FluxPainter
p = FluxPainter()
hwm = [int(x.split()[1]) for x in open("/proc/self/status") if x.startswith("VmHWM:")][0] * 1024
print("RESULT", hwm, p._arena.nbytes, flush=True)
p.close()
"""


def _painter_rss(mode="trim"):
    import os
    import subprocess
    import sys

    from vecworld.engine.stream import _trim

    import torch

    gc.collect()
    _trim()                         # hand the freed memory of earlier tests back first: two processes share 14 GB
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()    # and this process' cached GPU memory: the child needs ~1 GB of the 4 GB card
    env = {k: v for k, v in os.environ.items() if not k.startswith("VECWORLD_")}
    out = subprocess.run([sys.executable, "-c", PAINTER_RSS, mode], capture_output=True, text=True, env=env,
                         timeout=900)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT")]
    assert line, out.stderr[-2000:]
    peak, arena = map(int, line[0].split()[1:])
    return peak, arena


def test_painter_packs_the_arena_without_a_second_copy():
    """FluxPainter reads the GGUF into ordinary host memory and packs it into the arena block by block: the process'
    peak resident memory stays about one model (arena + ~1 GB of the rest), not two (13.8 GB, OOM-killed on this 14 GB
    laptop, when glibc kept the freed copies). Measured in a fresh process.
    (_painter_rss("notrim") is the red: the same load with the trim off, 9.10 GB.)"""
    peak, arena = _painter_rss()
    print(f"\npeak RSS {peak / 1e9:.2f} GB, arena {arena / 1e9:.3f} GB")
    assert arena > 6e9 and peak < arena + 2e9
