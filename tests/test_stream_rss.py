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


# The painter from the SSD (Task 10b): its peak and settled RSS, its page-locked bytes, and the GGUF's own pages in the
# page cache (mincore) before and after two passes, which read the whole 6.66 GB of blocks from the file twice: the
# file is evicted first (posix_fadvise DONTNEED), so a reader that went through the page cache would leave it there.
SSD_RSS = """
import importlib.util, sys
from pathlib import Path
import os
import torch
from vecworld.config import FLUX_GGUF
from vecworld.engine.uring import resident_bytes
from vecworld.flux.painter import FluxPainter
def evict(path):
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def cached():
    return [int(x.split()[1]) for x in open("/proc/meminfo") if x.startswith("Cached:")][0] * 1024
def status(k):
    return [int(x.split()[1]) for x in open("/proc/self/status") if x.startswith(k + ":")][0] * 1024
spec = importlib.util.spec_from_file_location("bench_flux", sys.argv[1])
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
p = FluxPainter()
hwm, rss = status("VmHWM"), status("VmRSS")
with torch.inference_mode():
    e = p._embed([("a quiet lake at dawn", 1.0)])
    gguf = os.path.realpath(FLUX_GGUF)
    evict(gguf)
    c0, r0 = cached(), resident_bytes(gguf)
    bench.run_tile512(p, e, 2)
    torch.cuda.synchronize()
    c1, r1 = cached(), resident_bytes(gguf)
print("RESULT", p.weights_source, hwm, rss, status("VmRSS"), p._slots.pinned_bytes, c0, c1, r0, r1, flush=True)
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
    if mode == "ssd":
        from pathlib import Path

        env["FLUX_WEIGHTS_SOURCE"] = "ssd"
        args = [SSD_RSS, str(Path(__file__).parents[1] / "scripts/bench_flux.py")]
    else:
        env["FLUX_WEIGHTS_SOURCE"] = "ram"                  # the arena (the default is the SSD since Task 10b)
        args = [PAINTER_RSS, mode]
    out = subprocess.run([sys.executable, "-c", *args], capture_output=True, text=True, env=env, timeout=900)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT")]
    assert line, out.stderr[-2000:]
    if mode == "ssd":
        return line[0].split()[1:]
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


def test_painter_from_ssd_holds_no_weights():
    """From the SSD (FLUX_WEIGHTS_SOURCE=ssd, Task 10b) the painter never reads the blocks' 6.6 GB of Q4_K weights into
    RAM: its load peaks under 4 GB and it settles under 3 GB (the arena path: 7.87 / 7.52 GB), page-locking the staging
    ring and the resident biases only (< 1 GB). A pass reads every block from the file with O_DIRECT, which doesn't go
    through the page cache: with the GGUF evicted first, its own resident pages (mincore) stay under 64 MB over two
    passes (13.3 GB read). Red check: FLUX_SSD_BUFFERED_TEST=1 (the reader without O_DIRECT) leaves ~6.6 GB of the file
    cached and fails here."""
    src, hwm, rss, rss_pass, pinned, c0, c1, r0, r1 = _painter_rss("ssd")
    hwm, rss, rss_pass, pinned, c0, c1, r0, r1 = map(int, (hwm, rss, rss_pass, pinned, c0, c1, r0, r1))
    print(f"\nSSD painter: peak RSS {hwm / 1e9:.2f} GB, after setup {rss / 1e9:.2f} GB, after 2 passes "
          f"{rss_pass / 1e9:.2f} GB, pinned {pinned / 1e9:.3f} GB, Cached {c0 / 1e9:.2f} -> {c1 / 1e9:.2f} GB, "
          f"GGUF resident {r0 / 1e6:.1f} -> {r1 / 1e6:.1f} MB")
    assert src == "ssd"
    assert hwm < 4e9 and rss < 3e9 and rss_pass < 3e9
    assert 0 < pinned < 1e9
    assert r0 < 64e6 and r1 < 64e6
