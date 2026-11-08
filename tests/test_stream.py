"""The layer streamer: FLUX blocks streamed from one page-locked host arena through a ring of device slots (Task 10a,
docs/engine/10-streamer.md; gpu, run with the game server stopped).

BlockStreamer builds on Task 7's BlockSlots: the weights of block i are one contiguous run of bytes, source(i), laid
out exactly as in its slot, so a block is one copy; the copies and the events that order them are issued by the
extension's SlotRing while holding the GIL. Streamed blocks must compute exactly what the same blocks compute with
their weights resident on the GPU (torch.equal).

The real-model tests share one transformer (module fixture `fx`), streamed as the painter streams it
(engine.graphs.stream_blocks(pack=True)).
"""
import gc
import statistics
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

H = 3072


# --- small blocks -----------------------------------------------------------------------------------------------------

def test_arena_packs_blocks_in_their_slot_layout():
    """HostArena moves every block's tensors into one page-locked buffer, block i a contiguous run laid out as its slot
    (engine.slots.block_layout): the blocks' tensors become views of it with the same values."""
    from vecworld.engine.ext import load_ext
    from vecworld.engine.slots import block_layout
    from vecworld.engine.stream import HostArena

    from test_graphs import _toys

    blocks, host = _toys(4)
    a = HostArena(blocks)
    base = a.buf.data_ptr()
    assert load_ext().host_pinned(base) and a.nbytes == a.buf.numel()
    for i, b in enumerate(blocks):
        lay, size = block_layout(b)
        blk = a.block(i)
        assert blk.numel() == size and blk.dtype == torch.uint8
        for name, t in (*b.named_parameters(), *b.named_buffers()):
            assert t.device.type == "cpu" and t.data_ptr() == blk.data_ptr() + lay[name], name
            assert torch.equal(t, host[i][name]), name
    a.close()
    a.close()                                                   # safe twice


def test_streamer_source_contract():
    """source(i) gives block i's bytes as a tensor or anything with the buffer protocol, page-locked and exactly the
    block's size; a wrong size or pageable memory is refused loudly when the block's copy is issued."""
    from vecworld.engine.stream import BlockStreamer, HostArena

    from test_graphs import Toy, _toys

    blocks, _ = _toys(3)
    ref = [Toy(i).cuda() for i in range(3)]
    a = HostArena(blocks)
    x = torch.randn(8, 64, device="cuda").bfloat16()
    for src in (a.block, lambda i: memoryview(a.block(i).numpy())):
        s = BlockStreamer(blocks, src, ring=2)
        s.install()
        with torch.inference_mode():
            for i in (0, 1, 2, 2, 0):
                assert torch.equal(blocks[i](x), ref[i](x)), i
        s.close()
    for bad, msg in ((lambda i: a.block(i)[:-256], "bytes"),
                     (lambda i: torch.zeros(a.block(i).numel(), dtype=torch.uint8), "page-locked")):
        s = BlockStreamer(blocks, bad, ring=2)
        s.install()
        with torch.inference_mode(), pytest.raises(ValueError, match=msg):
            blocks[0](x)
        s.close()
    a.close()


def test_streamer_close_then_install_again():
    """close() puts the arena views back in the blocks and frees the slots; install() streams them again, correctly."""
    from vecworld.engine.stream import BlockStreamer, HostArena

    from test_graphs import Toy, _toys

    blocks, _ = _toys(5)
    ref = [Toy(i).cuda() for i in range(5)]
    a = HostArena(blocks)
    s = BlockStreamer(blocks, a.block, ring=3)
    x = torch.randn(8, 64, device="cuda").bfloat16()
    for _ in range(2):
        s.install()
        with torch.inference_mode():
            for i in [*range(5), *range(5)]:
                assert torch.equal(blocks[i](x), ref[i](x)), i
        s.close()
        assert all(t.device.type == "cpu" for b in blocks for t in b.parameters())
        assert s.slots == []
    a.close()


def test_slot_hooks_keep_the_gil():
    """The per-block hooks issue their copies and events through the extension without releasing the GIL: with two
    Python threads spinning, a block's before + after costs about what it costs alone (a released and re-taken GIL
    waits for a spinner's 5 ms switch interval)."""
    import time

    from vecworld.engine.stream import BlockStreamer, HostArena

    from test_graphs import _toys

    blocks, _ = _toys(6)
    a = HostArena(blocks)
    s = BlockStreamer(blocks, a.block, ring=3)
    s.install()

    def per_block(n=60):
        t = time.perf_counter()
        for k in range(n):
            i = k % len(blocks)
            s.before(i)
            s.after(i)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    per_block()
    alone = per_block()
    stop = [False]

    def spin():
        while not stop[0]:
            pass

    th = [threading.Thread(target=spin, daemon=True) for _ in range(2)]
    for t in th:
        t.start()
    try:
        busy = per_block()
    finally:
        stop[0] = True
        for t in th:
            t.join()
    print(f"\nper block before + after: alone {alone * 1e6:.0f} us, 2 spinning threads {busy * 1e6:.0f} us")
    assert busy < 1e-3                                       # a single GIL hand-off would be ~5 ms
    s.close()
    a.close()


# --- the real model ---------------------------------------------------------------------------------------------------

def _trim():
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


@pytest.fixture(scope="module")
def fx():
    """The transformer as the painter runs it under the default flags (w4a8 + keep-list, fused glue, rowwise), its
    blocks packed into a host arena and streamed through FLUX_SLOT_RING slots (stream_blocks(pack=True)), eager."""
    from test_graphs import _bench
    from vecworld.engine.graphs import stream_blocks
    from vecworld.flux.painter import rowwise

    with pytest.MonkeyPatch.context() as mp:
        for k in ("VECWORLD_KERNEL", "VECWORLD_FUSE", "VECWORLD_GRAPHS"):
            mp.delenv(k, raising=False)
        tr, _ = _bench().load_transformer()
    rowwise(tr)
    st, runner = stream_blocks(tr, "cuda", graphs=False, pack=True)
    e = SimpleNamespace(tr=tr, st=st, runner=runner, blocks=[*tr.transformer_blocks, *tr.single_transformer_blocks])
    yield e
    torch.cuda.synchronize()
    e.runner.close()
    e.st.close()
    e.st.arena.close()
    del e, tr, st, runner
    _trim()
    torch.cuda.empty_cache()


@contextmanager
def _resident(blocks):
    """Each block's weights on the GPU for its own call (a fresh device copy of its host tensors, one block at a time:
    the whole model doesn't fit), put back after it. The blocks must not be streamed meanwhile."""
    saved = {}

    def pre(m, a):
        saved[m] = [(t, t.data) for t in (*m.parameters(), *m.buffers())]
        for t, h in saved[m]:
            t.data = h.to("cuda")

    def post(m, a, o):
        for t, h in saved.pop(m):
            t.data = h

    hooks = [h for b in blocks for h in (b.register_forward_pre_hook(pre),
                                         b.register_forward_hook(post, always_call=True))]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def test_stream_matches_resident(fx):
    """Whole transformer passes (1 and 2 windows; eager, and replayed as graphs) with the blocks streamed from the
    arena equal the same passes with each block's weights resident on the GPU, bit for bit."""
    from test_graphs import _inputs

    def run(kw, graphs):
        fx.runner.enabled = graphs
        with torch.inference_mode():
            out = fx.tr(**kw)[0]
        torch.cuda.synchronize()
        return out

    got = {}
    for b in (1, 2):
        kw = _inputs(b, 256, seed=10 + b)
        got[b] = (run(kw, False), run(kw, True), run(kw, False))     # the last: after the graphs, still the same
    fx.runner.clear()
    fx.runner.enabled = False
    fx.st.close()                                            # the blocks back on their arena views, no hooks
    try:
        with _resident(fx.blocks):
            for b in (1, 2):
                want = run(_inputs(b, 256, seed=10 + b), False)
                assert torch.isfinite(want).all()
                for k, g in enumerate(got[b]):
                    assert torch.equal(g, want), (b, k)
    finally:
        fx.st.install()


def _single_inputs(tr, b=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    L, T = 1024, 256
    ids = torch.zeros(T + L, 3, device="cuda")
    ids[T:, 1], ids[T:, 2] = torch.arange(L, device="cuda") // 32, torch.arange(L, device="cuda") % 32
    with torch.inference_mode():
        return dict(hidden_states=torch.randn(b, L, H, generator=g).to("cuda", torch.bfloat16),
                    encoder_hidden_states=torch.randn(b, T, H, generator=g).to("cuda", torch.bfloat16),
                    temb=torch.randn(b, H, generator=g).to("cuda", torch.bfloat16),
                    image_rotary_emb=tr.pos_embed(ids))


def _chain(blocks, kw, passes):
    """passes back to back over blocks (each block's output the next one's input), GPU time per pass (CUDA events)."""
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(passes + 1)]
    with torch.inference_mode():
        ev[0].record()
        for k in range(passes):
            es, hs = kw["encoder_hidden_states"], kw["hidden_states"]
            for b in blocks:
                es, hs = b(hidden_states=hs, encoder_hidden_states=es, temb=kw["temb"],
                           image_rotary_emb=kw["image_rotary_emb"])
            ev[k + 1].record()
    torch.cuda.synchronize()
    return [ev[k].elapsed_time(ev[k + 1]) / 1000 for k in range(passes)]


def test_stream_overlaps(fx):
    """Copy hides behind compute: 12 single blocks streamed through a ring of 3 slots (each block's copy issued while
    the block 2 ahead computes) take <= 1.1 x the time of the same 12 blocks with their weights resident (compute
    only), at one window a pass (the least compute per copied byte). On AC power only."""
    from pathlib import Path

    from vecworld.engine.stream import BlockStreamer

    if any(p.read_text().strip() == "0" for p in Path("/sys/class/power_supply").glob("A*/online")):
        pytest.skip("speed is measured on AC power only")
    first = len(fx.tr.transformer_blocks)
    blocks = fx.blocks[first:first + 12]
    kw = _single_inputs(fx.tr)
    fx.st.close()
    try:
        saved = [(t, t.data) for b in blocks for t in (*b.parameters(), *b.buffers())]
        for t, h in saved:                                   # compute only: all 12 resident
            t.data = h.to("cuda")
        _chain(blocks, kw, 2)
        resident = _chain(blocks, kw, 7)
        for t, h in saved:
            t.data = h
        del saved
        torch.cuda.empty_cache()
        sub = BlockStreamer(blocks, lambda i: fx.st.arena.block(first + i), ring=3)
        sub.install()
        try:
            _chain(blocks, kw, 2)
            streamed = _chain(blocks, kw, 7)
            copies = sub.copies
        finally:
            sub.close()
    finally:
        fx.st.install()
    r, s = statistics.median(resident), statistics.median(streamed)
    gb = sum(fx.st.arena.block(first + i).numel() for i in range(12)) / 1e9
    print(f"\n12 single blocks, 1 window: resident {r * 1e3:.1f} ms, streamed {s * 1e3:.1f} ms ({s / r:.3f}x; "
          f"{gb:.2f} GB copied a pass, {gb / s:.1f} GB/s needed); copies {copies}")
    assert copies >= 9 * 12                                 # every block copied in every streamed pass
    assert s <= 1.1 * r


def test_stream_stop_then_reuse(fx):
    """Review Focus 3: a job Stopped between the passes of a step (check_stop), and one Stopped inside a pass (a block
    raises it, mid-way through the blocks), then a second job on the same ring: the second job equals the same job
    run uninterrupted, and page-locked host and allocated device memory are what they were before."""
    from vecworld.config import FLUX_WIN
    from vecworld.dream import gpu
    from vecworld.dream.gpu import Stopped
    from vecworld.flux.painter import FluxPainter

    passes = []
    p = FluxPainter.__new__(FluxPainter)
    p._setup(fx.tr, SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)), None,
             lambda d, n: passes.append(d), dev="cuda", dtype=torch.bfloat16)
    p.batch = 2                                             # 4 windows: 2 passes a step, so a stop can fall mid-step
    p._slots = fx.st
    fx.runner.enabled = False
    g = torch.Generator().manual_seed(5)
    e = (torch.randn(1, 256, 4096, generator=g).to("cuda", torch.bfloat16),
         torch.randn(1, 768, generator=g).to("cuda", torch.bfloat16))
    grid = torch.randn(1, 64, FLUX_WIN, 2 * FLUX_WIN, generator=g).to("cuda")

    def job():
        with torch.inference_mode():
            out = p._denoise(grid.clone(), p._sigmas(3), [e] * 4)
        torch.cuda.synchronize()
        return out

    want = job()
    torch.cuda.synchronize()
    mem0, pin0 = torch.cuda.memory_allocated(), fx.st.pinned_bytes
    slots0 = [t.data_ptr() for t in fx.st.slots]

    # 1. Stopped at check_stop between the two passes of step 1 (after pass 3)
    stop = threading.Event()
    passes.clear()
    p.progress = lambda d, n: (passes.append(d), d == 3 and stop.set())
    gpu._job.stop = stop
    stopped = False
    try:
        job()
    except Stopped:
        stopped = True
    finally:
        gpu._job.stop = None
    assert stopped and passes == [1, 2, 3]

    # 2. Stopped inside a pass: block 20 raises it on the job's second pass
    blk, calls = fx.blocks[20], []
    inner = blk.forward

    def boom(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise Stopped()
        return inner(*a, **k)

    blk.forward = boom
    p.progress = None
    stopped = False
    try:
        job()
    except Stopped:
        stopped = True
    finally:
        blk.forward = inner
    assert stopped and len(calls) == 2

    got = job()
    assert torch.equal(got, want)
    del got
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == mem0
    assert fx.st.pinned_bytes == pin0 and [t.data_ptr() for t in fx.st.slots] == slots0


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
    laptop, when glibc kept the freed copies). Run alone, in a fresh process."""
    peak, arena = _painter_rss()
    print(f"\npeak RSS {peak / 1e9:.2f} GB, arena {arena / 1e9:.3f} GB")
    assert arena > 6e9 and peak < arena + 2e9
