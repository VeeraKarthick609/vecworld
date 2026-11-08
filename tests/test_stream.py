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


def test_graph_launch_keeps_the_gil():
    """The extension's graph_launch replays a captured graph like CUDAGraph.replay() (same result) without releasing
    the GIL, so a graph replay per block costs microseconds with other Python threads busy (replay() waits for one)."""
    import time

    from vecworld.engine.ext import load_ext

    launch = load_ext().graph_launch
    x = torch.randn(4096, device="cuda")
    y = torch.zeros_like(x)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y.copy_(x * 2 + 1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y.copy_(x * 2 + 1)
    x.normal_()
    launch(g.raw_cuda_graph_exec())
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2 + 1)

    def per_call(f, n=50):
        t = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    stop = [False]

    def spin():
        while not stop[0]:
            pass

    th = [threading.Thread(target=spin, daemon=True) for _ in range(2)]
    for t in th:
        t.start()
    try:
        ex = g.raw_cuda_graph_exec()                        # the runner keeps each graph's handle too
        ours = per_call(lambda: launch(ex))
        theirs = per_call(g.replay)
    finally:
        stop[0] = True
        for t in th:
            t.join()
    print(f"\nwith 2 spinning threads: graph_launch {ours * 1e6:.0f} us, CUDAGraph.replay {theirs * 1e6:.0f} us")
    assert ours < 1e-3


def test_reader_may_reuse_a_buffer_only_once_its_copy_is_done():
    """The 10b reader's contract: the bytes source(i) handed over may be changed once copy_done(i) is True, and not
    before. A reader that gives each block a fresh page-locked buffer and scribbles on every buffer whose copy is done
    (while the GPU is held back, so the copies queue up) must leave every block computing on its own weights; the
    streamer keeps each buffer alive until its copy has run; and copy_done(i) is False from inside source(i) on (a
    reader thread that asks in the window between handing the bytes over and the copy being issued)."""
    import weakref

    from vecworld.engine.stream import BlockStreamer, HostArena

    from test_graphs import Toy, _toys

    blocks, _ = _toys(6)
    ref = [Toy(i).cuda() for i in range(6)]
    a = HostArena(blocks)
    handed, early = [], []                                  # (block, weakref to its buffer); copy_done inside source

    def source(i):
        early.append(st.copy_done(i))
        buf = torch.empty(a.sizes[i], dtype=torch.uint8, pin_memory=True)
        buf.copy_(a.block(i))
        handed.append((i, weakref.ref(buf)))
        return buf                                          # the reader keeps no reference of its own

    st = BlockStreamer(blocks, source, ring=3)
    st.install()
    x = torch.randn(8, 64, device="cuda").bfloat16()
    outs = []
    with torch.inference_mode():
        for _ in range(3):                                  # three passes, the GPU held back before every block
            for i in range(6):
                torch.cuda._sleep(2_000_000)
                outs.append((i, blocks[i](x)))
        # the reader, before any sync: reuse every buffer whose copy is done. The k-th source() call is the k-th
        # load, ticket k + 1 (BlockStreamer.load returns it; the hooks issue them in this order)
        assert st._ring.issued() == len(handed)
        for k, (i, ref_buf) in enumerate(handed):
            buf, done = ref_buf(), st.done(k + 1)
            if done and buf is not None:
                buf.fill_(0xFF)
            elif not done:
                assert buf is not None, f"block {i}'s buffer (load {k + 1}) was released before its copy ran"
            if k + 1 == st._ticket[i]:
                assert st.copy_done(i) == done
        pending = sum(not st.done(k + 1) for k in range(len(handed)))
        torch.cuda.synchronize()
        for i, y in outs:
            assert torch.equal(y, ref[i](x)), i
    assert early and not any(early)
    assert pending > 0                                      # copies were still queued when the reader looked
    st.close()
    with pytest.raises(RuntimeError, match="closed"):
        st.copy_done(0)
    a.close()


def test_failed_install_leaves_the_blocks_and_frees_the_arena(monkeypatch):
    """BlockStreamer.packed(): an install() that fails (here the SlotRing; on the 4 GB card most likely the slots'
    cudaMalloc) leaves every block on its host tensors and the arena unregistered (a registered range freed later
    breaks other copies); a BlockSlots / BlockStreamer whose install() fails leaves the blocks' tensors as they were."""
    from types import SimpleNamespace as NS

    import vecworld.engine.ext as ext_mod
    from vecworld.engine.stream import BlockStreamer, HostArena

    from test_graphs import _toys

    real = ext_mod.load_ext()

    class Boom:
        def __init__(self, *a):
            raise RuntimeError("SlotRing failed")

    monkeypatch.setattr(ext_mod, "load_ext", lambda: NS(SlotRing=Boom, host_pinned=real.host_pinned))
    blocks, host = _toys(4)
    with pytest.raises(RuntimeError, match="SlotRing failed"):
        BlockStreamer.packed(blocks, ring=2)
    for b in blocks:
        for t in (*b.parameters(), *b.buffers()):
            assert t.device.type == "cpu"
            assert not real.host_pinned(t.data_ptr(), t.numel() * t.element_size())
    for b, h in zip(blocks, host):
        assert torch.equal(b.w, h["w"])
    a = HostArena(blocks)
    st = BlockStreamer(blocks, a.block, ring=2)
    before = [t.data_ptr() for b in blocks for t in b.parameters()]
    with pytest.raises(RuntimeError, match="SlotRing failed"):
        st.install()
    assert [t.data_ptr() for b in blocks for t in b.parameters()] == before and not st._hooks
    monkeypatch.setattr(ext_mod, "load_ext", lambda: real)
    st.install()                                            # and a later install works
    x = torch.randn(8, 64, device="cuda").bfloat16()
    with torch.inference_mode():
        assert blocks[0](x).isfinite().all()
    st.close()
    a.close()


class Sleepy(torch.nn.Module):
    """A toy block that holds the GPU back, then reads its weight into a buffer kept outside it, then (once) raises."""

    def __init__(self, i, n=256):
        super().__init__()
        g = torch.Generator().manual_seed(i)
        self.w = torch.nn.Parameter(torch.randn(n, n, generator=g).bfloat16(), requires_grad=False)
        self.out = None
        self.boom = False

    def forward(self, x):
        torch.cuda._sleep(50_000_000)                      # tens of ms: the next copy would overtake it
        self.out = x @ self.w.T
        if self.boom:
            raise RuntimeError("boom")
        return self.out


def test_a_raising_block_keeps_its_slot_until_its_kernels_have_read_it():
    """after() runs for a block whose forward raised (always_call), so its slot's next copy waits for the kernels it
    queued: the raising block's queued output is still computed from its own weights, even though the slot's next
    user runs straight after it (without always_call that copy waits only for the slot's previous pass and overwrites
    the weights under the sleeping kernel)."""
    from vecworld.engine.stream import BlockStreamer, HostArena

    blocks = [Sleepy(i) for i in range(4)]
    ref = [b.w.detach().clone().cuda() for b in blocks]
    a = HostArena(blocks)
    st = BlockStreamer(blocks, a.block, ring=2)
    st.install()
    x = torch.randn(64, 256, device="cuda").bfloat16()
    with torch.inference_mode():
        for b in blocks:                                    # a first pass: every slot used once
            b(x)
        torch.cuda.synchronize()
        blocks[0].boom = True
        with pytest.raises(RuntimeError, match="boom"):
            blocks[0](x)
        kept = blocks[0].out
        blocks[2](x)                                        # slot 0's next user, at once
        torch.cuda.synchronize()
        assert torch.equal(kept, x @ ref[0].T)
    st.close()
    a.close()


def test_graph_capture_refuses_a_block_that_draws_random_numbers():
    """graph_launch skips CUDAGraph.replay()'s RNG prologue, so a captured block drawing random numbers would repeat
    the capture's numbers on every replay: the runner refuses it at capture, naming the block; dropout_p = 0
    attention and blocks without randomness capture as before."""
    from vecworld.engine.graphs import GraphRunner

    class Blk(torch.nn.Module):
        def __init__(self, noise):
            super().__init__()
            self.noise = noise

        def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                    joint_attention_kwargs=None):
            h = hidden_states * 2
            if self.noise == "rand":
                h = h + torch.rand_like(h)
            q = h.view(h.shape[0], 1, h.shape[1], h.shape[2])
            h = torch.nn.functional.scaled_dot_product_attention(q, q, q, dropout_p=0.0).view_as(h)
            return encoder_hidden_states, h

    def call(b):
        hs = torch.randn(1, 64, 128, device="cuda").bfloat16()
        es = torch.randn(1, 8, 128, device="cuda").bfloat16()
        temb = torch.randn(1, 128, device="cuda").bfloat16()
        rot = (torch.ones(72, 128, device="cuda"), torch.zeros(72, 128, device="cuda"))
        return b(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot)

    ok, bad = Blk(None), Blk("rand")
    r = GraphRunner([ok, bad], None, enabled=True)
    call(ok)
    assert r.stats["captures"] == 1
    with pytest.raises(RuntimeError, match="block 1.*random"):
        call(bad)
    r.close()


# --- the real model ---------------------------------------------------------------------------------------------------

def _trim():
    from vecworld.engine.stream import _trim as trim

    gc.collect()
    trim()


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

    from vecworld.engine.ext import load_ext

    def pinned():
        """torch's page-locked pool in use (a reader's or a stray pin_memory buffer), and whether the arena is still
        registered with the driver (asked of the driver, not of the streamer)."""
        st = torch.cuda.host_memory_stats()
        return st.get("active_bytes.current", 0), load_ext().host_pinned(fx.st.arena.buf.data_ptr(),
                                                                         fx.st.arena.nbytes)

    want = job()
    torch.cuda.synchronize()
    mem0, pin0 = torch.cuda.memory_allocated(), pinned()
    slots0 = [t.data_ptr() for t in fx.st.slots]
    assert pin0[1]

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
    assert pinned() == pin0 and [t.data_ptr() for t in fx.st.slots] == slots0

