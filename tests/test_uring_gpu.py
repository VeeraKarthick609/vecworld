"""The FLUX blocks streamed from the SSD (engine.uring + BlockStreamer.from_ssd; Task 10b, docs/engine/11-io-uring.md;
gpu, run with the game server stopped).

The bytes are the file's, so the bar is equality: every block lands in its slot exactly as the RAM path puts it, and
the painter paints the same picture from the SSD as from RAM (torch.equal, batch 1 and 8; the plan's floor was PSNR
>= 40 dB).
"""
import gc

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

BASE = [("a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns", 1.0)]


def _bench():
    from test_graphs import _bench as b

    return b()


def _free():
    from vecworld.engine.stream import _trim

    gc.collect()
    _trim()
    torch.cuda.empty_cache()


def test_ssd_blocks_match_the_ram_bytes():
    """Every block read from the SSD lands in its slot byte for byte as the blocks' tensors loaded into RAM (the
    painter's full read of the GGUF): each weight from the file through the staging ring, each bias and norm from the
    resident copy. Two passes, so every staging buffer is read into again. The reader page-locks the staging ring and
    the resident tensors only (< 1 GB, not the 6.66 GB arena), and after close() the blocks hold placeholders."""
    from vecworld.engine.slots import _tensors
    from vecworld.engine.stream import BlockStreamer

    tr, path = _bench().load_transformer()
    blocks = [*tr.transformer_blocks, *tr.single_transformer_blocks]
    orig = [{n: t.data for n, t in _tensors(b).items()} for b in blocks]
    st = BlockStreamer.from_ssd(blocks, path, ring=3, buffers=3, ahead=2)
    try:
        assert 0 < st.pinned_bytes < 1e9 and st.reader.pinned_bytes == st.pinned_bytes
        for rnd in range(2):
            for i, b in enumerate(blocks):
                st.load(i)
                torch.cuda.synchronize()
                assert st.copy_done(i)
                slot, lay = st.slots[i % st.ring], st._layout[i]
                for name, h in orig[i].items():
                    want = h.contiguous().view(-1).view(torch.uint8)
                    got = slot[lay[name]:lay[name] + want.numel()]
                    assert torch.equal(got, want.to("cuda")), (rnd, i, name)
        s = st.reader.stats
        print(f"\nSSD reader: {s['reads']} reads, {s['bytes'] / 1e9:.2f} GB, {s['hits']} hits, "
              f"read wait {s['read_wait_s']:.2f} s, buffer wait {s['buffer_wait_s']:.2f} s")
    finally:
        st.close()
        st.reader.close()
        st.reader.close()                                   # safe twice
    for b in blocks:
        for name, t in _tensors(b).items():
            assert t.device.type == "cpu" and t.untyped_storage().nbytes() <= t.element_size(), name
    del st, tr, blocks, orig
    _free()


def _paint(source: str):
    """(batch-1 512 x 512 tile image, its latent, batch-8 panorama, the reader's stats) from a painter whose blocks'
    weights come from `source`."""
    from vecworld.config import FLUX_STEPS
    from vecworld.flux.painter import FluxPainter

    bench = _bench()
    with pytest.MonkeyPatch.context() as mp:
        for k in ("VECWORLD_KERNEL", "VECWORLD_FUSE", "VECWORLD_GRAPHS", "VECWORLD_BATCH", "VECWORLD_CACHE_THRESHOLD"):
            mp.delenv(k, raising=False)
        mp.setenv("FLUX_WEIGHTS_SOURCE", source)
        p = FluxPainter()
    try:
        assert p.weights_source == source and p.batch == 8
        with torch.inference_mode():
            e = p._embed(BASE)
            g = bench.run_tile512(p, e, FLUX_STEPS, seed=5)
            tile = p._decode(g)
        pano = p.panorama(BASE, seed=21)
        stats = dict(p._slots.reader.stats) if source == "ssd" else None
        pinned = p._slots.pinned_bytes
        return tile, g.cpu(), pano, stats, pinned
    finally:
        p.close()
        del p
        _free()


def test_flux_from_ssd_matches():
    """The painter with its blocks read from the SSD (FLUX_WEIGHTS_SOURCE=ssd) paints exactly what it paints from the
    RAM arena: a 512 x 512 tile at batch 1 (its latent too) and a panorama at batch 8, the same seeds."""
    ssd = _paint("ssd")
    ram = _paint("ram")
    print(f"\nSSD reader over a tile and a panorama: {ssd[3]}; pinned {ssd[4] / 1e9:.2f} GB (RAM: {ram[4] / 1e9:.2f})")
    ssd[0].save("/tmp/t10b_tile_ssd.png")
    ssd[2].save("/tmp/t10b_pano_ssd.png")
    assert torch.equal(ssd[1], ram[1])
    assert np.array_equal(np.asarray(ssd[0]), np.asarray(ram[0]))
    assert np.array_equal(np.asarray(ssd[2]), np.asarray(ram[2]))
    assert ssd[4] < 1e9 < 6e9 < ram[4]


def test_gather_must_cover_the_block_exactly():
    """A Gather whose copies miss a tensor (the slot would keep the previous block's bytes there), overlap, or put a
    tensor anywhere but its own range (N4: shifted into its padding, the byte total unchanged) is refused when the
    block's copy is issued (ValueError), as a copy outside the block is; an exact one streams."""
    from vecworld.engine.slots import _tensors, block_layout
    from vecworld.engine.stream import BlockStreamer, Gather, HostArena

    from test_graphs import _toys

    blocks, _ = _toys(3)
    a = HostArena(blocks)
    lays = [block_layout(b) for b in blocks]

    def exact(i):
        lay, base = lays[i][0], a.block(i).data_ptr()
        out = []
        for name, t in _tensors(blocks[i]).items():
            out += [lay[name], base + lay[name], t.numel() * t.element_size()]
        return out

    def shifted(c):
        # N4: the smallest tensor's copy moved 16 bytes into its own padding: no overlap, the same byte total, but
        # the tensor's first 16 bytes would keep the previous block's
        c = list(c)
        k = min(range(0, len(c), 3), key=lambda k: c[k + 2] if c[k + 2] else 1 << 60)
        assert c[k + 2] + 16 <= 256
        c[k] += 16
        return c

    bad = {"miss": lambda c: c[3:], "overlap": lambda c: c + c[:3], "shifted": shifted}
    for kind, f in [("exact", lambda c: c), *bad.items()]:
        keep = {i: Gather(f(exact(i)), a) for i in range(3)}
        st = BlockStreamer(blocks, lambda i: keep[i], ring=3)
        st.install()
        try:
            if kind == "exact":
                for i in range(3):
                    st.load(i)
                torch.cuda.synchronize()
            else:
                with pytest.raises(ValueError, match="cover"):
                    st.load(0)
        finally:
            st.close()
    a.close()


def test_ssd_stop_then_reuse():
    """Review Focus 3 on the shipped default (the blocks read from the SSD, FLUX_WEIGHTS_SOURCE=ssd; the RAM arena's
    version is test_stream.py::test_stream_stop_then_reuse): a job Stopped between the passes of a step (check_stop)
    and one Stopped inside a pass (block 20 raises it), then the same job again on the same streamer and reader: it
    equals the job run uninterrupted, and the reader's page-locked staging buffers, the slots and allocated device
    memory are what they were before (reused, not leaked or re-made)."""
    import threading
    from types import SimpleNamespace

    from test_graphs import _bench as b
    from vecworld.config import FLUX_SSD_AHEAD, FLUX_SSD_BUFFERS, FLUX_SSD_CHUNK, FLUX_SSD_QUEUE_DEPTH, FLUX_WIN
    from vecworld.dream import gpu
    from vecworld.dream.gpu import Stopped
    from vecworld.engine.ext import load_ext
    from vecworld.engine.graphs import stream_blocks
    from vecworld.flux.painter import FluxPainter, rowwise

    tr, path = b().load_transformer()
    rowwise(tr)
    st, runner = stream_blocks(tr, "cuda", graphs=False, ssd=path,
                               reader=dict(buffers=FLUX_SSD_BUFFERS, ahead=FLUX_SSD_AHEAD,
                                           queue_depth=FLUX_SSD_QUEUE_DEPTH, chunk=FLUX_SSD_CHUNK))
    blocks = [*tr.transformer_blocks, *tr.single_transformer_blocks]
    reader = st.reader
    try:
        passes = []
        p = FluxPainter.__new__(FluxPainter)
        p._setup(tr, SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)), None,
                 lambda d, n: passes.append(d), dev="cuda", dtype=torch.bfloat16)
        p.batch = 2                                     # 4 windows: 2 passes a step, so a stop can fall mid-step
        p._slots = st
        g = torch.Generator().manual_seed(5)
        e = (torch.randn(1, 256, 4096, generator=g).to("cuda", torch.bfloat16),
             torch.randn(1, 768, generator=g).to("cuda", torch.bfloat16))
        grid = torch.randn(1, 64, FLUX_WIN, 2 * FLUX_WIN, generator=g).to("cuda")

        def job():
            with torch.inference_mode():
                out = p._denoise(grid.clone(), p._sigmas(3), [e] * 4)
            torch.cuda.synchronize()
            return out

        def pinned():
            """The reader's page-locked bytes, whether its staging buffers are still registered (asked of the driver)
            and torch's page-locked pool in use."""
            return (reader.pinned_bytes, load_ext().host_pinned(reader._stage.data_ptr(), reader._stage.numel()),
                    torch.cuda.host_memory_stats().get("active_bytes.current", 0))

        want = job()
        torch.cuda.synchronize()
        mem0, pin0 = torch.cuda.memory_allocated(), pinned()
        slots0 = [t.data_ptr() for t in st.slots]
        assert pin0[0] > 0 and pin0[1]

        # 1. Stopped at check_stop between the two passes of step 1 (after pass 3)
        stop = threading.Event()
        passes.clear()
        p.progress = lambda d, n: (passes.append(d), d == 3 and stop.set())
        gpu._job.stop = stop
        with pytest.raises(Stopped):
            try:
                job()
            finally:
                gpu._job.stop = None
        assert passes == [1, 2, 3]

        # 2. Stopped inside a pass: block 20 raises it on the job's second pass (blocks 21.. never run)
        blk, calls = blocks[20], []
        inner = blk.forward

        def boom(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise Stopped()
            return inner(*a, **k)

        blk.forward = boom
        p.progress = None
        with pytest.raises(Stopped):
            try:
                job()
            finally:
                blk.forward = inner
        assert len(calls) == 2

        got = job()
        assert torch.equal(got, want)
        del got
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == mem0
        assert pinned() == pin0 and [t.data_ptr() for t in st.slots] == slots0
    finally:
        torch.cuda.synchronize()
        runner.close()
        st.close()
        reader.close()
        del st, runner, reader, tr, blocks
        _free()
