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
