"""Batch invariance beyond the default kernel (Task 9 review I1, M1; run with the game server stopped).

Under an engine kernel a window of a batched pass equals that window alone, bit for bit (rowwise outer layers; the
engine's kernels sum each row the same way at any M): test_graphs covers the default w4a8, this covers mma. Under
VECWORLD_KERNEL=off the blocks' cuBLAS GEMMs round by M, so the reference is pinned to one window a pass
(engine.engine_batch), whatever VECWORLD_BATCH says.

Each test loads its own model and frees it before the next: two 6.7 GB page-locked transformers don't fit in RAM.
"""
import gc

import pytest
import torch

from test_graphs import _fwd, _inputs, _trim

pytestmark = pytest.mark.gpu


def _free(*objs):
    torch.cuda.synchronize()
    for o in objs:
        del o
    _trim()
    torch.cuda.empty_cache()


def test_mma_batched_windows_match_single_windows(monkeypatch):
    """Task 9 review M1: under the lossless VECWORLD_KERNEL=mma, each window of a batch of FLUX_BATCH, 5 and 2 equals
    it alone, bit for bit, eager on the slots and as CUDA graphs."""
    from types import SimpleNamespace

    from test_graphs import _bench

    from vecworld.config import FLUX_BATCH
    from vecworld.engine.graphs import stream_blocks
    from vecworld.flux.painter import _pin_in_place, _unpin, rowwise

    for k in ("VECWORLD_FUSE", "VECWORLD_GRAPHS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VECWORLD_KERNEL", "mma")
    tr, _ = _bench().load_transformer()
    rowwise(tr)
    pinned = _pin_in_place([*tr.transformer_blocks, *tr.single_transformer_blocks])
    slots, runner = stream_blocks(tr, "cuda", graphs=True)
    e = SimpleNamespace(tr=tr, slots=slots, runner=runner)
    try:
        for graphs in (False, True):
            for b in sorted({FLUX_BATCH, 5, 2}):
                kw = _inputs(b, 256, seed=70 + b)
                out = _fwd(e, kw, graphs).clone()
                for i in range(b):
                    one = {k: (v[i:i + 1] if k in ("hidden_states", "encoder_hidden_states", "pooled_projections",
                                                   "timestep") else v) for k, v in kw.items()}
                    assert torch.equal(out[i:i + 1], _fwd(e, one, graphs=False)), (graphs, b, i)
                del kw, out
            runner.clear()
    finally:
        torch.cuda.synchronize()
        runner.close()
        slots.close()
        _unpin(pinned)
        del e, tr, slots, runner
        _free()


def test_reference_paints_a_window_a_pass_whatever_the_batch(monkeypatch):
    """Task 9 review I1: a VECWORLD_KERNEL=off painter configured with VECWORLD_BATCH=8 runs batch 1, so every window it
    paints equals the same window alone, bit for bit. Why it must: forced to batch 2 the reference's windows are NOT
    their alone selves (diffusers' cuBLAS GEMMs in the blocks round by M; recorded here so nobody relies on it)."""
    from vecworld.config import FLUX_WIN
    from vecworld.flux.painter import FluxPainter

    for k in ("VECWORLD_FUSE", "VECWORLD_GRAPHS", "VECWORLD_CACHE_THRESHOLD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VECWORLD_KERNEL", "off")
    monkeypatch.setenv("VECWORLD_BATCH", "8")
    p = FluxPainter()
    try:
        assert p.kernel is None and p.batch == 1
        g = torch.Generator().manual_seed(9)
        embs = [(torch.randn(1, 256, 4096, generator=g).to("cuda", torch.bfloat16),
                 torch.randn(1, 768, generator=g).to("cuda", torch.bfloat16)) for _ in range(8)]
        grid = torch.randn(1, 64, FLUX_WIN, 8 * FLUX_WIN // 2, generator=g).to("cuda")
        seen, fwd = [], p._forward

        def rec(tiles, es, s, **k):
            out = fwd(tiles, es, s, **k)
            seen.append((tiles.clone(), [x for x in es], s, out.clone()))
            return out

        p._forward = rec
        with torch.inference_mode():
            p._denoise(grid, [1.0, 0.0], embs)               # one step, 8 windows
            assert [t.shape[0] for t, *_ in seen] == [1] * 8
            for t, es, s, out in seen[::3]:
                assert torch.equal(fwd(t, es, s), out)        # the window alone, again: the same bits
            t2 = torch.cat([seen[0][0], seen[1][0]])
            both = fwd(t2, seen[0][1] + seen[1][1], 1.0)      # the reference forced to batch 2
            assert not (torch.equal(both[:1], seen[0][3]) and torch.equal(both[1:], seen[1][3]))
    finally:
        p._forward = fwd
        p.close()
        del p
        gc.collect()
        _free()
