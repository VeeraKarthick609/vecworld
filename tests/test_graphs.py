"""CUDA graphs over the FLUX blocks from fixed-address weight slots (Task 7, docs/engine/07-cuda-graphs.md; gpu, run
with the game server stopped).

BlockSlots streams each block's weights from page-locked host memory into one of `ring` device slots of fixed
address; GraphRunner replays one CUDA graph per (block, shape) from them. Graph replay runs exactly the kernels eager
runs, on the same weights and inputs, so everything here is held to bit equality (torch.equal).

The real-model tests share one transformer (module fixture `eng`). test_slots_match_offload, which switches that
transformer to diffusers' group offload, runs last.
"""
import gc
import threading
import types
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu

H = 3072


# --- slots on small blocks --------------------------------------------------------------------------------------------

class Toy(torch.nn.Module):
    """A block with tensors of awkward sizes (an odd-length bias, a 1-byte buffer) between aligned ones."""

    def __init__(self, i, n=64):
        super().__init__()
        g = torch.Generator().manual_seed(i)
        self.w = torch.nn.Parameter(torch.randn(n, n, generator=g).bfloat16(), requires_grad=False)
        self.b = torch.nn.Parameter(torch.randn(3, generator=g).bfloat16(), requires_grad=False)
        self.register_buffer("flag", torch.tensor([i % 7], dtype=torch.uint8))
        self.v = torch.nn.Parameter(torch.randn(n, generator=g).bfloat16(), requires_grad=False)

    def forward(self, x):
        return x @ self.w.T + self.v + self.b.sum() + self.flag.to(x.dtype)


def _toys(n):
    blocks = [Toy(i) for i in range(n)]
    host = [{k: t.detach().clone() for k, t in (*b.named_parameters(), *b.named_buffers())} for b in blocks]
    return blocks, host


def test_slot_layout_aligns_every_tensor():
    """Every tensor of every block views into its slot (block i in slot i % ring) at an aligned offset, so the Q4_K
    kernels' 16-byte vector loads hold; the slots are sized to the largest block."""
    from vecworld.engine.slots import ALIGN, BlockSlots

    blocks, host = _toys(5)
    s = BlockSlots(blocks, lambda i: host[i], ring=2)
    assert ALIGN % 16 == 0 and len(s.slots) == 2
    for i, b in enumerate(blocks):
        base = s.slots[i % 2].data_ptr()
        for name, t in (*b.named_parameters(), *b.named_buffers()):
            assert t.is_cuda and t.data_ptr() % 16 == 0, name
            assert 0 <= t.data_ptr() - base < s.slot_bytes and (t.data_ptr() - base) % ALIGN == 0, name
    s.close()


def test_slot_misalignment_is_loud(monkeypatch):
    """A layout that would put a tensor at an address off 16 bytes raises, naming it, never copies it aside."""
    import vecworld.engine.slots as slots

    monkeypatch.setattr(slots, "ALIGN", 1)                  # packs the 6-byte bias and 1-byte flag back to back
    blocks, host = _toys(2)
    with pytest.raises(RuntimeError, match="aligned"):
        slots.BlockSlots(blocks, lambda i: host[i], ring=2)


def test_slots_stream_blocks_in_any_order():
    """Run in order, twice round (the prefetch wraps to the next pass), then out of order: every block computes on
    its own weights, as the same blocks resident on the GPU do."""
    from vecworld.engine.slots import BlockSlots

    blocks, host = _toys(5)
    ref = [Toy(i).cuda() for i in range(5)]
    s = BlockSlots(blocks, lambda i: host[i], ring=2)
    x = torch.randn(8, 64, device="cuda").bfloat16()
    with torch.inference_mode():
        for order in (range(5), range(5), (3, 0, 4, 4, 1)):
            for i in order:
                assert torch.equal(blocks[i](x), ref[i](x)), i
    assert s.copies >= 5
    s.close()
    for b, h in zip(blocks, host):                           # close() puts the host tensors back
        assert all(t.device.type == "cpu" for t in b.parameters())
        assert torch.equal(b.w, h["w"])


def test_slots_refuse_a_non_contiguous_host_tensor():
    """A non-contiguous host tensor would need a pageable copy (no async transfer, and close() would put the copy back
    in place of the page-locked original), so BlockSlots raises instead of copying it."""
    from vecworld.engine.slots import BlockSlots

    blocks, host = _toys(2)
    host[1]["w"] = host[1]["w"].t().contiguous().t()                 # same shape, transposed strides
    with pytest.raises(ValueError, match="contiguous"):
        BlockSlots(blocks, lambda i: host[i], ring=2)


def test_slot_is_released_after_its_block_when_the_block_raises():
    """A block whose forward raises still marks its slot free after the kernels it queued, and starts the next user's
    copy (an always-called hook), so a later copy is ordered after them; the next pass computes correctly."""
    from vecworld.engine.slots import BlockSlots

    blocks, host = _toys(5)
    ref = [Toy(i).cuda() for i in range(5)]
    s = BlockSlots(blocks, lambda i: host[i], ring=2)
    x = torch.randn(8, 64, device="cuda").bfloat16()
    orig = Toy.forward

    def boom(self, x):
        orig(self, x)                                       # queue its kernels, then fail
        raise RuntimeError("boom")

    with torch.inference_mode():
        blocks[0](x)
        blocks[1].forward = types.MethodType(boom, blocks[1])
        with pytest.raises(RuntimeError, match="boom"):
            blocks[1](x)
        del blocks[1].forward
        assert s._used[1] and s.loaded[1] == s.next_user(1)  # released, the slot's next user on its way in
        for i in range(5):
            assert torch.equal(blocks[i](x), ref[i](x)), i
    s.close()


def test_slots_refuse_another_thread():
    """BlockSlots' state and event ordering assume one thread (the game's GPU queue worker): a block run from a
    second thread raises instead of interleaving."""
    from vecworld.engine.slots import BlockSlots

    blocks, host = _toys(3)
    s = BlockSlots(blocks, lambda i: host[i], ring=2)
    x = torch.randn(8, 64, device="cuda").bfloat16()
    with torch.inference_mode():
        blocks[0](x)
    err = []

    def other():
        try:
            with torch.inference_mode():
                blocks[1](x)
        except RuntimeError as e:
            err.append(str(e))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert err and "thread" in err[0]
    s.close()


# --- the attached quantization's staleness guard ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def gg():
    from vecworld.config import FLUX_GGUF
    from vecworld.engine.q4k import read_gguf

    return read_gguf(FLUX_GGUF)


def test_stale_attached_quantization_is_never_used(gg):
    """attach_quantized records x's version; once x is written in place (as a graph's static buffer is) the attached
    quantization is stale and quantized_of returns None, so a w4a8 linear quantizes the new x itself. An inference
    tensor tracks no version, so its attachment is never trusted (the GraphRunner runs blocks outside inference
    mode for this)."""
    from vecworld.engine.q4k import Q4KLinear, attach_quantized, quantize_act, quantized_of

    lin = Q4KLinear(gg["double_blocks.0.img_mlp.0.weight"], None, "w4a8").cuda()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, H, generator=g).to("cuda", torch.bfloat16)
    with torch.no_grad():
        attach_quantized(x, quantize_act(x))
        assert quantized_of(x) is not None
        x.mul_(2)                                           # written in place: the attachment is stale
        assert quantized_of(x) is None
        assert torch.equal(lin(x), lin(x.clone()))          # computed from the new x, not the stale quantization
    with torch.inference_mode():
        xi = torch.randn(300, H, device="cuda").bfloat16()
        attach_quantized(xi, quantize_act(xi))
        assert quantized_of(xi) is None


# --- the real model -------------------------------------------------------------------------------------------------

def _bench():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bench_flux", Path(__file__).parents[1] / "scripts/bench_flux.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def _trim():
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


@pytest.fixture(scope="module")
def eng():
    """The transformer as the painter runs it under the default flags (w4a8 + keep-list, fused glue), its blocks on
    BlockSlots with a GraphRunner (vecworld.engine.graphs.stream_blocks), host weights page-locked."""
    from vecworld.engine.graphs import stream_blocks
    from vecworld.flux.painter import _pin_in_place, _unpin, rowwise

    with pytest.MonkeyPatch.context() as mp:
        for k in ("VECWORLD_KERNEL", "VECWORLD_FUSE", "VECWORLD_GRAPHS"):
            mp.delenv(k, raising=False)
        tr, _ = _bench().load_transformer()
    rowwise(tr)                                             # as the painter (Task 9)
    pinned = _pin_in_place([*tr.transformer_blocks, *tr.single_transformer_blocks])
    slots, runner = stream_blocks(tr, "cuda", graphs=True)
    e = SimpleNamespace(tr=tr, slots=slots, runner=runner, pinned=pinned)
    yield e
    torch.cuda.synchronize()
    e.runner.close()
    e.slots.close()
    _unpin(e.pinned)
    del e, tr, slots, runner
    _trim()
    torch.cuda.empty_cache()


def _inputs(b, t, seed=0, n=32):
    g = torch.Generator().manual_seed(seed)
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    return dict(hidden_states=torch.randn(b, n * n, 64, generator=g).to("cuda", torch.bfloat16),
                encoder_hidden_states=torch.randn(b, t, 4096, generator=g).to("cuda", torch.bfloat16),
                pooled_projections=torch.randn(b, 768, generator=g).to("cuda", torch.bfloat16),
                timestep=torch.full((b,), 0.75, device="cuda", dtype=torch.bfloat16),
                img_ids=torch.stack([torch.zeros_like(i), i, j], -1).reshape(-1, 3).to("cuda", torch.bfloat16),
                txt_ids=torch.zeros(t, 3, device="cuda", dtype=torch.bfloat16), guidance=None, return_dict=False)


def _fwd(e, kw, graphs: bool):
    e.runner.enabled = graphs
    with torch.inference_mode():
        out = e.tr(**kw)[0]
    torch.cuda.synchronize()
    return out


def test_graph_matches_eager(eng):
    """Graph replay equals eager on the same slots, bit for bit: on the pass that captures (warm-up, capture, replay)
    and on a pure replay. The counters show the fused block forwards and the fused quantize ran (Python-time counts:
    eager and capture runs, never a replay) and that the second pass replayed every block."""
    from vecworld.engine import blocks
    from vecworld.engine import q4k

    kw = _inputs(1, 256, seed=1)
    want = _fwd(eng, kw, graphs=False)
    assert torch.isfinite(want).all()
    c0, q0, s0 = blocks.CALLS.copy(), q4k.CALLS.copy(), eng.runner.stats.copy()
    first = _fwd(eng, kw, graphs=True)
    c1, q1, s1 = blocks.CALLS.copy(), q4k.CALLS.copy(), eng.runner.stats.copy()
    again = _fwd(eng, _inputs(1, 256, seed=1), graphs=True)          # new input tensors: copied into the graph's
    c2, q2, s2 = blocks.CALLS.copy(), q4k.CALLS.copy(), eng.runner.stats.copy()
    assert torch.equal(first, want) and torch.equal(again, want)
    nd, ns = len(eng.tr.transformer_blocks), len(eng.tr.single_transformer_blocks)
    # capture pass: each block ran twice in Python (warm-up + capture), its quantized x_mod consumed from the glue
    assert (c1["double"] - c0["double"], c1["single"] - c0["single"]) == (2 * nd, 2 * ns)
    assert q1["attached"] - q0["attached"] > 0 and q1["stale"] == q0["stale"]
    assert s1["captures"] - s0["captures"] == nd + ns
    # replay pass: no Python block forward at all, every block replayed
    assert c2 == c1 and q2 == q1
    assert s2["replays"] - s1["replays"] == nd + ns and s2["captures"] == s1["captures"]


def _block0_inputs(e, seed=0):
    """Inference-tensor inputs for block 0 called on its own (1 window, 1024 + 256 tokens), as the painter's pass makes
    them: hidden / text streams, temb, rotary tables."""
    g = torch.Generator().manual_seed(seed)
    L, T = 1024, 256
    ids = torch.zeros(T + L, 3, device="cuda")
    ids[T:, 1], ids[T:, 2] = torch.arange(L, device="cuda") // 32, torch.arange(L, device="cuda") % 32
    with torch.inference_mode():
        return dict(hidden_states=torch.randn(1, L, H, generator=g).to("cuda", torch.bfloat16),
                    encoder_hidden_states=torch.randn(1, T, H, generator=g).to("cuda", torch.bfloat16),
                    temb=torch.randn(1, H, generator=g).to("cuda", torch.bfloat16),
                    image_rotary_emb=e.tr.pos_embed(ids))


def _block0(e, kw, graphs):
    e.runner.enabled = graphs
    with torch.inference_mode():
        out = e.tr.transformer_blocks[0](**kw)
        out = [t.clone() for t in out]                      # graph outputs are reused buffers: keep a copy
    torch.cuda.synchronize()
    return out


def test_graph_input_written_in_place_is_copied_again(eng):
    """The same input tensors written in place between calls (inference tensors: no version counter) are copied into
    the graph's static buffers again: the replay equals eager on the new values, not the old ones."""
    kw = _block0_inputs(eng, seed=21)
    _block0(eng, kw, graphs=True)                           # capture (or replay) with the first values
    with torch.inference_mode():
        for t in (kw["hidden_states"], kw["temb"], kw["image_rotary_emb"][0]):
            t.mul_(1.5)
    got = _block0(eng, kw, graphs=True)
    want = _block0(eng, kw, graphs=False)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_runner_refuses_another_thread(eng):
    """A block run through the GraphRunner from a thread other than the one that first ran it raises."""
    kw = _block0_inputs(eng, seed=22)
    _block0(eng, kw, graphs=True)
    err = []

    def other():
        try:
            with torch.inference_mode():
                eng.tr.transformer_blocks[0].forward(**kw)  # the runner's forward, past the slots' hooks
        except RuntimeError as e:
            err.append(str(e))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert err and "thread" in err[0]


def test_graph_shape_change(eng):
    """Review Focus 8, at the batch sizes Task 9 runs. Key A (batch 1, 1024 + 256), B (batch 3), its ragged last batch
    R (2: 8 windows at 3 are 3, 3, 2), C (batch 1, 1024 + 77), then A again; then (the small keys dropped: a 4 GB card
    holds a few large keys' buffers, not all of them) P (FLUX_BATCH: a panorama step's every window), W (5: a walk
    step's partial active set, the ragged batch at FLUX_BATCH), and P again. Each matches eager bit for bit, each new
    shape captures its own graphs, and A's and P's second runs replay the graphs captured for them."""
    from vecworld.config import FLUX_BATCH

    r = eng.runner
    n = len(eng.tr.transformer_blocks) + len(eng.tr.single_transformer_blocks)
    for runs in ([("A", 1, 256), ("B", 3, 256), ("R", 2, 256), ("C", 1, 77), ("A", 1, 256)],
                 [("P", FLUX_BATCH, 256), ("W", 5, 256), ("P", FLUX_BATCH, 256)]):
        r.clear()
        torch.cuda.empty_cache()
        caps = {}
        for name, b, t in runs:
            kw = _inputs(b, t, seed=b * 1000 + t)
            want = _fwd(eng, kw, graphs=False)
            before = r.stats.copy()
            got = _fwd(eng, kw, graphs=True)
            assert torch.equal(got, want), name
            caps.setdefault(name, []).append(r.stats["captures"] - before["captures"])
            assert (b, 1024, t) in r.keys()
            del kw, want, got
        first = runs[0][0]
        assert caps[first] == [n, 0], first                  # its second run reused its graphs
        for name in set(caps) - {first}:
            assert caps[name] == [n], name
    r.clear()


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_batched_windows_match_single_windows(eng, graphs):
    """Task 9: each window of a batched pass (FLUX_BATCH, and a ragged 2 and 5) equals that window run alone, bit for
    bit, on the eager slot path and on the graphs (the painter's rowwise outer layers; the blocks' kernels sum each
    row the same way whatever M). And it is that window's: far from any other window alone."""
    from vecworld.config import FLUX_BATCH

    eng.runner.clear()
    for b in sorted({FLUX_BATCH, 2, 5}):
        kw = _inputs(b, 256, seed=50 + b)
        out = _fwd(eng, kw, graphs).clone()
        refs = []
        for i in range(b):
            one = {k: (v[i:i + 1] if k in ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep")
                       else v) for k, v in kw.items()}
            refs.append(_fwd(eng, one, graphs=False).clone())
        for i in range(b):
            assert torch.equal(out[i:i + 1], refs[i]), (b, i)
            j = (i + 1) % b
            assert float((out[i:i + 1].float() - refs[j].float()).norm() / refs[j].float().norm()) > 0.1, (b, i)
        del kw, out, refs
    eng.runner.clear()


def test_graph_lru_evicts_beyond_max(eng):
    """Beyond max_graphs shape keys the least recently used key's graphs are dropped (and recaptured on its next
    use, still matching eager)."""
    r = eng.runner
    old = r.max_graphs
    try:
        r.max_graphs = 2
        for t in (256, 77, 100):
            _fwd(eng, _inputs(1, t, seed=t), graphs=True)
        assert len(r.keys()) == 2 and (1, 1024, 256) not in r.keys()
        kw = _inputs(1, 256, seed=5)
        assert torch.equal(_fwd(eng, kw, graphs=True), _fwd(eng, kw, graphs=False))
    finally:
        r.max_graphs = old


def test_graph_clear_then_recapture(eng):
    """Dropping every graph (clear(), or evicting the last key) releases the shared pool; the next capture starts a
    fresh one and still matches eager (capturing into the released pool's handle trips an allocator assert)."""
    kw = _inputs(1, 256, seed=11)
    want = _fwd(eng, kw, graphs=False)
    for _ in range(2):
        eng.runner.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        assert torch.equal(_fwd(eng, kw, graphs=True), want)


def test_graph_stop_between_steps(eng):
    """A job stopped between denoising steps (Stopped at check_stop, before a pass), then a second job: the second is
    correct (= the same job run eagerly) and allocated memory is the same after both jobs (graphs reused, nothing
    leaked by the stopped one)."""
    from vecworld.config import FLUX_WIN
    from vecworld.dream import gpu
    from vecworld.dream.gpu import Stopped
    from vecworld.flux.painter import FluxPainter

    passes = []
    p = FluxPainter.__new__(FluxPainter)
    p._setup(eng.tr, SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)), None,
             lambda d, n: passes.append(d), dev="cuda", dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(3)
    e = (torch.randn(1, 256, 4096, generator=g).to("cuda", torch.bfloat16),
         torch.randn(1, 768, generator=g).to("cuda", torch.bfloat16))
    grid = torch.randn(1, 64, FLUX_WIN, 2 * FLUX_WIN, generator=g).to("cuda")   # 4 windows: one pass of 4 a step

    def job():
        with torch.inference_mode():
            out = p._denoise(grid.clone(), p._sigmas(4), [e] * 4)
        torch.cuda.synchronize()
        return out

    eng.runner.enabled = False
    want = job()
    eng.runner.enabled = True
    job()                                                   # captures the batch-4 key
    torch.cuda.synchronize()
    mem0 = torch.cuda.memory_allocated()
    stop = threading.Event()
    passes.clear()
    p.progress = lambda d, n: (passes.append(d), d == 2 and stop.set())
    gpu._job.stop = stop
    stopped = False
    try:
        job()
    except Stopped:                                         # (not pytest.raises: its traceback would hold tensors)
        stopped = True
    finally:
        gpu._job.stop = None
    assert stopped and passes == [1, 2]
    torch.cuda.synchronize()
    mem1 = torch.cuda.memory_allocated()
    p.progress = None
    got = job()
    assert torch.equal(got, want)
    del got
    torch.cuda.synchronize()
    mem2 = torch.cuda.memory_allocated()
    print(f"\nallocated after: a full job {mem0 / 1e9:.3f} GB, the stopped job {mem1 / 1e9:.3f}, the next "
          f"{mem2 / 1e9:.3f}")
    assert mem0 == mem1 == mem2


def test_graph_peak_memory_at_flux_batch(eng):
    """Peak allocated < 3.6 GB at FLUX_BATCH windows a pass with graphs on (opt-in), the capture pass included."""
    from vecworld.config import FLUX_BATCH

    eng.runner.clear()                                       # drop earlier keys: the first FLUX_BATCH use captures
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _fwd(eng, _inputs(FLUX_BATCH, 256, seed=4), graphs=True)
    _fwd(eng, _inputs(FLUX_BATCH, 256, seed=4), graphs=True)
    peak = torch.cuda.max_memory_allocated()
    print(f"\npeak allocated at batch {FLUX_BATCH} with graphs: {peak / 1e9:.3f} GB")
    assert peak < 3.6e9
    # a walk's worst case: the FLUX_GRAPH_MAX largest batch sizes all kept (each key holds its own buffers)
    for b in range(FLUX_BATCH - 1, max(FLUX_BATCH - eng.runner.max_graphs, 0), -1):
        _fwd(eng, _inputs(b, 256, seed=b), graphs=True)
    peak = torch.cuda.max_memory_allocated()
    print(f"peak allocated with keys {eng.runner.keys()}: {peak / 1e9:.3f} GB")
    assert peak < 3.6e9
    eng.runner.clear()


def test_slots_match_offload(eng):
    """Block outputs with BlockSlots equal those with diffusers' group offload (Task 6's path), bit for bit: the whole
    transformer forward, eager and graphs, against group offload on the same weights. Runs last: it hands the
    transformer to group offload."""
    kw = _inputs(2, 256, seed=7)
    ours_eager = _fwd(eng, kw, graphs=False)
    ours_graph = _fwd(eng, kw, graphs=True)
    eng.runner.close()
    eng.slots.close()                                        # host tensors back in the blocks, hooks gone
    tr = eng.tr
    for name, child in tr.named_children():
        if name not in ("transformer_blocks", "single_transformer_blocks"):
            child.to("cpu")
    tr.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                            offload_type="block_level", num_blocks_per_group=1, use_stream=True)
    with torch.inference_mode():
        ref = tr(**kw)[0]
    torch.cuda.synchronize()
    assert torch.equal(ours_eager, ref) and torch.equal(ours_graph, ref)
