"""The first-block cache (Task 8, docs/engine/08-block-cache.md): BlockCache on its own and in the painter's pass loop
on a small fake FLUX (CPU), then on the real transformer (gpu, run with the game server stopped)."""
import math
import threading
from types import SimpleNamespace

import pytest
import torch

from vecworld.config import FLUX_BATCH, FLUX_STEPS, FLUX_STRIDE, FLUX_WIN
from vecworld.dream.pano import windows
from vecworld.dream import gpu
from vecworld.dream.gpu import Stopped
from vecworld.engine.cache import BlockCache, cached_forward
from vecworld.flux.painter import FluxPainter

H = 8


def _r(seed, shape=(1, 4, H)):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# --- BlockCache -------------------------------------------------------------------------------------------------------

def test_cache_never_skips_first_step():
    c = BlockCache(10.0)
    c.begin_job(1)
    k, r0 = (1, (0, 1)), _r(0)
    assert c.probe(k, r0, 0) is False                      # step 0: compute
    c.store(k, _r(1))
    assert c.probe(k, r0, 1) is True                       # no change at all: skip
    assert c.probe(k, r0, 0) is False                      # step 0 never skips, whatever is stored
    c.begin_job(2)
    assert c.probe((2, (0, 1)), r0, 0) is False


def test_cache_threshold_zero_never_skips():
    c = BlockCache(0.0)
    c.begin_job(1)
    k, r0 = (1, (0,)), _r(0)
    for step in range(4):
        assert c.probe(k, r0, step) is False
        c.store(k, r0)
    assert not c.active and c.stats["probes"] == 0         # off: no probe work, no host sync
    with pytest.raises(ValueError):
        BlockCache(-0.1)


def test_cache_reset_per_job():
    """Review Focus 7: begin_job drops the last job's entries, and a key of another job is refused (never served)."""
    c = BlockCache(10.0)
    c.begin_job("a")
    ka = ("a", (0, 1, 2, 3))
    c.probe(ka, _r(0), 0)
    c.store(ka, _r(1))
    assert len(c) == 1
    c.begin_job("b")
    assert len(c) == 0
    with pytest.raises(ValueError, match="current job"):
        c.probe(ka, _r(0), 1)
    kb = ("b", (0, 1, 2, 3))
    assert c.probe(kb, _r(0), 1) is False                  # same windows, new job: cold


def test_cache_keyed_by_windows():
    """A batch with a different window set never uses another batch's residual."""
    c = BlockCache(10.0)
    c.begin_job(1)
    a, b = (1, (0, 1, 2, 3)), (1, (4, 5, 6, 7))
    r0 = _r(0)
    c.probe(a, r0, 0)
    c.store(a, _r(1))
    assert c.probe(b, r0, 1) is False                      # b has no residual of its own: compute
    with pytest.raises(KeyError):
        c.apply(b, r0)
    assert c.probe(a, r0, 1) is True
    assert torch.equal(c.apply(a, r0), r0 + _r(1))
    assert c.probe((1, (0, 1, 2)), r0, 2) is False        # a subset of a's windows is another batch


def test_cache_stop_clears():
    c = BlockCache(10.0)
    c.begin_job(1)
    c.probe((1, (0,)), _r(0), 0)
    c.store((1, (0,)), _r(1))
    c.clear()
    assert len(c) == 0 and c.job is None
    with pytest.raises(ValueError):
        c.probe((1, (0,)), _r(0), 1)


def test_cache_rel_is_per_job_and_bounded():
    """Task 8 review M1: the probe's relative changes (rel) are the current job's only, so a long-lived painter with the
    cache on doesn't grow the list forever; begin_job starts it again."""
    c = BlockCache(10.0)
    for job in range(50):
        c.begin_job(job)
        k = (job, (0,))
        for step in range(4):
            c.probe(k, _r(step), step)
            c.store(k, _r(9))
        assert len(c.rel) == 3                             # steps 1..3 of this job, none of the jobs before
    c.clear()                                              # a job's end keeps its rel for the bench to read
    assert len(c.rel) == 3


def test_cache_accumulates_change():
    """Two small changes, each under the threshold, that sum above it force a recompute; then the sum starts again."""
    c = BlockCache(0.1)
    c.begin_job(1)
    k = (1, (0,))
    base = torch.ones(1, 4, H)
    c.probe(k, base, 0)
    c.store(k, _r(1))
    r1 = base * 1.06                                       # |r1 - r0| / |r0| = 0.06
    assert c.probe(k, r1, 1) is True
    r2 = r1 * 1.06                                         # another 0.06: the sum 0.12 > 0.1
    assert c.probe(k, r2, 2) is False
    assert c.probe(k, r2 * 1.06, 3) is True                # summed from 0 again after the computed pass
    assert c.rel == pytest.approx([0.06, 0.06, 0.06], abs=1e-5)


# --- the pass loop on a fake FLUX (CPU) -------------------------------------------------------------------------------

class _Block(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        g = torch.Generator().manual_seed(100 + i)
        self.w = torch.nn.Parameter(torch.randn(H, H, generator=g) / H ** 0.5, requires_grad=False)
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        self.calls += 1
        t = temb[:, None, :]
        return (encoder_hidden_states + 0.1 * torch.tanh(encoder_hidden_states @ self.w + t),
                hidden_states + 0.1 * torch.tanh(hidden_states @ self.w + t))


class FakeFlux(torch.nn.Module):
    """FluxTransformer2DModel's structure in miniature (hidden size 8; 2 double and 3 single blocks)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.x_embedder = torch.nn.Linear(64, H)
        self.context_embedder = torch.nn.Linear(8, H)
        self.pool = torch.nn.Linear(8, H)
        self.proj_out = torch.nn.Linear(H, 64)
        self.transformer_blocks = torch.nn.ModuleList([_Block(i) for i in range(2)])
        self.single_transformer_blocks = torch.nn.ModuleList([_Block(10 + i) for i in range(3)])

    def time_text_embed(self, timestep, pooled):
        return self.pool(pooled) + timestep[:, None] / 1000

    def pos_embed(self, ids):
        return ids.cos(), ids.sin()

    def norm_out(self, h, temb):
        return h * (1 + 0.01 * temb[:, None, :])

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids,
                guidance=None, return_dict=False):
        """The diffusers loop, plainly."""
        hs = self.x_embedder(hidden_states)
        temb = self.time_text_embed(timestep.to(hs.dtype) * 1000, pooled_projections)
        es = self.context_embedder(encoder_hidden_states)
        rot = self.pos_embed(torch.cat((txt_ids, img_ids), 0))
        for b in [*self.transformer_blocks, *self.single_transformer_blocks]:
            es, hs = b(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot)
        return (self.proj_out(self.norm_out(hs, temb)),)

    def block_calls(self):
        return [b.calls for b in (*self.transformer_blocks, *self.single_transformer_blocks)]


class _Vae:
    config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)


def _painter(threshold, tr=None):
    p = FluxPainter.__new__(FluxPainter)
    p.cache_threshold = threshold
    p._setup(tr or FakeFlux(), _Vae(), None, None, dev="cpu", dtype=torch.float32)
    return p


def _job(p, n_win=8, steps=FLUX_STEPS, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.randn(1, 64, FLUX_WIN, n_win * FLUX_WIN // 2, generator=g)
    e = (torch.randn(1, 4, 8, generator=g), torch.randn(1, 8, generator=g))
    with torch.inference_mode():
        return p._denoise(grid, p._sigmas(steps), [e] * n_win)


def test_cached_forward_computing_equals_the_plain_loop():
    tr = FakeFlux()
    c = BlockCache(1.0)
    c.begin_job(1)
    g = torch.Generator().manual_seed(1)
    kw = dict(hidden_states=torch.randn(2, 16, 64, generator=g),
              encoder_hidden_states=torch.randn(2, 4, 8, generator=g),
              pooled_projections=torch.randn(2, 8, generator=g), timestep=torch.full((2,), 0.75),
              img_ids=torch.randn(16, 3, generator=g), txt_ids=torch.zeros(4, 3))
    want = tr(**kw)[0]
    assert torch.equal(cached_forward(tr, c, (1, (0, 1)), 0, **kw)[0], want)
    assert torch.equal(cached_forward(tr, BlockCache(0.0), (1, (0, 1)), 0, **kw)[0], want)


class _HoldSlots:
    """BlockSlots' hold bookkeeping without a GPU: block 0's forward hook holds its slot's next user (block 3) while 0
    is in hold, as BlockSlots.after does; release issues it."""

    def __init__(self, tr):
        self.hold, self._held, self.loads = set(), {}, []
        tr.transformer_blocks[0].register_forward_hook(lambda m, a, o: self._after())

    def _after(self):
        if 0 in self.hold:
            self._held[0] = 3
        else:
            self.loads.append(3)

    def release(self, i):
        j = self._held.pop(i, None)
        if j is not None:
            self.loads.append(j)

    def unhold(self):
        from vecworld.engine.slots import BlockSlots

        BlockSlots.unhold(self)


def test_cached_forward_leaves_nothing_held_after_a_skip_or_an_error():
    """Task 8 review M2: after a skipped pass (block 0 only, its prefetch held back) and after a pass that raised, the
    slots hold nothing: a later release(0) issues no stray copy into block 0's slot."""
    tr = FakeFlux()
    slots = _HoldSlots(tr)
    c = BlockCache(100.0)
    c.begin_job(1)
    g = torch.Generator().manual_seed(2)
    kw = dict(hidden_states=torch.randn(1, 16, 64, generator=g), encoder_hidden_states=torch.randn(1, 4, 8, generator=g),
              pooled_projections=torch.randn(1, 8, generator=g), timestep=torch.full((1,), 0.75),
              img_ids=torch.randn(16, 3, generator=g), txt_ids=torch.zeros(4, 3))
    cached_forward(tr, c, (1, (0,)), 0, slots=slots, **kw)          # computed: block 0's prefetch released
    assert slots.loads == [3] and slots._held == {} and slots.hold == set()
    cached_forward(tr, c, (1, (0,)), 1, slots=slots, **kw)          # skipped: nothing copied, nothing left held
    assert c.stats["skips"] == 1
    assert slots.loads == [3] and slots._held == {} and slots.hold == set()
    slots.release(0)
    assert slots.loads == [3]
    c.probe = lambda *a: (_ for _ in ()).throw(RuntimeError("probe failed"))
    with pytest.raises(RuntimeError):
        cached_forward(tr, c, (1, (0,)), 2, slots=slots, **kw)      # raised after block 0 held its prefetch
    assert slots._held == {} and slots.hold == set()


def test_painter_cache_threshold_zero_is_the_plain_transformer():
    """threshold 0: the painter's job equals the job without any cache, and every pass runs every block."""
    off, plain = _painter(0.0), _painter(0.0)
    plain.cache = None                                     # no cache object at all: the transformer called as is
    a, b = _job(off), _job(plain)
    assert torch.equal(a, b)
    passes = FLUX_STEPS * math.ceil(8 / FLUX_BATCH)
    assert off.transformer.block_calls() == [passes] * 5
    assert off.last_cache == {"passes": passes, "skips": 0}


def test_painter_skips_at_a_high_threshold_but_never_at_step_0():
    p = _painter(100.0)
    out = _job(p)
    per_step = math.ceil(8 / FLUX_BATCH)
    passes = FLUX_STEPS * per_step
    assert p.last_cache == {"passes": passes, "skips": passes - per_step}   # all but each batch's step 0
    calls = p.transformer.block_calls()
    assert calls[0] == passes and calls[1:] == [per_step] * 4
    assert torch.isfinite(out).all()
    assert len(p.cache) == 0                              # the job's entries are dropped when it ends


def test_painter_keys_batches_by_job_and_windows():
    """Every probe's key is (this job, the batch's window ids), and a new job gets a new id."""
    p = _painter(100.0)
    seen = []
    probe = p.cache.probe
    p.cache.probe = lambda key, r0, step: seen.append((key, step)) or probe(key, r0, step)
    _job(p)
    jobs = {k[0] for k, _ in seen}
    assert len(jobs) == 1
    assert {k[1] for k, _ in seen} == {tuple(range(i, min(i + FLUX_BATCH, 8))) for i in range(0, 8, FLUX_BATCH)}
    assert [s for _, s in seen] == [k for k in range(FLUX_STEPS) for _ in range(math.ceil(8 / FLUX_BATCH))]
    _job(p)
    assert len({k[0] for k, _ in seen}) == 2


@pytest.mark.parametrize("batch", [3, 8])
def test_painter_batches_never_reuse_another_compositions_residual(batch):
    """Task 9, cache forced on: with the windows batched (ragged at 3; all of a step's active windows at 8) and the
    active set changing from step to step, a pass only ever skips with the residual stored under its own exact window
    tuple, of its own batch size; a batch whose composition changed computes."""
    p = _painter(100.0)
    p.batch = batch
    probes, stores, applied = [], [], []
    probe, store, apply = p.cache.probe, p.cache.store, p.cache.apply
    p.cache.probe = lambda key, r0, step: probes.append((key, step, probe(key, r0, step))) or probes[-1][2]
    p.cache.store = lambda key, res: (stores.append((key, res.shape[0])), store(key, res))[1]
    p.cache.apply = lambda key, x: (applied.append((key, x.shape[0], p.cache._e[key].residual.shape[0])),
                                    apply(key, x))[1]
    g = torch.Generator().manual_seed(4)
    w = 8 * FLUX_WIN // 2
    grid, orig, noise = (torch.randn(1, 64, FLUX_WIN, w, generator=g) for _ in range(3))
    e = [(torch.randn(1, 4, 8, generator=g), torch.randn(1, 8, generator=g)) for _ in range(8)]
    rw = [torch.zeros(1, 1, 1, w) for _ in range(5)]
    rw[1][..., :w // 2] = 1.0                              # step 0: windows 0..2 idle (reset again at step 1)
    rw[3][..., w // 4:] = 1.0                              # step 2: only windows 0, 1, 7
    with torch.inference_mode():
        out = p._denoise(grid, p._sigmas(4), e, keep=(orig, noise, rw))
    assert torch.isfinite(out).all()
    sets = [{k for k, s, _ in probes if s == step} for step in range(4)]
    assert sets[0] != sets[1] != sets[2]                    # the composition does change
    skipped = [(k, s) for k, s, skip in probes if skip]
    assert skipped and len(applied) == len(skipped)
    for key, step in skipped:
        assert key in sets[step]
        assert any(k == key for k, _ in stores), key        # stored under this very key, at an earlier step
    for key, n, rn in applied:
        assert n == rn == len(key[1])                       # its own batch's residual, window for window
    for step in (1, 2):                                     # a window tuple new at this step computes
        for key in sets[step] - set().union(*sets[:step]):
            assert (key, step, False) in probes


def test_painter_stop_clears_the_cache_and_the_next_job_starts_cold():
    p = _painter(100.0)
    stop = threading.Event()
    n = []
    p.progress = lambda d, t: (n.append(d), d == 3 and stop.set())
    gpu._job.stop = stop
    try:
        with pytest.raises(Stopped):
            _job(p)
    finally:
        gpu._job.stop = None
    assert len(p.cache) == 0 and p.cache.job is None
    p.progress = None
    before = p.transformer.block_calls()[1]
    _job(p)
    per_step = math.ceil(8 / FLUX_BATCH)
    assert p.transformer.block_calls()[1] - before == per_step          # only the cold step 0 computes block 1
    assert p.last_cache["skips"] == (FLUX_STEPS - 1) * per_step


def test_engine_cache_threshold_rule(monkeypatch):
    """VECWORLD_CACHE_THRESHOLD (engine.engine_cache_threshold): unset: config.FLUX_CACHE_THRESHOLD; a number >= 0
    otherwise; under VECWORLD_KERNEL=off (kernel None: diffusers' group offload) always 0."""
    from vecworld.config import FLUX_CACHE_THRESHOLD
    from vecworld.engine import engine_cache_threshold

    monkeypatch.delenv("VECWORLD_CACHE_THRESHOLD", raising=False)
    assert engine_cache_threshold("w4a8") == FLUX_CACHE_THRESHOLD
    assert engine_cache_threshold(None) == 0.0
    monkeypatch.setenv("VECWORLD_CACHE_THRESHOLD", "0.2")
    assert engine_cache_threshold("w4a8") == 0.2 and engine_cache_threshold(None) == 0.0
    monkeypatch.setenv("VECWORLD_CACHE_THRESHOLD", "0")
    assert engine_cache_threshold("w4a8") == 0.0
    for bad in ("-1", "lots"):
        monkeypatch.setenv("VECWORLD_CACHE_THRESHOLD", bad)
        with pytest.raises(ValueError):
            engine_cache_threshold("w4a8")


# --- slots: a skipped pass copies nothing (gpu, small blocks) ---------------------------------------------------------

@pytest.mark.gpu
def test_slots_hold_defers_block0_prefetch_until_released():
    """With hold = {0}, block 0's after-hook does not prefetch its slot's next user; release(0) does. A skipped pass
    (block 0 only) copies nothing, and a full pass after it is still right."""
    from vecworld.engine.slots import BlockSlots

    from test_graphs import _toys

    blocks, host = _toys(7)
    ref = [type(b)(i).cuda() for i, b in enumerate(blocks)]
    s = BlockSlots(blocks, lambda i: host[i], ring=3)
    s.hold = {0}
    x = torch.randn(4, 64, device="cuda").bfloat16()

    def full():
        y = blocks[0](x)
        s.release(0)
        for b in blocks[1:]:
            y = b(y)
        return y

    want = x
    for b in ref:
        want = b(want)
    assert torch.equal(full(), want)
    c0 = s.copies
    blocks[0](x)                                           # a skipped pass: block 0 only, no release
    assert s.copies == c0 and s.loaded[0] == 0
    c1 = s.copies
    assert torch.equal(full(), want)
    assert s.copies - c1 == len(blocks)                    # every block once: 3..6 now, 0..2 for the next pass
    blocks[0](x)
    assert torch.equal(full(), want)
    blocks[0](x)                                           # a skipped pass leaves block 0's prefetch held ...
    assert s._held == {0: 3}
    s.unhold()                                             # ... until unhold drops it (and the hold set)
    assert s._held == {} and s.hold == set()
    c2 = s.copies
    s.release(0)                                           # nothing held: no stray copy into block 0's slot
    assert s.copies == c2 and s.loaded[0] == 0
    torch.cuda.synchronize()
    s.close()


# --- the real transformer (gpu) ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eng():
    """The transformer as the painter runs it (w4a8 + keep-list, fused), blocks on BlockSlots with a GraphRunner."""
    from test_graphs import _bench, _trim

    from vecworld.engine.graphs import stream_blocks
    from vecworld.flux.painter import _pin_in_place, _unpin

    with pytest.MonkeyPatch.context() as mp:
        for k in ("VECWORLD_KERNEL", "VECWORLD_FUSE", "VECWORLD_GRAPHS", "VECWORLD_CACHE_THRESHOLD"):
            mp.delenv(k, raising=False)
        tr, _ = _bench().load_transformer()
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


def _kw(b=1, seed=0, s=0.75):
    from test_graphs import _inputs

    kw = _inputs(b, 256, seed=seed)
    kw["timestep"] = torch.full((b,), s, device="cuda", dtype=torch.bfloat16)
    kw.pop("return_dict")
    return kw


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_cached_forward_matches_diffusers(eng, graphs):
    """A computing pass of the cache's loop equals diffusers' forward bit for bit, eager and with graphs."""
    eng.runner.enabled = graphs
    kw = _kw(1, seed=2)
    with torch.inference_mode():
        want = eng.tr(**kw, return_dict=False)[0]
        c = BlockCache(1.0)
        c.begin_job(1)
        got = cached_forward(eng.tr, c, (1, (0,)), 0, slots=eng.slots, **kw)[0]
    torch.cuda.synchronize()
    assert eng.slots.hold == set()                         # held only during a cached pass
    assert torch.equal(got, want)


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_full_skip_full_matches_no_cache(eng, graphs):
    """Full, skip, full: the skipped pass runs block 0 only and copies no weights; the full pass after it equals the
    pass without a cache (the slot ring and its events stay right), eager and with graphs."""
    eng.runner.enabled = graphs
    k1, k2 = _kw(1, seed=3, s=0.75), _kw(1, seed=4, s=0.5)
    with torch.inference_mode():
        want = eng.tr(**k2, return_dict=False)[0].clone()
        c = BlockCache(1e6)
        c.begin_job(7)
        cached_forward(eng.tr, c, (7, (0,)), 0, slots=eng.slots, **k1)                 # full
        torch.cuda.synchronize()
        n0 = eng.slots.copies
        skipped = cached_forward(eng.tr, c, (7, (0,)), 1, slots=eng.slots, **k1)[0]    # same input: skips
        torch.cuda.synchronize()
        assert c.stats["skips"] == 1 and eng.slots.copies == n0
        got = cached_forward(eng.tr, c, (7, (1,)), 1, slots=eng.slots, **k2)[0]        # new batch: full
    torch.cuda.synchronize()
    assert torch.isfinite(skipped).all()
    assert torch.equal(got, want)


@pytest.mark.gpu
def test_cache_skips_on_flux(eng):
    """At a high threshold, at least one pass of a 4-step 512 job skips; at threshold 0 none does and the
    job equals the uncached one."""
    eng.runner.enabled = False
    p = FluxPainter.__new__(FluxPainter)
    p._slots, p._runner = eng.slots, eng.runner
    p.cache_threshold = 0.0
    p._setup(eng.tr, SimpleNamespace(config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)), None, None,
             dev="cuda", dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(5)
    e = (torch.randn(1, 256, 4096, generator=g).to("cuda", torch.bfloat16),
         torch.randn(1, 768, generator=g).to("cuda", torch.bfloat16))
    grid = torch.randn(1, 64, FLUX_WIN, FLUX_WIN, generator=g).to("cuda")    # 512 x 512: its windows in one pass
    n = len(windows(FLUX_WIN, FLUX_WIN, FLUX_WIN, FLUX_STRIDE))

    def job():
        with torch.inference_mode():
            out = p._denoise(grid.clone(), p._sigmas(4), [e] * n)
        torch.cuda.synchronize()
        return out

    want = job()
    assert p.last_cache == {"passes": 4, "skips": 0}
    p.cache = BlockCache(5.0)
    got = job()
    print(f"\nrelative r0 changes per step: {[round(x, 3) for x in p.cache.rel]}; {p.last_cache}")
    assert p.last_cache["skips"] >= 1
    assert torch.isfinite(got).all() and not torch.equal(got, want)
