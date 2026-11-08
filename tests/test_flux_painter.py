"""FluxPainter's MultiDiffusion loop on fakes (CPU, no models): windows, progress and stopping."""
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vecworld.config import FLUX_BATCH, FLUX_STEPS, FLUX_STRIDE, FLUX_WIN, PANO_H, PANO_W
from vecworld.dream import gpu
from vecworld.dream.gpu import Stopped
from vecworld.flux.painter import FluxPainter

GH, GW = PANO_H // 16, PANO_W // 16          # the packed token grid
N_WIN = GW // FLUX_STRIDE


class FakeTransformer:
    """Predicts zero velocity; records every call's packed tokens. on_call(n) runs after the n-th call."""

    def __init__(self, on_call=None):
        self.calls, self.on_call = [], on_call

    def __call__(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids,
                 guidance=None, return_dict=False):
        assert hidden_states.shape[1:] == (FLUX_WIN * FLUX_WIN, 64)
        assert encoder_hidden_states.shape[0] == pooled_projections.shape[0] == hidden_states.shape[0]
        assert img_ids.shape == (FLUX_WIN * FLUX_WIN, 3)
        self.calls.append(hidden_states.detach().clone())
        if self.on_call:
            self.on_call(len(self.calls))
        return (torch.zeros_like(hidden_states),)


class FakeVae:
    """Decode: the first 3 latent channels nearest-upsampled; encode: 8x8 means (16 channels)."""
    config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, z):
        return SimpleNamespace(sample=z[:, :3].repeat_interleave(8, -2).repeat_interleave(8, -1))

    def encode(self, x):
        lat = torch.nn.functional.avg_pool2d(x[:, :1], 8).expand(-1, 16, -1, -1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=lat))


def fake_painter(transformer=None, progress=None, recipe="t1"):
    p = FluxPainter.__new__(FluxPainter)
    p._setup(transformer or FakeTransformer(), FakeVae(),
             lambda text: (torch.zeros(1, 4, 8, dtype=torch.bfloat16), torch.zeros(1, 8, dtype=torch.bfloat16)),
             progress, dev="cpu", dtype=torch.float32)
    p.recipe = recipe
    return p


class TimedTransformer(FakeTransformer):
    """Also records each call's sigma."""

    def __init__(self):
        super().__init__()
        self.sigmas = []

    def __call__(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, *a, **k):
        self.sigmas.append(round(float(timestep[0]), 4))
        return super().__call__(hidden_states, encoder_hidden_states, pooled_projections, timestep, *a, **k)


def test_windows_wrap_and_cover():
    tr = FakeTransformer()
    p = fake_painter(tr)
    grid = torch.zeros(1, 64, GH, GW)
    grid[:, 0] = torch.arange(GW, dtype=torch.float32)          # channel 0 holds each token's column
    emb = p._embed([("x", 1.0)])
    p._denoise(grid, p._sigmas(1), [emb] * N_WIN)
    seen, wrapped = set(), False
    for call in tr.calls:
        for tile in call:
            cols = tile[:, 0].view(FLUX_WIN, FLUX_WIN)[0].round().long().tolist()   # one row of the window
            seen.update(cols)
            wrapped |= cols[0] > cols[-1]                         # runs past the right edge back to column 0
            assert all((b - a) % GW == 1 for a, b in zip(cols, cols[1:]))
    assert seen == set(range(GW)), "every packed column is covered"
    assert wrapped, "a window wraps across the seam"
    assert sum(len(c) for c in tr.calls) == N_WIN


def test_progress_called_per_step():
    got = []
    p = fake_painter(progress=lambda k, total: got.append((k, total)))
    img = p.panorama([("a forest", 0.5), ("a lake", 0.5)], seed=3)
    assert img.size == (PANO_W, PANO_H)
    total = FLUX_STEPS * math.ceil(N_WIN / FLUX_BATCH)
    assert got == [(k, total) for k in range(1, total + 1)]


def test_progress_counts_only_active_windows_in_a_step():
    got = []
    tr = FakeTransformer()
    p = fake_painter(tr, progress=lambda k, total: got.append((k, total)))
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    p.step(prev, [("a forest", 1.0)], None, 0.5, seed=4, bg_strength=0.15)
    total = got[-1][1]
    assert got == [(k, total) for k in range(1, total + 1)]
    assert len(tr.calls) == total
    sig, _ = p._reset_weights(np.zeros(GW, np.float32), 0.5, 0.15)
    assert sum(len(c) for c in tr.calls) < (len(sig) - 1) * N_WIN     # background windows skipped early


class WindowwiseTransformer(FakeTransformer):
    """A velocity that depends on each window's own tokens, text and sigma only (elementwise, so exact in any batch):
    batching must not change any window's output. Records each pass's batch size."""

    def __call__(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, *a, **k):
        super().__call__(hidden_states, encoder_hidden_states, pooled_projections, timestep, *a, **k)
        t = encoder_hidden_states.float().mean((1, 2))[:, None, None] + pooled_projections.float().mean(1)[:, None, None]
        return (torch.tanh(hidden_states * 1.5 + t) - timestep[:, None, None] * hidden_states,)


def _distinct_embs(n):
    """One (t5, pooled) per window, all different (the step's focus / base split, exaggerated)."""
    return [(torch.full((1, 4, 8), 0.1 * i, dtype=torch.bfloat16), torch.full((1, 8), -0.05 * i, dtype=torch.bfloat16))
            for i in range(n)]


@pytest.mark.parametrize("batch", [1, 2, 3, 5, 8, 16])
def test_batched_equals_unbatched(batch):
    """Task 9: the same window outputs whatever the batch size, in a panorama and in a walk step (whose active
    windows change from step to step), ragged last batches included (8 windows at 3: 3, 3, 2), with each window's own
    text embedding stacked in its place."""
    g = torch.Generator().manual_seed(5)
    grid = torch.randn(1, 64, GH, GW, generator=g)
    embs = _distinct_embs(N_WIN)
    orig, noise = torch.randn(1, 64, GH, GW, generator=g), torch.randn(1, 64, GH, GW, generator=g)
    rw = [torch.zeros(1, 1, 1, GW) for _ in range(5)]
    rw[1][..., :GW // 2] = 1.0                              # the left half reset again at step 1: windows idle at 0
    rw[3][..., GW // 4:] = 1.0                              # most of it at step 3: only the windows over 0..1/4
    outs, sizes = {}, {}
    for b in (1, batch):
        tr = WindowwiseTransformer()
        p = fake_painter(tr)
        p.batch = b
        pano = p._denoise(grid.clone(), p._sigmas(FLUX_STEPS), embs)
        step = p._denoise(grid.clone(), p._sigmas(4), embs, keep=(orig, noise, rw))
        outs[b], sizes[b] = (pano, step), [c.shape[0] for c in tr.calls]
    assert torch.equal(outs[batch][0], outs[1][0]) and torch.equal(outs[batch][1], outs[1][1])
    assert max(sizes[batch]) == min(batch, N_WIN)
    assert sum(sizes[batch]) == sum(sizes[1])               # the same window passes, fewer transformer passes


def test_every_active_window_of_a_step_in_as_few_passes_as_the_batch_allows():
    """Each step's active windows go through the transformer as ceil(active / batch) passes of up to `batch` windows
    (the painter's batch: FLUX_BATCH unless VECWORLD_BATCH says otherwise); a panorama's 8 windows at FLUX_BATCH."""
    tr = TimedTransformer()
    p = fake_painter(tr)
    assert p.batch == FLUX_BATCH
    p.panorama([("a forest", 1.0)], seed=3)
    assert [c.shape[0] for c in tr.calls] == [min(FLUX_BATCH, N_WIN)] * (FLUX_STEPS * math.ceil(N_WIN / FLUX_BATCH))
    for b in (3, N_WIN):
        tr.calls.clear(), tr.sigmas.clear()
        p.batch = b
        p.step(Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60)), [("a forest", 1.0)], None, 0.5, seed=4)
        per_sigma = {}
        for s, c in zip(tr.sigmas, tr.calls):
            per_sigma.setdefault(s, []).append(c.shape[0])
        for s, sizes in per_sigma.items():
            n = sum(sizes)
            assert sizes == [b] * (n // b) + ([n % b] if n % b else []), (b, s, sizes)


class _RecLayer(torch.nn.Module):
    """A layer that records each call's batch and adds its y (per row) and its bias (shared, not batched)."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, y=None, bias=None, scale=1.0):
        self.seen.append(x.shape[0])
        out = x * 2 + (0 if y is None else y.sum(-1, keepdim=True))
        return out * scale + (0 if bias is None else bias)


def _outer_layers():
    from vecworld.flux.painter import ROWWISE

    tr = torch.nn.Module()
    tr.norm_out = torch.nn.Module()
    for name in ROWWISE:
        parent, _, leaf = name.rpartition(".")
        setattr(tr.get_submodule(parent), leaf, _RecLayer())
    return tr


def test_rowwise_runs_the_outer_layers_a_window_at_a_time():
    """Task 9: cuBLAS picks its bf16 GEMM by M, so the transformer's layers outside the blocks (x / context embedders,
    the time-text embedder, the output norm's linear and the projection out, M = batch or batch x tokens) round a window
    differently in a batch than alone, and under w4a8 that one ulp became a different sample (25.7 dB). rowwise() runs
    each a window at a time: the same rows, the same result, whatever the batch."""
    from vecworld.flux.painter import ROWWISE, rowwise

    tr = _outer_layers()
    rowwise(tr)
    x, y = torch.randn(3, 5), torch.randn(3, 4)
    for name in ROWWISE:
        m = tr.get_submodule(name)
        m.seen.clear()
        got = m(x, y)
        assert torch.equal(got, x * 2 + y.sum(-1, keepdim=True)) and m.seen == [1, 1, 1], name
    rowwise(tr)                                              # twice: still one level of rows
    m = tr.get_submodule(ROWWISE[0])
    m.seen.clear()
    m(x)
    assert m.seen == [1, 1, 1]


def test_rowwise_splits_only_batched_tensors_and_takes_keywords():
    """Task 9 review M3 (a, b): keyword arguments are split like positional ones; a tensor whose dim 0 isn't the
    batch (a shared bias) and a non-tensor (a scale) pass through whole to every row."""
    from vecworld.flux.painter import rowwise

    tr = _outer_layers()
    rowwise(tr)
    m = tr.x_embedder
    x, y, bias = torch.randn(3, 5), torch.randn(3, 4), torch.randn(5)
    want = (x * 2 + y.sum(-1, keepdim=True)) * 0.5 + bias
    m.seen.clear()
    assert torch.equal(m(x, y=y, bias=bias, scale=0.5), want) and m.seen == [1, 1, 1]
    m.seen.clear()
    assert torch.equal(m(x=x, y=y, bias=bias, scale=0.5), want) and m.seen == [1, 1, 1]   # the batch from a keyword


def test_rowwise_is_safe_twice_after_a_rewrap_and_unwraps():
    """Task 9 review M3 (c): rowwise() marks the module, so a second call after something else rewrapped the forward
    (a hook registry) doesn't wrap again; rowwise(tr, False) puts the original forward back."""
    from vecworld.flux.painter import ROWWISE, rowwise

    tr = _outer_layers()
    rowwise(tr)
    m = tr.proj_out
    inner = m.forward
    m.forward = lambda *a, **k: inner(*a, **k)               # a hook rewraps it
    rowwise(tr)
    x = torch.randn(4, 5)
    m.seen.clear()
    m(x)
    assert m.seen == [1, 1, 1, 1]                            # one level of rows, not rows of rows
    tr2 = _outer_layers()
    rowwise(tr2)
    rowwise(tr2, False)
    for name in ROWWISE:
        assert "forward" not in tr2.get_submodule(name).__dict__, name
        layer = tr2.get_submodule(name)
        layer.seen.clear()
        layer(x)
        assert layer.seen == [4]


def test_rowwise_keeps_no_reference_cycle():
    """Task 9 review M3 (d): the wrapper holds its module weakly, so a wrapped layer is freed by refcount alone."""
    import gc
    import weakref

    from vecworld.flux.painter import rowwise

    gc.disable()
    try:
        tr = _outer_layers()
        rowwise(tr)
        ref = weakref.ref(tr.proj_out)
        del tr
        assert ref() is None
    finally:
        gc.enable()


def test_engine_batch_rule(monkeypatch):
    """VECWORLD_BATCH (vecworld.engine.engine_batch, one rule for the painter, the walk and the bench): unset or empty,
    config.FLUX_BATCH; otherwise a whole number >= 1, under the engine kernels. Under VECWORLD_KERNEL=off (kernel None)
    always 1, whatever FLUX_BATCH or VECWORLD_BATCH say: diffusers' cuBLAS GEMMs round by M inside the blocks too, so
    the reference is pinned to one window a pass, its batch-invariant picture (Task 9 review I1). Anything else is
    refused."""
    from vecworld.engine import engine_batch

    monkeypatch.delenv("VECWORLD_BATCH", raising=False)
    assert engine_batch() == engine_batch("w4a8") == engine_batch("mma") == FLUX_BATCH
    assert engine_batch(None) == 1
    for value, want in (("", FLUX_BATCH), ("1", 1), ("6", 6), (" 8 ", 8)):
        monkeypatch.setenv("VECWORLD_BATCH", value)
        assert engine_batch("w4a8") == want, value
        assert engine_batch(None) == 1, value
    for bad in ("0", "-2", "2.5", "lots"):
        monkeypatch.setenv("VECWORLD_BATCH", bad)
        with pytest.raises(ValueError, match="VECWORLD_BATCH"):
            engine_batch("w4a8")


def test_painter_batch_follows_the_kernel(monkeypatch):
    """FluxPainter reads its batch with its kernel: FLUX_BATCH (or VECWORLD_BATCH) under an engine kernel, 1 under
    VECWORLD_KERNEL=off even with VECWORLD_BATCH=8 (stops at the VAE load: no models)."""
    import diffusers

    import vecworld.engine.ext as ext_mod
    import vecworld.engine.graphs as graphs_mod
    import vecworld.engine.swap as swap_mod
    import vecworld.engine.blocks as blocks_mod
    import vecworld.flux.painter as painter_mod

    class Loaded(Exception):
        pass

    seen = []
    tr = SimpleNamespace(transformer_blocks=[], single_transformer_blocks=[], enable_group_offload=lambda **k: None)
    monkeypatch.setattr(ext_mod, "load_ext", lambda: None)
    monkeypatch.setattr(painter_mod, "_read_gguf", lambda path, unread=None: {})
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", lambda *a, **k: tr)
    monkeypatch.setattr(swap_mod, "gguf_linears", lambda tr: {})
    monkeypatch.setattr(swap_mod, "swap_linears", lambda *a, **k: 1)
    monkeypatch.setattr(blocks_mod, "patch_blocks", lambda tr: 57)
    monkeypatch.setattr(painter_mod, "rowwise", lambda tr: None)
    monkeypatch.setattr(painter_mod, "_pin_in_place", lambda mods: [])
    monkeypatch.setattr(graphs_mod, "stream_blocks",
                        lambda tr, dev, graphs, **k: (SimpleNamespace(arena="arena", reader="reader"), "runner"))

    def stop_at_vae(*a, **k):
        raise Loaded

    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", stop_at_vae)
    real_init = FluxPainter.__init__

    def init(self, *a, **k):
        try:
            real_init(self, *a, **k)
        finally:
            seen.append(self.batch)

    monkeypatch.setattr(FluxPainter, "__init__", init)
    for k in ("VECWORLD_FUSE", "VECWORLD_GRAPHS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VECWORLD_BATCH", "8")
    for kernel, want in (("w4a8", 8), ("mma", 8), ("off", 1)):
        monkeypatch.setenv("VECWORLD_KERNEL", kernel)
        with pytest.raises(Loaded):
            FluxPainter()
        assert seen[-1] == want, kernel


def test_check_stop_between_steps():
    stop = threading.Event()
    tr = FakeTransformer(on_call=lambda n: stop.set())        # asked to stop during the first pass
    p = fake_painter(tr)
    gpu._job.stop = stop
    try:
        with pytest.raises(Stopped):
            p.panorama([("a forest", 1.0)], seed=1)
    finally:
        gpu._job.stop = None
    assert len(tr.calls) == 1


def test_prompt_blend_is_a_weighted_sum():
    p = FluxPainter.__new__(FluxPainter)
    vals = {"a": 1.0, "b": 3.0}
    p._setup(FakeTransformer(), FakeVae(),
             lambda t: (torch.full((1, 4, 8), vals[t], dtype=torch.bfloat16),
                        torch.full((1, 8), 2 * vals[t], dtype=torch.bfloat16)),
             None, dev="cpu", dtype=torch.float32)
    t5, pooled = p._embed([("a", 0.25), ("b", 0.75)])
    assert torch.allclose(t5.float(), torch.full((1, 4, 8), 2.5))
    assert torch.allclose(pooled.float(), torch.full((1, 8), 5.0))


def test_pack_roundtrip():
    p = fake_painter()
    lat = torch.randn(1, 16, PANO_H // 8, PANO_W // 8)
    g = p._pack(lat)
    assert g.shape == (1, 64, GH, GW)
    assert torch.equal(p._unpack(g), lat)
    # token (i, j) channel c*4 + dy*2 + dx is latent (c, 2i + dy, 2j + dx), as diffusers packs FLUX latents
    assert g[0, 5 * 4 + 1 * 2 + 0, 3, 7] == lat[0, 5, 7, 14]


def test_step_strength_zero_keeps_the_sides_exactly():
    """Strength 0 never releases the sides (between the focus band and the band behind the camera, which is
    always repainted at FLUX_BACK_STRENGTH), so they come back as the original."""
    p = fake_painter()
    prev = Image.new("RGB", (PANO_W, PANO_H), (128, 128, 128))
    out = np.asarray(p.step(prev, [("a forest", 1.0)], None, 0.5, seed=5, bg_strength=0.0), np.float64)
    ref = np.asarray(p._decode(p._encode(prev)), np.float64)
    side = slice(round(PANO_W * 0.27), round(PANO_W * 0.33))       # 0.17-0.23 of a turn from the focus at 0.5
    assert np.abs(out[:, side] - ref[:, side]).max() <= 1


def test_step_schedule_puts_the_background_sigmas_on_it():
    p = fake_painter()
    assert p._step_sigmas(0.25) == ([1.0, 0.75, 0.5, 0.25, 0.0], [3])           # on schnell's grid: unchanged
    assert p._step_sigmas(0.15, 0.5) == ([1.0, 0.75, 0.5, 0.25, 0.15, 0.0], [4, 2])   # off it: added
    assert p._step_sigmas(0.0) == ([1.0, 0.75, 0.5, 0.25, 0.0], [FLUX_STEPS])   # 0: never released


def test_step_releases_the_band_behind_the_camera_earlier():
    """Behind the camera the dolly shrank the view and mirrored rows in at the top and bottom: those columns are
    released at FLUX_BACK_STRENGTH (repainted harder), the sides at the step's background strength."""
    from vecworld.config import FLUX_BACK_FEATHER, FLUX_BACK_FROM, FLUX_BACK_STRENGTH, FLUX_BG_STRENGTH

    p = fake_painter()
    m = np.zeros(GW, np.float32)                                  # no focus band: background only
    sig, rw = p._reset_weights(m, 0.5, FLUX_BG_STRENGTH)
    rw = np.stack([r.reshape(-1).cpu().numpy() for r in rw])     # [step, column]: weight of the reset to orig
    d = np.abs((np.arange(GW) + 0.5) / GW - 0.5)
    back = d >= FLUX_BACK_FROM + FLUX_BACK_FEATHER
    sides = d <= FLUX_BACK_FROM
    k_back, k_side = sig.index(FLUX_BACK_STRENGTH), sig.index(FLUX_BG_STRENGTH)
    assert k_back < k_side
    assert (rw[:k_back + 1] == 1).all()                           # everything held to the original up to there
    assert (rw[k_back + 1:, back] == 0).all()                     # behind: released at FLUX_BACK_STRENGTH
    assert (rw[:k_side + 1, sides] == 1).all() and (rw[k_side + 1:, sides] == 0).all()
    between = ~back & ~sides
    assert ((rw[k_back + 1:k_side + 1, between] > 0) & (rw[k_back + 1:k_side + 1, between] < 1)).all()   # feathered


def test_a_stronger_step_is_stronger_everywhere():
    """A mood change asks for 0.8: the band behind is never released later than the sides."""
    p = fake_painter()
    sig, rw = p._reset_weights(np.zeros(GW, np.float32), 0.5, 0.8)
    k = sig.index(0.8)
    assert all((r == 0).all() for r in rw[k + 1:]) and all((r == 1).all() for r in rw[:k + 1])


def test_read_gguf_matches_diffusers_loader(tmp_path):
    import gguf
    import numpy as np
    from diffusers.models.model_loading_utils import load_gguf_checkpoint

    from vecworld.flux.painter import _read_gguf

    path = str(tmp_path / "t.gguf")
    w = gguf.GGUFWriter(path, "flux")
    rng = np.random.default_rng(0)
    w.add_tensor("a.weight", rng.standard_normal((4, 8)).astype(np.float32))
    w.add_tensor("b.weight", rng.standard_normal((3, 16)).astype(np.float16))
    q = gguf.quants.quantize(rng.standard_normal((2, 256)).astype(np.float32), gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("c.weight", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    mine, ref = _read_gguf(path), load_gguf_checkpoint(path)
    assert mine.keys() == ref.keys()
    for k in ref:
        assert type(mine[k]) is type(ref[k]) and mine[k].dtype == ref[k].dtype and torch.equal(mine[k], ref[k]), k
        assert getattr(mine[k], "quant_type", None) == getattr(ref[k], "quant_type", None)


def test_windows_crossfade_without_steps_at_their_edges():
    """Windows that disagree (as FLUX's do in haze, each with its own exposure) must blend smoothly: a box average
    jumps wherever the pair of windows covering a column changes, a hard vertical line every FLUX_STRIDE."""
    def velocity(hidden_states, **kw):
        x0 = hidden_states[:, 0, 0].round().long()              # channel 0 of the first token = the window's column
        v = torch.cos(2 * math.pi * (x0 // FLUX_STRIDE).float() / N_WIN)
        return (v[:, None, None].expand_as(hidden_states).clone(),)

    p = fake_painter(transformer=lambda **kw: velocity(**kw))
    grid = torch.zeros(1, 64, GH, GW)
    grid[:, 0] = torch.arange(GW, dtype=torch.float32)
    out = p._denoise(grid.clone(), [1.0, 0.0], [p._embed([("x", 1.0)])] * N_WIN)
    v = (out - grid)[0, 1, 0]                                    # a channel the fake didn't seed, one row
    jumps = (torch.roll(v, -1) - v).abs()
    assert jumps.max() < 0.15, f"step {jumps.max():.2f} at column {int(jumps.argmax())}"


def test_stages_name_the_work_around_the_passes():
    """stage(name) says what a job is doing between the passes progress() counts: the text, the view it starts
    from (a step), then the decode."""
    log = []
    p = fake_painter(progress=lambda k, total: log.append(("pass", k)))
    p.stage = lambda name: log.append(("stage", name))
    p.panorama([("a forest", 1.0)], seed=3)
    names = [x for x in log if x[0] == "stage"]
    assert names == [("stage", "text"), ("stage", "finishing")]
    assert log[0] == ("stage", "text") and log[-1] == ("stage", "finishing") and ("pass", 1) in log
    log.clear()
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    p.step(prev, [("a forest", 1.0)], None, 0.5, seed=4, bg_strength=0.15)
    assert [x for x in log if x[0] == "stage"] == [("stage", "text"), ("stage", "image"), ("stage", "finishing")]
    assert log.index(("stage", "image")) < log.index(("pass", 1)) < log.index(("stage", "finishing"))


def test_close_unpins_once_and_ends_the_painter(monkeypatch):
    from vecworld.flux import painter as fp

    gone = []
    monkeypatch.setattr(fp, "_unpin", lambda ptrs: gone.extend(ptrs))
    p = fake_painter()
    p._pinned = [11, 22]
    p.close()
    p.close()                                   # safe twice
    assert gone == [11, 22] and p._pinned == []
    with pytest.raises(RuntimeError):
        p.panorama([("a forest", 1.0)], seed=1)


@pytest.mark.parametrize("recipe", ["t1", "repaint", "carry"])
def test_a_step_without_focus_denoises_from_its_strength(recipe):
    """Without a focus band a step is img2img: its first pass is at sigma = the background strength (it started
    one schedule point later, i.e. at 0.25 for 0.5, before Task 2b)."""
    tr = TimedTransformer()
    p = fake_painter(tr, recipe=recipe)
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    p.step(prev, [("a forest", 1.0)], None, None, seed=4, bg_strength=0.5)
    assert tr.sigmas[0] == 0.5
    assert sorted(set(tr.sigmas), reverse=True) == [0.5, 0.25]


def _rw(rw):
    return np.stack([r.reshape(-1).cpu().numpy() for r in rw])     # [step, column]


def test_graded_release_releases_each_column_once_at_its_strength():
    from vecworld.config import FLUX_BACK_FEATHER, FLUX_BACK_FROM
    from vecworld.dream.pano import focus_mask

    p = fake_painter(recipe="carry")
    m = focus_mask(GW, 0.5, 0.15, 0.15)
    sig, rw = p._graded_weights(m, 0.5, 0.5, 0.75, 0.9)
    rw = _rw(rw)
    assert 0.9 in sig and 0.5 in sig and 0.75 in sig
    assert (np.diff(rw, axis=0) <= 0).all(), "a column is never held again once released"
    assert (((rw > 0) & (rw < 1)).sum(0) <= 1).all(), "at most one partial hold per column"
    d = np.abs((np.arange(GW) + 0.5) / GW - 0.5)
    first_free = (rw >= 1).sum(0)          # the step from which a column is (at least partly) free
    focus, side = m >= 1, (m == 0) & (d < FLUX_BACK_FROM)
    back = d >= FLUX_BACK_FROM + FLUX_BACK_FEATHER
    assert (first_free[focus] == sig.index(0.9) + 1).all()
    assert (first_free[side] == sig.index(0.5) + 1).all()
    assert (first_free[back] == sig.index(0.75) + 1).all()
    feather = (m > 0) & (m < 1) & (d < FLUX_BACK_FROM)
    released = sig.index(0.9) + 1 - rw[sig.index(0.9) + 1:].sum(0)   # fractional release step
    assert (np.diff(released[feather][: feather.sum() // 2]) != 0).any(), "the feather grades between them"


def test_graded_step_runs_every_column_together_after_its_release():
    """The double exposure: with Task 1's release a feather column is pulled back to the original at every step
    up to the background's release, so the band ahead and the background never denoise together there."""
    from vecworld.dream.pano import focus_mask

    m = focus_mask(GW, 0.5, 0.15, 0.15)
    feather = (m > 0.2) & (m < 0.8)
    lin = _rw(fake_painter()._reset_weights(m, 0.5, 0.5)[1])
    gr = _rw(fake_painter(recipe="repaint")._graded_weights(m, 0.5, 0.5, 0.5, 1.0)[1])
    held_partly = lambda rw: ((rw > 0) & (rw < 1)).sum(0)                    # noqa: E731
    assert (held_partly(lin)[feather] >= 2).all()
    assert (held_partly(gr)[feather] <= 1).all()


def test_fresh_step_is_a_new_panorama():
    p = fake_painter(recipe="fresh")
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    out = p.step(prev, [("a forest", 1.0)], None, 0.5, seed=9)
    assert np.array_equal(np.asarray(out), np.asarray(p.panorama([("a forest", 1.0)], seed=9)))


def test_every_recipe_steps(monkeypatch):
    from vecworld.config import FLUX_STEP_RECIPE, FLUX_STEP_RECIPES

    assert FLUX_STEP_RECIPE in FLUX_STEP_RECIPES and "t1" in FLUX_STEP_RECIPES
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    for name in FLUX_STEP_RECIPES:
        assert fake_painter(recipe=name).step(prev, [("a", 1.0)], None, 0.5, seed=1, focus_half=0.18).size == \
            (PANO_W, PANO_H)


def test_a_recipe_name_is_checked_when_set():
    from vecworld.config import check_recipe

    with pytest.raises(ValueError, match="unknown FLUX step recipe 'sharpp'"):
        check_recipe("sharpp")
    p = fake_painter()
    with pytest.raises(ValueError):
        p.recipe = "sharpp"                     # fails here, not at the first step
    assert p.recipe == "t1"


def test_config_checks_the_configured_recipe_at_import(tmp_path):
    """A typo in FLUX_STEP_RECIPE stops the game when config is imported (startup), not at its first walk step."""
    import subprocess
    import sys
    from pathlib import Path

    import vecworld.config as cfg

    src = Path(cfg.__file__).read_text().replace('FLUX_STEP_RECIPE = "', 'FLUX_STEP_RECIPE = "x', 1)
    (tmp_path / "config.py").write_text(src)
    r = subprocess.run([sys.executable, str(tmp_path / "config.py")], capture_output=True, text=True)
    assert r.returncode != 0 and "unknown FLUX step recipe" in r.stderr


@pytest.mark.parametrize("bad", [{"release": "stepped", "side": 0.5, "back": 0.5, "focus": 1.0, "feather": 0.1},
                                 {"release": "graded", "side": 0.5, "back": 0.5, "focus": 1.0},
                                 {"release": "linear", "side": 0.15, "back": 0.5, "focus": 0.9, "feather": 0.06},
                                 {"release": "graded", "side": 1.5, "back": 0.5, "focus": 1.0, "feather": 0.1}])
def test_malformed_recipes_are_refused(monkeypatch, bad):
    from vecworld import config

    monkeypatch.setitem(config.FLUX_STEP_RECIPES, "bad", bad)
    with pytest.raises(ValueError):
        config.check_recipe("bad")


def test_a_linear_recipe_releases_behind_at_its_own_back_strength(monkeypatch):
    """The linear path used the global FLUX_BACK_STRENGTH whatever the recipe said."""
    from vecworld import config

    monkeypatch.setitem(config.FLUX_STEP_RECIPES, "t1b",
                        {**config.FLUX_STEP_RECIPES["t1"], "back": 0.6})
    tr = TimedTransformer()
    p = fake_painter(tr, recipe="t1b")
    prev = Image.new("RGB", (PANO_W, PANO_H), (90, 120, 60))
    p.step(prev, [("a forest", 1.0)], None, 0.5, seed=4, focus_half=0.15)
    assert 0.6 in tr.sigmas


def test_kernel_defaults_to_w4a8(monkeypatch):
    """VECWORLD_KERNEL unset or empty: the FLUX painter (and so the game) runs its Q4_K linears on config.FLUX_KERNEL,
    since Task 5b w4a8 (with FLUX_A16_KEEP's linears on mma). "off" (or "reference") keeps the diffusers path; a kernel
    name picks that kernel; anything else is refused before the minute-long model load."""
    import vecworld.flux.painter as painter_mod
    from vecworld.config import FLUX_KERNEL
    from vecworld.engine import engine_kernel

    assert painter_mod.engine_kernel is engine_kernel      # one rule for the painter, the game and the bench
    assert FLUX_KERNEL == "w4a8"
    monkeypatch.delenv("VECWORLD_KERNEL", raising=False)
    assert engine_kernel() == "w4a8"
    for value, kernel in (("", "w4a8"), ("off", None), ("reference", None), ("naive", "naive"), ("tiled", "tiled"),
                          ("mma", "mma"), ("w4a8", "w4a8")):
        monkeypatch.setenv("VECWORLD_KERNEL", value)
        assert engine_kernel() == kernel, value
    monkeypatch.setenv("VECWORLD_KERNEL", "fastest")
    with pytest.raises(ValueError, match="fastest"):
        engine_kernel()


def test_keep_for_applies_the_keep_list_under_w4a8_only():
    """vecworld.engine.keep_for: config.FLUX_A16_KEEP under w4a8, nothing under any other kernel (mma stays pure)."""
    from vecworld.config import FLUX_A16_KEEP
    from vecworld.engine import keep_for

    assert keep_for("w4a8") == tuple(FLUX_A16_KEEP) and FLUX_A16_KEEP
    assert keep_for("mma") == keep_for("tiled") == keep_for("naive") == keep_for(None) == ()


class _Stop(Exception):
    pass


def test_painter_swaps_with_the_kernels_keep_list(monkeypatch):
    """FluxPainter hands swap_linears keep_for(its kernel): FLUX_A16_KEEP under w4a8 (the default), nothing under mma
    or tiled; and records it as painter.keep_a16 (walk_sheet writes it into each walk's timings.json)."""
    import diffusers

    import vecworld.engine.ext as ext_mod
    import vecworld.engine.swap as swap_mod
    import vecworld.flux.painter as painter_mod
    from vecworld.config import FLUX_A16_KEEP

    seen = []

    def swap(tr, gg, kernel, keep_a16=()):
        seen.append((kernel, tuple(keep_a16)))
        raise _Stop                              # far enough: nothing after the swap is under test

    monkeypatch.setattr(ext_mod, "load_ext", lambda: None)
    monkeypatch.setattr(painter_mod, "_read_gguf", lambda path, unread=None: {})
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", lambda *a, **k: "transformer")
    monkeypatch.setattr(swap_mod, "gguf_linears", lambda tr: {})
    monkeypatch.setattr(swap_mod, "swap_linears", swap)
    for value, want in ((None, ("w4a8", tuple(FLUX_A16_KEEP))), ("mma", ("mma", ())), ("tiled", ("tiled", ()))):
        if value is None:
            monkeypatch.delenv("VECWORLD_KERNEL", raising=False)
        else:
            monkeypatch.setenv("VECWORLD_KERNEL", value)
        seen.clear()
        with pytest.raises(_Stop):
            FluxPainter()
        assert seen == [want], value


def test_keep_patterns_match_real_flux_linears():
    """Every FLUX_A16_KEEP pattern names at least one linear of the real FLUX transformer (built on the meta device
    from its config, no weights), and none of them a modulation linear. (swap_linears also raises on a pattern that
    matches no Q4_K linear, so a typo can't silently keep nothing.)"""
    import fnmatch

    import torch
    from diffusers import FluxTransformer2DModel

    from vecworld.config import FLUX_A16_KEEP, FLUX_BASE

    cfg = FluxTransformer2DModel.load_config(FLUX_BASE, subfolder="transformer", local_files_only=True)
    with torch.device("meta"):
        tr = FluxTransformer2DModel.from_config(cfg)
    names = [n for n, m in tr.named_modules() if isinstance(m, torch.nn.Linear) and m.in_features % 256 == 0]
    assert len(names) > 400
    for p in FLUX_A16_KEEP:
        hits = [n for n in names if fnmatch.fnmatchcase(n, p)]
        assert hits, p
        assert all("norm" not in n for n in hits) or p.endswith(".*"), p


def test_a_broken_extension_build_fails_before_the_model_load(monkeypatch):
    """With an engine kernel (the default), FluxPainter builds/loads the CUDA extension first: a failure stops it at
    once, before the minute-long transformer load, and says how to paint without the engine."""
    import diffusers

    import vecworld.engine.ext as ext_mod

    def broken():
        raise OSError("nvcc not found")

    def no_load(*a, **k):
        raise AssertionError("the transformer load started before the extension was checked")

    monkeypatch.setattr(ext_mod, "load_ext", broken)
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", no_load)
    monkeypatch.delenv("VECWORLD_KERNEL", raising=False)
    with pytest.raises(RuntimeError, match=r"VECWORLD_KERNEL=off") as e:
        FluxPainter()
    assert "nvcc not found" in str(e.value) and isinstance(e.value.__cause__, OSError)


def test_fuse_defaults_on_with_an_engine_kernel(monkeypatch):
    """VECWORLD_FUSE (vecworld.engine.engine_fuse, one rule for the painter and the bench): unset or empty, the glue
    is fused under every engine kernel and not under "off" (the diffusers reference stays diffusers all the way);
    "0" / "off" turns it off under any kernel, "1" / "on" turns it on under any kernel, "off" included (to test the
    fused blocks on diffusers' linears); anything else is refused."""
    from vecworld.engine import engine_fuse

    monkeypatch.delenv("VECWORLD_FUSE", raising=False)
    for kernel in ("naive", "tiled", "mma", "w4a8"):
        assert engine_fuse(kernel) is True
    assert engine_fuse(None) is False
    for value, want in (("", None), ("0", False), ("off", False), ("1", True), ("on", True)):
        monkeypatch.setenv("VECWORLD_FUSE", value)
        for kernel in ("w4a8", "mma", None):
            assert engine_fuse(kernel) is (kernel is not None if want is None else want), (value, kernel)
    monkeypatch.setenv("VECWORLD_FUSE", "maybe")
    with pytest.raises(ValueError, match="maybe"):
        engine_fuse("w4a8")


def test_painter_patches_blocks_when_fused(monkeypatch):
    """FluxPainter patches the transformer's blocks (vecworld.engine.blocks.patch_blocks) after the swap and before
    group offload (whose hooks wrap each block's forward as it is then) when engine_fuse says so, and records it as
    painter.fused."""
    import diffusers

    import vecworld.engine.blocks as blocks_mod
    import vecworld.engine.ext as ext_mod
    import vecworld.engine.swap as swap_mod
    import vecworld.flux.painter as painter_mod

    order = []
    monkeypatch.setattr(ext_mod, "load_ext", lambda: None)
    monkeypatch.setattr(painter_mod, "_read_gguf", lambda path, unread=None: {})
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", lambda *a, **k: "transformer")
    monkeypatch.setattr(swap_mod, "gguf_linears", lambda tr: {})
    monkeypatch.setattr(swap_mod, "swap_linears", lambda *a, **k: order.append("swap") or 1)
    monkeypatch.setattr(blocks_mod, "patch_blocks", lambda tr: order.append("patch") or 57)

    # the transformer is a string here, so the painter stops at the pinning that follows (AttributeError)
    for kernel, fuse, want in ((None, None, ["swap", "patch"]), ("mma", "off", ["swap"]), ("off", None, []),
                               ("off", "on", ["patch"])):
        monkeypatch.delenv("VECWORLD_KERNEL", raising=False)
        monkeypatch.delenv("VECWORLD_FUSE", raising=False)
        if kernel:
            monkeypatch.setenv("VECWORLD_KERNEL", kernel)
        if fuse:
            monkeypatch.setenv("VECWORLD_FUSE", fuse)
        order.clear()
        with pytest.raises(AttributeError):
            FluxPainter()
        assert order == want, (kernel, fuse)


def test_graphs_default_off_opt_in(monkeypatch):
    """VECWORLD_GRAPHS (vecworld.engine.engine_graphs, one rule for the painter and the bench): unset, empty, "0" or
    "off", the blocks run eagerly on the fixed weight slots (the default: graphs tie with eager when quiet, cost +8.4 s
    on the first panorama and retain 0.9 GB of graph pools that can run the 4 GB card out of memory on a fifth shape,
    Task 10a-g); "1" / "on" replays them as CUDA graphs under an engine kernel. Under VECWORLD_KERNEL=off there are no
    slots (diffusers' group offload), so never graphs. Anything else is refused."""
    from vecworld.engine import engine_graphs

    monkeypatch.delenv("VECWORLD_GRAPHS", raising=False)
    for kernel in ("w4a8", "mma", "tiled", "naive"):
        assert engine_graphs(kernel) is False
    assert engine_graphs(None) is False
    for value, want in (("", False), ("0", False), ("off", False), ("1", True), ("on", True)):
        monkeypatch.setenv("VECWORLD_GRAPHS", value)
        assert engine_graphs("w4a8") is want, value
        assert engine_graphs(None) is False, value
    monkeypatch.setenv("VECWORLD_GRAPHS", "maybe")
    with pytest.raises(ValueError, match="maybe"):
        engine_graphs("w4a8")


def test_painter_streams_blocks_from_slots_under_an_engine_kernel(monkeypatch):
    """Under an engine kernel FluxPainter puts the blocks on fixed-address slots with a GraphRunner
    (vecworld.engine.graphs.stream_blocks; graphs as engine_graphs says) instead of diffusers' group offload, after
    the swap, the fused patch and the rowwise outer layers (Task 9, every kernel), packing the blocks into one
    page-locked host arena (pack=True, Task 10a; FLUX_WEIGHTS_SOURCE=ram) or reading them from the GGUF on the SSD
    (ssd=<path>, Task 10b; the default), when the load leaves the blocks' Q4_K weights unread; under
    VECWORLD_KERNEL=off it reads everything, pins in place and keeps group offload (the reference), whatever the source."""
    import diffusers

    import vecworld.engine.blocks as blocks_mod
    import vecworld.engine.ext as ext_mod
    import vecworld.engine.graphs as graphs_mod
    import vecworld.engine.swap as swap_mod
    import vecworld.flux.painter as painter_mod

    class Loaded(Exception):
        pass

    order = []
    tr = SimpleNamespace(transformer_blocks=[], single_transformer_blocks=[],
                         enable_group_offload=lambda **k: order.append("offload"))
    monkeypatch.setattr(ext_mod, "load_ext", lambda: None)
    monkeypatch.setattr(painter_mod, "_read_gguf", lambda path, unread=None: order.append(f"read unread={unread}") or {})
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", lambda *a, **k: tr)
    monkeypatch.setattr(swap_mod, "gguf_linears", lambda tr: {})
    monkeypatch.setattr(swap_mod, "swap_linears", lambda *a, **k: order.append("swap") or 1)
    monkeypatch.setattr(blocks_mod, "patch_blocks", lambda tr: order.append("patch") or 57)
    monkeypatch.setattr(painter_mod, "rowwise", lambda tr: order.append("rows"))
    monkeypatch.setattr(painter_mod, "_pin_in_place", lambda mods: order.append("pin") or [])
    monkeypatch.setattr(graphs_mod, "stream_blocks",
                        lambda tr, dev, graphs, **k: order.append(f"slots graphs={graphs} pack={k.get('pack')} "
                                                                  f"ssd={k.get('ssd') is not None}")
                        or (SimpleNamespace(arena="arena", reader="reader"), "runner"))

    def stop_at_vae(*a, **k):
        raise Loaded

    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", stop_at_vae)
    ssd, ram = "read unread=<function _block_q4k", "read unread=None"
    for kernel, graphs, source, want in (
            (None, None, None, [ssd, "swap", "patch", "rows", "slots graphs=False pack=False ssd=True"]),
            (None, None, "ram", [ram, "swap", "patch", "rows", "slots graphs=False pack=True ssd=False"]),
            ("mma", "1", "ssd", [ssd, "swap", "patch", "rows", "slots graphs=True pack=False ssd=True"]),
            ("mma", "1", "ram", [ram, "swap", "patch", "rows", "slots graphs=True pack=True ssd=False"]),
            ("off", None, None, [ram, "rows", "pin", "offload"]), ("off", "1", "ssd", [ram, "rows", "pin", "offload"])):
        for k in ("VECWORLD_KERNEL", "VECWORLD_FUSE", "VECWORLD_GRAPHS", "FLUX_WEIGHTS_SOURCE"):
            monkeypatch.delenv(k, raising=False)
        if kernel:
            monkeypatch.setenv("VECWORLD_KERNEL", kernel)
        if graphs:
            monkeypatch.setenv("VECWORLD_GRAPHS", graphs)
        if source:
            monkeypatch.setenv("FLUX_WEIGHTS_SOURCE", source)
        order.clear()
        with pytest.raises(Loaded):
            FluxPainter()
        got = [o.split(" at 0x")[0] for o in order]
        assert got == want, (kernel, graphs, source)


def test_painter_releases_the_arena_if_construction_fails_after_the_streamer(monkeypatch):
    """N2 (Task 10a re-review): once stream_blocks has registered the host arena, anything that raises later in
    FluxPainter.__init__ (a CUDA OOM moving the VAE or CLIP onto the card) must close the runner, the slots and the
    arena, or the transformer's collection frees registered memory. The original error is the one that propagates,
    even when closing fails too."""
    import diffusers

    import vecworld.engine.blocks as blocks_mod
    import vecworld.engine.ext as ext_mod
    import vecworld.engine.graphs as graphs_mod
    import vecworld.engine.swap as swap_mod
    import vecworld.flux.painter as painter_mod

    class Boom(Exception):
        pass

    done = []
    tr = SimpleNamespace(transformer_blocks=[], single_transformer_blocks=[], enable_group_offload=lambda **k: None)
    arena = SimpleNamespace(close=lambda: done.append("arena"))
    slots = SimpleNamespace(arena=arena, reader=arena, close=lambda: done.append("slots"))   # (either source)
    runner = SimpleNamespace(close=lambda: done.append("runner"))
    monkeypatch.setattr(ext_mod, "load_ext", lambda: None)
    monkeypatch.setattr(painter_mod, "_read_gguf", lambda path, unread=None: {})
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_single_file", lambda *a, **k: tr)
    monkeypatch.setattr(swap_mod, "gguf_linears", lambda tr: {})
    monkeypatch.setattr(swap_mod, "swap_linears", lambda *a, **k: 1)
    monkeypatch.setattr(blocks_mod, "patch_blocks", lambda tr: 57)
    monkeypatch.setattr(painter_mod, "rowwise", lambda tr: None)
    monkeypatch.setattr(graphs_mod, "stream_blocks", lambda tr, dev, graphs, **k: (slots, runner))

    def oom_on_vae(*a, **k):
        raise Boom("CUDA out of memory (moving the VAE)")

    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", oom_on_vae)
    for k in ("VECWORLD_FUSE", "VECWORLD_GRAPHS", "VECWORLD_KERNEL", "FLUX_WEIGHTS_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(Boom) as ei:         # held: the traceback keeps the half-built painter alive, as pytest's does
        FluxPainter()
    assert done == ["runner", "slots", "arena"]
    del ei

    done.clear()
    slots.close = lambda: (_ for _ in ()).throw(RuntimeError("sticky"))   # a failing close must not hide Boom
    with pytest.raises(Boom) as ei:
        FluxPainter()
    assert "arena" in done
    del ei


def test_close_frees_slots_and_unpins_even_if_the_runner_fails(monkeypatch):
    """FluxPainter.close() still closes the weight slots and unregisters the page-locked host memory when closing the
    graph runner raises (e.g. a sticky CUDA error surfacing in its sync); the error is raised after that."""
    import vecworld.flux.painter as painter_mod

    done = []

    class Runner:
        def close(self):
            raise RuntimeError("sticky CUDA error")

    class Slots:
        def close(self):
            done.append("slots")

    class Arena:
        def close(self):
            done.append("arena")

    monkeypatch.setattr(painter_mod, "_unpin", lambda ptrs: done.append(("unpin", tuple(ptrs))))
    p = fake_painter()
    p._runner, p._slots, p._arena, p._pinned = Runner(), Slots(), Arena(), [1, 2]
    with pytest.raises(RuntimeError, match="sticky"):
        p.close()
    assert done == ["slots", "arena", ("unpin", (1, 2))]
    assert p._runner is None and p._slots is None and p._arena is None and p._pinned == []
    p.close()                                               # safe twice
