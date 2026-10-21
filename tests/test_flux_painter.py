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
