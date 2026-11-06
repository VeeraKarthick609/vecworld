"""Glue fusion (Task 6): residual add, LayerNorm and modulate in one pass, and the FLUX block forwards that call it
(gpu; run with the game server stopped). The fused pass follows the reference's bf16 rounding points, so it is held
to the eager fp32 arithmetic within a bf16 step, and the patched blocks to the unpatched ones."""
import contextlib
import gc

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

H = 3072


def _rand(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to("cuda", torch.bfloat16)


def _inputs(m, b, seed=0, with_y=True):
    res = _rand((m, H), seed, 4.0)
    res[:, 11] *= 40                                    # an outlier channel, as FLUX's residual stream has
    y = _rand((m, H), seed + 1) if with_y else None
    emb = _rand((b, 6 * H), seed + 2, 0.5)              # chunks of one modulation output: strided rows, as in FLUX
    shift, scale, gate = emb[:, :H], emb[:, H:2 * H], emb[:, 2 * H:3 * H]
    return res, y, gate if with_y else None, shift, scale


def eager_fp32(res, y, gate, shift, scale, eps=1e-6):
    """The definition in fp32 from the bf16 inputs: r = res + gate * y; x = LayerNorm(r) * (1 + scale) + shift, the
    [B, H] rows broadcast over each image's M / B tokens."""
    b = shift.shape[0]
    r = res.float().view(b, -1, H)
    if y is not None:
        r = r + gate.float()[:, None] * y.float().view(b, -1, H)
    x = torch.nn.functional.layer_norm(r, (H,), eps=eps) * (1 + scale.float()[:, None]) + shift.float()[:, None]
    return r.reshape(-1, H), x.reshape(-1, H)


def _check(ours, ref):
    err = float((ours.float() - ref).abs().max())
    bar = 2 ** -7 * float(ref.abs().max())
    return err, bar


# M = 5120 is a walk pass's 4 windows x 1280 tokens: B = 4 rows of modulation, each broadcast over its 1280 tokens.
@pytest.mark.parametrize("m", [1, 333, 1280, 5120], ids=lambda m: f"M={m}")
def test_rnm_matches_eager(m):
    from vecworld.engine.glue import residual_norm_modulate

    b = 4 if m == 5120 else 1
    res, y, gate, shift, scale = _inputs(m, b, seed=m)
    res_out, x = residual_norm_modulate(res, y, gate, shift, scale)
    r_ref, x_ref = eager_fp32(res, y, gate, shift, scale)
    assert res_out.dtype == x.dtype == torch.bfloat16 and res_out.shape == x.shape == (m, H)
    for name, ours, ref in (("res_out", res_out, r_ref), ("x_mod", x, x_ref)):
        err, bar = _check(ours, ref)
        print(f"\nM={m} {name}: max|d| {err:.3e}, bar {bar:.3e}")
        assert err <= bar, name


def test_rnm_no_residual():
    """y = None: the norm and modulate alone (a block's first norm); res comes back as it is, not copied."""
    from vecworld.engine.glue import residual_norm_modulate

    res, _, _, shift, scale = _inputs(1024, 1, seed=3, with_y=False)
    res_out, x = residual_norm_modulate(res, None, None, shift, scale)
    assert res_out is res
    _, x_ref = eager_fp32(res, None, None, shift, scale)
    err, bar = _check(x, x_ref)
    assert err <= bar


@pytest.mark.parametrize("b", [1, 2, 8])
def test_rnm_broadcasts_per_image(b):
    """Each image's tokens take that image's modulation row (B up to 8 windows a pass)."""
    from vecworld.engine.glue import residual_norm_modulate

    res, y, gate, shift, scale = _inputs(b * 64, b, seed=b)
    res_out, x = residual_norm_modulate(res, y, gate, shift, scale)
    for i in range(b):
        sl = slice(64 * i, 64 * (i + 1))
        r1, x1 = residual_norm_modulate(res[sl].contiguous(), y[sl].contiguous(), gate[i:i + 1], shift[i:i + 1],
                                        scale[i:i + 1])
        assert torch.equal(res_out[sl], r1) and torch.equal(x[sl], x1)


def test_rnm_matches_the_reference_rounding():
    """The fused pass rounds where the reference rounds (gate * y, the sum, the norm, 1 + scale, the product, the
    shift) and takes torch's own Welford statistics in torch's order, so it is bit-identical to the eager bf16 chain
    (a two-pass mean / variance flipped ~3 in a million last bits)."""
    from vecworld.engine.glue import residual_norm_modulate

    for b, m in ((1, 1280), (4, 4 * 333)):
        res, y, gate, shift, scale = _inputs(m, b, seed=5 + b)
        res_out, x = residual_norm_modulate(res, y, gate, shift, scale)
        r = res.view(b, -1, H) + gate[:, None] * y.view(b, -1, H)
        xe = torch.nn.functional.layer_norm(r, (H,), eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
        assert torch.equal(res_out, r.view(-1, H))
        assert torch.equal(x, xe.view(-1, H))


def test_rnm_rejects():
    from vecworld.engine.glue import residual_norm_modulate

    res, y, gate, shift, scale = _inputs(64, 2, seed=9)
    with pytest.raises((RuntimeError, ValueError)):
        residual_norm_modulate(res[:63], y[:63], gate, shift, scale)        # 63 tokens don't split over 2 images
    with pytest.raises((RuntimeError, ValueError)):
        residual_norm_modulate(res.float(), y, gate, shift, scale)          # not bf16
    with pytest.raises((RuntimeError, ValueError)):
        residual_norm_modulate(res, y, None, shift, scale)                  # y without a gate


@pytest.mark.parametrize("m", [333, 1280])
def test_rnm_quantize_matches_quantize_act(m):
    """quantize=True also emits quantize_act's outputs for x_mod, bit for bit, attached for the next w4a8 linear."""
    from vecworld.engine.glue import residual_norm_modulate
    from vecworld.engine.q4k import quantize_act, quantized_of

    res, y, gate, shift, scale = _inputs(m, 1, seed=m + 7)
    res[3] = 0
    plain = residual_norm_modulate(res, y, gate, shift, scale)
    r2, x2 = residual_norm_modulate(res, y, gate, shift, scale, quantize=True)
    assert torch.equal(plain[0], r2) and torch.equal(plain[1], x2)
    assert quantized_of(plain[1]) is None
    q = quantized_of(x2)
    ref = quantize_act(x2)
    for a, b in zip(q, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)


def _rope_tables(T, L):
    """FLUX's rotary cos / sin (fp32 [T + L, 128]) for T text tokens and a 32-wide image grid of L tokens."""
    from diffusers.models.transformers.transformer_flux import FluxPosEmbed

    ids = torch.zeros(T + L, 3, device="cuda")
    ids[T:, 1], ids[T:, 2] = torch.arange(L, device="cuda") // 32, torch.arange(L, device="cuda") % 32
    return FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])(ids)


def test_qk_norm_rope_matches_diffusers():
    """q / k RMSNorm + rotary, text and image written into one joint buffer, against diffusers' rms_norm,
    apply_rotary_emb (fp32) and torch.cat: the same bf16 values but for rare last-bit flips (the sum of squares'
    order)."""
    from diffusers.models.embeddings import apply_rotary_emb

    from vecworld.engine.glue import qk_norm_rope

    B, T, L, nh = 2, 256, 1024, 24
    cos, sin = _rope_tables(T, L)
    assert cos.dtype == torch.float32 and cos.shape == (T + L, 128)
    xt, xi = _rand((B, T, nh * 128), 1, 3.0), _rand((B, L, nh * 128), 2, 3.0)
    w = (torch.rand(128, generator=torch.Generator().manual_seed(3)) * 2).to("cuda", torch.bfloat16)
    out = torch.empty(B, T + L, nh, 128, device="cuda", dtype=torch.bfloat16)
    qk_norm_rope(xt, w, cos, sin, out, 0, 1e-6)
    qk_norm_rope(xi, w, cos, sin, out, T, 1e-6)
    n = torch.cat([torch.nn.functional.rms_norm(t.unflatten(-1, (nh, 128)), (128,), w, 1e-6) for t in (xt, xi)], 1)
    ref = apply_rotary_emb(n, (cos, sin), sequence_dim=1)
    d = (out.float() - ref.float()).abs()
    frac = float((d > 0).float().mean())
    step = ref.float().abs().amax(-1, keepdim=True) * 2 ** -7          # a bf16 step of the row's largest value
    print(f"\nqk_norm_rope vs diffusers: {frac:.2e} of values differ, max {float((d / step).max()):.2f} steps")
    assert frac < 1e-3 and float((d / step).max()) <= 1.0
    assert torch.equal(out, ref)                                      # measured: bit-identical


def test_qk_norm_rope_rejects():
    from vecworld.engine.glue import qk_norm_rope

    cos, sin = _rope_tables(4, 4)
    x = _rand((1, 8, 256), 0)
    with pytest.raises(RuntimeError):
        qk_norm_rope(x, None, cos, sin, torch.empty(1, 8, 2, 128, device="cuda", dtype=torch.bfloat16), 1, 1e-6)
    with pytest.raises(RuntimeError):
        qk_norm_rope(x[..., :200].contiguous(), None, cos, sin,
                     torch.empty(1, 8, 2, 128, device="cuda", dtype=torch.bfloat16), 0, 1e-6)


@pytest.fixture(scope="module")
def gg():
    from vecworld.config import FLUX_GGUF
    from vecworld.engine.q4k import read_gguf

    return read_gguf(FLUX_GGUF)


def test_attached_quantized_input_only_feeds_w4a8(gg):
    """A Q4KLinear on w4a8 takes the quantized x attached to its input (same output as quantizing itself); one on
    mma (a keep-listed linear) ignores it and runs on bf16 x, even when the attached tuple is wrong."""
    from vecworld.engine.q4k import Q4KLinear, attach_quantized, quantize_act

    w = gg["double_blocks.0.img_mlp.0.weight"]
    x = _rand((300, w.shape[1]), 1)
    for kernel in ("w4a8", "mma"):
        lin = Q4KLinear(w, None, kernel).cuda()
        want = lin(x)
        x2 = x.clone()
        good = quantize_act(x2)
        attach_quantized(x2, good if kernel == "w4a8" else tuple(torch.zeros_like(t) for t in good))
        assert torch.equal(lin(x2), want), kernel


# --- patched blocks -------------------------------------------------------------------------------------------------

def _bench():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bench_flux", Path(__file__).parents[1] / "scripts/bench_flux.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def _trim():
    """gc, then hand the freed heap back to the OS (glibc malloc_trim). diffusers' GGUF load allocates the whole
    transformer tensor by tensor; freed, it stays resident as fragmented heap (~5 GB measured after this module),
    which pushed the next GPU module (the Qwen dreamer) out of RAM in the full suite."""
    import ctypes

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class _Two(torch.nn.Module):
    """Double block 0 and single block 0 under their names in the transformer (so the keep-list patterns match)."""

    def __init__(self, d, s):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([d])
        self.single_transformer_blocks = torch.nn.ModuleList([s])


@pytest.fixture(scope="module")
def two():
    """Double block 0 and single block 0 on diffusers (kernel off), plus seeded inputs: 2 windows (B = 2) of 1024
    image + 256 text tokens, and their rotary embedding."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VECWORLD_KERNEL", "off")
        tr, _ = _bench().load_transformer()
    two = _Two(tr.transformer_blocks[0], tr.single_transformer_blocks[0])
    dev, L, T, B = "cuda", 1024, 256, 2
    g = torch.Generator().manual_seed(0)
    hs, es = (torch.randn(B, n, H, generator=g).to(dev, torch.bfloat16) for n in (L, T))
    temb = torch.randn(B, H, generator=g).to(dev, torch.bfloat16)
    ids = torch.zeros(T + L, 3, device=dev)
    ids[T:, 1], ids[T:, 2] = torch.arange(L, device=dev) // 32, torch.arange(L, device=dev) % 32
    with torch.inference_mode():
        rot = tr.pos_embed.to(dev)(ids)
    del tr
    _trim()
    yield two, dict(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot)
    del two
    _trim()
    torch.cuda.empty_cache()


def _run(blk, kw):
    with torch.inference_mode():
        o = blk(**kw)
    return torch.cat([x.float().flatten() for x in o])


@contextlib.contextmanager
def _swapped(m, kernel):
    """m with its Q4_K linears on `kernel` (w4a8: with the keep-list) for the duration; "off" leaves diffusers'."""
    from vecworld.engine import keep_for
    from vecworld.engine.swap import gguf_linears, swap_linears

    orig = dict(m.named_modules())
    try:
        if kernel != "off":
            assert swap_linears(m, gguf_linears(m), kernel, keep_a16=keep_for(kernel)) > 0
        yield m
    finally:
        for n, mod in list(m.named_modules()):
            if n and orig.get(n) is not mod:
                parent, _, child = n.rpartition(".")
                setattr(m.get_submodule(parent) if parent else m, child, orig[n])


@pytest.mark.parametrize("kernel", ["off", "w4a8"])
def test_patched_block_matches_unpatched(two, kernel):
    """The fused double and single block forwards vs diffusers' own, the same linears and inputs: max|d| / max|ref|
    < 2e-2 (2 windows, so each window's modulation row is broadcast over its own tokens)."""
    from vecworld.engine.blocks import patch_blocks

    m, kw = two
    with _swapped(m, kernel):
        m.to("cuda")
        try:
            refs = [_run(b, kw) for b in (m.transformer_blocks[0], m.single_transformer_blocks[0])]
            assert patch_blocks(m) == 2
            outs = [_run(b, kw) for b in (m.transformer_blocks[0], m.single_transformer_blocks[0])]
        finally:
            for b in (m.transformer_blocks[0], m.single_transformer_blocks[0]):
                b.__dict__.pop("forward", None)             # back to the class's (diffusers') forward
            m.to("cpu")
    for name, ours, ref in zip(("double0", "single0"), outs, refs):
        err = float((ours - ref).abs().max() / ref.abs().max())
        print(f"\n{kernel} {name}: patched vs unpatched max-rel {err:.2e}")
        assert torch.isfinite(ours).all()
        assert err < 2e-2
        assert torch.equal(ours, ref)                       # the glue rounds as diffusers does: bit-identical


def test_patch_blocks_counts_and_skips_patched(two):
    from vecworld.engine.blocks import patch_blocks

    m = two[0]
    try:
        assert patch_blocks(m) == 2
        assert patch_blocks(m) == 0                         # already patched: nothing to do
    finally:
        for b in (m.transformer_blocks[0], m.single_transformer_blocks[0]):
            b.__dict__.pop("forward", None)


def test_joint_reuses_adjacent_halves():
    """A single block given the previous single block's two output slices takes the buffer they share (no copy);
    anything else is concatenated."""
    from vecworld.engine.blocks import _joint

    buf = _rand((2, 10, 64), 4)
    es, hs = buf[:, :3], buf[:, 3:]
    j = _joint(es, hs)
    assert j.data_ptr() == buf.data_ptr() and torch.equal(j, torch.cat([es, hs], 1))
    es2, hs2 = es.clone(), hs.clone()
    j2 = _joint(es2, hs2)
    assert j2.data_ptr() not in (es2.data_ptr(), hs2.data_ptr()) and torch.equal(j2, j)
    assert torch.equal(_joint(hs[:, :3], es), torch.cat([hs[:, :3], es], 1))     # wrong order: concatenated


# --- the picture ----------------------------------------------------------------------------------------------------

PROMPT = "a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns"


def _paint_with(kernel: str, fuse: str, monkeypatch):
    from vecworld.config import FLUX_STEPS, FLUX_STYLE
    from vecworld.flux.painter import FluxPainter

    monkeypatch.setenv("VECWORLD_KERNEL", kernel)
    monkeypatch.setenv("VECWORLD_FUSE", fuse)
    p = FluxPainter()
    try:
        assert p.kernel == kernel and p.fused == (fuse != "off")
        with torch.inference_mode():
            e = p._embed([(f"{PROMPT}, {FLUX_STYLE}", 1.0)])
            grid = _bench().run_tile512(p, e, FLUX_STEPS, seed=0)
            lat = p._unpack(grid).to(p._dtype) / p._sf + p._shift
            px = p.vae.decode(lat).sample[0].float().clamp(-1, 1)
        return ((px.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    finally:
        p.close()
        del p
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("kernel", ["mma", "w4a8"])
def test_flux_fused_image_matches(monkeypatch, tmp_path, kernel):
    """Same seed at 512, fused vs unfused on the same kernel: PSNR >= 30 dB (the brief's bar). The fused glue is
    bit-identical to diffusers', so the pictures are identical too (infinite PSNR) under both kernels, the lossy w4a8
    included; the sheet (unfused | fused | 8x |difference|) is saved for an eye check."""
    from PIL import Image

    ref = _paint_with(kernel, "off", monkeypatch)
    ours = _paint_with(kernel, "on", monkeypatch)
    diff = np.clip(np.abs(ref.astype(np.int16) - ours.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    sheet = tmp_path / f"fused_vs_unfused_{kernel}.png"
    Image.fromarray(np.concatenate([ref, ours, diff], 1)).save(sheet)
    mse = np.mean((ref.astype(np.float64) - ours.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
    print(f"\nPSNR fused vs unfused ({kernel}): {psnr:.2f} dB, identical: {np.array_equal(ref, ours)}; sheet: {sheet}")
    assert psnr >= 30
    assert np.array_equal(ref, ours)
