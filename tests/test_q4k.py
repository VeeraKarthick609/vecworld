"""The hand-written Q4_K kernels against gguf-py's dequant + torch.matmul, on the real FLUX tensors (gpu; run
with the game server stopped)."""
import gc
import re

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

# The GEMM stages built so far. Every unit test runs on each; "naive" (Task 3) is the reference the later ones are
# also checked against directly.
KERNELS = ["naive", "tiled"]

# One real Q4_K tensor per in_features the file holds (3072, 12288, 15360).
REAL = ["double_blocks.0.img_attn.proj.weight", "double_blocks.0.img_mlp.2.weight", "single_blocks.0.linear2.weight"]


@pytest.fixture(scope="module")
def gg():
    from vecworld.config import FLUX_GGUF
    from vecworld.engine.q4k import read_gguf

    return read_gguf(FLUX_GGUF)


def deq(t) -> torch.Tensor:
    """gguf-py's reference dequant of a GGUFTensor, fp32 [out, in] on the CPU."""
    import gguf

    return torch.from_numpy(gguf.quants.dequantize(np.asarray(t.data), t.type).reshape(t.shape))


def rows(t, n):
    """The first n rows (outputs) of a Q4_K tensor: rows are independent super-block runs."""
    from vecworld.engine.q4k import GGUFTensor

    return GGUFTensor(f"{t.name}[:{n}]", t.type, (n, t.shape[1]), np.asarray(t.data[:n]))


def rel_err(out, ref) -> float:
    return float((out.float() - ref).abs().max() / ref.abs().max())


@pytest.mark.parametrize("name", REAL)
def test_q4k_dequant_matches_gguf(gg, name):
    """Our device decode (the one the GEMM uses) is bit-exact against gguf.quants.dequantize."""
    import gguf

    from vecworld.engine.ext import load_ext

    t = gg[name]
    assert t.type == gguf.GGMLQuantizationType.Q4_K
    packed = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    ours = load_ext().q4k_dequant(packed, t.shape[1]).cpu()
    ref = deq(t)
    assert ours.shape == ref.shape == t.shape
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("name", REAL)
def test_q4k_gemm_matches_reference(gg, name, kernel):
    from vecworld.engine.q4k import Q4KLinear

    t = gg[name]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, t.shape[1], generator=g).to("cuda", torch.bfloat16)
    bias = (torch.randn(t.shape[0], generator=g) * 0.1).to("cuda", torch.bfloat16)
    lin = Q4KLinear(t, bias, kernel).cuda()
    ref = x.float() @ deq(t).cuda().T + bias.float()
    out = lin(x)
    assert out.dtype == torch.bfloat16 and out.shape == (256, t.shape[0])
    assert rel_err(out, ref) < 2e-2


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("m", [1, 7, 32, 333, 1792])
def test_q4k_gemm_ragged(gg, m, kernel):
    """Ragged M (tokens x windows) and an out_features that is no tile multiple (1000), with leading dims."""
    from vecworld.engine.q4k import Q4KLinear

    t = rows(gg[REAL[0]], 1000)
    g = torch.Generator().manual_seed(m)
    x = torch.randn(1, m, t.shape[1], generator=g).to("cuda", torch.bfloat16)
    lin = Q4KLinear(t, None, kernel).cuda()
    out = lin(x)
    ref = x.float() @ deq(t).cuda().T
    assert out.shape == (1, m, 1000)
    assert rel_err(out, ref) < 2e-2
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("m", [7, 333])
def test_q4k_gemm_odd_n(gg, m, kernel):
    """out_features 1001 (not a multiple of 4): the tiled epilogue's per-element store and bias tail, on both
    kernels, against x @ deq(W)^T + bias."""
    from vecworld.engine.q4k import Q4KLinear

    t = rows(gg[REAL[0]], 1001)
    g = torch.Generator().manual_seed(m)
    x = torch.randn(m, t.shape[1], generator=g).to("cuda", torch.bfloat16)
    bias = (torch.randn(1001, generator=g) * 0.1).to("cuda", torch.bfloat16)
    out = Q4KLinear(t, bias, kernel).cuda()(x)
    ref = x.float() @ deq(t).cuda().T + bias.float()
    assert out.shape == (m, 1001)
    assert rel_err(out, ref) < 2e-2


def test_q4k_tiled_small_m_runs_naive(gg, monkeypatch):
    """Q4KLinear on "tiled" sends M < TILED_MIN_M (the M = 1 modulation linears) to the naive kernel, and M >= it to
    the tiled one. (The raw tiled kernel is checked at small M in test_q4k_gemm_matches_naive.)"""
    import vecworld.engine.ext as ext_mod
    from vecworld.engine.q4k import TILED_MIN_M, Q4KLinear

    real, calls = ext_mod.load_ext(), []

    class Spy:
        def __getattr__(self, name):
            calls.append(name)
            return getattr(real, name)

    monkeypatch.setattr(ext_mod, "load_ext", lambda: Spy())
    lin = Q4KLinear(rows(gg[REAL[0]], 64), None, "tiled").cuda()
    for m in (1, TILED_MIN_M - 1, TILED_MIN_M, 1280):
        lin(torch.randn(m, 3072, device="cuda", dtype=torch.bfloat16))
    assert TILED_MIN_M == 32
    assert calls == ["q4k_gemm_naive", "q4k_gemm_naive", "q4k_gemm_tiled", "q4k_gemm_tiled"]


# Every later stage against the naive reference directly, on the same real weights. Both decode each weight
# bit-exactly and sum in fp32; only the summation order differs, so the bf16 outputs agree to about one bf16 step of
# the largest output (2^-8 = 3.9e-3 relative to it). A decode or indexing bug is orders of magnitude above this bar.
NAIVE_BAR = 8e-3


@pytest.mark.parametrize("kernel", [k for k in KERNELS if k != "naive"])
@pytest.mark.parametrize("name, m, n", [(REAL[0], 256, None), (REAL[1], 256, None), (REAL[2], 256, None),
                                        (REAL[0], 1, 1000), (REAL[0], 7, 1000), (REAL[0], 333, 1000),
                                        (REAL[0], 1792, 1000), (REAL[0], 31, 1000), (REAL[0], 128, 1000),
                                        (REAL[0], 129, 1000), (REAL[0], 7, 1001), (REAL[0], 333, 1001)])
def test_q4k_gemm_matches_naive(gg, name, m, n, kernel):
    from vecworld.engine.ext import load_ext

    t = gg[name] if n is None else rows(gg[name], n)
    g = torch.Generator().manual_seed(m)
    x = torch.randn(m, t.shape[1], generator=g).to("cuda", torch.bfloat16)
    bias = (torch.randn(t.shape[0], generator=g) * 0.1).to("cuda", torch.bfloat16)
    packed = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    ext = load_ext()
    ref = ext.q4k_gemm_naive(x, packed, bias)
    out = getattr(ext, f"q4k_gemm_{kernel}")(x, packed, bias)      # the kernel itself, whatever Q4KLinear picks
    assert out.shape == ref.shape == (m, t.shape[0])
    err = rel_err(out, ref.float())
    print(f"\n{kernel} vs naive, {t.name} M={m}: max-rel {err:.2e}")
    assert err < NAIVE_BAR


@pytest.mark.parametrize("kernel", KERNELS)
def test_q4k_gemm_misaligned(gg, kernel):
    """The kernels read x and the packed weight with 16-byte vector loads. The extension rejects a view whose base
    address is not 16-byte aligned (instead of a misaligned-address fault or a silent wrong read). Q4KLinear copies a
    misaligned activation and computes the right answer, but raises on a misaligned packed weight, naming it: a
    weight layout that breaks alignment must fail loudly, not cost a weight-sized copy on every call."""
    from vecworld.engine.ext import load_ext
    from vecworld.engine.q4k import Q4KLinear

    t = rows(gg[REAL[0]], 64)
    gemm = getattr(load_ext(), f"q4k_gemm_{kernel}")
    packed = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    x = torch.randn(64, t.shape[1], device="cuda", dtype=torch.bfloat16)     # 64 rows: "tiled" runs tiled
    xm = torch.empty(x.numel() + 1, device="cuda", dtype=torch.bfloat16)[1:].view_as(x)      # 2 bytes off
    xm.copy_(x)
    wm = torch.empty(packed.numel() + 4, device="cuda", dtype=torch.uint8)[4:].view_as(packed)  # 4 bytes off
    wm.copy_(packed)
    assert xm.is_contiguous() and wm.is_contiguous()
    with pytest.raises(RuntimeError, match="aligned"):
        gemm(xm, packed)
    with pytest.raises(RuntimeError, match="aligned"):
        gemm(x, wm)
    lin = Q4KLinear(t, None, kernel).cuda()
    ref = x.float() @ deq(t).cuda().T
    assert rel_err(lin(xm), ref) < 2e-2              # a misaligned activation is copied once, then computed
    lin.qweight.data = wm
    with pytest.raises(RuntimeError, match=re.escape(t.name) + ".*aligned"):   # an error, never copied
        lin(x)


def test_swap_skips_non_q4k(gg):
    """Review Focus 1: only Linear modules whose weight is Q4_K, with a matching shape, are replaced. Other GGUF
    types (Q5_K, F16, F32), a shape mismatch, a missing entry and non-Linear modules stay as they were."""
    import gguf

    from vecworld.engine.q4k import GGUFTensor, Q4KLinear
    from vecworld.engine.swap import swap_linears

    Q = gguf.GGMLQuantizationType
    q4 = rows(gg[REAL[0]], 64)                                         # 3072 -> 64, real Q4_K
    m = torch.nn.Module()
    m.q4 = torch.nn.Linear(3072, 64, dtype=torch.bfloat16)
    m.f16 = torch.nn.Linear(3072, 64, dtype=torch.bfloat16)
    m.f32 = torch.nn.Linear(3072, 64, dtype=torch.bfloat16)
    m.q5 = torch.nn.Linear(256, 32, dtype=torch.bfloat16)
    m.ragged = torch.nn.Linear(300, 64, dtype=torch.bfloat16)          # in_features not a multiple of 256
    m.missing = torch.nn.Linear(3072, 64, dtype=torch.bfloat16)
    m.norm = torch.nn.LayerNorm(3072)
    table = {
        "q4.weight": GGUFTensor("q4.weight", Q.Q4_K, q4.shape, q4.data),
        "f16.weight": GGUFTensor("f16.weight", Q.F16, (64, 3072), np.zeros((64, 3072), np.float16)),
        "f32.weight": GGUFTensor("f32.weight", Q.F32, (64, 3072), np.zeros((64, 3072), np.float32)),
        "q5.weight": GGUFTensor("q5.weight", Q.Q5_K, (32, 256), np.zeros((32, 176), np.uint8)),
        "ragged.weight": GGUFTensor("ragged.weight", Q.Q4_K, (64, 3072), q4.data),     # file shape != module's
        "norm.weight": GGUFTensor("norm.weight", Q.Q4_K, (64, 3072), q4.data),
    }
    before = {n: mod for n, mod in m.named_children() if n != "q4"}
    bias = m.q4.bias.detach().clone()
    assert swap_linears(m, table, "naive") == 1
    assert isinstance(m.q4, Q4KLinear)
    assert {n: mod for n, mod in m.named_children() if n != "q4"} == before        # the very same modules
    assert all(type(mod) is not Q4KLinear for n, mod in m.named_children() if n != "q4")
    # The packed bytes and the bias are frozen parameters (diffusers' streamed group offload moves parameters both
    # ways, but never offloads a block's buffers), and the bias is kept.
    params = dict(m.q4.named_parameters())
    assert set(params) == {"qweight", "bias"} and not list(m.q4.buffers())
    assert not any(p.requires_grad for p in params.values()) and params["qweight"].dtype == torch.uint8
    assert torch.equal(m.q4.bias, bias)
    m.cuda()
    x = torch.randn(5, 3072, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ deq(q4).cuda().T + bias.cuda().float()
    assert rel_err(m.q4(x), ref) < 2e-2


def test_swap_rejects_unknown_kernel(gg):
    from vecworld.engine.q4k import Q4KLinear

    with pytest.raises(ValueError):
        Q4KLinear(rows(gg[REAL[0]], 8), None, "fastest")


# --- Gates against the diffusers GGUF reference ------------------------------------------------------------------
# Three gates, from tight to loose, because errors compound. Diffusers dequantizes each weight and rounds it to bf16
# (via fp16) before cuBLAS; the engine keeps the exact fp32 weight and sums in its own order. Neither is wrong, but
# the last-bit differences grow through 57 blocks and again through 4 denoising steps. Measured 2026-10-24, naive vs
# diffusers (and plain torch.matmul on the exact gguf fp32 weights vs diffusers, the plan's own reference check):
#   one block (double 0 / 9, single 0 / 20): max-rel 5.5e-3 .. 8.3e-3 (the exact fp32 path: the same)
#   one whole forward pass (57 blocks):         max-rel 0.127, rel L2 4.2% (exact fp32 path: 0.139, 4.5%)
#   the 4-step picture at seed 0:               PSNR 29.87 dB (exact fp32 path 29.4 dB; naive vs it 30.7 dB)
# So a whole-pass max-rel bar or a 30 dB picture bar measures rounding policy, not kernel correctness. The primary
# gate is per block (max-rel < 2e-2, reusable by kernel name); the whole pass is held to rel L2 < 0.06 and the
# picture to 28 dB, with a sheet to judge by eye.

BLOCK_KERNELS = KERNELS            # the same gate holds every kernel to diffusers


@pytest.fixture(scope="module")
def blocks():
    """Double block 0 and single block 0 of the bare diffusers transformer (no offload), their inputs (seeded
    random activations, 1024 image + 256 text tokens, rotary embedding) and diffusers' outputs, fp32 on the CPU."""
    import os

    os.environ.pop("VECWORLD_KERNEL", None)    # load_transformer would swap the linears: this is the reference
    tr, _ = _bench().load_transformer()
    dev, L, T = "cuda", 1024, 256
    g = torch.Generator().manual_seed(0)
    hs, es = (torch.randn(1, n, 3072, generator=g).to(dev, torch.bfloat16) for n in (L, T))
    temb = torch.randn(1, 3072, generator=g).to(dev, torch.bfloat16)
    ids = torch.zeros(T + L, 3, device=dev)
    ids[T:, 1], ids[T:, 2] = torch.arange(L, device=dev) // 32, torch.arange(L, device=dev) % 32
    with torch.inference_mode():
        rot = tr.pos_embed.to(dev)(ids)
    kw = dict(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot)
    out = {}
    for name, blk in (("double0", tr.transformer_blocks[0]), ("single0", tr.single_transformer_blocks[0])):
        blk.to(dev)
        out[name] = (blk, _run_block(blk, kw))
        blk.to("cpu")
    del tr                                      # keep the two blocks only: a FluxPainter needs the RAM later
    gc.collect()
    yield out, kw
    out.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _run_block(blk, kw) -> torch.Tensor:
    with torch.inference_mode():
        o = blk(**kw)
    return torch.cat([x.float().flatten().cpu() for x in (o if isinstance(o, tuple) else (o,))])


@pytest.mark.parametrize("kernel", BLOCK_KERNELS)
@pytest.mark.parametrize("name", ["double0", "single0"])
def test_q4k_block_matches_reference(blocks, name, kernel):
    """The primary gate: one FLUX block with its Q4_K linears on `kernel` vs the same block on diffusers, same
    inputs, max-relative error < 2e-2 (naive measured 8.3e-3 double, 5.5e-3 single)."""
    from vecworld.engine.swap import gguf_linears, swap_linears

    out, kw = blocks
    blk, ref = out[name]
    orig = dict(blk.named_modules())
    try:
        assert swap_linears(blk, gguf_linears(blk), kernel) > 0
        blk.to("cuda")
        ours = _run_block(blk, kw)
    finally:                                   # put diffusers' linears back for the next kernel
        for n, mod in list(blk.named_modules()):
            if n and orig.get(n) is not mod:
                parent, _, child = n.rpartition(".")
                setattr(blk.get_submodule(parent) if parent else blk, child, orig[n])
        blk.to("cpu")
    err = rel_err(ours, ref)
    print(f"\n{name} {kernel}: max-rel {err:.2e}")
    assert torch.isfinite(ours).all()
    assert err < 2e-2


PROMPT = "a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns"


def _bench():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("bench_flux", Path(__file__).parents[1] / "scripts/bench_flux.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def _with_painter(kernel: str | None, monkeypatch, fn):
    """fn(painter, embedding) on a FluxPainter built with VECWORLD_KERNEL=kernel (None: the diffusers path), freed
    afterwards: two painters don't fit in RAM side by side."""
    from vecworld.config import FLUX_STYLE
    from vecworld.flux.painter import FluxPainter

    if kernel is None:
        monkeypatch.delenv("VECWORLD_KERNEL", raising=False)
    else:
        monkeypatch.setenv("VECWORLD_KERNEL", kernel)
    p = FluxPainter()
    try:
        assert p.kernel == kernel
        with torch.inference_mode():
            return fn(p, p._embed([(f"{PROMPT}, {FLUX_STYLE}", 1.0)]))
    finally:
        p.close()
        del p
        gc.collect()
        torch.cuda.empty_cache()


def _forward(p, e) -> torch.Tensor:
    """One transformer pass on a seed-0 noise window at sigma 1 (no denoise loop): the velocity, fp32 on the CPU."""
    from vecworld.config import FLUX_WIN

    g = torch.randn(1, 64, FLUX_WIN, FLUX_WIN, device=p._dev, generator=torch.Generator(p._dev).manual_seed(0))
    return p._forward(g, [e], 1.0).cpu()


_REFERENCE: dict = {}               # the diffusers results, computed once per module for every kernel


def _reference(fn, monkeypatch):
    if fn.__name__ not in _REFERENCE:
        _REFERENCE[fn.__name__] = _with_painter(None, monkeypatch, fn)
    return _REFERENCE[fn.__name__]


@pytest.mark.parametrize("kernel", KERNELS)
def test_flux_forward_matches_reference(monkeypatch, kernel):
    """One whole forward pass (57 blocks, no denoise loop), same inputs, `kernel` vs diffusers: relative L2 < 0.06
    (naive 4.2%; the exact fp32 path 4.5%). Max-rel is printed, not gated (0.127: compounding, see the note above)."""
    ref = _reference(_forward, monkeypatch)
    ours = _with_painter(kernel, monkeypatch, _forward)
    l2 = float((ours - ref).norm() / ref.norm())
    print(f"\nforward {kernel} vs reference: rel L2 {l2:.2e}, max-rel {rel_err(ours, ref):.2e}")
    assert torch.isfinite(ours).all()
    assert l2 < 0.06


def _paint(p, e) -> np.ndarray:
    """One 512 x 512 window, FLUX_STEPS steps from seed 0, decoded: RGB uint8."""
    from vecworld.config import FLUX_STEPS

    grid = _bench().run_tile512(p, e, FLUX_STEPS, seed=0)
    lat = p._unpack(grid).to(p._dtype) / p._sf + p._shift
    px = p.vae.decode(lat).sample[0].float().clamp(-1, 1)
    return ((px.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)


@pytest.mark.parametrize("kernel", KERNELS)
def test_flux_image_matches_reference(monkeypatch, tmp_path, kernel):
    """The 4-step sanity gate: same seed at 512, `kernel` vs diffusers, PSNR >= 28 dB (see the note above). The
    sheet (reference | kernel | 4x |difference|) is saved for an eye check."""
    from PIL import Image

    ref = _reference(_paint, monkeypatch)
    ours = _with_painter(kernel, monkeypatch, _paint)
    diff = np.clip(np.abs(ref.astype(np.int16) - ours.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    sheet = tmp_path / f"q4k_{kernel}_vs_reference.png"
    Image.fromarray(np.concatenate([ref, ours, diff], 1)).save(sheet)
    mse = np.mean((ref.astype(np.float64) - ours.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
    print(f"\nPSNR {kernel} vs reference: {psnr:.2f} dB; sheet: {sheet}")
    assert psnr >= 28
