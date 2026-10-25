"""W4A8 (Task 5b): Q4_K weights x per-token INT8 activations on the INT8 tensor cores, against torch (gpu; run with
the game server stopped). W4A8 is lossy by design, so its GEMM is held to the arithmetic it is meant to do, a
fake-quantized x times the exact gguf weight, not to x @ W; the picture is judged by the walk gate (04-w4a8.md)."""
import re

import numpy as np
import pytest
import torch
from test_q4k import REAL, deq, rows

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gg():
    from vecworld.config import FLUX_GGUF
    from vecworld.engine.q4k import read_gguf

    return read_gguf(FLUX_GGUF)


def torch_quantize(x: torch.Tensor):
    """The definition, in torch: per token s = max|x_t| / 127, xq = round(x_t / s) (half to even, as rintf), and
    xsum = the sum of xq over each 32-wide sub-block. A zero row gets s = 0 and xq = 0."""
    xf = x.float()
    amax = xf.abs().amax(1)
    s = amax / torch.full_like(amax, 127)           # a true division (torch turns "/ 127" into "* (1 / 127)")
    safe = torch.where(s > 0, s, torch.ones_like(s))
    xq = torch.where(s[:, None] > 0, torch.round(xf / safe[:, None]), 0).clamp(-127, 127).to(torch.int8)
    xsum = xq.int().view(x.shape[0], -1, 32).sum(-1, dtype=torch.int32)
    return xq, s, xsum


def _x(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    x[:, 7] *= 30                                   # an outlier channel, as FLUX's activations have
    return x.to("cuda", torch.bfloat16)


# K = 16640 is past the 16384 values a block holds in registers: the two-pass path.
@pytest.mark.parametrize("m, k", [(1, 3072), (7, 3072), (333, 12288), (1280, 15360), (5, 16640)])
def test_quantize_act_matches_torch(m, k):
    from vecworld.engine.q4k import quantize_act

    x = _x(m, k, m)
    if m > 1:
        x[1] = 0                                    # a zero token
        x[-1] *= 1000                               # a loud one
    xq, s, xsum = quantize_act(x)
    rq, rs, rsum = torch_quantize(x)
    assert xq.dtype == torch.int8 and s.dtype == torch.float32 and xsum.dtype == torch.int32
    assert xq.shape == (m, k) and s.shape == (m,) and xsum.shape == (m, k // 32)
    assert torch.equal(s, rs)
    assert torch.equal(xq, rq)
    assert torch.equal(xsum, rsum)


def fake_quant_ref(xq, s, w, bias):
    """(xq * s) @ deq(W)^T (+ bias) in fp64: what W4A8 computes, exactly."""
    y = (xq.double() * s.double()[:, None]) @ deq(w).cuda().double().T
    return y if bias is None else y + bias.double()


# The integer part (xq . q per sub-block) is exact; only the fp32 rescale (d * sc, dmin * m, s) and its order differ
# from the reference, so the fp32 output agrees to ~1e-6. A misplaced fragment or a wrong scale is far above 1e-3.
FAKE_QUANT_BAR = 1e-3


def _w4a8(w, m, seed, bias=True, tile=0):
    from vecworld.engine.ext import load_ext
    from vecworld.engine.q4k import quantize_act

    x = _x(m, w.shape[1], seed)
    g = torch.Generator().manual_seed(seed + 1)
    b = (torch.randn(w.shape[0], generator=g) * 0.1).to("cuda", torch.bfloat16) if bias else None
    packed = torch.from_numpy(np.ascontiguousarray(w.data)).cuda()
    xq, s, xsum = quantize_act(x)
    out = load_ext().q4k_gemm_w4a8(xq, s, xsum, packed, b, tile_n=tile, out_fp32=True)
    return out, fake_quant_ref(xq, s, w, b), b


def _err(out, ref):
    return float((out.double() - ref).abs().max() / ref.abs().max())


@pytest.mark.parametrize("tile", [0, 64, 128])
@pytest.mark.parametrize("name, m", [(REAL[0], 256), (REAL[1], 256), (REAL[2], 256), (REAL[1], 1333),
                                     (REAL[2], 1280)])
def test_w4a8_gemm_matches_fake_quant(gg, name, m, tile):
    out, ref, _ = _w4a8(gg[name], m, m, tile=tile)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    err = _err(out, ref)
    print(f"\nw4a8 {tile or 'auto'} vs fake quant, {name} M={m}: max-rel {err:.2e}")
    assert err < FAKE_QUANT_BAR


@pytest.mark.parametrize("tile", [0, 64, 128])
@pytest.mark.parametrize("n", [1000, 1001])
@pytest.mark.parametrize("m", [1, 7, 333, 1792])
def test_w4a8_gemm_ragged(gg, m, n, tile):
    """Review Focus 2: ragged M (tokens x windows), N no tile multiple, even (the bf16 pair store) and odd (the bias
    tail), with a bias, both tile shapes forced and the launcher's pick; and Q4KLinear's bf16 output, leading dims
    kept, is the kernel's fp32 output rounded."""
    from vecworld.engine.q4k import GEMV_MAX_M, Q4KLinear

    t = rows(gg[REAL[0]], n)
    out, ref, b = _w4a8(t, m, m, tile=tile)
    assert out.shape == (m, n)
    err = _err(out, ref)
    print(f"\nw4a8 {tile or 'auto'} M={m} N={n}: max-rel {err:.2e}")
    assert err < FAKE_QUANT_BAR
    if tile == 0:
        x = _x(m, t.shape[1], m)
        y = Q4KLinear(t, b.cpu(), "w4a8").cuda()(x.view(1, m, -1))
        assert y.shape == (1, m, n) and y.dtype == torch.bfloat16 and torch.isfinite(y).all()
        if m > GEMV_MAX_M:                          # up to it the linear runs the GEMV (bf16, exact)
            assert torch.equal(y[0], out.to(torch.bfloat16))


def test_w4a8_outlier_token(gg):
    """Review Focus 6: one token x1000 changes only its own scale, so every other token's output is bit-identical."""
    from vecworld.engine.q4k import Q4KLinear

    t = rows(gg[REAL[0]], 1001)
    lin = Q4KLinear(t, (torch.randn(1001) * 0.1).to(torch.bfloat16), "w4a8").cuda()
    x = _x(333, 3072, 6)
    loud = x.clone()
    loud[100] *= 1000
    a, b = lin(x), lin(loud)
    keep = torch.arange(333, device="cuda") != 100
    assert torch.equal(a[keep], b[keep])
    assert not torch.equal(a[100], b[100])


def test_w4a8_keep_a16(gg):
    """swap_linears(..., keep_a16=patterns): linears whose module name matches an fnmatch pattern stay on mma, the
    rest of the Q4_K linears go to the asked kernel; the same weights, the same parameters."""
    import gguf

    from vecworld.engine.q4k import GGUFTensor, Q4KLinear
    from vecworld.engine.swap import swap_linears

    q4 = rows(gg[REAL[0]], 64)
    names = ["blocks.0.ff.net.2", "blocks.1.ff.net.2", "blocks.0.attn.to_q", "single.0.proj_out", "single.1.proj_out"]
    m = torch.nn.Module()
    for n in names:
        parent = m
        *path, leaf = n.split(".")
        for p in path:
            if not hasattr(parent, p):
                parent.add_module(p, torch.nn.Module())
            parent = getattr(parent, p)
        parent.add_module(leaf, torch.nn.Linear(3072, 64, dtype=torch.bfloat16))
    table = {f"{n}.weight": GGUFTensor(f"{n}.weight", gguf.GGMLQuantizationType.Q4_K, q4.shape, q4.data)
             for n in names}
    assert swap_linears(m, table, "w4a8", keep_a16=("blocks.*.ff.net.2", "single.1.*")) == 5
    got = {n: m.get_submodule(n).kernel for n in names}
    assert all(isinstance(m.get_submodule(n), Q4KLinear) for n in names)
    assert got == {"blocks.0.ff.net.2": "mma", "blocks.1.ff.net.2": "mma", "blocks.0.attn.to_q": "w4a8",
                   "single.0.proj_out": "w4a8", "single.1.proj_out": "mma"}
    m2 = torch.nn.Module()
    m2.lin = torch.nn.Linear(3072, 64, dtype=torch.bfloat16)
    assert swap_linears(m2, {"lin.weight": table[f"{names[0]}.weight"]}, "w4a8") == 1   # default: keep nothing
    assert m2.lin.kernel == "w4a8"
    m3 = torch.nn.Module()
    m3.lin = torch.nn.Linear(3072, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"blocks\.\*\.ff\.net\.3"):           # a typo keeps nothing: refused
        swap_linears(m3, {"lin.weight": table[f"{names[0]}.weight"]}, "w4a8", keep_a16=("blocks.*.ff.net.3",))
    assert type(m3.lin) is torch.nn.Linear                                     # nothing swapped


def test_w4a8_nonfinite_token(gg):
    """An inf or NaN in a token makes that token's outputs NaN (as on the bf16 paths), not clamped finite numbers,
    and leaves every other token bit-identical."""
    from vecworld.engine.q4k import Q4KLinear, quantize_act

    lin = Q4KLinear(rows(gg[REAL[0]], 1000), None, "w4a8").cuda()
    x = _x(64, 3072, 9)
    ref = lin(x)
    for bad in (float("inf"), float("nan")):
        xb = x.clone()
        xb[10, 5] = bad
        _, s, _ = quantize_act(xb)
        y = lin(xb)
        assert torch.isnan(s[10]) and torch.isnan(y[10]).all(), bad
        keep = torch.arange(64, device="cuda") != 10
        assert torch.equal(y[keep], ref[keep])


def test_w4a8_misaligned(gg):
    """As Task 4: Q4KLinear copies a misaligned activation and computes the same answer, but a misaligned packed
    weight raises naming the module, never copied; the raw kernels reject both."""
    from vecworld.engine.ext import load_ext
    from vecworld.engine.q4k import Q4KLinear, quantize_act

    t = rows(gg[REAL[0]], 64)
    packed = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
    x = _x(64, 3072, 3)
    xm = torch.empty(x.numel() + 1, device="cuda", dtype=torch.bfloat16)[1:].view_as(x)
    xm.copy_(x)
    wm = torch.empty(packed.numel() + 4, device="cuda", dtype=torch.uint8)[4:].view_as(packed)
    wm.copy_(packed)
    with pytest.raises(RuntimeError, match="aligned"):
        quantize_act(xm)
    xq, s, xsum = quantize_act(x)
    with pytest.raises(RuntimeError, match="aligned"):
        load_ext().q4k_gemm_w4a8(xq, s, xsum, wm)
    lin = Q4KLinear(t, None, "w4a8").cuda()
    assert torch.equal(lin(xm), lin(x))
    lin.qweight.data = wm
    with pytest.raises(RuntimeError, match=re.escape(t.name) + ".*aligned"):
        lin(x)


def test_w4a8_is_a_kernel(monkeypatch):
    """VECWORLD_KERNEL=w4a8 selects it through the one dispatch rule (vecworld.engine.engine_kernel)."""
    from vecworld.engine import engine_kernel

    monkeypatch.setenv("VECWORLD_KERNEL", "w4a8")
    assert engine_kernel() == "w4a8"
