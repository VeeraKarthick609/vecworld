"""Q4_K linears on the engine's kernels, and the GGUF tensors they are built from."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from gguf import GGMLQuantizationType

QK_K, Q4K_BYTES = 256, 144          # weights and bytes per Q4_K super-block
KERNELS = ("naive", "tiled", "mma", "w4a8")
# Up to this many rows (tokens x windows) every kernel hands the call to q4k_gemv, a memory-bound GEMV that streams
# the Q4_K bytes once with bf16 activations (exact, as naive). Only the modulation linears are this small: one row per
# image, and a pass holds at most 8 windows (docs/engine/05-modulation-gemv.md).
GEMV_MAX_M = 8
# Below this many rows "tiled" runs the naive kernel: a 128-row tile would be mostly padding (crossover measured at
# M = 24..32, docs/engine/02-q4k-tiled.md). (mma and w4a8 crossed naive at M = 6..8, now under GEMV_MAX_M.)
TILED_MIN_M = 32
SMALL_M = {"tiled": TILED_MIN_M}     # GEMV_MAX_M < M < this: the kernel hands the call to naive


@dataclass(frozen=True)
class GGUFTensor:
    """One GGUF tensor. shape is in torch order ((out, in) for a weight); data holds the raw bytes: uint8
    [rows, bytes per row] for a quantized type, the values themselves for F16 / F32 (numpy or torch)."""
    name: str
    type: GGMLQuantizationType
    shape: tuple[int, ...]
    data: object


def read_gguf(path) -> dict[str, GGUFTensor]:
    """Every tensor of a GGUF file by its GGUF name. data are memory-mapped views of the file (read when used), so
    this costs no RAM up front."""
    import gguf

    r = gguf.GGUFReader(str(path))
    return {t.name: GGUFTensor(t.name, t.tensor_type, tuple(int(d) for d in reversed(t.shape)), t.data)
            for t in r.tensors}


def _packed(weight: GGUFTensor) -> torch.Tensor:
    rows, cols = weight.shape
    d = weight.data
    t = d if isinstance(d, torch.Tensor) else torch.from_numpy(np.array(d, np.uint8))  # copies a read-only map
    t = t.view(torch.uint8)
    if t.shape != (rows, cols // QK_K * Q4K_BYTES):
        raise ValueError(f"{weight.name}: {tuple(t.shape)} bytes don't hold a Q4_K {rows} x {cols} weight")
    return t


def quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W4A8's activation side, one kernel: x bf16 [M, K] -> (xq int8 [M, K], s fp32 [M], xsum int32 [M, K // 32]).
    Per token s = max|x_t| / 127 and xq = round(x_t / s) (half to even); xsum is the sum of xq over each 32-wide
    sub-block, for the dmin * m_j term (docs/engine/04-w4a8.md). A zero token gets s = 0, xq = 0; a token
    holding an inf or NaN gets s = NaN (xq = 0), so its outputs come out NaN."""
    from vecworld.engine.ext import load_ext

    return tuple(load_ext().q4k_quantize_act(x))


_QUANTIZED = "_vecworld_w4a8_act"     # the tensor attribute attach_quantized sets
# Python-time counts of what w4a8 Q4KLinears did with their input (Task 7: proves the fused quantize is used, also
# under CUDA graphs). They count eager calls and graph captures, never a replay (a replay runs no Python):
# "attached" used the glue's attached quantization, "stale" found one but refused it (x written since, or an
# inference tensor, whose writes are not tracked), "quantized" quantized x itself.
CALLS: Counter = Counter()


def attach_quantized(x: torch.Tensor, q: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Attach quantize_act(x)'s outputs (xq, s, xsum) to the tensor object x, for a w4a8 Q4KLinear that is handed x
    itself: it then skips its own quantize. The fused glue (glue.residual_norm_modulate, quantize=True) computes them in
    the pass that writes x. Linears on any other kernel (mma: the keep-list) ignore them and read x in bf16.

    The attachment records x's version counter: once x is written in place (x.mul_(...), or a CUDA graph's static
    buffer refilled with copy_), it is stale and quantized_of no longer returns it. An inference tensor
    (torch.inference_mode) has no version counter, so its attachment is never returned: the GraphRunner runs the
    blocks outside inference mode for this (engine/graphs.py). Returns x."""
    setattr(x, _QUANTIZED, (q, None if x.is_inference() else x._version))
    return x


def quantized_of(x: torch.Tensor):
    """The quantization attach_quantized put on x, or None: also None when x was written since (its version moved)
    or when x is an inference tensor (no version to check)."""
    rec = getattr(x, _QUANTIZED, None)
    if rec is None:
        return None
    q, version = rec
    if version is None or x.is_inference() or x._version != version:
        return None
    return q


def check_aligned(t: torch.Tensor, what: str) -> None:
    """Raise unless t starts on a 16-byte boundary (the kernels' vector loads). Loud on purpose: a misaligned weight
    means whatever laid it out (e.g. engine.slots carving a block's tensors out of a slot) is wrong, and copying it
    aside would hide that behind a weight-sized copy on every call."""
    if t.data_ptr() % 16:
        raise RuntimeError(f"{what} must be 16-byte aligned (vector loads); its address is {t.data_ptr() % 16} "
                           f"bytes off")


def _aligned_activation(x: torch.Tensor) -> torch.Tensor:
    """x, or an aligned copy of it: the kernels read x with 16-byte vector loads. x is an activation, made fresh each
    call, and fresh CUDA allocations are always aligned; only an exotic view that starts mid-allocation gets copied,
    which costs one activation-sized copy. (A misaligned packed *weight* is never copied: see Q4KLinear.forward.)"""
    return x if x.data_ptr() % 16 == 0 else x.clone()


class Q4KLinear(torch.nn.Module):
    """A linear layer computed straight from its packed Q4_K super-blocks: forward(x bf16 [..., in]) ->
    bf16 [..., out]. kernel picks the stage (KERNELS) for M > GEMV_MAX_M flattened rows; up to it every kernel runs
    the GEMV (q4k_gemv).

    The packed bytes (qweight, uint8 [out, in / 256 * 144]) and the bias are frozen Parameters, not buffers:
    diffusers' streamed group offload (0.41, use_stream=True) copies a block's buffers to the GPU on onload but
    only puts its parameters back on offload, so buffers pile up on the GPU block after block (out of memory by
    the 20th block). As parameters they come and go like the GGUF weights they replace."""

    def __init__(self, weight: GGUFTensor, bias: torch.Tensor | None, kernel: str):
        super().__init__()
        if kernel not in KERNELS:
            raise ValueError(f"unknown Q4_K kernel {kernel!r}; built: {', '.join(KERNELS)}")
        if weight.type != GGMLQuantizationType.Q4_K or len(weight.shape) != 2 or weight.shape[1] % QK_K:
            raise ValueError(f"{weight.name}: not a 2-D Q4_K weight ({weight.type.name}, {weight.shape})")
        self.out_features, self.in_features = weight.shape
        if bias is not None and tuple(bias.shape) != (self.out_features,):
            raise ValueError(f"{weight.name}: bias {tuple(bias.shape)} for {self.out_features} outputs")
        self.kernel = kernel
        self.weight_name = weight.name
        self.qweight = torch.nn.Parameter(_packed(weight), requires_grad=False)
        self.bias = None if bias is None else torch.nn.Parameter(bias.detach(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vecworld.engine.ext import load_ext

        lead = x.shape[:-1]
        x2 = _aligned_activation(x.reshape(-1, self.in_features).to(torch.bfloat16).contiguous())
        b = None if self.bias is None else self.bias.to(torch.bfloat16).contiguous()
        check_aligned(self.qweight, f"Q4KLinear {self.weight_name}: the packed weight")
        m = x2.shape[0]
        kernel = "gemv" if m <= GEMV_MAX_M else "naive" if m < SMALL_M.get(self.kernel, 0) else self.kernel
        if kernel == "gemv":
            y = load_ext().q4k_gemv(x2, self.qweight, b)
        elif kernel == "w4a8":
            gemm = load_ext().q4k_gemm_w4a8
            q = quantized_of(x)                    # the fused glue's quantization of this very x, if it made one
            if q is None or q[0].shape != x2.shape:
                CALLS["stale" if hasattr(x, _QUANTIZED) else "quantized"] += 1
                q = quantize_act(x2)
            else:
                CALLS["attached"] += 1
            y = gemm(*q, self.qweight, b)
        else:
            y = getattr(load_ext(), f"q4k_gemm_{kernel}")(x2, self.qweight, b)
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, kernel={self.kernel}"
