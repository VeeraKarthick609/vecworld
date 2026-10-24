"""Q4_K linears on the engine's kernels, and the GGUF tensors they are built from."""
from dataclasses import dataclass

import numpy as np
import torch
from gguf import GGMLQuantizationType

QK_K, Q4K_BYTES = 256, 144          # weights and bytes per Q4_K super-block
KERNELS = ("naive", "tiled")       # later stages add "mma", "w4a8"
# Below this many rows (tokens x windows) "tiled" runs the naive kernel: a 128-row tile would be mostly padding. FLUX's
# modulation linears run at M = 1 per window (4.0 ms tiled vs 0.7 ms naive for 3072 -> 18432); the crossover measured
# at M = 24..32 (docs/engine/02-q4k-tiled.md).
TILED_MIN_M = 32


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


def _aligned_activation(x: torch.Tensor) -> torch.Tensor:
    """x, or an aligned copy of it: the kernels read x with 16-byte vector loads. x is an activation, made fresh each
    call, and fresh CUDA allocations are always aligned; only an exotic view that starts mid-allocation gets copied,
    which costs one activation-sized copy. (A misaligned packed *weight* is never copied: see Q4KLinear.forward.)"""
    return x if x.data_ptr() % 16 == 0 else x.clone()


class Q4KLinear(torch.nn.Module):
    """A linear layer computed straight from its packed Q4_K super-blocks: forward(x bf16 [..., in]) ->
    bf16 [..., out]. kernel picks the stage (KERNELS).

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
        kernel = "naive" if self.kernel == "tiled" and x2.shape[0] < TILED_MIN_M else self.kernel
        if self.qweight.data_ptr() % 16:
            # Loud on purpose: a misaligned weight means whatever laid it out (e.g. a slot buffer carving views) is
            # wrong. Copying it here would hide that behind a weight-sized copy on every call.
            raise RuntimeError(f"Q4KLinear {self.weight_name}: the packed weight must be 16-byte aligned (vector "
                               f"loads); its address is {self.qweight.data_ptr() % 16} bytes off")
        y = getattr(load_ext(), f"q4k_gemm_{kernel}")(x2, self.qweight, b)
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, kernel={self.kernel}"
