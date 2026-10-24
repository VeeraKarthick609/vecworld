"""Put Q4KLinear in place of a model's Q4_K linears, leaving every other layer on the reference path."""
import torch
from gguf import GGMLQuantizationType

from vecworld.engine.q4k import QK_K, GGUFTensor, Q4KLinear


def gguf_linears(model: torch.nn.Module) -> dict[str, GGUFTensor]:
    """The quantized weights of a model loaded through diffusers' GGUF quantizer, keyed "<module path>.weight".
    The data share the parameters' storage (no copy).

    diffusers renames and splits the file's tensors when it converts the checkpoint (a double block's qkv becomes
    to_q / to_k / to_v, a single block's linear1 four linears), so the file's own names don't name the model's
    linears; the converted GGUFParameters do, and they keep the file's raw super-blocks and quant type.
    """
    out = {}
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        qt = getattr(w, "quant_type", None)
        if isinstance(mod, torch.nn.Linear) and qt is not None:
            out[f"{name}.weight"] = GGUFTensor(f"{name}.weight", GGMLQuantizationType(qt),
                                               (mod.out_features, mod.in_features), w.as_tensor())
    return out


def swap_linears(model: torch.nn.Module, gguf: dict[str, GGUFTensor], kernel: str) -> int:
    """Replace each nn.Linear whose weight gguf["<module path>.weight"] is Q4_K, of the linear's own shape, with
    a Q4KLinear on `kernel` (keeping its bias). Every other module stays as it was: other GGUF types (Q5_K, F16,
    F32, ...), a weight missing from gguf, a shape mismatch, in_features not a multiple of 256. Returns how many
    linears were replaced."""
    todo = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        t = gguf.get(f"{name}.weight")
        if (t is None or t.type != GGMLQuantizationType.Q4_K
                or tuple(t.shape) != (mod.out_features, mod.in_features) or mod.in_features % QK_K):
            continue
        todo.append((name, mod, t))
    for name, mod, t in todo:
        parent, _, child = name.rpartition(".")
        q = Q4KLinear(t, mod.bias, kernel)
        setattr(model.get_submodule(parent) if parent else model, child, q)
    return len(todo)
