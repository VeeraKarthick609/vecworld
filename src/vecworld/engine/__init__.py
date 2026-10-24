"""The FLUX engine: hand-written CUDA kernels that replace pieces of the diffusers reference path.

ext.load_ext builds the CUDA extension (csrc/) once and caches it; q4k.Q4KLinear runs a linear straight from its
packed Q4_K super-blocks; swap.swap_linears puts it in place of a model's Q4_K linears (docs/engine/).
"""
import os

KERNEL_OFF = ("off", "reference")    # VECWORLD_KERNEL values that keep the diffusers path


def engine_kernel() -> str | None:
    """The engine kernel VECWORLD_KERNEL asks for, the one rule for the painter, the game and the bench: unset or
    empty, config.FLUX_KERNEL (the default, "w4a8"); "off" or "reference", None (diffusers' dequantize-then-matmul);
    otherwise a name from q4k.KERNELS, checked here so a typo fails before the minute-long model load."""
    from vecworld.config import FLUX_KERNEL

    name = os.environ.get("VECWORLD_KERNEL") or FLUX_KERNEL
    if name in KERNEL_OFF:
        return None
    from vecworld.engine.q4k import KERNELS

    if name not in KERNELS:
        raise ValueError(f"VECWORLD_KERNEL={name!r}; built kernels: {', '.join(KERNELS)} (or off)")
    return name


def load_engine_ext():
    """ext.load_ext(), with a hint on failure: the first call in a process may compile the extension (CUDA 13.4
    toolkit, ninja), and the painter calls this before its minute-long model load so a broken build fails at once."""
    from vecworld.engine.ext import load_ext

    try:
        return load_ext()
    except Exception as e:
        raise RuntimeError(f"the engine's CUDA extension failed to build or load ({type(e).__name__}: {e}); set "
                           f"VECWORLD_KERNEL=off to paint on the diffusers path instead") from e
