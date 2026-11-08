"""The FLUX engine: hand-written CUDA kernels that replace pieces of the diffusers reference path.

ext.load_ext builds the CUDA extension (csrc/) once and caches it; q4k.Q4KLinear runs a linear straight from its
packed Q4_K super-blocks; swap.swap_linears puts it in place of a model's Q4_K linears (docs/engine/).
"""
import os

KERNEL_OFF = ("off", "reference")    # VECWORLD_KERNEL values that keep the diffusers path
FUSE_ON, FUSE_OFF = ("1", "on", "true", "yes"), ("0", "off", "false", "no")   # VECWORLD_FUSE / _GRAPHS values


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


def engine_fuse(kernel: str | None) -> bool:
    """Whether the FLUX blocks run their glue fused (vecworld.engine.blocks, docs/engine/06-glue-fusion.md), the one
    rule for the painter and the bench. VECWORLD_FUSE unset or empty: yes under every engine kernel, no under "off"
    (kernel None: the diffusers reference stays diffusers all the way); "0" / "off": no, under any kernel; "1" / "on":
    yes, under any kernel, "off" included (the fused blocks on diffusers' own linears, to test them apart)."""
    v = (os.environ.get("VECWORLD_FUSE") or "").strip().lower()
    if not v:
        return kernel is not None
    if v in FUSE_ON:
        return True
    if v in FUSE_OFF:
        return False
    raise ValueError(f"VECWORLD_FUSE={v!r}; use 1 / on or 0 / off (unset: on with an engine kernel)")


def engine_graphs(kernel: str | None) -> bool:
    """Whether the FLUX blocks replay as CUDA graphs (vecworld.engine.graphs, docs/engine/07-cuda-graphs.md), the one
    rule for the painter and the bench. Under an engine kernel the blocks always run from fixed-address weight slots
    (engine.slots); VECWORLD_GRAPHS unset, empty, "1" / "on": replayed as graphs (the default since Task 10a-g: other
    threads keep running while a pass replays, at about 3.9 s of first-use captures and some memory); "0" / "off": run
    eagerly on the slots. Under "off" (kernel None) the blocks stay on diffusers' group offload, so never graphs."""
    v = (os.environ.get("VECWORLD_GRAPHS") or "").strip().lower()
    if v and v not in FUSE_ON + FUSE_OFF:
        raise ValueError(f"VECWORLD_GRAPHS={v!r}; use 1 / on or 0 / off (unset: on with an engine kernel)")
    return kernel is not None and v not in FUSE_OFF


def engine_cache_threshold(kernel: str | None) -> float:
    """The first-block cache's threshold (vecworld.engine.cache, docs/engine/08-block-cache.md), the one rule for the
    painter, the walk and the bench: VECWORLD_CACHE_THRESHOLD (a number >= 0) or, unset or empty,
    config.FLUX_CACHE_THRESHOLD; 0 is off. Under "off" (kernel None: diffusers' group offload, whose prefetch order
    assumes every block runs) always 0."""
    from vecworld.config import FLUX_CACHE_THRESHOLD

    v = (os.environ.get("VECWORLD_CACHE_THRESHOLD") or "").strip()
    try:
        t = float(v) if v else float(FLUX_CACHE_THRESHOLD)
    except ValueError:
        raise ValueError(f"VECWORLD_CACHE_THRESHOLD={v!r}; use a number >= 0 (0: off)") from None
    if t < 0:
        raise ValueError(f"VECWORLD_CACHE_THRESHOLD={v!r}; use a number >= 0 (0: off)")
    return t if kernel is not None else 0.0


def engine_batch(kernel: str | None = "engine") -> int:
    """Windows per transformer pass (Task 9, docs/engine/09-batching.md), the one rule for the painter, the walk and the
    bench. Under an engine kernel (any name; the default, for callers without one such as the CPU fakes):
    VECWORLD_BATCH (a whole number >= 1) or, unset or empty, config.FLUX_BATCH. Under "off" (kernel None: diffusers'
    reference) always 1, whatever they say: diffusers' cuBLAS GEMMs inside the blocks round by M, so a batched window
    would paint another picture there, and the reference is pinned to its one-window-a-pass, batch-invariant picture
    (the engine's rowwise outer layers make any batch paint that same picture)."""
    from vecworld.config import FLUX_BATCH

    v = (os.environ.get("VECWORLD_BATCH") or "").strip()
    if v and (not v.isdigit() or int(v) < 1):
        raise ValueError(f"VECWORLD_BATCH={v!r}; use a whole number >= 1 (unset: config.FLUX_BATCH)")
    if kernel is None:
        return 1
    return int(v) if v else int(FLUX_BATCH)


def keep_for(kernel: str | None) -> tuple[str, ...]:
    """The linears (fnmatch patterns of module names, for swap_linears' keep_a16) that stay on "mma" under `kernel`:
    config.FLUX_A16_KEEP under "w4a8" (W4A8's per-layer policy, docs/engine/04-w4a8.md), nothing under any other
    kernel, so VECWORLD_KERNEL=mma / tiled / naive stay pure. The one rule for the painter and the bench."""
    from vecworld.config import FLUX_A16_KEEP

    return tuple(FLUX_A16_KEEP) if kernel == "w4a8" else ()


def load_engine_ext():
    """ext.load_ext(), with a hint on failure: the first call in a process may compile the extension (CUDA 13.4
    toolkit, ninja), and the painter calls this before its minute-long model load so a broken build fails at once."""
    from vecworld.engine.ext import load_ext

    try:
        return load_ext()
    except Exception as e:
        raise RuntimeError(f"the engine's CUDA extension failed to build or load ({type(e).__name__}: {e}); set "
                           f"VECWORLD_KERNEL=off to paint on the diffusers path instead") from e
