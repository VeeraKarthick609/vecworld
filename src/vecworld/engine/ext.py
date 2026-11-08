"""Build and load the engine's CUDA extension (JIT through torch.utils.cpp_extension, cached on disk)."""
import functools
import os
from pathlib import Path

CUDA_HOME = "/usr/local/cuda-13.4"   # torch is cu130; the nvcc on PATH (12.4) is too old for it
ARCH = "8.6"                         # RTX 3050 Ti (sm_86)
CSRC = Path(__file__).parent / "csrc"


@functools.cache
def load_ext():
    """The compiled module (q4k_dequant, q4k_gemm_naive / tiled / mma / w4a8, q4k_quantize_act, q4k_gemv; glue_rnm,
    glue_gated_residual, glue_qk_norm_rope; SlotRing, host_pinned, graph_launch). The first call in a process builds it, or loads the build cached under
    ~/.cache/torch_extensions when the sources haven't changed."""
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH
    import torch.utils.cpp_extension as cpp

    cpp.CUDA_HOME = CUDA_HOME        # read once at import; it may have been imported before CUDA_HOME was set
    sources = [str(CSRC / f) for f in ("bindings.cpp", "q4k.cu", "glue.cu", "stream.cpp")]   # + w4a8_quant.cuh, included
    return cpp.load(name="vecworld_engine", sources=sources, extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
                    verbose=False)
