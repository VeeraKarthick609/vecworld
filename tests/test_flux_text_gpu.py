"""T5-XXL on the engine (vecworld.flux.text; Task 10c, docs/engine/12-t5-and-results.md; gpu, run with the game server
stopped).

The engine path streams the encoder's 24 blocks from the GGUF on the SSD (BlockStreamer + the 10b SsdSource, O_DIRECT)
and runs their linears as dequantize-then-matmul on the GPU (the file's linears are Q5_K / Q6_K: no engine kernel takes
those). Its bar is the plan's: cosine >= 0.999 against the Task 1 path (T5Text engine="pread": every tensor pread into
host memory, then dequantized on the GPU), and a cold encode in <= 5 s on AC that leaves the file out of the page cache.
"""
import glob
import os
import time

import pytest
import torch

pytestmark = pytest.mark.gpu

TEXTS = ["dense haunted forest at night, digital painting, detailed",
         "desert oasis with palm trees at dusk",
         "snowy mountain monastery, prayer flags, a narrow trail in the foreground"]


def _on_ac() -> bool:
    return "1" in [open(p).read().strip() for p in glob.glob("/sys/class/power_supply/A*/online")]


def _evict(path) -> None:
    fd = os.open(os.path.realpath(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def test_t5_engine_matches_cpu():
    """The engine's embeddings against the Task 1 path's, three prompts: cosine >= 0.999 over the whole sequence
    (the plan's bar), and per token."""
    from vecworld.config import FLUX_TEXT_LEN
    from vecworld.flux.text import T5Text

    ref, eng = T5Text(cache=0, engine="pread"), T5Text(cache=0, engine="engine")
    assert (ref.engine, eng.engine) == ("pread", "engine")
    for text in TEXTS:
        a, b = ref.encode(text), eng.encode(text)
        assert b.dtype == torch.bfloat16 and b.device.type == "cpu" and tuple(b.shape) == (1, FLUX_TEXT_LEN, 4096)
        a, b = a.double(), b.double()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        tok = torch.nn.functional.cosine_similarity(a[0], b[0], dim=-1).min().item()
        print(f"\n{text[:30]!r}: cosine {cos:.6f}, worst token {tok:.6f}, max |diff| {(a - b).abs().max().item():.4g}, "
              f"identical {torch.equal(a, b)}")
        assert cos >= 0.999 and tok >= 0.999, (text, cos, tok)


def test_t5_encode_fast():
    """A cold encode (the GGUF evicted from the page cache, nothing cached in the encoder) on the default path takes
    <= 5 s on AC, and reads the file with O_DIRECT: its pages stay out of the page cache (no resident weights)."""
    if not _on_ac():
        pytest.skip("speed counts only on AC power")
    from vecworld.config import FLUX_T5_GGUF
    from vecworld.engine.ext import load_ext, load_uring
    from vecworld.engine.uring import resident_bytes
    from vecworld.flux.text import T5Text

    load_ext(), load_uring()                    # built and loaded at the painter's start, not per encode
    torch.zeros(1, device="cuda")
    t5 = T5Text(cache=0)
    times = []
    for text in TEXTS[:2]:
        _evict(FLUX_T5_GGUF)
        t = time.perf_counter()
        t5.encode(text)
        times.append(time.perf_counter() - t)
    res = resident_bytes(os.path.realpath(FLUX_T5_GGUF))
    print(f"\nT5 cold encodes {[round(x, 2) for x in times]} s ({getattr(t5, "engine", "pread")}); GGUF resident {res / 1e6:.1f} MB")
    assert max(times) <= 5.0, times
    assert res < 64e6, res
