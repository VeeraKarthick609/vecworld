import os

import pytest

from PIL import Image

# The allocator the painter runs (FluxPainter.__init__ sets it before CUDA starts): GPU tests that build the
# transformer without the painter (test_graphs, test_cache) must see the same fragmentation behaviour, or batch-8 graph
# pools that fit the game run out of memory in the tests. Set before anything touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@pytest.fixture(autouse=True, scope="module")
def _hand_host_memory_back():
    """After each test module: collect, and hand glibc's freed heap back to the OS (malloc_trim). Model loads leave
    gigabytes of freed but still mapped heap (each tensor under glibc's 32 MB mmap threshold), and some GPU tests peak
    at ~12.8 GB of this 14 GB laptop on their own (test_glue's patched-block test): leftovers from the modules before
    them got the suite OOM-killed."""
    yield
    import gc

    gc.collect()
    try:
        from vecworld.engine.stream import _trim

        _trim()
    except Exception:  # noqa: BLE001 - best effort (no torch / no glibc)
        pass


def pytest_collection_modifyitems(config, items):
    """Tests marked card_limit run first. They need nearly all of the 4 GB card (the spot walks peak at 3.32 GiB
    allocated of 3.68 GiB usable), and every GPU module before them leaves device memory this process can't give back:
    not PyTorch tensors (allocated is back to ~0.01 GB between modules) but CUDA libraries' code and handles loaded
    into the context, 0.14 GB alone -> 0.24 GB after the GPU modules (measured with a per-test tracker, the same at
    fcbfe73 and fe9e0a2). The walk then ran out of memory 5 MB short. A fresh process can't host it either: the
    suite's own context holds the card's last 0.24 GB."""
    first = [it for it in items if it.get_closest_marker("card_limit")]
    if first:
        items[:] = first + [it for it in items if not it.get_closest_marker("card_limit")]


class FakePainter:
    """Solid tiny panoramas; records every call; fail_next raises a CUDA OOM once."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.fail_next = False

    def _maybe_fail(self):
        if self.fail_next:
            import torch

            self.fail_next = False
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    def _img(self, colour):
        from vecworld.config import PANO_H, PANO_W

        return Image.new("RGB", (PANO_W // 16, PANO_H // 16), colour)

    def panorama(self, base, seed):
        self.calls.append(("panorama", {"base": base, "seed": seed}))
        self._maybe_fail()
        return self._img("navy")

    def step(self, prev, base, focus, focus_at, seed, bg_strength=None, focus_half=None):
        self.calls.append(("step", {"prev": prev, "base": base, "focus": focus, "focus_at": focus_at, "seed": seed,
                                    "bg_strength": bg_strength, "focus_half": focus_half}))
        self._maybe_fail()
        return self._img("teal")


class FakeDreamer:
    """Canned places; records every call."""

    def __init__(self):
        self.place_calls: list[tuple[str, list[str], list[str], int]] = []

    PLACES = ["old mill", "reed marsh", "pine ridge", "stone bridge"]

    def places(self, place, journey, moods, n):
        self.place_calls.append((place, list(journey), list(moods), n))
        return [(self.PLACES[i % len(self.PLACES)],
                 (f"a path of {i} stones", "a mossy fence post", "a crooked old tree")) for i in range(n)]


class FakeDepth:
    """Deterministic ramp: near (1) at the bottom, far (0) at the top, at the panorama's size."""

    def depth360(self, pano):
        import numpy as np

        w, h = pano.size
        return np.repeat(np.linspace(0, 1, h, dtype=np.float32)[:, None], w, axis=1)


def firefox(width: int, height: int):
    """Headless Firefox with WebGL: headless has no GLX display, so WebGL runs through surfaceless EGL on Mesa's
    software renderer (llvmpipe)."""
    import os

    from selenium import webdriver
    from selenium.webdriver.firefox.service import Service

    opts = webdriver.FirefoxOptions()
    opts.add_argument("-headless")
    for k in ("webgl.force-enabled", "gfx.x11-egl.force-enabled"):
        opts.set_preference(k, True)
    opts.set_preference("webgl.out-of-process", False)
    env = {**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1", "EGL_PLATFORM": "surfaceless", "MOZ_WEBGL_FORCE_EGL": "1"}
    d = webdriver.Firefox(options=opts, service=Service("/snap/bin/geckodriver", env=env))
    d.set_window_size(width, height)
    return d
