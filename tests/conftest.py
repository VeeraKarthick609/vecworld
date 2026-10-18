import numpy as np

from vecworld.space.index import MeaningSpace


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


TOY_WORDS = ["ocean", "sea", "ship", "desert", "sand"]
TOY_VECS = np.stack([
    unit([1, 0, 0]),
    unit([0.95, 0.1, 0]),
    unit([0.8, 0.6, 0]),
    unit([0, 0, 1]),
    unit([0, 0.1, 0.95]),
])


def make_space() -> MeaningSpace:
    return MeaningSpace(list(TOY_WORDS), TOY_VECS.copy())


# --- fakes for the game service and API --------------------------------------

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from vecworld.scenes.hotspots import Hotspot  # noqa: E402


class FakeEmbedder:
    """Toy-space vector for known words; anything else lands just off 'ocean'."""

    def __init__(self):
        self.last_texts: list[str] = []

    def embed(self, texts):
        from vecworld.config import MAX_TEXT_CHARS

        self.last_texts = [t[:MAX_TEXT_CHARS] for t in texts]
        lookup = dict(zip(TOY_WORDS, TOY_VECS))
        return np.stack([lookup.get(t, unit([1, 0.05, 0.05])) for t in self.last_texts])


class FakeGenerator:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.sources: list[tuple[int, int, int]] = []   # a pixel of each img2img source image

    def fresh(self, prompt, seed):
        self.calls.append(("fresh", prompt))
        return Image.new("RGB", (64, 64), "navy")

    def evolve(self, prev, prompt, seed):
        self.calls.append(("evolve", prompt))
        self.sources.append(prev.convert("RGB").getpixel((0, 0)))
        return Image.new("RGB", (64, 64), "teal")


class OomGenerator(FakeGenerator):
    def fresh(self, prompt, seed):
        import torch

        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    evolve = lambda self, prev, prompt, seed: self.fresh(prompt, seed)  # noqa: E731


class FakeFinder:
    def find(self, img, concepts):
        n = len(concepts)
        return [Hotspot(c, i / n, 0.4, 1 / n, 0.2) for i, c in enumerate(concepts)]


@pytest.fixture
def svc_factory(tmp_path):
    from vecworld.game.service import GameService

    def make(generator=None, **kw):
        return GameService(make_space(), np.zeros(3, np.float32), FakeEmbedder(),
                           generator or FakeGenerator(), FakeFinder(), tmp_path / "scenes", **kw)
    return make


@pytest.fixture
def svc(svc_factory):
    return svc_factory()


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

    def step(self, prev, base, focus, focus_at, seed):
        self.calls.append(("step", {"prev": prev, "base": base, "focus": focus, "focus_at": focus_at, "seed": seed}))
        self._maybe_fail()
        return self._img("teal")


class FakeDreamer:
    """Canned discoveries; records every call."""

    IDEAS = ["an old lantern on a post", "a ruined stone chapel", "a misty pond",
             "a lone wooden bench", "a mossy stone well", "a crooked fence"]

    def __init__(self):
        self.calls: list[tuple[str, list[str], int]] = []

    def ideas(self, theme, moods, n):
        self.calls.append((theme, list(moods), n))
        return [self.IDEAS[i % len(self.IDEAS)] for i in range(n)]
