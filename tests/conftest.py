from PIL import Image


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

    def step(self, prev, base, focus, focus_at, seed, bg_strength=None):
        self.calls.append(("step", {"prev": prev, "base": base, "focus": focus, "focus_at": focus_at, "seed": seed,
                                    "bg_strength": bg_strength}))
        self._maybe_fail()
        return self._img("teal")


class FakeDreamer:
    """Canned discoveries; records every call."""

    IDEAS = ["an old lantern on a post", "a ruined stone chapel", "a misty pond",
             "a lone wooden bench", "a mossy stone well", "a crooked fence"]

    def __init__(self):
        self.calls: list[tuple[str, list[str], int]] = []
        self.place_calls: list[tuple[str, list[str], list[str], int]] = []

    def ideas(self, theme, moods, n):
        self.calls.append((theme, list(moods), n))
        return [self.IDEAS[i % len(self.IDEAS)] for i in range(n)]

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
