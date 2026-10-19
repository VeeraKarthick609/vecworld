"""Dream Roam on fake models, for looking at the viewer without the GPU.

The painter hands out the stand-in fixture spots (web/roam/fixtures: the free-roam spike's panorama and two derived
from it) and the depth model their depth maps, so the world looks real while nothing touches CUDA. Neighbours are
the built-in fallback places (no LLM).

    uv run python scripts/dev_fake_server.py --host 100.86.11.109 --port 8094 [--step-delay 6]

--step-delay slows every step (pre-dreams included), so walking out quickly meets the dream fog.
"""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image

from vecworld.__main__ import WEB_DIR
from vecworld.api.app import create_app
from vecworld.roam.service import RoamService

FIXTURES = WEB_DIR / "roam" / "fixtures"


class FixturePainter:
    """The first panorama is fixture 0; steps alternate fixtures 1 and 2."""

    def __init__(self, delay: float):
        self.delay = delay
        self.images = [Image.open(FIXTURES / f"spot_{i}.jpg").convert("RGB") for i in range(3)]
        self.steps = 0
        self.last = 0

    def panorama(self, base, seed):
        self.last = 0
        return self.images[0].copy()

    def step(self, prev, base, focus, focus_at, seed, bg_strength=None, focus_half=None):
        time.sleep(self.delay)
        self.steps += 1
        self.last = 1 + self.steps % 2
        return self.images[self.last].copy()


class FixtureDepth:
    """The depth map of the fixture the painter handed out last (the GPU queue runs one build at a time)."""

    def __init__(self, painter: FixturePainter):
        self.painter = painter
        self.maps = [np.asarray(Image.open(FIXTURES / f"depth_{i}.png").convert("L"), np.float32) / 255
                     for i in range(3)]

    def depth360(self, pano):
        return self.maps[self.painter.last]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--step-delay", type=float, default=0.0)
    a = ap.parse_args()
    painter = FixturePainter(a.step_delay)
    scenes = Path(tempfile.mkdtemp(prefix="roam-fake-scenes-"))
    service = RoamService(painter, FixtureDepth(painter), None, scenes, spawn=lambda fn: None)
    uvicorn.run(create_app(service, WEB_DIR), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
