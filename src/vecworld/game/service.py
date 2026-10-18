import math
import secrets
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from vecworld.config import MAX_STEP_ANGLE, MAX_TEXT_CHARS, N_NEIGHBOURS
from vecworld.game.session import Scene, Session
from vecworld.scenes.generator import SceneGenerator
from vecworld.scenes.hotspots import Hotspot, HotspotFinder
from vecworld.scenes.prompt import scene_prompt, zoom_crop
from vecworld.space.embedder import Embedder
from vecworld.space.index import MeaningSpace
from vecworld.space.vocab import centre

MINDMAP_NEIGHBOURS = 30


class GameError(Exception):
    status = 500
    code = "error"


class BadInput(GameError):
    status, code = 422, "bad_input"


class NotStarted(GameError):
    status, code = 409, "not_started"


class Busy(GameError):
    status, code = 409, "busy"


class UnknownConcept(GameError):
    status, code = 400, "unknown_concept"


class GpuUnavailable(GameError):
    status, code = 503, "gpu_unavailable"


class GameService:
    def __init__(self, space: MeaningSpace, vocab_mean: np.ndarray, embedder: Embedder,
                 generator: SceneGenerator, finder: HotspotFinder, scene_dir: Path):
        self.space = space
        self.vocab_mean = vocab_mean
        self.embedder = embedder
        self.generator = generator
        self.finder = finder
        self.scene_dir = Path(scene_dir)
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()   # guards busy flags
        self._gpu = threading.Lock()    # one generation at a time on the shared GPU

    def new_session(self) -> Session:
        s = Session(id=secrets.token_urlsafe(8), seed=secrets.randbelow(2**31))
        self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session:
        return self._sessions[sid]

    def jump(self, sid: str, text: str) -> Scene:
        s = self.get(sid)
        text = text.strip()
        if not text:
            raise BadInput("Type a word or phrase to jump to.")
        with self._claim(s):
            position = centre(self.embedder.embed([text[:MAX_TEXT_CHARS]]), self.vocab_mean)[0]
            (centre_word, _), *rest = self.space.neighbours(position, k=1 + N_NEIGHBOURS)
            others = [w for w, _ in rest]
            img, hotspots = self._render(lambda: self.generator.fresh(scene_prompt(centre_word, others), s.seed),
                                         [centre_word] + others)
            return self._commit(s, position, img, hotspots)

    def step(self, sid: str, concept: str) -> Scene:
        s = self.get(sid)
        if s.position is None:
            raise NotStarted("Jump somewhere first by typing a word.")
        clicked = next((h for h in s.hotspots if h.concept == concept), None)
        if clicked is None:
            raise UnknownConcept(f"'{concept}' isn't in this scene.")
        with self._claim(s):
            position = self.space.step_toward(s.position, self.space.vector(concept), MAX_STEP_ANGLE)
            # The clicked concept stays pinned at the centre even if the cap stopped us short of it.
            others = [w for w, _ in self.space.neighbours(position, k=N_NEIGHBOURS, exclude={concept})]
            prev = zoom_crop(s.image, clicked.x + clicked.w / 2, clicked.y + clicked.h / 2)
            img, hotspots = self._render(lambda: self.generator.evolve(prev, scene_prompt(concept, others), s.seed),
                                         [concept] + others)
            return self._commit(s, position, img, hotspots)

    def _claim(self, s: Session):
        service = self

        class _Claim:
            def __enter__(self):
                with service._lock:
                    if s.busy:
                        raise Busy("Still loading the last move.")
                    s.busy = True

            def __exit__(self, *exc):
                s.busy = False

        return _Claim()

    def _render(self, make_image, concepts: list[str]) -> tuple[Image.Image, list[Hotspot]]:
        import torch

        with self._gpu:
            try:
                img = make_image()
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise GpuUnavailable("The GPU is out of memory, try again in a moment.") from e
        return img, self.finder.find(img, concepts)

    def _commit(self, s: Session, position: np.ndarray, img: Image.Image, hotspots: list[Hotspot]) -> Scene:
        scene_id = secrets.token_urlsafe(8)
        path = self.scene_dir / f"{scene_id}.png"
        img.save(path)
        s.position, s.image, s.hotspots = position, img, hotspots
        me = self.space.project(position)
        s.trail.append(me)

        reach = [self.space.project(self.space.step_toward(position, self.space.vector(h.concept), MAX_STEP_ANGLE))
                 for h in hotspots]
        reach_radius = float(np.mean([math.dist(me, r) for r in reach])) if reach else 0.0
        neighbours = [(w, *self.space.project(self.space.vector(w)))
                      for w, _ in self.space.neighbours(position, k=MINDMAP_NEIGHBOURS)]
        return Scene(scene_id, path, hotspots, me, list(s.trail), neighbours, reach_radius)
