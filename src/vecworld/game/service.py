import math
import secrets
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from vecworld.config import (HISTORY, MAX_SESSIONS, MAX_STEP_ANGLE, MAX_TEXT_CHARS, N_CANDIDATES, N_NEIGHBOURS,
                             N_PROMPT_NEIGHBOURS, RECENT, SESSION_TTL, ZOOM_FRAC)
from vecworld.game.session import Scene, Session, Snapshot
from vecworld.scenes.generator import SceneGenerator
from vecworld.scenes.hotspots import Hotspot, HotspotFinder, choose
from vecworld.scenes.prompt import scene_prompt, zoom_crop
from vecworld.space.embedder import Embedder
from vecworld.space.index import MeaningSpace
from vecworld.space.vocab import centre, is_concrete

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


class NoHistory(GameError):
    status, code = 409, "no_history"


class UnknownSession(GameError, KeyError):
    status, code = 404, "unknown_session"

    def __str__(self):
        return "That session has expired, reload the page."


class GameService:
    def __init__(self, space: MeaningSpace, vocab_mean: np.ndarray, embedder: Embedder,
                 generator: SceneGenerator, finder: HotspotFinder, scene_dir: Path, clock=time.monotonic,
                 concrete=is_concrete):
        self.space = space
        self.vocab_mean = vocab_mean
        self.embedder = embedder
        self.generator = generator
        self.finder = finder
        self.scene_dir = Path(scene_dir)
        self.clock = clock
        self.concrete = concrete
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()   # guards busy flags
        self._gpu = threading.Lock()    # one generation at a time on the shared GPU

    def new_session(self) -> Session:
        now = self.clock()
        s = Session(id=secrets.token_urlsafe(8), seed=secrets.randbelow(2**31), last_used=now)
        with self._lock:
            # Each session holds a full scene image, so drop idle ones and cap the rest.
            for sid in [k for k, v in self._sessions.items() if now - v.last_used > SESSION_TTL and not v.busy]:
                del self._sessions[sid]
            while len(self._sessions) >= MAX_SESSIONS:
                del self._sessions[min(self._sessions, key=lambda k: self._sessions[k].last_used)]
            self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session:
        try:
            return self._sessions[sid]
        except KeyError:
            raise UnknownSession(sid) from None

    def jump(self, sid: str, text: str) -> Scene:
        s = self.get(sid)
        text = text.strip()
        if not text:
            raise BadInput("Type a word or phrase to jump to.")
        with self._claim(s):
            position = centre(self.embedder.embed([text[:MAX_TEXT_CHARS]]), self.vocab_mean)[0]
            (centre_word, _), *rest = self.space.neighbours(position, k=1 + N_CANDIDATES)
            others = [w for w, _ in rest]
            img, hotspots = self._render(
                lambda: self.generator.fresh(self._prompt(centre_word, others), s.seed), centre_word, others)
            return self._commit(s, position, img, hotspots, recent=[centre_word], heading=None)

    def step(self, sid: str, concept: str) -> Scene:
        s = self._started(sid)
        clicked = next((h for h in s.hotspots if h.concept == concept), None)
        if clicked is None:
            raise UnknownConcept(f"'{concept}' isn't in this scene.")
        with self._claim(s):
            position = self.space.step_toward(s.position, self.space.vector(concept), MAX_STEP_ANGLE)
            # The clicked concept stays pinned at the centre even if the cap stopped us short of it.
            return self._evolve(s, position, concept, clicked.x + clicked.w / 2, clicked.y + clicked.h / 2,
                                heading=self.space.heading(s.position, position))

    def forward(self, sid: str) -> Scene:
        """Keep walking the way you're heading; the nearest unvisited concept becomes the centre."""
        s = self._started(sid)
        with self._claim(s):
            if s.heading is None:
                # No direction yet (just jumped or went back): head for the closest unvisited concept.
                (target, _), = self.space.neighbours(s.position, k=1, exclude=set(s.recent))
                position = self.space.step_toward(s.position, self.space.vector(target), MAX_STEP_ANGLE)
                heading = self.space.heading(s.position, position)
            else:
                position, heading = self.space.walk(s.position, s.heading, MAX_STEP_ANGLE)
            (centre_word, _), = self.space.neighbours(position, k=1, exclude=set(s.recent))
            return self._evolve(s, position, centre_word, 0.5, 0.5, heading=heading)

    def back(self, sid: str) -> Scene:
        """Stand where you stood before the last move. Instant: the scene is already on disk."""
        s = self._started(sid)
        with self._claim(s):
            if not s.history:
                raise NoHistory("Nothing behind you yet.")
            snap = s.history[-1]
            image = Image.open(snap.scene.image_path).convert("RGB")
            s.history.pop()
            s.scene, s.position, s.heading, s.recent = snap.scene, snap.position, snap.heading, list(snap.recent)
            s.image, s.hotspots, s.trail = image, list(snap.scene.hotspots), list(snap.scene.trail)
            return snap.scene

    def _started(self, sid: str) -> Session:
        s = self.get(sid)
        if s.position is None:
            raise NotStarted("Jump somewhere first by typing a word.")
        return s

    def _evolve(self, s: Session, position: np.ndarray, centre_word: str, cx: float, cy: float,
                heading: np.ndarray | None) -> Scene:
        """img2img from a zoom-crop of the current scene, centred on (cx, cy), with centre_word pinned."""
        # Recently visited centres stay out of the neighbours, so a walk can't ping-pong in a cluster.
        others = [w for w, _ in self.space.neighbours(position, k=N_CANDIDATES,
                                                      exclude={centre_word, *s.recent})]
        prev = zoom_crop(s.image, cx, cy, ZOOM_FRAC)
        img, hotspots = self._render(
            lambda: self.generator.evolve(prev, self._prompt(centre_word, others), s.seed), centre_word, others)
        return self._commit(s, position, img, hotspots, recent=(s.recent + [centre_word])[-RECENT:],
                            heading=heading if heading is not None else s.heading)

    def _prompt(self, centre_word: str, others: list[str]) -> str:
        return scene_prompt(centre_word, others[:N_PROMPT_NEIGHBOURS], self.concrete(centre_word))

    def _claim(self, s: Session):
        service = self

        class _Claim:
            def __enter__(self):
                with service._lock:
                    if s.busy:
                        raise Busy("Still loading the last move.")
                    s.busy = True
                    s.last_used = service.clock()

            def __exit__(self, *exc):
                s.busy = False

        return _Claim()

    def _render(self, make_image, centre_word: str, candidates: list[str]) -> tuple[Image.Image, list[Hotspot]]:
        """Generate, then offer the pinned centre plus only the candidates CLIPSeg finds in the picture."""
        import torch

        with self._gpu:
            try:
                img = make_image()
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise GpuUnavailable("The GPU is out of memory, try again in a moment.") from e
        found = self.finder.find(img, [centre_word] + candidates)
        return img, choose(centre_word, found, N_NEIGHBOURS)

    def _commit(self, s: Session, position: np.ndarray, img: Image.Image, hotspots: list[Hotspot],
                recent: list[str], heading: np.ndarray | None) -> Scene:
        scene_id = secrets.token_urlsafe(8)
        path = self.scene_dir / f"{scene_id}.png"
        img.save(path)
        if s.scene is not None:
            s.history = (s.history + [Snapshot(s.scene, s.position, s.heading, list(s.recent))])[-HISTORY:]
        s.position, s.image, s.hotspots, s.recent, s.heading = position, img, hotspots, recent, heading
        me = self.space.project(position)
        s.trail.append(me)

        reach = [self.space.project(self.space.step_toward(position, self.space.vector(h.concept), MAX_STEP_ANGLE))
                 for h in hotspots]
        reach_radius = float(np.mean([math.dist(me, r) for r in reach])) if reach else 0.0
        neighbours = [(w, *self.space.project(self.space.vector(w)))
                      for w, _ in self.space.neighbours(position, k=MINDMAP_NEIGHBOURS)]
        s.scene = Scene(scene_id, path, hotspots, me, list(s.trail), neighbours, reach_radius, hotspots[0].concept)
        return s.scene
