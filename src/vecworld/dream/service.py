"""The Dream Walk game: sessions, walking a painted panorama, discoveries, theme drift, moods and prefetch."""
import logging
import math
import random
import secrets
import threading
import time
from collections import deque
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vecworld.config import (DRIFT_STEPS, FOCUS_HALF, KEEP_OFFSET, MAX_SESSIONS, MAX_THEME_CHARS, MOODS,
                             PALETTE_STRENGTH, PREFETCH_TOLERANCE, QUIET_EVERY, SESSION_TTL, STYLE)
from vecworld.dream.errors import BadInput, Busy, GpuUnavailable, NotStarted, UnknownLabel, UnknownSession
from vecworld.dream.gpu import GpuQueue
from vecworld.dream.ideas import FALLBACK_IDEAS, Dreamer
from vecworld.dream.labels import Label, LabelFinder, carry
from vecworld.dream.painter import Painter, Prompt
from vecworld.dream.pano import dolly, roll_to, wrap_dist, yaw_to_u
from vecworld.dream.session import DreamSession, View
from vecworld.scenes.palette import Stats, match_palette, palette_stats

log = logging.getLogger(__name__)

REFILL_N = 8        # ideas asked for per refill
REFILL_BELOW = 3    # refill when fewer than this remain


@dataclass(frozen=True)
class _Snap:
    """What a build reads, copied at submit time so later session changes can't reach an in-flight build."""
    pano: Image.Image
    labels: tuple[Label, ...]
    steps: int
    idea: str | None
    theme: str
    moods: tuple[str, ...]
    drift: tuple[str, int] | None
    version: int
    ref: Stats
    reset_ref: bool     # this spot becomes the palette reference
    seed: int


def _thread(fn) -> None:
    threading.Thread(target=fn, name="dream-ideas", daemon=True).start()


def _yaw_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


class DreamService:
    def __init__(self, painter: Painter, dreamer: Dreamer, labels: LabelFinder, scene_dir: Path,
                 queue: GpuQueue | None = None, clock=time.monotonic, spawn=None):
        self.painter = painter
        self.dreamer = dreamer
        self.labels = labels
        self.scene_dir = Path(scene_dir)
        self.queue = queue or GpuQueue()
        self.clock = clock
        self._spawn = spawn or _thread   # runs an ideas refill off the request thread (never on the GPU queue)
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, DreamSession] = {}
        self._lock = threading.Lock()    # guards busy flags and every session read/write a build or commit makes

    # --- sessions ----------------------------------------------------------------

    def new_session(self) -> DreamSession:
        now = self.clock()
        s = DreamSession(id=secrets.token_urlsafe(8), seed=secrets.randbelow(2**31), last_used=now)
        with self._lock:
            # Each session holds a full panorama, so drop idle ones and cap the rest.
            gone = [k for k, v in self._sessions.items() if now - v.last_used > SESSION_TTL and not v.busy]
            for sid in gone:
                del self._sessions[sid]
            while len(self._sessions) >= MAX_SESSIONS:
                sid = min(self._sessions, key=lambda k: self._sessions[k].last_used)
                del self._sessions[sid]
                gone.append(sid)
            self._sessions[s.id] = s
        for sid in gone:
            self.queue.cancel_owner(sid)
        return s

    def get(self, sid: str) -> DreamSession:
        try:
            return self._sessions[sid]
        except KeyError:
            raise UnknownSession(sid) from None

    # --- moves -------------------------------------------------------------------

    def start(self, sid: str, theme: str, moods: list[str]) -> View:
        """A fresh panorama for this theme, with no discovery."""
        s = self.get(sid)
        theme, moods = self._theme(theme), self._moods(moods)
        with self._claim(s):
            base = self._base(theme, moods, None)
            img = self._run(s, lambda: self._paint(lambda: self.painter.panorama(base, s.seed)))
            ref = palette_stats(img)
            path, view_id = self._save(img)
            with self._lock:
                self._drop_prefetch(s)
                s.version += 1
                s.theme, s.moods, s.pano, s.ref = theme, moods, img, ref
                s.labels, s.steps, s.drift, s.ref_pending = [], 0, None, False
                s.ideas, s.ideas_theme = deque(), theme
                view = self._view(s, view_id, path)
        self._maybe_refill(s)
        return view

    def walk(self, sid: str, yaw: float) -> View:
        return self._walk(sid, yaw, None)

    def back(self, sid: str, yaw: float) -> View:
        return self._walk(sid, self._yaw(yaw) + 180, None)

    def approach(self, sid: str, label_id: str) -> View:
        s = self._started(sid)
        label = next((l for l in s.labels if l.id == label_id), None)
        if label is None:
            raise UnknownLabel("That discovery isn't here any more.")
        return self._walk(sid, (label.u - 0.5) * 360, True)

    def _walk(self, sid: str, yaw: float, keep: bool | None) -> View:
        yaw = self._yaw(yaw)
        s = self._started(sid)
        with self._claim(s):
            built = self._take_prefetch(s, yaw, keep)
            if built is None:
                with self._lock:
                    snap = self._snap(s)
                built = self._run(s, lambda: self._build(snap, yaw, keep))
            return self._commit(s, *built)

    # --- theme, moods, prefetch ------------------------------------------------------

    def set_theme(self, sid: str, theme: str) -> None:
        """The new theme fades in over the next DRIFT_STEPS steps."""
        theme = self._theme(theme)
        s = self._started(sid)
        with self._lock:
            if theme == s.theme:
                return
            s.drift = (s.theme, 0)
            s.theme = theme
            s.version += 1
            s.last_used = self.clock()
            self._drop_prefetch(s)
        self._maybe_refill(s)

    def set_moods(self, sid: str, moods: list[str]) -> None:
        moods = self._moods(moods)
        s = self._started(sid)
        with self._lock:
            if moods == s.moods:
                return
            s.moods = moods
            s.version += 1
            s.ref_pending = True      # so night or snow actually show instead of being pulled back
            s.last_used = self.clock()
            self._drop_prefetch(s)

    def prefetch(self, sid: str, yaw: float) -> None:
        """Pre-build the spot ahead at low priority. Never takes the busy flag."""
        yaw = self._yaw(yaw)
        s = self._started(sid)
        with self._lock:
            s.last_used = self.clock()
            if s.busy:
                return                 # the move in flight will make anything built now stale
            pf = s.prefetch
            if (pf is not None and pf[1] == s.version and _yaw_diff(pf[0], yaw) <= PREFETCH_TOLERANCE
                    and not pf[2].cancelled()):
                return
            snap = self._snap(s)
            fut = self.queue.submit(lambda: self._build(snap, yaw, None), owner=s.id, prefetch=True)
            s.prefetch = (yaw, s.version, fut)

    def _take_prefetch(self, s: DreamSession, yaw: float, keep: bool | None):
        """The prefetched spot if it fits this move (waiting for it if still running), else None."""
        with self._lock:
            pf = s.prefetch
            if pf is None or pf[1] != s.version or _yaw_diff(pf[0], yaw) > PREFETCH_TOLERANCE:
                return None
            if keep and not self._near_label(s.labels, yaw_to_u(pf[0])):
                return None            # approaching a label needs a build that keeps it
        try:
            built = pf[2].result()
        except Exception as e:  # noqa: BLE001 - cancelled, out of memory or broken: build on demand instead
            if not isinstance(e, (CancelledError, GpuUnavailable)):
                log.warning("prefetch failed (%s); building on demand", type(e).__name__)
            with self._lock:
                if s.prefetch is pf:
                    s.prefetch = None
            return None
        if built[2]["steps"] != s.steps:
            return None
        return built

    def _drop_prefetch(self, s: DreamSession) -> None:
        # caller holds the lock
        if s.prefetch is not None:
            s.prefetch[2].cancel()     # only takes effect while it is still queued
            s.prefetch = None

    # --- building a spot -----------------------------------------------------------

    def _snap(self, s: DreamSession) -> _Snap:
        # caller holds the lock
        drift_done = s.drift is not None and s.drift[1] + 1 >= DRIFT_STEPS
        return _Snap(pano=s.pano, labels=tuple(s.labels), steps=s.steps, idea=s.ideas[0] if s.ideas else None,
                     theme=s.theme, moods=tuple(s.moods), drift=s.drift, version=s.version, ref=s.ref,
                     reset_ref=s.ref_pending or drift_done, seed=s.seed)

    def _build(self, snap: _Snap, yaw: float, keep: bool | None) -> tuple[Image.Image, list[Label], dict]:
        """Paint the spot one step toward yaw. Pure: reads only the snapshot, so nothing changes until _commit."""
        u = yaw_to_u(yaw)
        prev = dolly(roll_to(snap.pano, u))
        kept = self._near_label(snap.labels, u) if keep is None else keep
        # Keep what you walk toward (it is img2img'd closer) and put the next discovery beside it.
        focus_at = 0.5 + (KEEP_OFFSET if snap.steps % 2 == 0 else -KEEP_OFFSET) if kept else 0.5
        quiet = (snap.steps + 1) % QUIET_EVERY == 0
        discovery = None if quiet else (snap.idea or FALLBACK_IDEAS[snap.steps % len(FALLBACK_IDEAS)])
        base = self._base(snap.theme, list(snap.moods), snap.drift)
        focus = [(f"{t}, {discovery} ahead in the distance", w) for t, w in base] if discovery else None
        seed = (snap.seed + snap.steps + 1) % 2**31
        img = self._paint(lambda: self.painter.step(prev, base, focus, focus_at, seed))
        if not snap.reset_ref:
            # A spot that becomes the new reference keeps its own colours; the rest are pulled back.
            img = match_palette(img, snap.ref, PALETTE_STRENGTH)
        wanted = carry(list(snap.labels), u) + ([(discovery, focus_at)] if discovery else [])
        labels = self.labels.find(img, wanted)
        meta = {"steps": snap.steps, "version": snap.version, "idea": snap.idea if discovery else None,
                "reset_ref": snap.reset_ref, "kept": kept}
        return img, labels, meta

    def _commit(self, s: DreamSession, img: Image.Image, labels: list[Label], meta: dict) -> View:
        path, view_id = self._save(img)
        ref = palette_stats(img) if meta["reset_ref"] else None
        with self._lock:
            self._drop_prefetch(s)     # built from the spot we just left
            s.pano, s.labels, s.steps = img, labels, s.steps + 1
            if meta["idea"] is not None and s.ideas and s.ideas[0] == meta["idea"]:
                s.ideas.popleft()
            # A theme or mood change during this build applies from the next step instead.
            if meta["version"] == s.version:
                if s.drift is not None:
                    done = s.drift[1] + 1
                    s.drift = None if done >= DRIFT_STEPS else (s.drift[0], done)
                if ref is not None:
                    s.ref, s.ref_pending = ref, False
            view = self._view(s, view_id, path)
        self._maybe_refill(s)
        return view

    # --- ideas -------------------------------------------------------------------

    def _maybe_refill(self, s: DreamSession) -> None:
        with self._lock:
            stale = s.ideas_theme != s.theme
            if (len(s.ideas) >= REFILL_BELOW and not stale) or s.refilling == s.theme:
                return
            s.refilling = theme = s.theme
            moods = list(s.moods)

        def refill():
            try:
                new = [i for i in self.dreamer.ideas(theme, moods, REFILL_N) if i][:REFILL_N]
            except Exception as e:  # noqa: BLE001 - a refill must never die silently
                log.warning("ideas refill failed (%s); using fallback ideas", type(e).__name__)
                new = []
            if not new:
                new = random.sample(FALLBACK_IDEAS, min(REFILL_N, len(FALLBACK_IDEAS)))
            with self._lock:
                if s.refilling == theme:
                    s.refilling = None
                if s.theme != theme:
                    return
                if s.ideas_theme != theme:
                    s.ideas, s.ideas_theme = deque(new), theme
                else:
                    s.ideas.extend(new)

        self._spawn(refill)

    # --- helpers -----------------------------------------------------------------

    def _run(self, s: DreamSession, fn):
        """Run fn on the GPU queue at on-demand priority and wait for it."""
        try:
            return self.queue.submit(fn, owner=s.id).result()
        except CancelledError:
            raise UnknownSession(s.id) from None   # evicted while waiting

    @staticmethod
    def _paint(fn) -> Image.Image:
        import torch

        try:
            return fn()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise GpuUnavailable("The GPU is out of memory, try again in a moment.") from e

    @staticmethod
    def _base(theme: str, moods: list[str], drift: tuple[str, int] | None) -> Prompt:
        def text(t):
            return ", ".join([t, *moods, STYLE])

        if drift is None:
            return [(text(theme), 1.0)]
        k = drift[1] + 1
        return [(text(drift[0]), 1 - k / DRIFT_STEPS), (text(theme), k / DRIFT_STEPS)]

    @staticmethod
    def _near_label(labels, u: float) -> bool:
        return any(wrap_dist(l.u, u) <= FOCUS_HALF for l in labels)

    def _save(self, img: Image.Image) -> tuple[Path, str]:
        view_id = secrets.token_urlsafe(8)
        path = self.scene_dir / f"{view_id}.png"
        img.save(path)
        return path, view_id

    @staticmethod
    def _view(s: DreamSession, view_id: str, path: Path) -> View:
        return View(view_id, path, list(s.labels), s.theme, list(s.moods), s.steps)

    @staticmethod
    def _theme(theme: str) -> str:
        theme = (theme or "").strip()[:MAX_THEME_CHARS].strip()
        if not theme:
            raise BadInput("Type a theme to dream of.")
        return theme

    @staticmethod
    def _moods(moods: list[str]) -> list[str]:
        return list(dict.fromkeys(m for m in (moods or ()) if m in MOODS))

    @staticmethod
    def _yaw(yaw) -> float:
        try:
            yaw = float(yaw)
        except (TypeError, ValueError):
            raise BadInput("yaw must be a number.") from None
        if not math.isfinite(yaw):
            raise BadInput("yaw must be a number.")
        return yaw

    def _started(self, sid: str) -> DreamSession:
        s = self.get(sid)
        if s.pano is None:
            raise NotStarted("Type a theme to start dreaming first.")
        return s

    def _claim(self, s: DreamSession):
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
