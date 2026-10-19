"""The Dream Roam game: sessions, spots chained by heading, places with drift and arrival, and speculative
pre-dreaming of the next spot in three directions.

Headings are degrees, 0 = north (the first spot's forward), clockwise, in [0, 360).

Locking: one service lock guards every session read/write. It may be held while calling into the GPU queue
(queue lock nested inside), never while waiting on a Future, painting or calling the dreamer. A build reads only
an immutable _Snap taken at submit time and returns a _Built; nothing in the session changes until _commit.
"""
import logging
import math
import random
import secrets
import threading
import time
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vecworld.config import MAX_SESSIONS, MAX_THEME_CHARS, MOOD_BG_STRENGTH, MOOD_WORDS, MOODS, SESSION_TTL, STYLE
from vecworld.dream.errors import BadInput, Busy, GpuUnavailable, NotStarted, UnknownSession
from vecworld.dream.gpu import GpuQueue
from vecworld.dream.ideas import Dreamer
from vecworld.dream.painter import Painter, Prompt
from vecworld.roam.depth import DepthEstimator, depth_png
from vecworld.roam.places import (GENERIC_SCENERY, MIN_SEPARATION, Journey, Neighbour, assign_bearings,
                                  fallback_neighbours)
from vecworld.roam.session import Predream, RoamSession, SpotView
from vecworld.roam.spot import Spot, first_spot, next_spot
from vecworld.scenes.palette import Stats, palette_stats

log = logging.getLogger(__name__)

SPREAD = 60.0           # the pre-dreamed set: the heading, then heading + SPREAD and heading - SPREAD
TAKE_WITHIN = 45.0      # a pre-dreamed spot serves an exit (or a candidate direction) within this
MAX_NEIGHBOURS = 4
KEEP_SPOTS = 3          # spots whose files stay on disk per session
JPEG_QUALITY = 90
OPEN_GROUND = "open ground in the foreground"   # the first panorama: something to stand on, not a tree trunk


@dataclass(frozen=True)
class _Snap:
    """What a build reads, copied at submit time so later session changes can't reach an in-flight build."""
    spot: Spot
    spot_stats: Stats
    journey: Journey
    moods: tuple[str, ...]
    ref: Stats
    ref_pending: bool
    version: int
    seed: int
    steps: int


@dataclass(frozen=True)
class _Built:
    spot: Spot
    journey: Journey        # the snapshot's journey, stepped (and arrived) on a copy
    arrived: str | None
    version: int
    parent_id: str
    ref_pending: bool       # built as the new palette reference after a mood change


def _thread(fn) -> None:
    threading.Thread(target=fn, name="roam-places", daemon=True).start()


def _diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _ok(f: Future) -> bool:
    """Finished with a result (never blocks)."""
    return f.done() and not f.cancelled() and f.exception(timeout=0) is None


def _alive(f: Future) -> bool:
    """Queued, running or finished with a result."""
    return not f.done() or _ok(f)


def _blend(a: Stats, b: Stats, w: float) -> Stats:
    return a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])


def _copy(j: Journey) -> Journey:
    c = Journey(j.place, j.neighbours, j.came_from)
    c.drift = dict(j.drift)
    return c


def _rebuilt(j: Journey, neighbours: list[Neighbour], carry: dict[str, str]) -> Journey:
    """A new journey with these neighbours; drift carried over from the old neighbour named carry[new name]."""
    out = Journey(j.place, neighbours, j.came_from)
    out.drift = {n.name: j.drift.get(carry.get(n.name, n.name), 0) for n in neighbours}
    return out


def _with_place(j: Journey, place: str, bearing: float) -> Journey:
    """The typed place becomes a neighbour at `bearing`; neighbours too close to it make way."""
    if place.lower() == j.place.lower():
        return j
    keep = [n for n in j.neighbours if n.name.lower() != place.lower() and _diff(n.bearing, bearing) >= MIN_SEPARATION]
    new = Neighbour(place, round(bearing % 360, 6), GENERIC_SCENERY)
    return _rebuilt(j, sorted(keep + [new], key=lambda n: n.bearing), {})


def _renamed(j: Journey, place: str, found: list[tuple[str, tuple[str, ...]]]) -> Journey:
    """The LLM's places take over the fallback neighbours' slots (bearings and drift kept); the came-from stays."""
    if j.place != place:
        return j
    taken = {place.lower(), (j.came_from or "").lower()}
    fresh: list[tuple[str, tuple[str, ...]]] = []
    for name, scenery in found:
        if name and name.lower() not in taken:
            taken.add(name.lower())
            fresh.append((name, tuple(scenery[:3]) + GENERIC_SCENERY[len(scenery[:3]):3]))
    it = iter(fresh)
    out, carry = [], {}
    for n in j.neighbours:
        if n.name == j.came_from:
            out.append(n)
            continue
        nxt = next(it, None)
        if nxt is None:
            if n.name.lower() not in {o.name.lower() for o in out} | {f[0].lower() for f in fresh}:
                out.append(n)
            continue
        out.append(Neighbour(nxt[0], n.bearing, nxt[1]))
        carry[nxt[0]] = n.name
    return _rebuilt(j, out, carry)


class RoamService:
    def __init__(self, painter: Painter, depth: DepthEstimator, dreamer: Dreamer, scene_dir: Path,
                 queue: GpuQueue | None = None, clock=time.monotonic, spawn=None):
        self.painter = painter
        self.depth = depth
        self.dreamer = dreamer
        self.scene_dir = Path(scene_dir)
        self.queue = queue or GpuQueue()
        self.clock = clock
        self._spawn = spawn or _thread   # runs a places refill off the request thread (never on the GPU queue)
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, RoamSession] = {}
        self._lock = threading.Lock()

    # --- sessions -------------------------------------------------------------------

    def new_session(self) -> RoamSession:
        now = self.clock()
        s = RoamSession(id=secrets.token_urlsafe(8), seed=secrets.randbelow(2**31), last_used=now)
        with self._lock:
            gone = [k for k, v in self._sessions.items() if now - v.last_used > SESSION_TTL and not v.busy]
            while len(self._sessions) - len(gone) >= MAX_SESSIONS:
                gone.append(min((k for k in self._sessions if k not in gone),
                                key=lambda k: self._sessions[k].last_used))
            dropped = [self._sessions.pop(sid) for sid in gone]
            for old in dropped:
                old.prefetch = {}
            self._sessions[s.id] = s
        for old in dropped:
            self.queue.cancel_owner(old.id)
            self._forget_files(old, keep=0)
        return s

    def get(self, sid: str) -> RoamSession:
        try:
            return self._sessions[sid]
        except KeyError:
            raise UnknownSession(sid) from None

    # --- moves ------------------------------------------------------------------------

    def start(self, sid: str, place: str, moods: list[str]) -> SpotView:
        """A fresh world at `place`: the first spot, fallback neighbours, and the set pre-dreamed around north."""
        place, moods = self._place(place), self._moods(moods)
        s = self.get(sid)
        with self._claim(s):
            prompt = [(self._text(place, OPEN_GROUND, moods), 1.0)]
            spot = self._run(s, lambda: self._paint(lambda: first_spot(self.painter, self.depth, prompt, s.seed)))
            names = fallback_neighbours(place)[:MAX_NEIGHBOURS]
            journey = Journey(place, assign_bearings(names, None, random.Random(s.seed)))
            stats = palette_stats(spot.image)
            paths = self._save(spot)
            with self._lock:
                self._check_alive(s, paths)
                s.version += 1
                s.prefetch, s.pending = {}, []
                s.spot, s.spot_stats, s.ref, s.ref_pending = spot, stats, stats, False
                s.journey, s.moods, s.steps, s.visited = journey, moods, 0, [place]
                s.places_for, s.hint = None, 0.0
                s.files.append(paths)
                self._submit_set(s)
                view = self._view(s, paths, None)
        self._forget_files(s)
        self._maybe_refill(s)
        return view

    def next(self, sid: str, exit_heading: float) -> SpotView:
        """Step to the next spot: a finished pre-dreamed one within TAKE_WITHIN of the exit (closest first), else
        the exact exit heading built at demand priority after cancelling the queued set."""
        exit_heading = self._heading(exit_heading)
        s = self._started(sid)
        with self._claim(s):
            with self._lock:
                hints = s.hints
                built = self._take(s, exit_heading)
                if built is None:
                    self.queue.submit_set([], s.id)   # the queued set would only delay the exact heading
                    snap = self._snap(s)
            if built is None:
                built = self._run(s, lambda: self._build(snap, exit_heading))
            return self._commit(s, built, exit_heading, hints)

    def heading(self, sid: str, heading: float) -> None:
        """The player's heading: re-target the pre-dreamed set around it. Never paints on this thread."""
        h = self._heading(heading)
        s = self._started(sid)
        with self._lock:
            s.hint, s.hints, s.last_used = h, s.hints + 1, self.clock()
            if not s.busy:            # a move in flight re-submits the set when it lands
                self._submit_set(s)

    # --- theme and moods ------------------------------------------------------------------

    def set_theme(self, sid: str, place: str) -> None:
        """Drift toward a typed place: it becomes a neighbour at the current heading."""
        place = self._place(place)
        s = self._started(sid)
        with self._lock:
            s.last_used = self.clock()
            if not s.busy and place.lower() == s.journey.place.lower():
                return
            bearing = s.hint
            self._edit(s, lambda j: _with_place(j, place, bearing))

    def set_moods(self, sid: str, moods: list[str]) -> None:
        moods = self._moods(moods)
        s = self._started(sid)
        with self._lock:
            s.last_used = self.clock()
            if moods == s.moods:
                return
            s.moods = moods
            s.ref_pending = True      # so night or snow actually show instead of being pulled back
            self._bump(s)

    # --- the pre-dreamed set ------------------------------------------------------------------

    def _key(self, s: RoamSession, heading: int) -> str:
        # The queue reuses a queued/running job by key with its original fn, so the key names everything the
        # build depends on: the world version (moods, theme, neighbours) and the spot it starts from.
        return f"{s.version}:{s.spot.id}:{heading}"

    def _current(self, s: RoamSession, e: Predream) -> bool:
        return e.version == s.version and e.spot_id == s.spot.id

    def _submit_set(self, s: RoamSession) -> None:
        """Make the queue's set for s: hint, hint + SPREAD, hint - SPREAD. A candidate within TAKE_WITHIN of a
        live pre-dream (finished, or queued/running and re-requested by its key) reuses it; the rest are new.
        Caller holds the lock."""
        if s.spot is None or self._sessions.get(s.id) is not s:
            return
        snap = self._snap(s)
        jobs: list[tuple[str, Callable]] = []
        chosen: dict[str, Predream | float] = {}
        for c in (s.hint % 360, (s.hint + SPREAD) % 360, (s.hint - SPREAD) % 360):
            near = [(_diff(e.heading, c), e.heading, k) for k, e in s.prefetch.items()
                    if k not in chosen and self._current(s, e) and _alive(e.future) and _diff(e.heading, c) <= TAKE_WITHIN]
            if near:
                k = min(near)[2]
                e = chosen[k] = s.prefetch[k]
                if not e.future.done():
                    jobs.append((k, self._job(snap, e.heading)))
                continue
            h = round(c) % 360
            k = self._key(s, h)
            if k not in chosen:
                chosen[k] = float(h)
                jobs.append((k, self._job(snap, float(h))))
        futs = self.queue.submit_set(jobs, owner=s.id)
        out = {}
        for k, e in chosen.items():
            f = futs.get(k)
            if isinstance(e, Predream):
                if f is not None and f is not e.future:
                    if _ok(e.future):     # it finished between our look and the queue's: keep it, drop the rerun
                        f.cancel()
                    else:
                        e = Predream(e.heading, e.version, e.spot_id, f)
                out[k] = e
            else:
                out[k] = Predream(e, s.version, s.spot.id, f)
        s.prefetch = out

    def _drop_set(self, s: RoamSession) -> None:
        # caller holds the lock; a running job finishes but is never used
        s.prefetch = {}
        self.queue.submit_set([], s.id)

    def _bump(self, s: RoamSession) -> None:
        """The world changed: every pre-dreamed spot is stale. Caller holds the lock."""
        s.version += 1
        self._drop_set(s)
        if not s.busy:
            self._submit_set(s)

    def _edit(self, s: RoamSession, fn: Callable[[Journey], Journey]) -> None:
        """Change the journey (pure fn: returns a new one). During a move it is applied when the move lands, on
        top of the journey that move built. Caller holds the lock."""
        if s.busy:
            s.pending.append(fn)
        else:
            s.journey = fn(s.journey)
        self._bump(s)

    def _take(self, s: RoamSession, exit_heading: float) -> _Built | None:
        """The finished pre-dreamed spot closest to the exit within TAKE_WITHIN, if any. Caller holds the lock."""
        best = min(((_diff(e.heading, exit_heading), e.heading, e) for e in s.prefetch.values()
                    if self._current(s, e) and _ok(e.future) and _diff(e.heading, exit_heading) <= TAKE_WITHIN),
                   key=lambda t: t[:2], default=None)
        return None if best is None else best[2].future.result(timeout=0)

    # --- building a spot --------------------------------------------------------------------

    def _snap(self, s: RoamSession) -> _Snap:
        # caller holds the lock
        return _Snap(spot=s.spot, spot_stats=s.spot_stats, journey=s.journey, moods=tuple(s.moods), ref=s.ref,
                     ref_pending=s.ref_pending, version=s.version, seed=s.seed, steps=s.steps)

    def _job(self, snap: _Snap, heading: float):
        return lambda: self._build(snap, heading)

    def _build(self, snap: _Snap, heading: float) -> _Built:
        """Paint the spot one step from snap.spot toward heading. Pure: reads only the snapshot."""
        j = _copy(snap.journey)
        ds = j.step(heading)
        w = 1.0 if ds.arrived else (ds.weights[1][1] if len(ds.weights) > 1 else 0.0)
        if ds.arrived:
            names = [n for n in fallback_neighbours(ds.arrived) if n not in (j.place, ds.arrived)]
            j.arrive(ds.arrived, names[:MAX_NEIGHBOURS - 1], random.Random(f"{snap.seed}:{snap.steps}"))
        base: Prompt = [(self._text(name, ds.scenery, snap.moods), wt) for name, wt in ds.weights]
        # The colour reference moves toward the place ahead as you drift, not in one jump on arrival.
        if snap.ref_pending:
            ref, bg = None, MOOD_BG_STRENGTH
        else:
            ref, bg = (_blend(snap.ref, snap.spot_stats, w) if w > 0 else snap.ref), None
        seed = (snap.seed + snap.steps + 1) % 2**31
        spot = self._paint(lambda: next_spot(self.painter, self.depth, snap.spot, heading, base, None, seed, ref,
                                             bg_strength=bg))
        return _Built(spot, j, ds.arrived, snap.version, snap.spot.id, snap.ref_pending)

    def _commit(self, s: RoamSession, built: _Built, exit_heading: float, hints: int) -> SpotView:
        stats = palette_stats(built.spot.image)
        paths = self._save(built.spot)
        with self._lock:
            self._check_alive(s, paths)
            assert built.parent_id == s.spot.id, "a build from another spot reached commit"
            j = built.journey
            for fn in s.pending:          # theme or neighbour changes made during this move
                j = fn(j)
            s.pending = []
            s.spot, s.spot_stats, s.journey, s.steps = built.spot, stats, j, s.steps + 1
            s.files.append(paths)
            if built.arrived:
                s.visited.append(built.arrived)
                s.ref = stats
            # A mood change during this build applies from the next spot instead.
            if built.ref_pending and built.version == s.version:
                s.ref, s.ref_pending = stats, False
            if s.hints == hints:          # no newer hint arrived during the move: keep walking this way
                s.hint = exit_heading
            s.prefetch = {}
            self._submit_set(s)
            view = self._view(s, paths, built.arrived)
        self._forget_files(s)
        self._maybe_refill(s)
        return view

    # --- places refill -------------------------------------------------------------------

    def _maybe_refill(self, s: RoamSession) -> None:
        """Ask the LLM for the current place's neighbours, once per place, one refill at a time."""
        with self._lock:
            if s.journey is None or s.refilling is not None or s.places_for == s.journey.place:
                return
            place = s.refilling = s.journey.place
            visited, moods = list(s.visited), list(s.moods)
            n = sum(1 for nb in s.journey.neighbours if nb.name != s.journey.came_from)

        def refill():
            try:
                found = list(self.dreamer.places(place, visited, moods, n))
            except Exception as e:  # noqa: BLE001 - a refill must never die silently; the fallback stays
                log.warning("places refill failed (%s); keeping fallback places", type(e).__name__)
                found = []
            with self._lock:
                s.refilling, s.places_for = None, place
                if (found and s.journey is not None and s.journey.place == place
                        and _renamed(s.journey, place, found).neighbours != s.journey.neighbours):
                    self._edit(s, lambda j: _renamed(j, place, found))
            self._maybe_refill(s)     # the place changed while this ran

        self._spawn(refill)

    # --- helpers ----------------------------------------------------------------------

    def _run(self, s: RoamSession, fn):
        """Run fn on the GPU queue at on-demand priority and wait for it (never under the lock)."""
        try:
            return self.queue.submit(fn, owner=s.id).result()
        except CancelledError:
            raise UnknownSession(s.id) from None   # evicted while waiting

    @staticmethod
    def _paint(fn):
        import torch

        oom = False
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            oom = True
        # Raised outside the except block, so the error doesn't chain the OOM's traceback, whose frames hold
        # the build's CUDA tensors, into the future that keeps it.
        if oom:
            torch.cuda.empty_cache()
            raise GpuUnavailable("The GPU is out of memory, try again in a moment.")

    @staticmethod
    def _text(name: str, scenery: str, moods) -> str:
        """{place}, {in-between scenery if drifting}, {mood words}, {STYLE}."""
        return ", ".join([name, *([scenery] if scenery else []), *(MOOD_WORDS[m] for m in moods), STYLE])

    def _save(self, spot: Spot) -> tuple[Path, Path]:
        jpg, png = self.scene_dir / f"{spot.id}.jpg", self.scene_dir / f"{spot.id}.png"
        spot.image.save(jpg, quality=JPEG_QUALITY)
        png.write_bytes(depth_png(spot.depth))
        return jpg, png

    def _check_alive(self, s: RoamSession, paths: tuple[Path, Path]) -> None:
        # caller holds the lock: a session evicted mid-move keeps no files and gets no set
        if self._sessions.get(s.id) is not s:
            for p in paths:
                p.unlink(missing_ok=True)
            raise UnknownSession(s.id)

    def _forget_files(self, s: RoamSession, keep: int = KEEP_SPOTS) -> None:
        """Delete the session's spot files but the newest `keep` spots'."""
        with self._lock:
            old = s.files[:max(0, len(s.files) - keep)]
            del s.files[:len(old)]
        for pair in old:
            for p in pair:
                p.unlink(missing_ok=True)

    @staticmethod
    def _view(s: RoamSession, paths: tuple[Path, Path], arrived: str | None) -> SpotView:
        sp = s.spot
        return SpotView(sp.id, paths[0], paths[1], sp.heading, sp.scale, sp.distance, s.journey.place,
                        list(s.journey.neighbours), arrived)

    @staticmethod
    def _place(place: str) -> str:
        place = (place or "").strip()[:MAX_THEME_CHARS].strip() if isinstance(place, str) else ""
        if not place:
            raise BadInput("Type a place to dream of.")
        return place

    @staticmethod
    def _moods(moods: list[str]) -> list[str]:
        return list(dict.fromkeys(m for m in (moods or ()) if m in MOODS))

    @staticmethod
    def _heading(heading) -> float:
        try:
            h = float(heading)
        except (TypeError, ValueError):
            raise BadInput("heading must be a number.") from None
        if not math.isfinite(h):
            raise BadInput("heading must be a number.")
        return h % 360.0

    def _after_failed_move(self, s: RoamSession) -> None:
        """The spot stays as it was, but changes made during the move still land. Caller holds the lock."""
        if s.pending and s.journey is not None:
            j = s.journey
            for fn in s.pending:
                j = fn(j)
            s.journey = j
        s.pending = []
        if s.spot is not None and not s.prefetch:   # a mood/theme change dropped the set mid-move
            self._submit_set(s)

    def _started(self, sid: str) -> RoamSession:
        s = self.get(sid)
        if s.spot is None:
            raise NotStarted("Type a place to start dreaming first.")
        return s

    def _claim(self, s: RoamSession):
        service = self

        class _Claim:
            def __enter__(self):
                with service._lock:
                    if s.busy:
                        raise Busy("Still loading the last move.")
                    s.busy = True
                    s.last_used = service.clock()

            def __exit__(self, exc_type, *rest):
                with service._lock:
                    s.busy = False
                    if exc_type is not None:
                        service._after_failed_move(s)

        return _Claim()
