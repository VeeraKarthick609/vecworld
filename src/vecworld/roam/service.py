"""The Dream Roam game: sessions, spots chained by heading, places with drift and arrival, speculative
pre-dreaming of the next spot (around the player's heading, in as many directions as the painter can afford), and
the painting progress of the spot a player waits for.

Headings are degrees, 0 = north (the first spot's forward), clockwise, in [0, 360).

Locking: one service lock guards every session read/write. It may be held while calling into the GPU queue
(queue lock nested inside), never while waiting on a Future, painting or calling the dreamer. A build reads only
an immutable _Snap taken at submit time and returns a _Built; nothing in the session changes until _commit.

Progress: a painter with a `progress` hook (FluxPainter) reports (done, total) during a job; the service takes the
hook over and files the latest report under the job painting now (_paint names it). The GPU queue runs one job at
a time, so that is the only job a report can come from.
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

from vecworld.config import (MAX_SESSIONS, MAX_THEME_CHARS, MOOD_BG_STRENGTH, MOOD_WORDS, MOODS, PREDREAM_HEADINGS,
                             SESSION_TTL, STYLE)
from vecworld.dream.gpu import GpuQueue, Stopped
from vecworld.dream.ideas import Dreamer
from vecworld.dream.painter import Painter, Prompt
from vecworld.roam.depth import DepthEstimator, depth_png
from vecworld.roam.errors import BadInput, Busy, GpuUnavailable, NotStarted, UnknownSession, UnknownSpot
from vecworld.roam.places import (GENERIC_SCENERY, MIN_SEPARATION, Journey, Neighbour, assign_bearings,
                                  fallback_neighbours)
from vecworld.roam.session import Predream, RoamSession, SpotView, Visit
from vecworld.roam.spot import Spot, first_spot, next_spot, spot_limits
from vecworld.scenes.palette import Stats, palette_stats

log = logging.getLogger(__name__)

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
    """The typed place becomes a neighbour at `bearing`; neighbours too close to it make way, except the place
    you came from, which always stays behind you (unless it is the typed place)."""
    if place.lower() == j.place.lower():
        return j
    keep = [n for n in j.neighbours if n.name.lower() != place.lower()
            and (n.name == j.came_from or _diff(n.bearing, bearing) >= MIN_SEPARATION)]
    new = Neighbour(place, round(bearing % 360, 6), GENERIC_SCENERY)
    return _rebuilt(j, sorted(keep + [new], key=lambda n: n.bearing), {})


Names = dict[str, tuple[str, tuple[str, ...]]]   # fallback slot name -> (LLM place, scenery)


def _name_map(j: Journey, place: str, found: list[tuple[str, tuple[str, ...]]], typed: set[str]) -> Names:
    """Which fallback-named slot of j each new LLM place takes, decided once, when the answer lands. The came-from
    place, typed places, and slots the LLM itself named stay as they are."""
    if j.place != place:
        return {}
    fallback = {n.lower() for n in fallback_neighbours(place)}
    answered = {name.lower() for name, _ in found if name}
    present = {n.name.lower() for n in j.neighbours} | {place.lower(), (j.came_from or "").lower()}
    fresh: list[tuple[str, tuple[str, ...]]] = []
    for name, scenery in found:
        if name and name.lower() not in present:
            present.add(name.lower())
            fresh.append((name, tuple(scenery[:3]) + GENERIC_SCENERY[len(scenery[:3]):3]))
    it = iter(fresh)
    out: Names = {}
    for n in j.neighbours:
        low = n.name.lower()
        if (n.name != j.came_from and low in fallback and low not in answered and low not in typed
                and (nxt := next(it, None)) is not None):
            out[n.name] = nxt
    return out


def _renamed(j: Journey, place: str, names: Names, typed: set[str]) -> Journey:
    """Rename the slots of j still carrying a mapped fallback name (bearings and drift kept). Idempotent, and
    never brings back an LLM place a typed place displaced: that slot's fallback name is gone too."""
    if j.place != place or not names:
        return j
    present = {n.name.lower() for n in j.neighbours}
    out, carry = [], {}
    for n in j.neighbours:
        nxt = names.get(n.name)
        if (nxt is not None and n.name != j.came_from and n.name.lower() not in typed
                and nxt[0].lower() not in present):
            present.add(nxt[0].lower())
            out.append(Neighbour(nxt[0], n.bearing, nxt[1]))
            carry[nxt[0]] = n.name
        else:
            out.append(n)
    return _rebuilt(j, out, carry) if carry else j


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
        # The pre-dreamed set: these offsets from the player's heading, in queue order (a painter without its own:
        # Turbo's). A painter attribute, like its style, so it can't disagree with the painter actually in use.
        self._headings = tuple(float(h) for h in getattr(painter, "predream_headings", PREDREAM_HEADINGS["turbo"]))
        # (job, {"done", "total"} or None): the job painting now (or last) and its latest report. Written only on
        # the GPU worker, read anywhere: one immutable tuple, swapped whole.
        self._painting: tuple[object, dict | None] | None = None
        if hasattr(painter, "progress"):
            painter.progress = self._painted

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
            job = self._await(s, object())
            spot = self._run(s, lambda: self._paint(job, lambda: first_spot(self.painter, self.depth, prompt, s.seed)))
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
                s.places_for, s.names, s.typed, s.hint = None, None, set(), 0.0
                s.files.append(paths)
                s.visits = {spot.id: Visit(spot, stats, journey, stats, paths, None)}
                self._submit_set(s)
                view = self._view(s, paths, None, "start")
        self._forget_files(s)
        self._maybe_refill(s)
        return view

    def next(self, sid: str, exit_heading: float, walked: float | None = None) -> SpotView:
        """Step to the next spot: a finished pre-dreamed one within TAKE_WITHIN of the exit (closest first; it was
        built for an exit at the walkable radius and says where it stands), else the exact exit heading built at
        demand priority after cancelling the queued set, dollied by `walked` (the distance from the current spot's
        centre to where the player leaves it, in its units; None = its walkable radius that way)."""
        exit_heading = self._heading(exit_heading)
        walked = self._walked(walked)
        s = self._started(sid)
        with self._claim(s):
            running, served = None, "predream"
            with self._lock:
                hints = s.hints
                built = self._take(s, exit_heading)
                if built is None:
                    running = self._keep_running(s, exit_heading)
                    if running is not None:
                        s.awaiting = (s.id, running[0])
            if running is not None:
                # On the single worker a demand build would only start after this job anyway.
                built, served = self._wait(s, running[1]), "waited"
                with self._lock:
                    if self._sessions.get(s.id) is not s:
                        raise UnknownSession(s.id)
                    # A mood or theme change during the wait can't stop a running job: rebuild instead.
                    if built is not None and (built.version != s.version or built.parent_id != s.spot.id):
                        built = None
            if built is None:
                served = "demand"
                with self._lock:
                    self._set([], s.id)   # the queued set (and a pre-dream painting elsewhere) would only delay it
                    snap = self._snap(s)
                    job = self._await(s, object())
                built = self._run(s, lambda: self._build(snap, exit_heading, walked, job=job))
            view = self._commit(s, built, exit_heading, hints, served)
            log.debug("next: %s, walked %s -> distance %.2f, scale %.2f", served, walked, view.distance, view.scale)
            return view

    def enter(self, sid: str, spot_id: str) -> SpotView:
        """Step back into a spot the player walked into without a move: one the current spot was built from (its
        parent, or further back while the files last: a move can land while the player walks back), one built
        from the current spot, or one built from its parent (a sibling: the player walked from one child of a spot
        into another). The journey and colour reference return to what they were when it was current; the
        pre-dreamed set is made around it. No GPU work."""
        s = self._started(sid)
        with self._claim(s):
            with self._lock:
                if spot_id != s.spot.id:
                    v = s.visits.get(spot_id)
                    up = self._ancestors(s)
                    if v is None or not (v.parent_id == s.spot.id or spot_id in up
                                         or (up and v.parent_id == up[0])):
                        raise UnknownSpot("That spot has faded from the dream.")
                    self._leave(s)
                    j = v.journey
                    if s.names is not None:   # the LLM's places, if they landed after it was current
                        j = _renamed(j, *s.names, s.typed)
                    # Places typed since stay where they were typed.
                    have = {n.name.lower() for n in j.neighbours}
                    for n in s.journey.neighbours:
                        if n.name.lower() in s.typed and n.name.lower() not in have:
                            j = _with_place(j, n.name, n.bearing)
                    s.spot, s.spot_stats, s.journey, s.ref = v.spot, v.spot_stats, j, v.ref
                    s.files.remove(v.paths)   # the most recent now: its files are the last to go
                    s.files.append(v.paths)
                    s.prefetch = {}
                    self._submit_set(s)
                view = self._view(s, s.visits[s.spot.id].paths, None, "enter")
            self._forget_files(s)
            return view

    def _ancestors(self, s: RoamSession) -> list[str]:
        """The ids the current spot was built from, nearest first, as far back as the session still holds."""
        out, v = [], s.visits.get(s.spot.id)
        while v is not None and v.parent_id is not None and v.parent_id not in out:
            out.append(v.parent_id)
            v = s.visits.get(v.parent_id)
        return out

    def _leave(self, s: RoamSession) -> None:
        """Remember the current spot's journey and reference as they are now. Caller holds the lock."""
        v = s.visits.get(s.spot.id)
        if v is not None:
            s.visits[s.spot.id] = Visit(v.spot, v.spot_stats, s.journey, s.ref, v.paths, v.parent_id)

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
            s.typed.add(place.lower())
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
        # build depends on: the world version (moods, theme, typed places) and the spot it starts from.
        # One exception: the LLM's neighbour names landing do not bump the version (see _rename). Such a
        # pre-dream's journey is renamed at commit; its prompt may name a fallback place unless the player was
        # already drifting toward it, which does bump.
        return f"{s.version}:{s.spot.id}:{heading}"

    def _current(self, s: RoamSession, e: Predream) -> bool:
        return e.version == s.version and e.spot_id == s.spot.id

    def _submit_set(self, s: RoamSession) -> None:
        """Make the queue's set for s: hint + each of the painter's pre-dream headings. A candidate within
        TAKE_WITHIN of a live pre-dream (finished, or queued/running and re-requested by its key) reuses it; the
        rest are new. Caller holds the lock."""
        if s.spot is None or self._sessions.get(s.id) is not s:
            return
        snap = self._snap(s)
        jobs: list[tuple[str, Callable]] = []
        chosen: dict[str, Predream | float] = {}
        for c in ((s.hint + d) % 360 for d in self._headings):
            near = [(_diff(e.heading, c), e.heading, k) for k, e in s.prefetch.items()
                    if k not in chosen and self._current(s, e) and _alive(e.future) and _diff(e.heading, c) <= TAKE_WITHIN]
            if near:
                k = min(near)[2]
                e = chosen[k] = s.prefetch[k]
                if not e.future.done():
                    jobs.append((k, self._job(s, k, snap, e.heading)))
                continue
            h = round(c) % 360
            k = self._key(s, h)
            if k not in chosen:
                chosen[k] = float(h)
                jobs.append((k, self._job(s, k, snap, float(h))))
        futs = self._set(jobs, s.id)
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

    def _set(self, jobs: list, owner: str) -> dict:
        """Make the queue's pre-dream set for a session. A pre-dream still painting that is not in it is stopped (at
        its next check_stop: FLUX checks between transformer passes): built from a spot the player left, or toward a
        direction no longer wanted, it would only hold up the GPU (a FLUX step is ~61 s) for the spot the player
        walks toward next."""
        return self.queue.submit_set(jobs, owner, stop_running=True)

    def _drop_set(self, s: RoamSession) -> None:
        # caller holds the lock; a running job is stopped (if it checks in) and never used
        s.prefetch = {}
        self._set([], s.id)

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

    def _keep_running(self, s: RoamSession, exit_heading: float) -> tuple[str, Future] | None:
        """(key, future) of the running pre-dream closest to the exit within TAKE_WITHIN, with the rest of the
        queued set cancelled; None if there is none. Queued (not started) pre-dreams lose to a demand build. Caller
        holds the lock."""
        best = min(((_diff(e.heading, exit_heading), e.heading, k) for k, e in s.prefetch.items()
                    if self._current(s, e) and e.future.running() and _diff(e.heading, exit_heading) <= TAKE_WITHIN),
                   default=None)
        if best is None:
            return None
        k = best[2]
        e = s.prefetch[k]
        f = self._set([(k, self._job(s, k, self._snap(s), e.heading))], s.id).get(k)
        if f is not None and f is not e.future:
            f.cancel()    # it ended between our look and the queue's; its own future still holds the result
        return k, e.future

    def _wait(self, s: RoamSession, f: Future) -> _Built | None:
        """The pre-dream's result, waited for outside the lock; None if it failed (build on demand instead)."""
        try:
            return f.result()
        except CancelledError:
            raise UnknownSession(s.id) from None
        except Exception as e:  # noqa: BLE001 - out of memory or broken: build on demand instead
            if not isinstance(e, (GpuUnavailable, Stopped)):
                log.warning("pre-dream failed (%s); building on demand", type(e).__name__)
            return None

    # --- building a spot --------------------------------------------------------------------

    def _snap(self, s: RoamSession) -> _Snap:
        # caller holds the lock
        return _Snap(spot=s.spot, spot_stats=s.spot_stats, journey=s.journey, moods=tuple(s.moods), ref=s.ref,
                     ref_pending=s.ref_pending, version=s.version, seed=s.seed, steps=s.steps)

    def _job(self, s: RoamSession, key: str, snap: _Snap, heading: float):
        job = (s.id, key)          # what a move waiting on this pre-dream awaits (see next)
        return lambda: self._build(snap, heading, job=job)

    def _build(self, snap: _Snap, heading: float, walked: float | None = None, job: object = None) -> _Built:
        """Paint the spot one step from snap.spot toward heading, for a player who walked `walked` (None: to the
        walkable radius, as a pre-dream expects), as the job `job` (its progress is filed under it). Pure: reads
        only the snapshot."""
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
        spot = self._paint(job, lambda: next_spot(self.painter, self.depth, snap.spot, heading, base, None, seed,
                                                  ref, bg_strength=bg, walked=walked))
        return _Built(spot, j, ds.arrived, snap.version, snap.spot.id, snap.ref_pending)

    def _commit(self, s: RoamSession, built: _Built, exit_heading: float, hints: int, served: str) -> SpotView:
        stats = palette_stats(built.spot.image)
        paths = self._save(built.spot)   # first: a failed write leaves the session on its spot
        with self._lock:
            self._check_alive(s, paths)
            assert built.parent_id == s.spot.id, "a build from another spot reached commit"
            j = built.journey
            if s.names is not None:       # the LLM's places, if they landed after this build's snapshot
                j = _renamed(j, *s.names, s.typed)
            for fn in s.pending:          # theme or neighbour changes made during this move
                j = fn(j)
            s.pending = []
            self._leave(s)
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
            s.visits[built.spot.id] = Visit(built.spot, stats, j, s.ref, paths, built.parent_id)
            s.prefetch = {}
            self._submit_set(s)
            view = self._view(s, paths, built.arrived, served)
        self._forget_files(s)
        self._maybe_refill(s)
        return view

    # --- painting progress ---------------------------------------------------------------

    def progress(self, sid: str) -> dict | None:
        """How far the painter has got with the spot this session's player is waiting for, as {"done", "total"}
        (the painter's own units: FLUX counts transformer passes), or None when that spot isn't painting (queued
        behind other work, finished, or a painter that doesn't report). During a move, that spot is the job the move
        waits on: the running pre-dream it took over, or its own build. Otherwise it is the spot the player would
        get at the edge: the pre-dream nearest the way they face."""
        s = self.get(sid)
        with self._lock:
            job = s.awaiting
            if job is None and not s.busy and s.spot is not None:
                near = min(((_diff(e.heading, s.hint), k) for k, e in s.prefetch.items()
                            if self._current(s, e) and _diff(e.heading, s.hint) <= TAKE_WITHIN), default=None)
                job = None if near is None else (s.id, near[1])
        painting = self._painting
        if job is None or painting is None or painting[0] != job:
            return None
        return painting[1]

    def _painted(self, done: int, total: int) -> None:
        """The painter's progress hook (GPU worker thread): file the report under the job painting now."""
        painting = self._painting
        if painting is not None:
            self._painting = (painting[0], {"done": int(done), "total": int(total)})

    @staticmethod
    def _await(s: RoamSession, job: object) -> object:
        """The session's move now waits on `job` (cleared when the move ends). Caller holds the lock or the claim."""
        s.awaiting = job
        return job

    # --- places refill -------------------------------------------------------------------

    def _maybe_refill(self, s: RoamSession) -> None:
        """Ask the LLM for the current place's neighbours, once per place, one refill at a time."""
        with self._lock:
            if (self._sessions.get(s.id) is not s or s.journey is None or s.refilling is not None
                    or s.places_for == s.journey.place):
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
                if found and self._sessions.get(s.id) is s and s.journey is not None and s.journey.place == place:
                    self._rename(s, place, found)
            self._maybe_refill(s)     # the place changed while this ran

        self._spawn(refill)

    def _rename(self, s: RoamSession, place: str, found: list) -> None:
        """Apply the LLM's places without discarding the pre-dreamed set: every move re-applies them to the
        journey it built (see _commit). Only a player mid-drift toward a renamed neighbour, heading its way, has
        pre-dreams whose prompts name the wrong place, so only then is the set invalidated. Caller holds the lock."""
        names = _name_map(s.journey, place, found, s.typed)
        s.names = (place, names)
        old, new = s.journey, _renamed(s.journey, place, names, s.typed)
        if new is old:
            return
        s.journey = new
        kept = {n.name for n in new.neighbours}
        if any(old.drift.get(n.name, 0) >= 1 and n.name not in kept and _diff(n.bearing, s.hint) <= TAKE_WITHIN
               for n in old.neighbours):
            self._bump(s)

    # --- helpers ----------------------------------------------------------------------

    def _run(self, s: RoamSession, fn):
        """Run fn on the GPU queue at on-demand priority and wait for it (never under the lock)."""
        try:
            return self.queue.submit(fn, owner=s.id).result()
        except CancelledError:
            raise UnknownSession(s.id) from None   # evicted while waiting

    def _paint(self, job: object, fn):
        """Run fn (painting and depth, on the GPU worker) as the job `job`: the painter's progress until the next
        job starts is this job's."""
        import torch

        self._painting = (job, None)
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

    def _text(self, name: str, scenery: str, moods) -> str:
        """{place}, {in-between scenery if drifting}, {mood words}, {the painter's style, else STYLE}."""
        style = getattr(self.painter, "style", STYLE)
        return ", ".join([name, *([scenery] if scenery else []), *(MOOD_WORDS[m] for m in moods), style])

    def _paths(self, spot: Spot) -> tuple[Path, Path]:
        return self.scene_dir / f"{spot.id}.jpg", self.scene_dir / f"{spot.id}.png"

    @staticmethod
    def _write(spot: Spot, paths: tuple[Path, Path]) -> None:
        spot.image.save(paths[0], quality=JPEG_QUALITY)
        paths[1].write_bytes(depth_png(spot.depth))

    def _save(self, spot: Spot) -> tuple[Path, Path]:
        paths = self._paths(spot)
        self._write(spot, paths)
        return paths

    def _check_alive(self, s: RoamSession, paths: tuple[Path, Path]) -> None:
        # caller holds the lock: a session evicted mid-move keeps no files and gets no set
        if self._sessions.get(s.id) is not s:
            for p in paths:
                p.unlink(missing_ok=True)
            raise UnknownSession(s.id)

    def _forget_files(self, s: RoamSession, keep: int = KEEP_SPOTS) -> None:
        """Delete the session's spot files but the `keep` most recently made or entered spots', and (keep > 0)
        always every spot the viewer keeps placed (world.js prune): the current spot, its parent, and every child
        of either (the player can walk into any of them)."""
        with self._lock:
            wanted = set(s.files[max(0, len(s.files) - keep):]) if keep else set()
            if keep and s.spot is not None and (cur := s.visits.get(s.spot.id)) is not None:
                wanted.add(cur.paths)
                if cur.parent_id in s.visits:
                    wanted.add(s.visits[cur.parent_id].paths)
                family = {s.spot.id} | ({cur.parent_id} if cur.parent_id is not None else set())
                wanted.update(v.paths for v in s.visits.values() if v.parent_id in family)
            old = [pair for pair in s.files if pair not in wanted]
            s.files = [pair for pair in s.files if pair in wanted]
            s.visits = {k: v for k, v in s.visits.items() if v.paths in wanted}
        for pair in old:
            for p in pair:
                p.unlink(missing_ok=True)

    @staticmethod
    def _view(s: RoamSession, paths: tuple[Path, Path], arrived: str | None, served: str) -> SpotView:
        sp = s.spot
        return SpotView(sp.id, paths[0], paths[1], sp.heading, sp.scale, sp.distance, s.journey.place,
                        list(s.journey.neighbours), arrived, [round(float(x), 4) for x in spot_limits(sp)], served)

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

    @staticmethod
    def _walked(walked) -> float | None:
        if walked is None:
            return None
        try:
            w = float(walked)
        except (TypeError, ValueError):
            raise BadInput("walked must be a distance.") from None
        if not math.isfinite(w) or w < 0:
            raise BadInput("walked must be a distance.")
        return w

    def _after_failed_move(self, s: RoamSession) -> None:
        """The spot stays as it was, but changes made during the move still land. Caller holds the lock."""
        if s.pending and s.journey is not None:
            j = s.journey
            for fn in s.pending:
                j = fn(j)
            s.journey = j
        s.pending = []
        # A mood/theme change dropped the set mid-move, or the move cancelled its queued part: pre-dream again.
        if s.spot is not None and (not s.prefetch or not all(_alive(e.future) for e in s.prefetch.values())):
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
                    s.awaiting = None
                    if exc_type is not None:
                        service._after_failed_move(s)

        return _Claim()
