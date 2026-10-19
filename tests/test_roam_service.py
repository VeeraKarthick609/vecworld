import threading
from concurrent.futures import Future, wait

import numpy as np
import pytest
from conftest import FakeDepth, FakeDreamer, FakePainter
from PIL import Image

from vecworld.config import MOOD_BG_STRENGTH, MOOD_WORDS, SESSION_TTL, STYLE
from vecworld.dream.errors import BadInput, Busy, GpuUnavailable, NotStarted, UnknownSession
from vecworld.roam.places import fallback_neighbours
from vecworld.roam.service import RoamService, SpotView
from vecworld.scenes.palette import palette_stats

NIGHT = MOOD_WORDS["night"]


class GatedPainter(FakePainter):
    """Once armed, step() signals `entered` and blocks until `gate` is set."""

    def __init__(self):
        super().__init__()
        self.armed = False
        self.entered, self.gate = threading.Event(), threading.Event()

    def step(self, *a, **kw):
        if self.armed:
            self.entered.set()
            assert self.gate.wait(5)
        return super().step(*a, **kw)


class TagPainter(FakePainter):
    """Step n (1-based) paints a solid (n * 20, 0, 0), so an unmatched spot names the call that made it."""

    def step(self, prev, base, focus, focus_at, seed, bg_strength=None):
        super().step(prev, base, focus, focus_at, seed, bg_strength=bg_strength)
        n = len(steps_of(self))
        return Image.new("RGB", prev.size, (n * 20 % 256, 0, 0))


class ManualQueue:
    """Runs on-demand jobs at once on the caller's thread; prefetch sets stay pending until run() (never, mostly)."""

    def __init__(self):
        self.sets: list[list] = []
        self.pending: dict = {}

    def submit(self, fn, owner, prefetch=False):
        assert not prefetch
        f = Future()
        try:
            f.set_result(fn())
        except Exception as e:  # noqa: BLE001
            f.set_exception(e)
        return f

    def submit_set(self, jobs, owner):
        self.sets.append([k for k, _ in jobs])
        out = {}
        for k, fn in jobs:
            f = self.pending.get(k)
            if f is None or f.done():
                f = Future()
                self.pending[k] = f
            out[k] = f
        for k in [k for k in self.pending if k not in out]:
            self.pending.pop(k).cancel()
        return out

    def cancel_owner(self, owner):
        for f in self.pending.values():
            f.cancel()
        self.pending.clear()


def run_now(fn):
    fn()


def never(fn):
    pass


def steps_of(painter):
    return [kw for name, kw in painter.calls if name == "step"]


@pytest.fixture
def make(tmp_path):
    def factory(painter=None, dreamer=None, spawn=never, **kw):
        return RoamService(painter or FakePainter(), FakeDepth(), dreamer or FakeDreamer(), tmp_path / "scenes",
                           spawn=spawn, **kw)
    return factory


@pytest.fixture
def svc(make):
    return make()


def started(svc, place="forest", moods=()):
    s = svc.new_session()
    svc.start(s.id, place, list(moods))
    return s


def set_futures(s):
    return {k: e.future for k, e in s.prefetch.items()}


def wait_set(s):
    for f in set_futures(s).values():
        f.result(timeout=5)


def spot_of(f):
    return f.result(timeout=0).spot


def block_worker(svc):
    """Occupy the GPU queue's worker until the returned event is set."""
    running, release = threading.Event(), threading.Event()
    svc.queue.submit(lambda: (running.set(), release.wait(5)), owner="blocker")
    assert running.wait(5)
    return release


def key_heading(k):
    return int(k.rsplit(":", 1)[1])


# --- start ---------------------------------------------------------------------------

def test_start_paints_a_first_spot_and_files(svc):
    s = svc.new_session()
    v = svc.start(s.id, "  forest ", ["night", "disco"])
    assert isinstance(v, SpotView)
    assert v.heading == 0 and v.scale == 1 and v.distance == 0 and v.arrived is None
    assert v.place == "forest"
    assert [n.name for n in v.neighbours] == fallback_neighbours("forest")
    base = svc.painter.calls[0][1]["base"]
    assert len(base) == 1 and base[0][0].startswith("forest, ") and NIGHT in base[0][0] and base[0][0].endswith(STYLE)
    assert Image.open(v.image_path).format == "JPEG"
    d = Image.open(v.depth_path)
    assert d.format == "PNG" and d.mode == "L" and d.size == Image.open(v.image_path).size


def test_start_submits_a_set_of_three_heading_first(make):
    q = ManualQueue()
    svc = make(queue=q)
    s = started(svc)
    assert len(q.sets) == 1
    keys = q.sets[0]
    assert [key_heading(k) for k in keys] == [0, 60, 300]
    assert all(k.startswith(f"{s.version}:{s.spot.id}:") for k in keys)
    assert set(s.prefetch) == set(keys)


def test_bad_input(svc):
    s = svc.new_session()
    with pytest.raises(BadInput):
        svc.start(s.id, "   ", [])
    with pytest.raises(NotStarted):
        svc.next(s.id, 0)
    with pytest.raises(NotStarted):
        svc.heading(s.id, 0)
    svc.start(s.id, "forest", [])
    for bad in (float("nan"), float("inf"), "north", None):
        with pytest.raises(BadInput):
            svc.next(s.id, bad)
        with pytest.raises(BadInput):
            svc.heading(s.id, bad)
    with pytest.raises(UnknownSession):
        svc.next("nope", 0)


# --- choosing a pre-dreamed spot at the exit -----------------------------------------------

def test_next_within_45_uses_the_finished_spot(svc):
    s = started(svc)
    wait_set(s)
    by_heading = {key_heading(k): f for k, f in set_futures(s).items()}
    assert len(steps_of(svc.painter)) == 3
    release = block_worker(svc)     # nothing can paint now: the move must not need the GPU
    try:
        v = svc.next(s.id, 20)
    finally:
        release.set()
    assert v.spot_id == spot_of(by_heading[0]).id and v.heading == 0
    assert len(steps_of(svc.painter)) == 3
    assert s.spot is spot_of(by_heading[0]) and s.steps == 1


def test_closest_finished_spot_wins(svc):
    s = started(svc)
    wait_set(s)
    by_heading = {key_heading(k): f for k, f in set_futures(s).items()}
    v = svc.next(s.id, 40)          # 20 from 60, 40 from 0
    assert v.spot_id == spot_of(by_heading[60]).id and v.heading == 60


def test_exit_far_from_all_prefetches_builds_on_demand(svc):
    s = started(svc)
    wait_set(s)
    prefetched = {spot_of(f).id for f in set_futures(s).values()}
    v = svc.next(s.id, 120)         # 60 from the nearest of 0, 60 and 300
    assert v.heading == pytest.approx(120)
    assert v.spot_id not in prefetched
    assert len(steps_of(svc.painter)) >= 4


def test_exit_cancels_the_queued_set_and_builds_at_demand_priority(make):
    svc = make(painter=GatedPainter())
    s = svc.new_session()
    svc.painter.armed = True
    svc.start(s.id, "forest", [])                 # the panorama is not gated; the set's first step is
    assert svc.painter.entered.wait(5)
    fut = {key_heading(k): f for k, f in set_futures(s).items()}
    submitted, submit = threading.Event(), svc.queue.submit

    def spy(fn, owner, prefetch=False):
        f = submit(fn, owner, prefetch=prefetch)
        if owner == s.id and not prefetch:
            submitted.set()
        return f

    svc.queue.submit = spy
    out = []
    t = threading.Thread(target=lambda: out.append(svc.next(s.id, 180)))
    t.start()
    ok = submitted.wait(5)
    svc.painter.gate.set()
    t.join(5)
    assert ok
    assert fut[60].cancelled() and fut[300].cancelled()
    assert not fut[0].cancelled()                 # it was already running
    assert out[0].heading == pytest.approx(180)
    assert s.steps == 1


def test_exit_backwards(svc):
    s = started(svc)
    wait_set(s)
    v1 = svc.next(s.id, 180)
    assert v1.heading == pytest.approx(180) and v1.distance > 0 and v1.scale > 0
    wait_set(s)
    v2 = svc.next(s.id, 0)                        # straight back the way we came
    assert v2.heading == pytest.approx(0) and s.steps == 2
    v3 = svc.next(s.id, -170)                     # normalised
    assert v3.heading == pytest.approx(190)


def test_failed_prefetch_is_not_used(svc, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    s = started(svc)
    wait_set(s)
    svc.painter.fail_next = True
    svc.heading(s.id, 180)                        # 180, 240, 120: all new, and 180 (first) fails
    futs = {key_heading(k): f for k, f in set_futures(s).items()}
    done, _ = wait(list(futs.values()), timeout=5)
    assert len(done) == 3
    assert isinstance(futs[180].exception(), GpuUnavailable)
    v = svc.next(s.id, 180)                       # 240 and 120 are 60 away: built on demand
    assert v.heading == pytest.approx(180)
    assert v.spot_id not in {spot_of(futs[h]).id for h in (120, 240)}


def test_after_a_move_the_new_set_is_around_the_exit_heading(make):
    q = ManualQueue()
    svc = make(queue=q)
    s = started(svc)
    v = svc.next(s.id, 100)                       # nothing ran: on demand
    assert v.heading == pytest.approx(100)
    keys = q.sets[-1]
    assert [key_heading(k) for k in keys] == [100, 160, 40]
    assert all(k.startswith(f"{s.version}:{v.spot_id}:") for k in keys)


# --- the heading hint ------------------------------------------------------------------

def test_heading_hint_retargets_the_set(make):
    svc = make(painter=GatedPainter())
    s = svc.new_session()
    svc.painter.armed = True
    svc.start(s.id, "forest", [])
    assert svc.painter.entered.wait(5)            # the 0 job is running, 60 and 300 are queued
    old = {key_heading(k): f for k, f in set_futures(s).items()}
    svc.heading(s.id, 180)
    assert old[60].cancelled() and old[300].cancelled() and not old[0].cancelled()
    new = set_futures(s)
    assert sorted(key_heading(k) for k in new) == [120, 180, 240]
    assert not any(f.cancelled() for f in new.values())
    svc.painter.gate.set()
    wait_set(s)


def test_heading_hint_keeps_close_directions(make):
    svc = make(painter=GatedPainter())
    s = svc.new_session()
    svc.painter.armed = True
    svc.start(s.id, "forest", [])
    assert svc.painter.entered.wait(5)
    old = set_futures(s)
    svc.heading(s.id, 20)                         # 20, 80, 320: each within 45 of 0, 60, 300
    assert set_futures(s) == old
    assert not any(f.cancelled() for f in old.values())
    svc.painter.gate.set()
    wait_set(s)


def test_finished_spots_near_a_candidate_are_kept(svc):
    s = started(svc)
    wait_set(s)
    old = {key_heading(k): f for k, f in set_futures(s).items()}
    n = len(steps_of(svc.painter))
    svc.heading(s.id, 120)                        # 120, 180, 60: only 60 has a finished spot near it
    new = {key_heading(k): f for k, f in set_futures(s).items()}
    assert new[60] is old[60]
    assert 0 not in new and 300 not in new
    assert sorted(new) == [60, 120, 180]
    wait_set(s)
    assert len(steps_of(svc.painter)) == n + 2


# --- moods and themes ------------------------------------------------------------------

def test_mood_change_discards_predreamed_spots(make):
    svc = make(painter=TagPainter())
    s = started(svc)
    wait_set(s)
    stale = {spot_of(f).id for f in set_futures(s).values()}
    version = s.version
    svc.set_moods(s.id, ["night", "disco"])
    assert s.moods == ["night"] and s.version == version + 1
    assert all(k.startswith(f"{s.version}:") for k in s.prefetch)
    v = svc.next(s.id, 0)
    assert v.spot_id not in stale
    # the spot shown was painted for the night (unmatched: the first spot after a mood change is the new reference)
    n = s.spot.image.getpixel((0, 0))[0] // 20
    call = steps_of(svc.painter)[n - 1]
    assert NIGHT in call["base"][0][0] and call["bg_strength"] == MOOD_BG_STRENGTH
    assert s.ref_pending is False


def test_same_moods_keep_the_set(svc):
    s = started(svc, moods=["night"])
    wait_set(s)
    old, version = set_futures(s), s.version
    svc.set_moods(s.id, ["night"])
    assert s.version == version and set_futures(s) == old


def test_mood_change_cancels_the_queued_set(make):
    svc = make(painter=GatedPainter())
    s = svc.new_session()
    svc.painter.armed = True
    svc.start(s.id, "forest", [])
    assert svc.painter.entered.wait(5)
    old = {key_heading(k): f for k, f in set_futures(s).items()}
    svc.set_moods(s.id, ["snow"])
    assert old[60].cancelled() and old[300].cancelled()
    assert all(k.startswith(f"{s.version}:") for k in s.prefetch)   # same headings, new world
    svc.painter.gate.set()
    wait_set(s)
    v = svc.next(s.id, 0)
    assert v.spot_id != spot_of(old[0]).id


def test_theme_becomes_a_neighbour_at_the_current_heading(make):
    q = ManualQueue()
    svc = make(queue=q)
    s = started(svc)
    svc.heading(s.id, 90)
    version = s.version
    svc.set_theme(s.id, "volcano")
    assert s.version == version + 1
    names = {n.name: n.bearing for n in s.journey.neighbours}
    assert names["volcano"] == pytest.approx(90)
    assert all(abs((b - 90 + 180) % 360 - 180) >= 60 for n, b in names.items() if n != "volcano")
    v = svc.next(s.id, 90)
    assert any(n.name == "volcano" for n in v.neighbours)
    base = steps_of(svc.painter)[-1]["base"]
    assert base[0][0].startswith("forest") and base[1][0].startswith("volcano")
    assert base[1][1] == pytest.approx(1 / 3)


def test_theme_during_a_move_applies_when_it_lands(make):
    svc = make(painter=GatedPainter())
    s = started(svc)
    wait_set(s)
    svc.painter.armed = True
    out = []
    t = threading.Thread(target=lambda: out.append(svc.next(s.id, 180)))
    t.start()
    assert svc.painter.entered.wait(5)
    svc.heading(s.id, 180)
    svc.set_theme(s.id, "volcano")
    svc.painter.armed = False
    svc.painter.gate.set()
    t.join(5)
    assert any(n.name == "volcano" and n.bearing == pytest.approx(180) for n in out[0].neighbours)
    assert all(k.startswith(f"{s.version}:") for k in s.prefetch)


def test_theme_during_a_failed_move_still_applies(make, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    svc = make(painter=GatedPainter())
    s = started(svc)
    wait_set(s)
    spot = s.spot
    svc.painter.armed = True
    errors = []

    def move():
        try:
            svc.next(s.id, 180)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=move)
    t.start()
    assert svc.painter.entered.wait(5)
    svc.heading(s.id, 180)
    svc.set_theme(s.id, "volcano")
    assert s.prefetch == {}                       # the stale set is gone
    svc.painter.armed = False
    svc.painter.fail_next = True
    svc.painter.gate.set()
    t.join(5)
    assert len(errors) == 1 and isinstance(errors[0], GpuUnavailable)
    assert s.spot is spot and s.pending == []
    assert any(n.name == "volcano" and n.bearing == pytest.approx(180) for n in s.journey.neighbours)
    assert sorted(key_heading(k) for k in s.prefetch) == [120, 180, 240]   # pre-dreaming resumes
    assert all(k.startswith(f"{s.version}:") for k in s.prefetch)


# --- drift, arrival, palette --------------------------------------------------------------

def walk_to(svc, s, name):
    bearing = next(n.bearing for n in s.journey.neighbours if n.name == name)
    return bearing, [svc.next(s.id, bearing) for _ in range(3)]


def test_drift_then_arrival(make):
    svc = make(queue=ManualQueue())
    s = started(svc, "forest")
    target = s.journey.neighbours[0].name
    bearing, (v1, v2, v3) = walk_to(svc, s, target)
    weights = [[w for _, w in c["base"]] for c in steps_of(svc.painter)]
    assert weights[0] == pytest.approx([2 / 3, 1 / 3]) and weights[1] == pytest.approx([1 / 3, 2 / 3])
    assert weights[2] == [1.0]
    assert steps_of(svc.painter)[2]["base"][0][0].startswith(target)
    assert (v1.place, v1.arrived) == ("forest", None) and (v2.place, v2.arrived) == ("forest", None)
    assert v3.place == target and v3.arrived == target
    back = next(n for n in v3.neighbours if n.name == "forest")
    assert back.bearing == pytest.approx((bearing + 180) % 360)
    assert s.visited == ["forest", target]


def test_scenery_is_in_drift_prompts(make):
    svc = make(queue=ManualQueue())
    s = started(svc, "forest")
    target = s.journey.neighbours[0]
    walk_to(svc, s, target.name)
    first = steps_of(svc.painter)[0]["base"]
    assert all(target.scenery[0] in text for text, _ in first)


def test_palette_reference_moves_gradually_on_arrival(make, monkeypatch):
    import vecworld.roam.service as service

    refs = []
    real = service.next_spot

    def spy(*a, **kw):
        refs.append(a[7] if len(a) > 7 else kw["ref"])
        return real(*a, **kw)

    monkeypatch.setattr(service, "next_spot", spy)
    svc = make(painter=TagPainter(), queue=ManualQueue())
    s = started(svc, "forest")
    start_ref = s.ref
    target = s.journey.neighbours[0].name
    bearing = s.journey.neighbours[0].bearing
    svc.next(s.id, bearing)
    st1 = s.spot_stats
    assert s.ref is start_ref
    svc.next(s.id, bearing)
    st2 = s.spot_stats
    assert s.ref is start_ref
    v = svc.next(s.id, bearing)
    assert v.arrived == target
    # drift spot 1 blends toward the start spot's own colours (the same as the reference), spot 2 2/3 of
    # the way to spot 1's, and the arrival spot takes spot 2's; the arrived spot becomes the reference
    close(refs[0], start_ref)
    close(refs[1], tuple(r + 2 / 3 * (t - r) for r, t in zip(start_ref, st1)))
    close(refs[2], st2)
    close(s.ref, palette_stats(s.spot.image))


def close(a, b):
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)


def test_walking_away_resets_drift(make):
    svc = make(queue=ManualQueue())
    s = started(svc, "forest")
    n0 = s.journey.neighbours[0]
    svc.next(s.id, n0.bearing)
    svc.next(s.id, n0.bearing)
    away = (n0.bearing + 180) % 360
    v = svc.next(s.id, away)
    assert v.arrived is None and s.journey.drift[n0.name] == 0


# --- places refill ------------------------------------------------------------------------

def test_fallback_until_the_llm_answers(make):
    jobs = []
    svc = make(spawn=jobs.append, queue=ManualQueue())
    s = started(svc, "forest")
    assert [n.name for n in s.journey.neighbours] == fallback_neighbours("forest")
    bearings = [n.bearing for n in s.journey.neighbours]
    version = s.version
    assert len(jobs) == 1
    jobs.pop()()
    assert svc.dreamer.place_calls == [("forest", ["forest"], [], 4)]
    assert [n.name for n in s.journey.neighbours] == FakeDreamer.PLACES
    assert [n.bearing for n in s.journey.neighbours] == bearings
    assert s.journey.neighbours[0].scenery == ("a path of 0 stones", "a mossy fence post", "a crooked old tree")
    assert s.version == version + 1                    # pre-dreamed fallback spots are stale now


def test_refill_skipped_while_one_runs_then_catches_up(make):
    jobs = []
    svc = make(spawn=jobs.append, queue=ManualQueue())
    s = started(svc, "forest")
    svc.start(s.id, "desert", [])
    assert len(jobs) == 1                              # the forest refill is still out
    jobs.pop()()                                       # it answers for a place we left: ignored, then catch up
    assert [n.name for n in s.journey.neighbours] == fallback_neighbours("desert")
    assert len(jobs) == 1
    jobs.pop()()
    assert svc.dreamer.place_calls[-1][0] == "desert"
    assert [n.name for n in s.journey.neighbours] == FakeDreamer.PLACES


def test_refill_failure_keeps_fallback(make):
    class Broken(FakeDreamer):
        def places(self, *a):
            raise RuntimeError("model fell over")

    svc = make(dreamer=Broken(), spawn=run_now, queue=ManualQueue())
    s = started(svc, "forest")
    assert [n.name for n in s.journey.neighbours] == fallback_neighbours("forest")
    assert s.refilling is None


def test_refill_runs_off_the_calling_thread(make):
    done, seen = threading.Event(), []

    class Recording(FakeDreamer):
        def places(self, *a):
            seen.append(threading.current_thread())
            done.set()
            return super().places(*a)

    svc = make(dreamer=Recording(), spawn=None, queue=ManualQueue())
    started(svc, "forest")
    assert done.wait(5)
    assert seen[0] is not threading.current_thread()


def test_arrival_asks_for_the_new_places_neighbours(make):
    svc = make(spawn=run_now, queue=ManualQueue())
    s = started(svc, "forest")
    target = s.journey.neighbours[1].name
    walk_to(svc, s, target)
    place, journey, moods, n = svc.dreamer.place_calls[-1]
    assert place == target and journey == ["forest", target]


# --- failures, concurrency, sessions -------------------------------------------------------

def test_oom_leaves_the_session_unchanged(svc, monkeypatch):
    import torch

    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    s = started(svc)
    wait_set(s)
    before = (s.spot, s.journey, s.steps, s.ref, s.version, list(s.files), s.hint, dict(s.prefetch))
    svc.painter.fail_next = True
    with pytest.raises(GpuUnavailable) as e:
        svc.next(s.id, 120)
    assert e.value.__context__ is None and e.value.__cause__ is None
    assert emptied
    assert (s.spot, s.journey, s.steps, s.ref, s.version, list(s.files), s.hint, dict(s.prefetch)) == before
    assert s.busy is False
    assert svc.next(s.id, 120).heading == pytest.approx(120) and s.steps == 1


def test_oom_on_start(svc, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    s = svc.new_session()
    svc.painter.fail_next = True
    with pytest.raises(GpuUnavailable):
        svc.start(s.id, "forest", [])
    assert s.spot is None and s.busy is False and not list(svc.scene_dir.iterdir())
    with pytest.raises(NotStarted):
        svc.next(s.id, 0)


def test_overlapping_next_is_busy(make):
    svc = make(painter=GatedPainter())
    s = started(svc)
    wait_set(s)
    svc.painter.armed = True
    out, errors = [], []

    def move():
        try:
            out.append(svc.next(s.id, 120))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=move)
    t.start()
    assert svc.painter.entered.wait(5)
    with pytest.raises(Busy):
        svc.next(s.id, 0)                         # even with a finished spot at 0
    with pytest.raises(Busy):
        svc.start(s.id, "desert", [])
    svc.heading(s.id, 200)                        # a hint is fine, and builds nothing yet
    svc.painter.armed = False
    svc.painter.gate.set()
    t.join(5)
    assert errors == [] and len(out) == 1 and s.steps == 1 and s.busy is False
    assert sorted(key_heading(k) for k in s.prefetch) == [140, 200, 260]   # the hint given mid-move wins


def test_files_kept_for_the_last_three_spots(make):
    svc = make(queue=ManualQueue())
    s = started(svc)
    views = [svc.next(s.id, 90) for _ in range(4)]
    want = sorted(p for v in views[1:] for p in (v.image_path, v.depth_path))
    assert sorted(svc.scene_dir.iterdir()) == want
    assert views[-1].image_path.name == f"{views[-1].spot_id}.jpg"
    assert views[-1].depth_path.name == f"{views[-1].spot_id}.png"


def test_eviction_cancels_jobs_and_deletes_files(make):
    now = [0.0]
    svc = make(clock=lambda: now[0])
    a = started(svc)
    wait_set(a)
    release = block_worker(svc)
    try:
        svc.heading(a.id, 180)                    # 180, 240, 120: all queued behind the blocker
        queued = list(set_futures(a).values())
        assert len(queued) == 3
        now[0] = SESSION_TTL + 1
        b = svc.new_session()                     # evicts a
        assert all(f.cancelled() for f in queued)
        assert list(svc.scene_dir.iterdir()) == []
        with pytest.raises(UnknownSession):
            svc.next(a.id, 0)
        assert svc.get(b.id) is b
    finally:
        release.set()


def test_lru_eviction_caps_sessions(make, monkeypatch):
    import vecworld.roam.service as service

    monkeypatch.setattr(service, "MAX_SESSIONS", 2)
    now = [0.0]
    svc = make(clock=lambda: now[0], queue=ManualQueue())
    a = started(svc)
    now[0] = 1
    b = svc.new_session()
    now[0] = 2
    svc.new_session()
    with pytest.raises(UnknownSession):
        svc.get(a.id)
    assert svc.get(b.id) is b
    assert list(svc.scene_dir.iterdir()) == []


def test_predreaming_never_touches_the_session(svc):
    s = started(svc)
    before = (s.spot, s.journey, dict(s.journey.drift), s.steps, s.ref, s.version)
    svc.heading(s.id, s.journey.neighbours[0].bearing)
    wait_set(s)
    assert (s.spot, s.journey, dict(s.journey.drift), s.steps, s.ref, s.version) == before
