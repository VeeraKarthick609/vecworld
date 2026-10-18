import threading

import numpy as np
import pytest
from conftest import FakeDreamer, FakePainter
from PIL import Image

from vecworld.config import (DRIFT_STEPS, KEEP_OFFSET, MOODS, MAX_THEME_CHARS, PANO_H, PANO_W, SESSION_TTL,
                             STYLE)
from vecworld.dream.errors import BadInput, Busy, GpuUnavailable, NotStarted, UnknownLabel, UnknownSession
from vecworld.dream.ideas import FALLBACK_IDEAS
from vecworld.dream.labels import Label
from vecworld.dream.pano import dolly, roll_to, yaw_to_u
from vecworld.dream.service import DreamService, View
from vecworld.scenes.palette import palette_stats

KEEPS = (pytest.approx(0.5 - KEEP_OFFSET), pytest.approx(0.5 + KEEP_OFFSET))


class FakeLabels:
    """Finds everything it is asked for, where it was asked for, unless the text is in `missing`."""

    def __init__(self):
        self.calls: list[list[tuple[str, float]]] = []
        self.missing: set[str] = set()
        self._n = 0

    def find(self, img, wanted):
        self.calls.append(list(wanted))
        out = []
        for text, u in wanted:
            if text not in self.missing:
                self._n += 1
                out.append(Label(f"L{self._n}", text, u, 0.5, 0.05, 0.2))
        return out


class TagPainter(FakePainter):
    """A horizontal-gradient panorama (so rolls are visible); step n returns a solid grey of n * 40."""

    def panorama(self, base, seed):
        super().panorama(base, seed)
        w, h = PANO_W // 16, PANO_H // 16
        a = np.zeros((h, w, 3), np.uint8)
        a[..., 0] = (np.arange(w) * 255 // (w - 1))[None, :]
        a[..., 1] = 255 - a[..., 0]
        return Image.fromarray(a)

    def step(self, prev, base, focus, focus_at, seed):
        super().step(prev, base, focus, focus_at, seed)
        n = len(steps_of(self))
        return Image.new("RGB", (PANO_W // 16, PANO_H // 16), (n * 40,) * 3)


class GatedPainter(FakePainter):
    """step() blocks until `gate` is set, after signalling `entered`."""

    def __init__(self):
        super().__init__()
        self.entered, self.gate = threading.Event(), threading.Event()

    def step(self, prev, base, focus, focus_at, seed):
        self.entered.set()
        assert self.gate.wait(5)
        return super().step(prev, base, focus, focus_at, seed)


def run_now(fn):
    fn()


def steps_of(painter):
    return [kw for name, kw in painter.calls if name == "step"]


@pytest.fixture
def svc_factory(tmp_path):
    def make(painter=None, dreamer=None, labels=None, spawn=run_now, **kw):
        return DreamService(painter or FakePainter(), dreamer or FakeDreamer(), labels or FakeLabels(),
                            tmp_path / "scenes", spawn=spawn, **kw)
    return make


@pytest.fixture
def svc(svc_factory):
    return svc_factory()


def started(svc, theme="a quiet forest", moods=("foggy",)):
    s = svc.new_session()
    svc.start(s.id, theme, list(moods))
    return s


def block_worker(svc):
    """Occupy the GPU queue's worker until the returned event is set."""
    running, release = threading.Event(), threading.Event()
    svc.queue.submit(lambda: (running.set(), release.wait(5)), owner="blocker")
    assert running.wait(5)
    return release


def arr(img):
    return np.asarray(img.convert("RGB"))


# --- start ---------------------------------------------------------------------

def test_start_makes_panorama_and_refills_ideas(svc):
    s = svc.new_session()
    v = svc.start(s.id, "  a quiet forest ", ["foggy"])
    assert isinstance(v, View)
    assert svc.painter.calls[0][0] == "panorama"
    assert svc.painter.calls[0][1]["base"] == [(f"a quiet forest, foggy, {STYLE}", 1.0)]
    assert svc.dreamer.calls == [("a quiet forest", ["foggy"], 8)]
    assert list(s.ideas) == FakeDreamer().ideas("", [], 8)
    assert v.labels == [] and v.steps == 0 and v.theme == "a quiet forest" and v.moods == ["foggy"]
    assert v.image_path.exists()
    assert s.ref is not None and s.pano is not None


def test_start_without_moods_has_no_empty_prompt_part(svc):
    s = svc.new_session()
    svc.start(s.id, "a quiet forest", [])
    assert svc.painter.calls[0][1]["base"] == [(f"a quiet forest, {STYLE}", 1.0)]


def test_theme_is_truncated_and_unknown_moods_dropped(svc):
    s = svc.new_session()
    v = svc.start(s.id, "x" * (MAX_THEME_CHARS + 50), ["night", "disco", "night", "rain"])
    assert v.theme == "x" * MAX_THEME_CHARS
    assert v.moods == ["night", "rain"]
    assert set(v.moods) <= set(MOODS)


def test_empty_theme_rejected(svc):
    s = svc.new_session()
    with pytest.raises(BadInput):
        svc.start(s.id, "   ", ["foggy"])
    s2 = started(svc)
    with pytest.raises(BadInput):
        svc.set_theme(s2.id, "")


def test_walk_before_start(svc):
    s = svc.new_session()
    for move in (lambda: svc.walk(s.id, 0), lambda: svc.back(s.id, 0), lambda: svc.approach(s.id, "x"),
                 lambda: svc.prefetch(s.id, 0)):
        with pytest.raises(NotStarted):
            move()


def test_unknown_session(svc):
    with pytest.raises(UnknownSession):
        svc.walk("nope", 0)


# --- walking -------------------------------------------------------------------

def test_walk_rolls_to_yaw_and_regenerates_ahead(svc_factory):
    svc = svc_factory(painter=TagPainter())
    s = started(svc)
    pano = s.pano
    v = svc.walk(s.id, 40)
    (kw,) = steps_of(svc.painter)
    assert np.array_equal(arr(kw["prev"]), arr(dolly(roll_to(pano, yaw_to_u(40)))))
    assert kw["focus_at"] == 0.5
    base_text = f"a quiet forest, foggy, {STYLE}"
    assert kw["base"] == [(base_text, 1.0)]
    assert kw["focus"] == [(f"{base_text}, {FakeDreamer.IDEAS[0]} ahead in the distance", 1.0)]
    assert svc.labels.calls[-1] == [(FakeDreamer.IDEAS[0], 0.5)]
    assert [l.text for l in v.labels] == [FakeDreamer.IDEAS[0]]
    assert v.steps == 1 and s.steps == 1 and v.image_path.exists()
    assert list(s.ideas)[0] == FakeDreamer.IDEAS[1]


def test_every_third_step_is_quiet(svc):
    s = started(svc)
    for yaw in (90, 180, 270):   # away from any label, so nothing is kept
        svc.walk(s.id, yaw)
    calls = steps_of(svc.painter)
    assert [c["focus"] is not None for c in calls] == [True, True, False]
    assert "ahead" in calls[1]["focus"][0][0]
    assert calls[2]["focus_at"] == 0.5          # still regenerated, with the base prompt
    assert FakeDreamer.IDEAS[0] in calls[0]["focus"][0][0]
    assert FakeDreamer.IDEAS[1] in calls[1]["focus"][0][0]
    assert list(s.ideas)[0] == FakeDreamer.IDEAS[2]   # the quiet step used no idea


def test_walk_toward_label_keeps_it(svc):
    s = started(svc)
    svc.walk(s.id, 0)                           # discovery found at the centre of the new frame
    (label,) = s.labels
    svc.walk(s.id, 0)
    kw = steps_of(svc.painter)[-1]
    assert kw["focus_at"] in KEEPS
    assert (label.text, pytest.approx(0.5)) in svc.labels.calls[-1]
    # and the next kept step goes to the other side
    svc.walk(s.id, 0)
    assert steps_of(svc.painter)[-1]["focus_at"] in KEEPS
    assert steps_of(svc.painter)[-1]["focus_at"] != kw["focus_at"]


def test_walk_away_from_label_regenerates_centre(svc):
    s = started(svc)
    svc.walk(s.id, 0)
    svc.walk(s.id, 90)
    assert steps_of(svc.painter)[-1]["focus_at"] == 0.5


def test_approach_label_walks_to_it_with_keep(svc_factory):
    svc = svc_factory(painter=TagPainter())
    s = started(svc)
    pano = s.pano
    s.labels = [Label("x", "a bench", 0.7, 0.5, 0.05, 0.2)]
    svc.approach(s.id, "x")
    kw = steps_of(svc.painter)[-1]
    assert np.array_equal(arr(kw["prev"]), arr(dolly(roll_to(pano, 0.7))))
    assert kw["focus_at"] in KEEPS
    assert ("a bench", pytest.approx(0.5)) in svc.labels.calls[-1]


def test_approach_unknown_label(svc):
    s = started(svc)
    with pytest.raises(UnknownLabel):
        svc.approach(s.id, "nope")
    assert steps_of(svc.painter) == []


def test_back_walks_the_other_way(svc_factory):
    svc = svc_factory(painter=TagPainter())
    s = started(svc)
    pano = s.pano
    svc.back(s.id, 30)
    (kw,) = steps_of(svc.painter)
    assert np.array_equal(arr(kw["prev"]), arr(dolly(roll_to(pano, yaw_to_u(210)))))
    assert not np.array_equal(arr(kw["prev"]), arr(dolly(roll_to(pano, yaw_to_u(30)))))


def test_bad_yaw_rejected(svc):
    s = started(svc)
    with pytest.raises(BadInput):
        svc.walk(s.id, float("nan"))


# --- ideas ---------------------------------------------------------------------

def test_ideas_refill_below_three(svc):
    s = started(svc)
    for i in range(7):
        svc.walk(s.id, 90 * (i % 2) + 90)       # never toward a label
    assert len(svc.dreamer.calls) == 1 and len(s.ideas) == 3
    svc.walk(s.id, 90)
    assert len(svc.dreamer.calls) == 2 and svc.dreamer.calls[1][2] == 8
    assert len(s.ideas) == 10


def test_steps_use_fallback_while_first_refill_runs(svc_factory):
    pending = []
    svc = svc_factory(spawn=pending.append)
    s = started(svc)
    assert svc.dreamer.calls == [] and len(pending) == 1
    svc.walk(s.id, 90)
    assert FALLBACK_IDEAS[0] in steps_of(svc.painter)[0]["focus"][0][0]
    svc.walk(s.id, 180)
    assert len(pending) == 1                    # one refill in flight at a time
    pending.pop()()
    assert len(s.ideas) == 8
    svc.walk(s.id, 270)                         # quiet
    svc.walk(s.id, 90)
    assert FakeDreamer.IDEAS[0] in steps_of(svc.painter)[-1]["focus"][0][0]


def test_refill_failure_falls_back(svc_factory):
    class Broken:
        calls = 0

        def ideas(self, theme, moods, n):
            Broken.calls += 1
            raise RuntimeError("model fell over")

    svc = svc_factory(dreamer=Broken())
    s = started(svc)
    assert Broken.calls == 1
    assert len(s.ideas) == 8 and set(s.ideas) <= set(FALLBACK_IDEAS)
    assert s.refilling is None


def test_refill_runs_off_the_calling_thread(svc_factory):
    done, seen = threading.Event(), []

    class Recording(FakeDreamer):
        def ideas(self, theme, moods, n):
            seen.append(threading.current_thread())
            done.set()
            return super().ideas(theme, moods, n)

    svc = svc_factory(dreamer=Recording(), spawn=None)
    started(svc)
    assert done.wait(5)
    assert seen[0] is not threading.current_thread() and seen[0].daemon


def test_theme_change_replaces_ideas(svc_factory):
    pending = []
    svc = svc_factory(spawn=pending.append)
    s = started(svc)
    pending.pop()()
    svc.set_theme(s.id, "frozen lake")
    assert len(pending) == 1
    assert list(s.ideas)[0] == FakeDreamer.IDEAS[0]    # old ideas serve until the new ones arrive
    pending.pop()()
    assert svc.dreamer.calls[-1] == ("frozen lake", ["foggy"], 8)
    assert len(s.ideas) == 8


# --- prefetch ------------------------------------------------------------------

def test_walk_uses_matching_prefetch_without_new_paint(svc):
    s = started(svc)
    svc.prefetch(s.id, 10)
    v = svc.walk(s.id, 30)
    assert len(steps_of(svc.painter)) == 1
    assert v.steps == 1 and s.prefetch is None


def test_prefetch_does_not_touch_session_until_used(svc):
    s = started(svc)
    pano, ideas = s.pano, list(s.ideas)
    svc.prefetch(s.id, 10)
    s.prefetch[2].result(timeout=5)
    assert s.pano is pano and s.steps == 0 and s.labels == [] and list(s.ideas) == ideas


def test_prefetch_within_tolerance_is_not_repeated(svc):
    s = started(svc)
    svc.prefetch(s.id, 10)
    first = s.prefetch[2]
    svc.prefetch(s.id, 40)
    assert s.prefetch[2] is first
    svc.prefetch(s.id, 100)
    assert s.prefetch[2] is not first


def test_walk_ignores_prefetch_for_another_direction(svc_factory, monkeypatch):
    import vecworld.dream.service as service

    monkeypatch.setattr(service, "PALETTE_STRENGTH", 0.0)   # so the saved image is the painter's output
    svc = svc_factory(painter=TagPainter())
    s = started(svc)
    pano = s.pano
    svc.prefetch(s.id, 10)
    s.prefetch[2].result(timeout=5)
    v = svc.walk(s.id, 100)
    calls = steps_of(svc.painter)
    assert len(calls) == 2
    assert np.array_equal(arr(calls[1]["prev"]), arr(dolly(roll_to(pano, yaw_to_u(100)))))
    assert Image.open(v.image_path).getpixel((0, 0)) == (80, 80, 80)


def test_walk_after_cancelled_prefetch_builds_on_demand(svc):
    s = started(svc)
    release = block_worker(svc)
    svc.prefetch(s.id, 0)
    fut = s.prefetch[2]
    assert fut.cancel()
    release.set()
    v = svc.walk(s.id, 0)
    assert v.steps == 1 and len(steps_of(svc.painter)) == 1


def test_failed_prefetch_is_dropped_and_walk_builds(svc, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    s = started(svc)
    svc.painter.fail_next = True
    svc.prefetch(s.id, 0)
    with pytest.raises(GpuUnavailable):
        s.prefetch[2].result(timeout=5)
    v = svc.walk(s.id, 0)
    assert v.steps == 1 and len(steps_of(svc.painter)) == 2


def test_mood_change_discards_prefetch(svc):
    s = started(svc)
    svc.prefetch(s.id, 0)
    s.prefetch[2].result(timeout=5)
    svc.set_moods(s.id, ["night", "disco"])
    assert s.prefetch is None and s.moods == ["night"]
    svc.walk(s.id, 0)
    calls = steps_of(svc.painter)
    assert len(calls) == 2
    assert "night" in calls[1]["base"][0][0] and "foggy" not in calls[1]["base"][0][0]


def test_prefetch_while_moving_is_skipped(svc_factory):
    svc = svc_factory(painter=GatedPainter())
    s = started(svc)
    t = threading.Thread(target=svc.walk, args=(s.id, 0))
    t.start()
    assert svc.painter.entered.wait(5)
    svc.prefetch(s.id, 90)                      # no error, and no build of a soon-stale spot
    assert s.prefetch is None
    svc.painter.gate.set()
    t.join(5)


# --- drift, moods, palette -------------------------------------------------------

def test_theme_drift_weights(svc):
    s = started(svc, theme="a quiet forest", moods=["foggy"])
    start_ref = s.ref
    svc.set_theme(s.id, "frozen lake")
    old, new = f"a quiet forest, foggy, {STYLE}", f"frozen lake, foggy, {STYLE}"
    for k in range(1, DRIFT_STEPS + 1):
        assert s.ref is start_ref
        svc.walk(s.id, 90)
        base = steps_of(svc.painter)[-1]["base"]
        assert [t for t, _ in base] == [old, new]
        assert [w for _, w in base] == [pytest.approx(1 - k / 3), pytest.approx(k / 3)]
    assert s.drift is None
    assert s.ref is not start_ref                # the colour reference resets once the theme takes over
    svc.walk(s.id, 90)
    assert steps_of(svc.painter)[-1]["base"] == [(new, 1.0)]


def test_set_theme_bumps_version_and_drops_prefetch(svc):
    s = started(svc)
    v0 = s.version
    svc.prefetch(s.id, 0)
    svc.set_theme(s.id, "frozen lake")
    assert s.version == v0 + 1 and s.prefetch is None and s.drift == ("a quiet forest", 0)


def test_mood_change_resets_palette_reference(svc):
    s = started(svc)
    start_ref = s.ref
    svc.walk(s.id, 90)
    assert s.ref is start_ref
    svc.set_moods(s.id, ["night"])
    svc.walk(s.id, 90)
    after = s.ref
    assert after is not start_ref
    np.testing.assert_allclose(after[0], palette_stats(s.pano)[0], atol=1e-3)
    svc.walk(s.id, 90)
    assert s.ref is after                        # only the first spot after the change


def test_theme_change_during_a_build_applies_from_the_next_step(svc_factory):
    svc = svc_factory(painter=GatedPainter())
    s = started(svc)
    svc.painter.gate.clear()
    t = threading.Thread(target=svc.walk, args=(s.id, 90))
    t.start()
    assert svc.painter.entered.wait(5)
    svc.set_theme(s.id, "frozen lake")
    svc.painter.gate.set()
    t.join(5)
    assert s.steps == 1 and s.drift == ("a quiet forest", 0)   # that spot was painted before the change
    svc.walk(s.id, 90)
    assert steps_of(svc.painter)[-1]["base"][1][1] == pytest.approx(1 / 3)


# --- failures and concurrency ------------------------------------------------------

def test_oom_keeps_previous_view(svc, monkeypatch):
    import torch

    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    s = started(svc)
    svc.walk(s.id, 0)
    pano, labels, ideas, ref = s.pano, list(s.labels), list(s.ideas), s.ref
    svc.painter.fail_next = True
    with pytest.raises(GpuUnavailable):
        svc.walk(s.id, 0)
    assert emptied
    assert s.pano is pano and s.labels == labels and s.steps == 1 and list(s.ideas) == ideas and s.ref is ref
    assert s.busy is False
    assert svc.walk(s.id, 0).steps == 2


def test_oom_on_start(svc, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    s = svc.new_session()
    svc.painter.fail_next = True
    with pytest.raises(GpuUnavailable):
        svc.start(s.id, "a quiet forest", [])
    assert s.pano is None and s.busy is False
    with pytest.raises(NotStarted):
        svc.walk(s.id, 0)


def test_overlapping_moves_get_busy(svc_factory):
    svc = svc_factory(painter=GatedPainter())
    s = started(svc)
    results, errors = [], []

    def move():
        try:
            results.append(svc.walk(s.id, 0))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=move)
    t.start()
    assert svc.painter.entered.wait(5)
    with pytest.raises(Busy):
        svc.walk(s.id, 90)
    with pytest.raises(Busy):
        svc.back(s.id, 0)
    svc.painter.gate.set()
    t.join(5)
    assert errors == [] and len(results) == 1 and isinstance(results[0], View)
    assert len(steps_of(svc.painter)) == 1 and s.busy is False


def test_evicted_session_prefetch_is_harmless(svc_factory):
    now = [0.0]
    svc = svc_factory(clock=lambda: now[0])
    a = started(svc)
    release = block_worker(svc)
    svc.prefetch(a.id, 0)
    fut = a.prefetch[2]
    now[0] = SESSION_TTL + 1
    b = svc.new_session()                        # evicts the idle session and cancels its jobs
    assert fut.cancelled()
    with pytest.raises(UnknownSession):
        svc.walk(a.id, 0)
    release.set()
    svc.start(b.id, "desert", [])
    assert svc.walk(b.id, 0).steps == 1
    assert len(steps_of(svc.painter)) == 1       # the evicted prefetch never painted


def test_lru_eviction_caps_sessions(svc_factory, monkeypatch):
    import vecworld.dream.service as service

    monkeypatch.setattr(service, "MAX_SESSIONS", 2)
    now = [0.0]
    svc = svc_factory(clock=lambda: now[0])
    a = svc.new_session()
    now[0] = 1
    b = svc.new_session()
    now[0] = 2
    svc.new_session()
    with pytest.raises(UnknownSession):
        svc.get(a.id)
    assert svc.get(b.id) is b
