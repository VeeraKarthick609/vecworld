import numpy as np
import pytest

from conftest import OomGenerator
from vecworld.config import MAX_TEXT_CHARS, N_NEIGHBOURS
from vecworld.game.service import BadInput, Busy, GpuUnavailable, NoHistory, NotStarted, UnknownConcept


def test_jump_lands_and_lists_clickables(svc):
    s = svc.new_session()
    sc = svc.jump(s.id, "ocean")
    assert [h.concept for h in sc.hotspots][0] == "ocean"
    assert 2 <= len(sc.hotspots) <= 1 + N_NEIGHBOURS
    assert sc.image_path.exists() and len(sc.trail) == 1


class HidingFinder:
    """Locates every concept except the hidden ones."""

    def __init__(self, hidden):
        self.hidden = set(hidden)
        self.asked: list[list[str]] = []

    def find(self, img, concepts):
        from vecworld.scenes.hotspots import Hotspot

        self.asked.append(list(concepts))
        n = len(concepts)
        return [None if c in self.hidden else Hotspot(c, i / n, 0.4, 1 / n, 0.2) for i, c in enumerate(concepts)]


def test_only_located_neighbours_are_offered(svc):
    from vecworld.config import CENTRE_BOX, N_CANDIDATES

    svc.finder = HidingFinder({"ocean", "sea"})
    s = svc.new_session()
    sc = svc.jump(s.id, "ocean")
    offered = [h.concept for h in sc.hotspots]
    assert offered[0] == "ocean" and "sea" not in offered and "ship" in offered
    c = sc.hotspots[0]
    assert (c.x, c.y, c.w, c.h) == CENTRE_BOX          # unlocated centre still pinned, in the middle
    assert len(svc.finder.asked[-1]) == 1 + min(N_CANDIDATES, len(svc.space.words) - 1)


def test_abstract_concepts_get_a_symbolic_scene(svc_factory):
    svc = svc_factory(concrete=lambda w: w != "desert")
    s = svc.new_session()
    svc.jump(s.id, "desert")
    assert svc.generator.calls[-1][1].startswith("a symbolic scene evoking desert")
    svc.jump(s.id, "ocean")
    assert svc.generator.calls[-1][1].startswith("a scene of an ocean")


def test_step_pins_clicked_concept_even_when_capped(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    target = [h.concept for h in svc.get(s.id).hotspots][-1]
    sc = svc.step(s.id, target)
    assert sc.hotspots[0].concept == target
    assert svc.generator.calls[-1][0] == "evolve" and len(sc.trail) == 2


def test_step_moves_no_further_than_cap(svc):
    from vecworld.config import MAX_STEP_ANGLE
    from vecworld.space.index import angle

    s = svc.new_session()
    svc.jump(s.id, "ocean")
    before = svc.get(s.id).position.copy()
    target = [h.concept for h in svc.get(s.id).hotspots][-1]
    svc.step(s.id, target)
    assert angle(before, svc.get(s.id).position) <= MAX_STEP_ANGLE + 1e-5


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_jump_rejected(svc, text):
    with pytest.raises(BadInput):
        svc.jump(svc.new_session().id, text)


def test_huge_and_emoji_jump_still_lands(svc):
    s = svc.new_session()
    assert svc.jump(s.id, "🌊" * 5000).hotspots
    assert len(svc.embedder.last_texts[0]) <= MAX_TEXT_CHARS


def test_step_before_jump(svc):
    with pytest.raises(NotStarted):
        svc.step(svc.new_session().id, "ship")


def test_step_unknown_concept(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    with pytest.raises(UnknownConcept):
        svc.step(s.id, "keyboard")


def test_busy_session_rejected(svc):
    s = svc.new_session()
    s.busy = True
    with pytest.raises(Busy):
        svc.jump(s.id, "ocean")


def test_unknown_session_is_keyerror(svc):
    with pytest.raises(KeyError):
        svc.jump("nope", "ocean")


def test_oom_keeps_previous_scene(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    before = (svc.get(s.id).position.copy(), list(svc.get(s.id).trail))
    svc.generator = OomGenerator()
    with pytest.raises(GpuUnavailable):
        svc.jump(s.id, "desert")
    assert np.allclose(svc.get(s.id).position, before[0]) and svc.get(s.id).trail == before[1]
    assert svc.get(s.id).busy is False


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_idle_sessions_expire(svc_factory):
    from vecworld.config import SESSION_TTL
    from vecworld.game.service import UnknownSession

    clock = Clock()
    svc = svc_factory(clock=clock)
    idle, active = svc.new_session(), svc.new_session()
    clock.t = SESSION_TTL - 10
    svc.jump(active.id, "ocean")
    clock.t = SESSION_TTL + 1
    svc.new_session()
    with pytest.raises(UnknownSession):
        svc.get(idle.id)
    assert svc.get(active.id).position is not None


def test_session_cap_evicts_least_recently_used(svc_factory):
    from vecworld.config import MAX_SESSIONS
    from vecworld.game.service import UnknownSession

    clock = Clock()
    svc = svc_factory(clock=clock)
    first = svc.new_session()
    others = []
    for i in range(MAX_SESSIONS - 1):
        clock.t += 1
        others.append(svc.new_session())
    clock.t += 1
    svc.jump(first.id, "ocean")          # first is now the most recently used
    clock.t += 1
    svc.new_session()
    svc.get(first.id)
    with pytest.raises(UnknownSession):
        svc.get(others[0].id)
    assert len(svc._sessions) == MAX_SESSIONS


def offered(sc):
    return [h.concept for h in sc.hotspots]


def test_previous_centres_are_not_offered_again(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    sc = svc.step(s.id, "sea")
    assert "ocean" not in offered(sc)          # build 1 ping-ponged ocean <-> sea here
    sc = svc.step(s.id, "ship")
    assert not {"ocean", "sea"} & set(offered(sc)[1:])
    assert svc.get(s.id).recent == ["ocean", "sea", "ship"]


def test_recent_is_bounded(svc):
    from vecworld.config import RECENT

    s = svc.new_session()
    svc.jump(s.id, "ocean")
    for _ in range(RECENT + 3):
        nxt = (svc.get(s.id).hotspots[1:] or svc.get(s.id).hotspots)[0].concept
        svc.step(s.id, nxt)
    assert len(svc.get(s.id).recent) == RECENT


def test_jump_resets_recent(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    svc.step(s.id, "sea")
    sc = svc.jump(s.id, "sea")
    assert svc.get(s.id).recent == ["sea"] and "ocean" in offered(sc)


# --- forward (W) and back (S) -------------------------------------------------

from vecworld.config import MAX_STEP_ANGLE  # noqa: E402
from vecworld.space.index import angle  # noqa: E402


def test_forward_keeps_walking_the_way_you_were_heading(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    start = svc.get(s.id).position.copy()
    svc.step(s.id, "ship")
    mid = svc.get(s.id).position.copy()
    sc = svc.forward(s.id)
    end = svc.get(s.id).position
    assert abs(angle(mid, end) - MAX_STEP_ANGLE) < 1e-4
    assert angle(start, end) > angle(start, mid) + 0.3      # onward, not sideways or back
    assert sc.centre == sc.hotspots[0].concept
    assert sc.centre == svc.space.neighbours(end, k=1, exclude={"ocean", "ship"})[0][0]
    assert svc.generator.calls[-1][0] == "evolve" and len(sc.trail) == 3


def test_forward_right_after_a_jump_heads_for_the_nearest_unvisited_concept(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    before = svc.get(s.id).position.copy()
    svc.forward(s.id)
    after = svc.get(s.id).position
    sea = svc.space.vector("sea")
    assert angle(after, sea) < angle(before, sea)
    assert angle(before, after) <= MAX_STEP_ANGLE + 1e-5
    assert svc.get(s.id).heading is not None


def test_forward_errors(svc):
    with pytest.raises(NotStarted):
        svc.forward(svc.new_session().id)
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    s.busy = True
    with pytest.raises(Busy):
        svc.forward(s.id)


def test_forward_oom_keeps_state(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    svc.step(s.id, "ship")
    before = (svc.get(s.id).position.copy(), svc.get(s.id).heading.copy(), list(svc.get(s.id).recent))
    svc.generator = OomGenerator()
    with pytest.raises(GpuUnavailable):
        svc.forward(s.id)
    now = svc.get(s.id)
    assert np.allclose(now.position, before[0]) and np.allclose(now.heading, before[1]) and now.recent == before[2]
    assert not now.busy


def snapshot(sess):
    return (sess.position.copy(), None if sess.heading is None else sess.heading.copy(), list(sess.recent),
            list(sess.hotspots), list(sess.trail), sess.scene.scene_id, sess.scene.centre)


def same(a, b):
    return (np.allclose(a[0], b[0]) and ((a[1] is None and b[1] is None) or np.allclose(a[1], b[1]))
            and a[2:] == b[2:])


def test_back_restores_the_previous_scene_without_the_gpu(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    svc.step(s.id, "ship")
    there = snapshot(svc.get(s.id))
    svc.forward(s.id)
    calls = len(svc.generator.calls)
    sc = svc.back(s.id)
    assert len(svc.generator.calls) == calls
    assert same(snapshot(svc.get(s.id)), there) and sc.scene_id == there[5]


def test_back_twice_returns_to_the_jump_then_runs_out(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")
    landed = snapshot(svc.get(s.id))
    svc.step(s.id, "ship")
    svc.back(s.id)
    assert same(snapshot(svc.get(s.id)), landed) and svc.get(s.id).heading is None
    with pytest.raises(NoHistory):
        svc.back(s.id)
    with pytest.raises(NotStarted):
        svc.back(svc.new_session().id)


def test_history_is_bounded(svc):
    from vecworld.config import HISTORY

    s = svc.new_session()
    svc.jump(s.id, "ocean")
    for _ in range(HISTORY + 5):
        svc.forward(s.id)
    for _ in range(HISTORY):
        svc.back(s.id)
    with pytest.raises(NoHistory):
        svc.back(s.id)


def test_after_back_the_next_step_grows_from_the_restored_scene(svc):
    s = svc.new_session()
    svc.jump(s.id, "ocean")              # navy scene
    svc.step(s.id, "ship")               # teal scene
    svc.back(s.id)
    svc.step(s.id, "sea")
    assert svc.generator.sources[-1] == (0, 0, 128)
