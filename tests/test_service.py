import numpy as np
import pytest

from conftest import OomGenerator
from vecworld.config import MAX_TEXT_CHARS, N_NEIGHBOURS
from vecworld.game.service import BadInput, Busy, GpuUnavailable, NotStarted, UnknownConcept


def test_jump_lands_and_lists_clickables(svc):
    s = svc.new_session()
    sc = svc.jump(s.id, "ocean")
    assert [h.concept for h in sc.hotspots][0] == "ocean"
    assert 2 <= len(sc.hotspots) <= 1 + N_NEIGHBOURS
    assert sc.image_path.exists() and len(sc.trail) == 1


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
