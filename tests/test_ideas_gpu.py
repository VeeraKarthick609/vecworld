import time

import pytest

from vecworld.dream.ideas import FALLBACK_IDEAS, QwenDreamer, is_visual


@pytest.mark.gpu
def test_qwen_dreamer_gives_themed_phrases():
    dreamer = QwenDreamer()
    t0 = time.time()
    ideas = dreamer.ideas("haunted forest", ["foggy"], 8)
    print(f"\nideas={ideas} cpu_time={time.time() - t0:.1f}s")
    assert len(ideas) == 8 and all(2 <= len(i.split()) <= 8 for i in ideas)
    assert any(i not in FALLBACK_IDEAS for i in ideas)
    assert all(is_visual(i) for i in ideas)


@pytest.mark.gpu
def test_qwen_dreamer_gives_places():
    from vecworld.dream.ideas import QwenDreamer as Q

    dreamer = Q()
    t0 = time.time()
    places = dreamer.places("haunted forest", ["old village"], ["foggy"], 3)
    print(f"\nplaces={places} cpu_time={time.time() - t0:.1f}s")
    assert len(places) == 3 and len({n for n, _ in places}) == 3
    for name, scenery in places:
        assert 1 <= len(name.split()) <= 6 and len(scenery) == 3
        assert all(2 <= len(p.split()) <= 8 for p in scenery)
