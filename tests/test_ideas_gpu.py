import time

import pytest


@pytest.mark.gpu
def test_qwen_dreamer_gives_places():
    from vecworld.dream.ideas import QwenDreamer

    dreamer = QwenDreamer()
    t0 = time.time()
    places = dreamer.places("haunted forest", ["old village"], ["foggy"], 3)
    print(f"\nplaces={places} cpu_time={time.time() - t0:.1f}s")
    assert len(places) == 3 and len({n for n, _ in places}) == 3
    for name, scenery in places:
        assert 1 <= len(name.split()) <= 6 and len(scenery) == 3
        assert all(2 <= len(p.split()) <= 8 for p in scenery)
