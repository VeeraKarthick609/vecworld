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
