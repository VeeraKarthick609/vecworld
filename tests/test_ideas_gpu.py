import time

import pytest

from vecworld.dream.ideas import QwenDreamer


@pytest.mark.gpu
def test_qwen_dreamer_gives_themed_phrases():
    dreamer = QwenDreamer()
    t0 = time.time()
    ideas = dreamer.ideas("haunted forest", ["foggy"], 6)
    print(f"\nideas={ideas} cpu_time={time.time() - t0:.1f}s")
    assert len(ideas) == 6 and all(2 <= len(i.split()) <= 8 for i in ideas)
