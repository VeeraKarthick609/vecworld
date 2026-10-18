import time

import pytest

from vecworld.scenes.prompt import scene_prompt


@pytest.mark.gpu
def test_clipseg_finds_concepts_in_generated_scene():
    from vecworld.scenes.generator import TurboGenerator
    from vecworld.scenes.hotspots import ClipSegFinder

    img = TurboGenerator().fresh(scene_prompt("ship", ["sun"]), seed=3)
    finder = ClipSegFinder()
    finder.find(img, ["warm", "up"])
    t = time.time()
    hs = finder.find(img, ["ship", "sun", "waves", "cloud"])
    print(f"\nclipseg 4 concepts {time.time() - t:.2f}s: {hs}")
    assert [h.concept for h in hs] == ["ship", "sun", "waves", "cloud"]
    for h in hs:
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 and h.y + h.h <= 1
