import time

import pytest

from vecworld.scenes.prompt import scene_prompt


@pytest.mark.gpu
def test_clipseg_finds_concepts_in_generated_scene():
    import torch

    from vecworld.scenes.generator import TurboGenerator
    from vecworld.scenes.hotspots import ClipSegFinder

    img = TurboGenerator().fresh(scene_prompt("ship", ["sun"], True), seed=3)
    finder = ClipSegFinder()
    concepts = ["ship", "sun", "waves", "cloud", "tiger", "piano", "castle"]
    finder.find(img, ["warm", "up"])
    t = time.time()
    hs = finder.find(img, concepts)
    took = time.time() - t
    print(f"\nclipseg {len(concepts)} concepts {took:.2f}s: {hs}")
    assert len(hs) == len(concepts) and hs[0] is not None and hs[0].concept == "ship"
    for h in filter(None, hs):
        assert 0 <= h.x and 0 <= h.y and h.x + h.w <= 1 and h.y + h.h <= 1
    assert took < 0.6

    # One shared vision pass must give the same heatmaps as the stock per-prompt forward.
    inputs = finder.processor(text=concepts, images=[img] * len(concepts), padding=True, return_tensors="pt")
    with torch.inference_mode():
        stock = torch.sigmoid(finder.model(**inputs).logits).numpy()
    assert abs(finder.heatmaps(img, concepts) - stock).max() < 1e-4
