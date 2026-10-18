import time

import pytest

from vecworld.config import STYLE


@pytest.mark.gpu
def test_clipseg_finds_concepts_in_generated_scene():
    import torch
    from diffusers import AutoPipelineForText2Image

    from vecworld.scenes.hotspots import ClipSegFinder

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    img = pipe(f"a ship on the waves under a warm sun, {STYLE}", num_inference_steps=1, guidance_scale=0.0,
               generator=torch.Generator("cuda").manual_seed(3)).images[0]
    del pipe
    torch.cuda.empty_cache()

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
