from typing import Protocol

from PIL import Image

from vecworld.config import STEP_STEPS, STEP_STRENGTH

SIZE = 512


class SceneGenerator(Protocol):
    def fresh(self, prompt: str, seed: int) -> Image.Image: ...

    def evolve(self, prev: Image.Image, prompt: str, seed: int) -> Image.Image: ...


class TurboGenerator:
    """SD-Turbo on the GPU: one-step txt2img for jumps, two-step img2img for steps."""

    def __init__(self, model_id: str = "stabilityai/sd-turbo"):
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        self._torch = torch
        self.txt2img = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self.img2img = AutoPipelineForImage2Image.from_pipe(self.txt2img)
        for pipe in (self.txt2img, self.img2img):
            pipe.enable_attention_slicing()
            pipe.set_progress_bar_config(disable=True)

    def _gen(self, seed: int):
        return self._torch.Generator("cuda").manual_seed(seed)

    def fresh(self, prompt: str, seed: int) -> Image.Image:
        return self.txt2img(
            prompt, num_inference_steps=1, guidance_scale=0.0,
            width=SIZE, height=SIZE, generator=self._gen(seed),
        ).images[0]

    def evolve(self, prev: Image.Image, prompt: str, seed: int) -> Image.Image:
        return self.img2img(
            prompt, image=prev.resize((SIZE, SIZE)), strength=STEP_STRENGTH,
            num_inference_steps=STEP_STEPS, guidance_scale=0.0, generator=self._gen(seed),
        ).images[0]
