"""Wrap-around MultiDiffusion on SD-Turbo: seamless 360° panoramas and region-focused walk steps."""
import os
from typing import Protocol

import numpy as np
from PIL import Image

from vecworld.config import BG_STRENGTH, PANO_H, PANO_W, STEP_STEPS, VAE_MARGIN, VAE_TILE, WIN
from vecworld.dream.pano import focus_mask, windows, wrap_cols

Prompt = list[tuple[str, float]]   # (text, weight); weights sum to 1
BATCH = 4                          # UNet windows per forward pass


class Painter(Protocol):
    def panorama(self, base: Prompt, seed: int) -> Image.Image: ...

    def step(self, prev: Image.Image, base: Prompt, focus: Prompt | None, focus_at: float | None,
             seed: int) -> Image.Image: ...


class TurboPainter(Painter):
    """SD-Turbo (fp16, default SDPA attention) denoising a PANO_W x PANO_H cylinder in wrapping windows.

    Attention slicing is deliberately off: it ran out of memory at this size. SD-Turbo has no CFG,
    so there is no negative prompt.
    """

    def __init__(self, model_id: str = "stabilityai/sd-turbo"):
        # The VAE decoder wants ~340 MB blocks that a fragmented 4 GB cache can't give without
        # failing and retrying (allocator OOM warnings on every step). Expandable segments avoid it;
        # it only applies if nothing has touched CUDA yet.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from diffusers import AutoPipelineForText2Image

        self._torch = torch
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        self._pipe.set_progress_bar_config(disable=True)
        self.unet, self.vae, self.sched = self._pipe.unet, self._pipe.vae, self._pipe.scheduler
        self._sf = self.vae.config.scaling_factor
        self._embs: dict = {}
        self._lat_shape = (1, 4, PANO_H // 8, PANO_W // 8)

    # --- text ----------------------------------------------------------------

    def _text(self, text: str):
        if text not in self._embs:
            self._embs[text] = self._pipe.encode_prompt(text, "cuda", 1, False)[0]
        return self._embs[text]

    def _embed(self, prompt: Prompt):
        return sum(w * self._text(t) for t, w in prompt)

    # --- VAE, tiled along the wrapping width -------------------------------------

    def _decode(self, lat) -> Image.Image:
        w, outs = lat.shape[-1], []
        for x0 in range(0, w, VAE_TILE):
            cols = wrap_cols(x0 - VAE_MARGIN, VAE_TILE + 2 * VAE_MARGIN, w)
            px = self.vae.decode(lat[..., cols] / self._sf).sample
            outs.append(px[..., VAE_MARGIN * 8:(VAE_MARGIN + VAE_TILE) * 8])
        img = self._torch.cat(outs, -1)[0].float().clamp(-1, 1)
        return Image.fromarray(((img + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())

    def _encode(self, im: Image.Image):
        torch = self._torch
        a = np.asarray(im.convert("RGB").resize((PANO_W, PANO_H)))
        x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].cuda().half() / 127.5 - 1
        w, outs, tile, margin = x.shape[-1], [], VAE_TILE * 8, VAE_MARGIN * 8
        for x0 in range(0, w, tile):
            cols = wrap_cols(x0 - margin, tile + 2 * margin, w)
            outs.append(self.vae.encode(x[..., cols]).latent_dist.mean[..., VAE_MARGIN:VAE_MARGIN + VAE_TILE])
        return torch.cat(outs, -1) * self._sf

    # --- MultiDiffusion -------------------------------------------------------------

    def _denoise(self, latents, timesteps, embs, keep=None):
        """Euler-denoise a wrap-around latent, averaging each step's noise prediction over windows.

        embs[i] is the text embedding for window i. keep = (orig, noise, mask, k_bg): up to and including
        step k_bg the unmasked part is reset to orig noised to the current timestep (differential
        diffusion), so it denoises only timesteps[k_bg:], exactly like img2img.
        """
        torch = self._torch
        h, w = latents.shape[-2:]
        wins = windows(h, w)
        for k, t in enumerate(timesteps):
            if keep is not None and k <= keep[3]:
                orig, noise, mask, _ = keep
                latents = mask * latents + (1 - mask) * self.sched.add_noise(orig, noise, t[None])
            lat_in = self.sched.scale_model_input(latents, t)
            eps, cnt = torch.zeros_like(latents), torch.zeros_like(latents[:, :1])
            for b in range(0, len(wins), BATCH):
                batch = wins[b:b + BATCH]
                cols = [wrap_cols(x, WIN, w) for _, x in batch]
                tiles = torch.cat([lat_in[..., y:y + WIN, :][..., c] for (y, _), c in zip(batch, cols)])
                out = self.unet(tiles, t, encoder_hidden_states=torch.cat(embs[b:b + BATCH])).sample
                for i, ((y, _), c) in enumerate(zip(batch, cols)):
                    eps[..., y:y + WIN, c] += out[i:i + 1]
                    cnt[..., y:y + WIN, c] += 1
            latents = self.sched.step(eps / cnt, t, latents).prev_sample
        return latents

    def _noise(self, seed: int):
        g = self._torch.Generator("cuda").manual_seed(seed)
        return self._torch.randn(self._lat_shape, generator=g, device="cuda", dtype=self._torch.float16)

    def panorama(self, base: Prompt, seed: int) -> Image.Image:
        with self._torch.inference_mode():
            self.sched.set_timesteps(1)
            self.sched.set_begin_index(0)
            n = len(windows(*self._lat_shape[-2:]))
            lat = self._noise(seed) * self.sched.init_noise_sigma
            return self._decode(self._denoise(lat, self.sched.timesteps, [self._embed(base)] * n))

    def step(self, prev: Image.Image, base: Prompt, focus: Prompt | None, focus_at: float | None,
             seed: int) -> Image.Image:
        torch = self._torch
        h, w = self._lat_shape[-2:]
        m = np.zeros(w, np.float32) if focus_at is None else focus_mask(w, focus_at)
        with torch.inference_mode():
            e_base = self._embed(base)
            e_focus = self._embed(focus) if focus is not None else e_base
            embs = [e_focus if m[(x + WIN // 2) % w] > 0 else e_base for _, x in windows(h, w)]
            self.sched.set_timesteps(STEP_STEPS)
            ts = self.sched.timesteps
            # The background denoises timesteps[k_bg:], as diffusers img2img does at BG_STRENGTH.
            # (Re-imposing only before k_bg left it 3 of 4 steps, i.e. ~0.75 strength, which repainted
            # the whole panorama.)
            k_bg = STEP_STEPS - int(STEP_STEPS * BG_STRENGTH)
            # With nothing to regenerate, every step before k_bg is overwritten by the re-imposed
            # background, so start there: plain img2img, fewer UNet passes.
            start = k_bg if not m.any() else 0
            self.sched.set_begin_index(start)
            noise, orig = self._noise(seed), self._encode(prev)
            mask = torch.from_numpy(m).to("cuda", torch.float16)[None, None, None, :]
            lat = self.sched.add_noise(orig, noise, ts[start:start + 1])
            out = self._denoise(lat, ts[start:], embs, keep=(orig, noise, mask, k_bg - start))
            return self._decode(out)
