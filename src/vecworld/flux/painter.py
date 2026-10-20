"""FLUX.1-schnell painter: wrap-around MultiDiffusion on FLUX's packed 2x2 latent tokens (the diffusers reference).

The approach is TurboPainter's: windows round the cylinder averaged per step, a walk step that repaints the band
ahead from scratch with the background at partial strength (differential diffusion), and a tiled VAE whose tiles
crossfade. The differences:

- Latents are 16 channels at 1/8 scale, packed 2x2 into 64-dim tokens. The loop works on a token grid
  [1, 64, PANO_H // 16, PANO_W // 16] (channel c*4 + dy*2 + dx, as diffusers packs them); a window is a
  FLUX_WIN x FLUX_WIN patch of it, flattened row by row into the transformer's sequence with its own local
  position ids, so every window looks like a 512 x 512 image to the model.
- Schnell is rectified flow: x_sigma = (1 - sigma) x0 + sigma noise, sigmas 1 -> 0 in FLUX_STEPS even steps, one
  Euler step per sigma on the window-averaged velocity. No guidance, no CFG.
- Text: T5-XXL sequence embeddings (vecworld.flux.text, GPU block by block, cached) plus CLIP-L pooled (GPU).
  A Prompt's texts blend as a weighted sum of both.
- The transformer is the 4-bit GGUF through diffusers, streamed block by block with diffusers' group offload.
"""
import math
import os
from collections import OrderedDict
from typing import Callable

import numpy as np
from PIL import Image

from vecworld.config import (FLUX_BASE, FLUX_BATCH, FLUX_BG_STRENGTH, FLUX_GGUF, FLUX_STEPS, FLUX_STRIDE, FLUX_STYLE,
                             FLUX_TEXT_LEN, FLUX_WIN, FOCUS_HALF, PANO_H, PANO_W, VAE_TILE)
from vecworld.dream.gpu import check_stop
from vecworld.dream.painter import Painter, Prompt, TurboPainter
from vecworld.dream.pano import focus_mask, windows, wrap_cols

POOLED_CACHE = 64     # CLIP pooled embeddings kept (least recently used dropped); T5 keeps its own


def _active(wins, cols, k: int, k_bg: int | None) -> list[int]:
    """The windows worth a pass at step k: before k_bg the unmasked part is reset to the noised original at the
    next step, so a window with no masked column would only be thrown away (as TurboPainter). cols: mask > 0."""
    if k_bg is None or k >= k_bg or cols is None:
        return list(range(len(wins)))
    return [i for i, (_, x) in enumerate(wins) if cols[wrap_cols(x, FLUX_WIN, len(cols))].any()]


class FluxPainter(Painter):
    """FLUX.1-schnell (Q4_K_S GGUF, bf16 compute) painting a PANO_W x PANO_H cylinder in wrapping windows.

    progress(done, total) is called after every transformer pass (one window batch at one step), total being
    the passes the job will make; check_stop() runs before every pass and every VAE tile.
    """

    style = FLUX_STYLE

    def __init__(self, progress: Callable[[int, int], None] | None = None):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from diffusers import AutoencoderKL, FluxTransformer2DModel, GGUFQuantizationConfig
        from transformers import CLIPTextModel, CLIPTokenizer

        from vecworld.flux.text import T5Text

        dev, bf16 = torch.device("cuda"), torch.bfloat16
        tr = FluxTransformer2DModel.from_single_file(
            str(FLUX_GGUF), quantization_config=GGUFQuantizationConfig(compute_dtype=bf16), torch_dtype=bf16,
            config=FLUX_BASE, subfolder="transformer")
        tr.enable_group_offload(onload_device=dev, offload_device=torch.device("cpu"), offload_type="block_level",
                                num_blocks_per_group=1, use_stream=True)
        vae = AutoencoderKL.from_pretrained(FLUX_BASE, subfolder="vae", torch_dtype=bf16).to(dev)
        self._clip = CLIPTextModel.from_pretrained(FLUX_BASE, subfolder="text_encoder", torch_dtype=bf16).to(dev)
        self._ctok = CLIPTokenizer.from_pretrained(FLUX_BASE, subfolder="tokenizer")
        self.t5 = T5Text(cache=64)
        self._pooled: OrderedDict = OrderedDict()
        self._setup(tr, vae, self._encode_text, progress, dev="cuda", dtype=bf16)

    def _setup(self, transformer, vae, encode_text, progress, dev, dtype):
        """Everything but model loading (tests build a painter from fakes through here)."""
        import torch

        self._torch = torch
        self.transformer, self.vae = transformer, vae
        self._encode_text, self.progress = encode_text, progress
        self._dev, self._dtype = dev, dtype
        self._sf, self._shift = vae.config.scaling_factor, vae.config.shift_factor
        self._lat_shape = (1, 16, PANO_H // 8, PANO_W // 8)
        i, j = torch.meshgrid(torch.arange(FLUX_WIN), torch.arange(FLUX_WIN), indexing="ij")
        self._img_ids = torch.stack([torch.zeros_like(i), i, j], -1).reshape(-1, 3).to(dev, dtype)
        self._txt_ids = torch.zeros(FLUX_TEXT_LEN, 3, device=dev, dtype=dtype)

    # --- text ------------------------------------------------------------------------------------------

    def _encode_text(self, text: str):
        """(T5 [1, 256, 4096], CLIP pooled [1, 768]) for one text, both bf16 on the CPU."""
        torch = self._torch
        if text in self._pooled:
            self._pooled.move_to_end(text)
        else:
            ids = self._ctok(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
            with torch.inference_mode():
                self._pooled[text] = self._clip(ids.to(self._dev)).pooler_output.to(torch.bfloat16).cpu()
            while len(self._pooled) > POOLED_CACHE:
                self._pooled.popitem(last=False)
        return self.t5.encode(text), self._pooled[text]

    def _embed(self, prompt: Prompt):
        """The prompt's (T5, pooled) conditioning: weighted sums over its texts, on the GPU."""
        t5 = pooled = 0
        for text, w in prompt:
            e, p = self._encode_text(text)
            t5, pooled = t5 + w * e.float(), pooled + w * p.float()
        return t5.to(self._dev, self._torch.bfloat16), pooled.to(self._dev, self._torch.bfloat16)

    # --- latents ---------------------------------------------------------------------------------------

    @staticmethod
    def _pack(lat):
        """[1, 16, H, W] VAE latent -> [1, 64, H/2, W/2] token grid (channel c*4 + dy*2 + dx)."""
        b, c, h, w = lat.shape
        return lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpack(grid):
        b, c4, h, w = grid.shape
        return grid.view(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c4 // 4, h * 2, w * 2)

    def _noise(self, seed: int):
        torch = self._torch
        g = torch.Generator(self._dev).manual_seed(seed)
        return self._pack(torch.randn(self._lat_shape, generator=g, device=self._dev, dtype=torch.float32))

    def _sigmas(self, steps: int):
        """Schnell's schedule: steps even sigmas from 1 down, then 0 (shift 1, no dynamic shifting)."""
        return [1 - k / steps for k in range(steps)] + [0.0]

    def _step_sigmas(self, bg_strength: float) -> tuple[list[float], int]:
        """(sigmas, k_bg): schnell's schedule with bg_strength on it, and the step where the background is
        released (len - 1, i.e. never, at strength 0)."""
        grid = self._sigmas(FLUX_STEPS)[:-1]
        if bg_strength <= 0:
            return grid + [0.0], FLUX_STEPS
        s = min(bg_strength, 1.0)
        sig = sorted({round(x, 6) for x in grid} | {round(s, 6)}, reverse=True) + [0.0]
        return sig, sig.index(round(s, 6))

    # --- VAE, tiled along the wrapping width (TurboPainter's crossfaded tiles) -------------------------------
    # TurboPainter decodes vae.decode(lat / sf) and encodes mean * sf; FLUX's VAE adds a shift factor:
    # decode (lat / sf + shift) = decode((lat + shift * sf) / sf), encode (mean - shift) * sf.

    _turbo_decode, _turbo_encode = TurboPainter._decode, TurboPainter._encode

    def _decode(self, grid, shift: int = 0) -> Image.Image:
        lat = self._unpack(grid).to(self._dtype)
        return self._turbo_decode(lat + self._shift * self._sf, shift)

    def _encode(self, im: Image.Image, shift: int = 0):
        return self._pack(self._turbo_encode(im, shift).float() - self._shift * self._sf)

    # --- MultiDiffusion --------------------------------------------------------------------------------

    def _denoise(self, grid, sigmas, embs, keep=None):
        """Euler-integrate a wrap-around token grid, averaging each step's velocity over windows.

        embs[i] = (t5, pooled) for window i. keep = (orig, noise, mask, k_bg): up to and including step k_bg
        the unmasked part is reset to orig noised to the current sigma (differential diffusion), so it only
        denoises sigmas[k_bg:], like img2img; before k_bg windows with no masked column are skipped (_active).
        """
        torch = self._torch
        h, w = grid.shape[-2:]
        wins = windows(h, w, FLUX_WIN, FLUX_STRIDE)
        masked = None if keep is None else (keep[2].reshape(-1) > 0).cpu().numpy()
        k_bg = None if keep is None else keep[3]
        acts = [_active(wins, masked, k, k_bg) for k in range(len(sigmas) - 1)]
        total, done = sum(math.ceil(len(a) / FLUX_BATCH) for a in acts), 0
        for k, act in enumerate(acts):
            s = sigmas[k]
            if keep is not None and k <= k_bg:
                orig, noise, mask, _ = keep
                grid = mask * grid + (1 - mask) * ((1 - s) * orig + s * noise)
            vel, cnt = torch.zeros_like(grid), torch.zeros_like(grid[:, :1])
            for b in range(0, len(act), FLUX_BATCH):
                check_stop()
                idx = act[b:b + FLUX_BATCH]
                cols = [wrap_cols(wins[i][1], FLUX_WIN, w) for i in idx]
                tiles = torch.cat([grid[..., y:y + FLUX_WIN, :][..., c] for (y, _), c in
                                   zip((wins[i] for i in idx), cols)])
                n = len(idx)
                out = self.transformer(
                    hidden_states=tiles.flatten(2).transpose(1, 2).to(self._dtype),
                    encoder_hidden_states=torch.cat([embs[i][0] for i in idx]),
                    pooled_projections=torch.cat([embs[i][1] for i in idx]),
                    timestep=torch.full((n,), s, device=self._dev, dtype=self._dtype),
                    img_ids=self._img_ids, txt_ids=self._txt_ids, guidance=None, return_dict=False)[0]
                out = out.float().transpose(1, 2).reshape(n, -1, FLUX_WIN, FLUX_WIN)
                for i, ((y, _), c) in enumerate(zip((wins[j] for j in idx), cols)):
                    vel[..., y:y + FLUX_WIN, c] += out[i:i + 1]
                    cnt[..., y:y + FLUX_WIN, c] += 1
                done += 1
                if self.progress is not None:
                    self.progress(done, total)
            # columns no window covered this step (skipped background) move by 0: they are reset next step
            grid = grid + (sigmas[k + 1] - s) * vel / cnt.clamp(min=1)
        if keep is not None and k_bg >= len(acts):     # strength 0: the background is never released, keep it
            orig, _, mask, _ = keep
            grid = mask * grid + (1 - mask) * orig
        return grid

    def panorama(self, base: Prompt, seed: int) -> Image.Image:
        with self._torch.inference_mode():
            e = self._embed(base)
            grid = self._noise(seed)
            n = len(windows(*grid.shape[-2:], FLUX_WIN, FLUX_STRIDE))
            out = self._denoise(grid, self._sigmas(FLUX_STEPS), [e] * n)
            return self._decode(out, shift=seed % VAE_TILE)

    def step(self, prev: Image.Image, base: Prompt, focus: Prompt | None, focus_at: float | None,
             seed: int, bg_strength: float | None = None, focus_half: float | None = None) -> Image.Image:
        torch = self._torch
        bg_strength = FLUX_BG_STRENGTH if bg_strength is None else bg_strength
        half = FOCUS_HALF if focus_half is None else focus_half
        h, w = PANO_H // 16, PANO_W // 16
        m = np.zeros(w, np.float32) if focus_at is None else focus_mask(w, focus_at, half)
        with torch.inference_mode():
            e_base = self._embed(base)
            e_focus = self._embed(focus) if focus is not None else e_base
            wins = windows(h, w, FLUX_WIN, FLUX_STRIDE)
            embs = [e_focus if m[(x + FLUX_WIN // 2) % w] > 0 else e_base for _, x in wins]
            # The background is released at sigma bg_strength and denoises from there, as img2img would. Schnell's
            # 4-step grid only offers 0.25 / 0.5 / 0.75, so the background's sigma is added to it when off the grid
            # (one more Euler point for the band ahead, which is cheap: only its windows run before k_bg).
            sig, k_bg = self._step_sigmas(bg_strength)
            # Nothing to regenerate: every step before k_bg would be overwritten, so start there (plain img2img).
            start = k_bg if not m.any() else 0
            shift = seed % VAE_TILE
            noise, orig = self._noise(seed), self._encode(prev, shift)
            mask = torch.from_numpy(m).to(self._dev, torch.float32)[None, None, None, :]
            s0 = sig[start]
            grid = (1 - s0) * orig + s0 * noise
            out = self._denoise(grid, sig[start:], embs, keep=(orig, noise, mask, k_bg - start))
            return self._decode(out, shift)
