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

from vecworld.config import (FLUX_BACK_FEATHER, FLUX_BACK_FROM, FLUX_BACK_STRENGTH, FLUX_BASE, FLUX_BATCH,
                             FLUX_GGUF, FLUX_STEP_RECIPE, FLUX_STEPS, FLUX_STRIDE, check_recipe,
                             FLUX_STYLE, FLUX_TEXT_LEN, FLUX_WIN, FOCUS_HALF, PANO_H, PANO_W, PREDREAM_HEADINGS, VAE_TILE)
from vecworld.dream.gpu import check_stop
from vecworld.dream.painter import Painter, Prompt, TurboPainter
from vecworld.dream.pano import focus_mask, ramp, windows, wrap_cols

POOLED_CACHE = 64     # CLIP pooled embeddings kept (least recently used dropped); T5 keeps its own


def _read_gguf(path) -> dict:
    """The GGUF's tensors as diffusers' load_gguf_checkpoint returns them, read with plain reads.

    diffusers copies each tensor out of a memmap of the whole file, so while it loads, the 6.8 GB of mapped file
    pages and the 6.8 GB of copies are both resident (a 12+ GB peak that pushed the game into swap).
    """
    import gguf
    import torch
    from diffusers.quantizers.gguf.utils import GGUFParameter

    r = gguf.GGUFReader(path)
    meta = [(t.name, t.tensor_type, t.data.shape, t.data.dtype, int(t.data_offset), int(t.n_bytes)) for t in r.tensors]
    del r
    plain = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    out = {}
    with open(path, "rb", buffering=0) as f:
        for name, qt, shape, dtype, off, n in meta:
            a = np.empty(shape, dtype)
            f.seek(off)
            assert f.readinto(memoryview(a).cast("B")) == n == a.nbytes, name
            w = torch.from_numpy(a)
            out[name] = w if qt in plain else GGUFParameter(w, quant_type=qt)
    return out


def _pin_in_place(modules) -> list[int]:
    """Page-lock the CPU parameters and buffers of modules where they already lie (cudaHostRegister); returns the
    registered pointers, which _unpin must release before the tensors are freed.

    diffusers' streamed group offload pins a second copy of each tensor (tensor.pin_memory()) and keeps the
    original until the first forward, so the 6.7 GB transformer briefly needs ~13.5 GB of RAM, and torch's pinned
    allocator rounds every block up to a power of two (8.8 GB held for 6.7). A tensor already pinned is returned
    as is by pin_memory(), so registering them first keeps one copy at its true size. Only modules whose CPU
    tensors live as long as the painter may be passed: a registered buffer freed and reused breaks later copies
    ("CUDA error: invalid argument ... Copy is larger than memobj size"; the offload group holding the transformer's
    non-block layers moves them with .to(), freeing the originals).
    """
    import torch

    cudart, seen, total = torch.cuda.cudart(), [], 0
    for t in [x for m in modules for x in (*m.parameters(), *m.buffers())]:
        st = t.untyped_storage()
        ptr, n = st.data_ptr(), st.nbytes()
        if t.device.type != "cpu" or n == 0 or ptr in seen or t.is_pinned():
            continue
        err = cudart.cudaHostRegister(ptr, n, 0)
        if int(err) != 0:
            _unpin(seen)
            raise RuntimeError(f"cudaHostRegister failed ({err}) after locking {total / 1e9:.1f} GB")
        seen.append(ptr)
        total += n
    return seen


def _unpin(ptrs) -> None:
    """Unregister host ranges _pin_in_place registered (cudaHostUnregister), while their tensors still live."""
    import torch

    cudart = torch.cuda.cudart()
    for ptr in ptrs:
        cudart.cudaHostUnregister(ptr)


def _wrap_dist(w: int, at: float) -> np.ndarray:
    """Per column (centre), the distance round the cylinder to fraction `at`, in fractions of a turn (0..0.5)."""
    d = np.abs((np.arange(w) + 0.5) / w - at) % 1.0
    return np.minimum(d, 1.0 - d)


def release_strengths(m: np.ndarray, focus_at: float | None, side: float, back: float, focus: float) -> np.ndarray:
    """Per column, the strength a graded step releases it at: `side`, rising to `back` behind the camera (from
    FLUX_BACK_FROM over FLUX_BACK_FEATHER), mixed toward `focus` by the focus mask m. (A linear step paints the
    focus from scratch and holds the rest to `side` / `back` the same way.)"""
    w = len(m)
    b = (np.zeros(w) if focus_at is None else
         np.clip((_wrap_dist(w, focus_at) - FLUX_BACK_FROM) / FLUX_BACK_FEATHER, 0.0, 1.0))
    bg = side + b * (back - side)
    return bg + m * (focus - bg)


def _active(wins, reset_next) -> list[int]:
    """The windows worth a pass at a step: a window all of whose columns are fully reset to the noised original
    at the next step would only be thrown away (as TurboPainter skips them). reset_next: bool per column, or None."""
    if reset_next is None:
        return list(range(len(wins)))
    return [i for i, (_, x) in enumerate(wins) if not reset_next[wrap_cols(x, FLUX_WIN, len(reset_next))].all()]


class FluxPainter(Painter):
    """FLUX.1-schnell (Q4_K_S GGUF, bf16 compute) painting a PANO_W x PANO_H cylinder in wrapping windows.

    progress(done, total) is called after every transformer pass (one window batch at one step), total being
    the passes the job will make; stage(name) names the work around them: "text" (T5 + CLIP, tens of seconds on a
    cache miss), "image" (a step's VAE encode of the view it starts from), "finishing" (the VAE decode).
    check_stop() runs before every pass and every VAE tile. A RoamService given this painter takes both hooks over
    to report each session's painting.

    The transformer's CPU weights are page-locked where they lie; close() (also run when the painter is freed)
    releases them, after which the painter can't paint.
    """

    style = FLUX_STYLE
    predream_headings = PREDREAM_HEADINGS["flux"]   # slow: pre-dream only the way the player faces

    def __init__(self, progress: Callable[[int, int], None] | None = None):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from diffusers import AutoencoderKL, FluxTransformer2DModel, GGUFQuantizationConfig
        from transformers import CLIPTextModel, CLIPTokenizer

        from vecworld.flux.text import T5Text

        dev, bf16 = torch.device("cuda"), torch.bfloat16
        tr = FluxTransformer2DModel.from_single_file(
            _read_gguf(str(FLUX_GGUF)), quantization_config=GGUFQuantizationConfig(compute_dtype=bf16),
            torch_dtype=bf16, config=FLUX_BASE, subfolder="transformer")
        self._pinned = _pin_in_place([*tr.transformer_blocks, *tr.single_transformer_blocks])
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
        self.stage: Callable[[str], None] | None = None
        self.recipe = FLUX_STEP_RECIPE
        self._pinned: list[int] = getattr(self, "_pinned", [])
        self._closed = False
        self._dev, self._dtype = dev, dtype
        self._sf, self._shift = vae.config.scaling_factor, vae.config.shift_factor
        self._lat_shape = (1, 16, PANO_H // 8, PANO_W // 8)
        i, j = torch.meshgrid(torch.arange(FLUX_WIN), torch.arange(FLUX_WIN), indexing="ij")
        self._img_ids = torch.stack([torch.zeros_like(i), i, j], -1).reshape(-1, 3).to(dev, dtype)
        self._txt_ids = torch.zeros(FLUX_TEXT_LEN, 3, device=dev, dtype=dtype)

    @property
    def recipe(self) -> str:
        """How step() repaints: a key of config.FLUX_STEP_RECIPES, checked when set (a typo fails here)."""
        return self._recipe

    @recipe.setter
    def recipe(self, name: str) -> None:
        check_recipe(name)
        self._recipe = name

    def close(self) -> None:
        """Unregister the page-locked weights (safe twice). The painter can't paint after it."""
        self._closed = True
        ptrs, self._pinned = getattr(self, "_pinned", []), []
        if ptrs:
            _unpin(ptrs)

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - at interpreter exit CUDA may already be gone
            pass

    def _stage(self, name: str) -> None:
        if self._closed:
            raise RuntimeError("this FluxPainter is closed")
        if self.stage is not None:
            self.stage(name)

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

    def _step_sigmas(self, *strengths: float) -> tuple[list[float], list[int]]:
        """(sigmas, ks): schnell's schedule with each strength put on it (as a sigma), and for each strength the
        last step at which a column released at it is still reset (the index of its sigma; strength 0: the final
        sigma 0, i.e. never released)."""
        r = [round(min(max(s, 0.0), 1.0), 6) for s in strengths]
        sig = sorted({round(x, 6) for x in self._sigmas(FLUX_STEPS)[:-1]} | {s for s in r if s > 0},
                     reverse=True) + [0.0]
        return sig, [sig.index(s) for s in r]

    def _reset_weights(self, m: np.ndarray, focus_at: float | None, bg_strength: float,
                       back_strength: float = FLUX_BACK_STRENGTH):
        """(sigmas, rw): rw[k] is, per token column, how much of the step-k latent is reset to the original noised
        to sigmas[k] (rw[-1], at sigma 0, pastes the original back). m: the focus mask (never reset).

        The sides are released at bg_strength. Behind the camera, from FLUX_BACK_FROM (a quarter turn, where the
        dolly starts shrinking the view) feathered over FLUX_BACK_FEATHER, the release is FLUX_BACK_STRENGTH:
        there dolly() shrank the view and mirrored the rows next to the top and bottom edges in (upside-down tree
        tops), which a light img2img keeps. Without a focus direction everything is a side.
        """
        torch = self._torch
        w = len(m)
        back = (np.zeros(w, np.float32) if focus_at is None else
                np.clip((_wrap_dist(w, focus_at) - FLUX_BACK_FROM) / FLUX_BACK_FEATHER, 0.0, 1.0).astype(np.float32))
        sig, (k_side, k_back) = self._step_sigmas(bg_strength, max(bg_strength, back_strength))
        rw = []
        for k in range(len(sig)):
            hold = back * (k <= k_back) + (1 - back) * (k <= k_side)
            rw.append(torch.from_numpy(((1 - m) * hold).astype(np.float32)).to(self._dev)[None, None, None, :])
        return sig, rw

    def _graded_weights(self, m: np.ndarray, focus_at: float | None, side: float, back: float, focus: float):
        """(sigmas, rw) as _reset_weights, but each column is released once, at its own strength: `side` on the
        sides, `back` behind the camera (from FLUX_BACK_FROM, feathered over FLUX_BACK_FEATHER), `focus` in the
        focus mask m, and in between their mix by the mask's weight. A strength between two sigmas of the schedule
        is held fully up to the earlier one and partly (by how far it lies toward the later one) at the later one.

        _reset_weights instead holds a feather column at a fraction (1 - m) of the original through every step up
        to the background's release: the band ahead and the background never denoise together there, so the
        feather comes out as the two images crossfaded (a double exposure). Here every column, feather included,
        denoises jointly with its neighbours from its release on.
        """
        torch = self._torch
        r = release_strengths(m, focus_at, side, back, focus)
        sig, _ = self._step_sigmas(side, back, focus)
        f = np.interp(-r, -np.asarray(sig), np.arange(len(sig)))       # its (fractional) release step
        rw = [torch.from_numpy(np.clip(f - k + 1, 0.0, 1.0).astype(np.float32)).to(self._dev)[None, None, None, :]
              for k in range(len(sig))]
        return sig, rw

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

    def _forward(self, tiles, embs, s: float):
        """The transformer's velocity for a batch of token-grid windows [n, 64, FLUX_WIN, FLUX_WIN] at sigma s;
        embs: one (t5, pooled) per window."""
        torch, n = self._torch, tiles.shape[0]
        out = self.transformer(
            hidden_states=tiles.flatten(2).transpose(1, 2).to(self._dtype),
            encoder_hidden_states=torch.cat([e[0] for e in embs]),
            pooled_projections=torch.cat([e[1] for e in embs]),
            timestep=torch.full((n,), s, device=self._dev, dtype=self._dtype),
            img_ids=self._img_ids, txt_ids=self._txt_ids, guidance=None, return_dict=False)[0]
        return out.float().transpose(1, 2).reshape(n, -1, FLUX_WIN, FLUX_WIN)

    def _denoise(self, grid, sigmas, embs, keep=None):
        """Euler-integrate a wrap-around token grid, averaging each step's velocity over windows.

        Each window's velocity is weighted by a ramp across its width (rising over its first half, falling over
        its second), so neighbouring windows crossfade. A plain average jumps wherever the pair of windows covering
        a column changes, and FLUX's windows disagree enough (each settles its own exposure, worst in haze) for
        that to show as a hard vertical line every FLUX_STRIDE.

        embs[i] = (t5, pooled) for window i. keep = (orig, noise, rw): at step k, rw[k] (per column) of the latent
        is reset to orig noised to sigmas[k] (differential diffusion), so a column held up to its release step
        denoises only from there, like img2img; rw[-1] pastes orig back after the last step (strength 0). Windows
        whose columns are all fully reset at the next step are skipped (_active).
        """
        torch = self._torch
        h, w = grid.shape[-2:]
        wins = windows(h, w, FLUX_WIN, FLUX_STRIDE)
        n_steps = len(sigmas) - 1
        full = None if keep is None else [(r.reshape(-1) >= 1).cpu().numpy() for r in keep[2]]
        acts = [_active(wins, None if full is None or k + 1 >= n_steps else full[k + 1]) for k in range(n_steps)]
        total, done = sum(math.ceil(len(a) / FLUX_BATCH) for a in acts), 0
        wt = torch.from_numpy(ramp(FLUX_WIN, FLUX_WIN // 2)).to(grid.device)
        for k, act in enumerate(acts):
            s = sigmas[k]
            if keep is not None:
                orig, noise, rw = keep
                grid = (1 - rw[k]) * grid + rw[k] * ((1 - s) * orig + s * noise)
            vel, cnt = torch.zeros_like(grid), torch.zeros_like(grid[:, :1])
            for b in range(0, len(act), FLUX_BATCH):
                check_stop()
                idx = act[b:b + FLUX_BATCH]
                cols = [wrap_cols(wins[i][1], FLUX_WIN, w) for i in idx]
                tiles = torch.cat([grid[..., y:y + FLUX_WIN, :][..., c] for (y, _), c in
                                   zip((wins[i] for i in idx), cols)])
                out = self._forward(tiles, [embs[i] for i in idx], s)
                for i, ((y, _), c) in enumerate(zip((wins[j] for j in idx), cols)):
                    vel[..., y:y + FLUX_WIN, c] += out[i:i + 1] * wt
                    cnt[..., y:y + FLUX_WIN, c] += wt
                done += 1
                if self.progress is not None:
                    self.progress(done, total)
            # columns no window covered this step (skipped background) move by 0: they are reset next step
            grid = grid + (sigmas[k + 1] - s) * vel / cnt.clamp(min=1e-6)
        if keep is not None and keep[2][-1].any():      # strength 0: a background never released is kept as is
            grid = (1 - keep[2][-1]) * grid + keep[2][-1] * keep[0]
        return grid

    def panorama(self, base: Prompt, seed: int) -> Image.Image:
        with self._torch.inference_mode():
            self._stage("text")
            e = self._embed(base)
            grid = self._noise(seed)
            n = len(windows(*grid.shape[-2:], FLUX_WIN, FLUX_STRIDE))
            out = self._denoise(grid, self._sigmas(FLUX_STEPS), [e] * n)
            self._stage("finishing")
            return self._decode(out, shift=seed % VAE_TILE)

    def step(self, prev: Image.Image, base: Prompt, focus: Prompt | None, focus_at: float | None,
             seed: int, bg_strength: float | None = None, focus_half: float | None = None) -> Image.Image:
        """One walk step from the dollied view prev, by self.recipe (config.FLUX_STEP_RECIPES): "fresh" paints a
        new panorama from base; the others repaint prev, the focus band ahead at the recipe's focus strength and
        the rest at its side and back strengths (bg_strength, if given, replaces the side strength and is the
        least any column gets)."""
        torch = self._torch
        rec = check_recipe(self.recipe)
        if rec["release"] == "fresh":
            return self.panorama(base, seed)
        side = rec["side"] if bg_strength is None else bg_strength
        half = FOCUS_HALF if focus_half is None else focus_half
        h, w = PANO_H // 16, PANO_W // 16
        m = np.zeros(w, np.float32) if focus_at is None else focus_mask(w, focus_at, half, rec["feather"])
        with torch.inference_mode():
            self._stage("text")
            e_base = self._embed(base)
            e_focus = self._embed(focus) if focus is not None else e_base
            wins = windows(h, w, FLUX_WIN, FLUX_STRIDE)
            embs = [e_focus if m[(x + FLUX_WIN // 2) % w] > 0 else e_base for _, x in wins]
            # The background is released at sigma bg_strength (the band behind the camera earlier, _reset_weights)
            # and denoises from there, as img2img would. Schnell's 4-step grid only offers 0.25 / 0.5 / 0.75, so a
            # strength off it is added to it (one more Euler point for the band ahead: cheap, since only windows
            # with a column not held to the original run before the release).
            if rec["release"] == "linear":
                sig, rw = self._reset_weights(m, focus_at, side, rec["back"])     # focus: from scratch (1.0)
            else:
                sig, rw = self._graded_weights(m, focus_at, side, max(side, rec["back"]), max(side, rec["focus"]))
            # Every step up to the first release resets the whole grid to the original noised to its sigma, so
            # start at the last of them (img2img): its reset is the starting latent, and its pass is made.
            start = max(0, next((k for k, r in enumerate(rw) if (r < 1).any()), len(sig)) - 1)
            shift = seed % VAE_TILE
            self._stage("image")
            noise, orig = self._noise(seed), self._encode(prev, shift)
            s0 = sig[start]
            grid = (1 - s0) * orig + s0 * noise
            out = self._denoise(grid, sig[start:], embs, keep=(orig, noise, rw[start:]))
            self._stage("finishing")
            return self._decode(out, shift)
