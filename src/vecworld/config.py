import os
from pathlib import Path

# A walk step schedules STEP_STEPS timesteps; the background is plain img2img at BG_STRENGTH of them.
STEP_STEPS = 4
# Each step is pulled this far back toward the reference panorama's LAB palette (reset by theme/mood).
# 0 bands the sky yellow/blue over a few steps, 1.0 washes everything grey; 0.5 holds it and stays lively.
PALETTE_STRENGTH = 0.5
SESSION_TTL = 3600     # seconds idle before a session is dropped; its view images are deleted with it
MAX_SESSIONS = 50
STYLE = "impressionist oil painting, visible brushstrokes, dreamlike atmosphere"

SCENE_DIR = Path("var/scenes")

# The painter: seamless cylindrical panorama
PANO_W = 2048
PANO_H = 512
WIN = 64                   # latent window size
STRIDE = 32
VAE_TILE = 64              # latents per VAE tile; tiles overlap and crossfade (per-tile GroupNorm shifts colour)
VAE_STRIDE = 48            # at most this far apart, so neighbours overlap >= 16 latents
VAE_MARGIN = 0             # extra context coded each side, then cropped (the crossfade already fades tile edges out)
DOLLY = 1.15
BG_STRENGTH = 0.6
FOCUS_HALF = 0.08
FOCUS_FEATHER = 0.06
MOOD_BG_STRENGTH = 0.8     # background img2img strength of the first step after a mood change
MOODS = ("foggy", "night", "rain", "snow", "eerie", "warm")   # the chips; their prompt words below
MOOD_WORDS = {
    "night": "at night, dark night sky, cold moonlight, deep shadows",
    "foggy": "thick fog, mist",
    "rain": "heavy rain, wet, overcast",
    "snow": "falling snow, snow-covered ground",
    "eerie": "dark, eerie, ominous, gloomy",
    "warm": "warm golden light",
}
MAX_THEME_CHARS = 120

# Dream Roam: 360-degree depth from overlapping slices of the panorama
DEPTH_SLICE = 512
DEPTH_STEP = 256
NEAR = 4.0
FAR = 50.0
EDGE_RATIO = 1.35

# Dream Roam pre-dreaming, per painter: the headings (degrees from where the player faces) each spot's next spots are
# pre-dreamed toward, in queue order. SD-Turbo paints a step in ~5 s, so three directions are cheap; a FLUX step is
# ~61 s on AC, so it pre-dreams only the way the player faces (spec §6) and a turn re-targets it.
PREDREAM_HEADINGS = {"flux": (0,), "turbo": (0, -60, 60)}

# Dream Roam pacing: the player walks up to WALK_LIMIT from a spot's centre before its edge, and the next spot is
# dollied by z = R / (R - walked) (R: the parent's median radius ahead), clamped to WALK_Z_MIN..WALK_Z_MAX.
# Walking speed: stride 1.1 per 0.958 s cycle = 1.15 units/s. SD-Turbo (2026-10-19): a spot built in ~5.5 s, and
# 6.5 (5.7 s rim to rim) hid it. FLUX (2026-10-20, AC): a step takes ~61 s, so no radius hides it; the limit is set
# so each wait buys as much walking as the scenery and the dolly allow. FLUX forests are deep (median radius ahead
# R ~ 15-20, dollies of z 1.5-1.6 at 6.5): 6.5 bound most directions, and at 9 the walls bind instead (scenery
# taller than OBSTACLE_H stands 7-11 units out: median limits 7.2-9.0 per spot, against 6.5). 9 is also about the
# most WALK_Z_MAX represents at FLUX's typical R ~ 15 (z = 15 / (15 - 9) = 2.5). 9 units is 7.8 s of walking per
# spot against 5.7 s at 6.5: the wait at the next rim drops from ~55 s to ~53 s, and a spot covers 40% more ground.
WALK_LIMIT = 9.0
MIN_WALK = 2.2             # never less than the spike's 0.55 * NEAR, whatever the depth says
WALL_FRACTION = 0.85       # stop this fraction of the way to scenery standing in the walking direction
OBSTACLE_H = 1.0           # scenery this far above the ground plane (spot units; the camera eye is ~3.2 up) blocks
WALL_HALF = 15.0           # degrees either side of a direction whose scenery bounds the walk that way
LIMIT_BINS = 36            # the walkable radius is sampled every 360 / LIMIT_BINS degrees, 0 = the spot's forward
WALK_Z_MIN = 1.15
# Past ~2.5 the depth re-read of the zoomed-in view stops agreeing with the parent's (scale clamped at MIN_STEP).
# Kept for FLUX: its deep scenes dolly by only 1.5-1.6 at the old limit, so a wider range would buy no walking.
WALK_Z_MAX = 2.5
# A bigger dolly magnifies more of the view ahead: the band regenerated from scratch grows to where the dolly
# magnifies less than FOCUS_MAG (beyond it img2img at BG_STRENGTH keeps it sharp), up to FOCUS_HALF_MAX.
FOCUS_MAG = 1.3
FOCUS_HALF_MAX = 0.2

# FLUX.1-schnell painter (vecworld.flux): 4-bit GGUF transformer, 16-channel VAE latents packed 2x2 into tokens.
# A PANO_W x PANO_H panorama is a (PANO_H // 16) x (PANO_W // 16) grid of packed tokens.
FLUX_BASE = "unsloth/FLUX.1-schnell"     # VAE, CLIP-L, tokenizers and configs (ungated mirror)
_HF_HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
FLUX_GGUF = _HF_HUB / ("models--city96--FLUX.1-schnell-gguf/snapshots/f495746ed9c5efcf4661f53ef05401dceadc17d2/"
                       "flux1-schnell-Q4_K_S.gguf")
FLUX_T5_GGUF = _HF_HUB / ("models--city96--t5-v1_1-xxl-encoder-gguf/snapshots/"
                          "005a6ea51a7d0b84d677b3e633bb52a8c85a83d9/t5-v1_1-xxl-encoder-Q5_K_M.gguf")
FLUX_TEXT_LEN = 256        # T5 tokens (schnell's max_sequence_length)
FLUX_WIN = 32              # window size in packed tokens (512 px), the full panorama height
FLUX_STRIDE = 16           # 8 windows round the cylinder, each column covered twice
FLUX_STEPS = 4
# The engine kernel FLUX's Q4_K linears run on when VECWORLD_KERNEL is unset (vecworld.engine.q4k.KERNELS). Since
# Task 5b "w4a8": per-token INT8 activations x the Q4_K codes on the INT8 tensor cores, lossy, admitted by the walk
# gate with FLUX_A16_KEEP below (docs/engine/04-w4a8.md). "mma" is the lossless W4A16 kernel (Task 5);
# VECWORLD_KERNEL=off runs the diffusers path.
FLUX_KERNEL = "w4a8"
# Under w4a8, the linears (fnmatch patterns of module names) kept on "mma" (W4A16); vecworld.engine.keep_for applies
# them under w4a8 only. Chosen by the walk gate and an ablation (docs/engine/04-w4a8.md, "the gate"):
#   - the study's (scripts/w4a8_study.py, bench/t5b-study.json) five worst families: the text stream's MLP, every
#     value projection, the text q / v projections. Alone they failed the gate (spot 1 seam 1.16, spot 3 1.14), as did
#     the study's next-worst 23 layers on top (1.19);
#   - plus the double blocks' residual-stream outputs, MLP-out (ff.net.2) and both attention-outs: the walk passed
#     (max seam 1.058). Without the attention-outs (1.24) or without ff.net.2 (1.22) it failed, and keeping the first
#     and last blocks instead did not help (1.16). The study ranks these outputs only mid to least hurt per layer:
#     the walk, not the per-layer error, picked them.
# 75.0% of the token-facing FLOPs stay on INT8.
FLUX_A16_KEEP: tuple[str, ...] = (
    "transformer_blocks.*.ff_context.net.0.proj", "transformer_blocks.*.ff_context.net.2", "*.attn.to_v",
    "transformer_blocks.*.attn.add_v_proj", "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.ff.net.2", "transformer_blocks.*.attn.to_out.0", "transformer_blocks.*.attn.to_add_out",
)
# Windows per transformer pass (Task 9, docs/engine/09-batching.md; VECWORLD_BATCH overrides): 8 = a panorama step's
# every window in one pass. Measured on AC at 1 / 2 / 4 / 6 / 8: panorama 30.3 / 30.6 / 30.3 / 30.4 / 30.2 s, walk step
# 40.3 / 39.9 / 39.4 / 39.5 / 39.3 s, peak 1.76-2.05 GB: compute-bound, so per-window time is flat (an interleaved
# A/B: 0.918 s at 4, 0.915 at 8), but 8 halves the passes, the per-block Python and every per-pass weight copy.
FLUX_BATCH = 8
# Under an engine kernel the transformer's blocks stream their weights through FLUX_SLOT_RING fixed-address device
# slots (vecworld.engine.slots, each the size of the largest block, 191 MB) and replay as one CUDA graph per block
# and input shape (vecworld.engine.graphs, VECWORLD_GRAPHS). 3, not 2: block i uses slot i % ring, and with 2 slots
# the 57 blocks (odd) put the last block and the next pass's first in the same slot, so every pass waited ~13.5 ms
# for block 0's copy (measured; 3 divides 57: tile512 pass 0.869 -> 0.859 s, +191 MB; docs/engine/07-cuda-graphs.md).
# FLUX_GRAPH_MAX shapes (batch, image tokens, text tokens) keep their graphs, the least recently used dropped beyond
# it: a walk uses up to FLUX_BATCH (1..8 windows a pass; graphs are opt-in).
FLUX_SLOT_RING = 3
FLUX_GRAPH_MAX = 4
# Where the streamed blocks' weights come from (Task 10b, docs/engine/11-io-uring.md; FLUX_WEIGHTS_SOURCE in the
# environment overrides it): "ssd" reads each block from the GGUF file with O_DIRECT through io_uring into
# FLUX_SSD_BUFFERS page-locked staging buffers (191 MB each), reading FLUX_SSD_AHEAD blocks ahead, so no weights stay in
# RAM; "ram" keeps all of them in one 6.66 GB page-locked arena (Task 10a). Under VECWORLD_KERNEL=off always "ram".
FLUX_WEIGHTS_SOURCE = "ssd"
FLUX_SSD_BUFFERS = 3
FLUX_SSD_AHEAD = 2
FLUX_SSD_QUEUE_DEPTH = 64       # 1 MB reads in the kernel at once (the SSD saturates at any depth from 4)
FLUX_SSD_CHUNK = 1 << 20
# First-block cache (Task 8, docs/engine/08-block-cache.md): a pass skips blocks 1..56 while block 0's residual has
# changed by less than this (summed relative L1 change over the window batch's steps); 0.0 is off.
# VECWORLD_CACHE_THRESHOLD overrides it; under VECWORLD_KERNEL=off it is always off. Off by the Task 8 sweep: 0.3 and
# 0.25 (1.3-1.6x per pass) fail the multi-scene relative walk gate, 0.2 and below skip nothing on its walks.
FLUX_CACHE_THRESHOLD = 0.0
FLUX_BG_STRENGTH = 0.15    # a walk step releases its background at this sigma (0.25: SSIM 0.58 outside the focus, 0.15: 0.67)
# Behind the camera (from a quarter turn off the walking direction, feathered over FLUX_BACK_FEATHER) the dolly
# shrank the view and mirrored rows in at the top and bottom edges; there the background is released at
# FLUX_BACK_STRENGTH instead, so the mirror images (upside-down tree tops) are repainted.
FLUX_BACK_FROM = 0.25
FLUX_BACK_FEATHER = 0.1
FLUX_BACK_STRENGTH = 0.5
# How a FLUX walk step repaints the dollied view (FluxPainter.recipe; the user picks by eye, Task 2b A/B sheets).
#   release "linear": Task 1's step. The band ahead (focus mask, `feather`) is painted from scratch while the
#     rest is held to the original until `side` (behind the camera `back`); a feather column is held at (1 - mask)
#     throughout, so the feather is a crossfade of the new band and the old view. Its focus is always from scratch,
#     so `focus` must be 1.0.
#   release "graded": every column is released once at its own strength (focus / side / back, mixed by the masks'
#     weights; the focus mask feathered over `feather` of a turn) and from then on denoises with its neighbours.
#   release "fresh": a new panorama from the step's prompt and seed; only the prompt blend carries the walk on.
FLUX_STEP_RECIPES = {
    "t1": {"release": "linear", "side": FLUX_BG_STRENGTH, "back": FLUX_BACK_STRENGTH, "focus": 1.0,
           "feather": FOCUS_FEATHER},
    "repaint": {"release": "graded", "side": 0.5, "back": 0.5, "focus": 1.0, "feather": 0.15},
    "carry": {"release": "graded", "side": 0.5, "back": 0.75, "focus": 0.75, "feather": 0.15},
    "renew": {"release": "graded", "side": 0.5, "back": 0.9, "focus": 0.9, "feather": 0.15},
    # sharp was side 0.5 / feather 0.15 until Task 2b's fix round. Between the band painted from scratch and a kept
    # view of another exposure, columns released at 0.6-0.85 came out as a featureless haze ending in a hard
    # full-height edge (walk spots 2, 4 and 5). A 0.25 feather removed the hard edges of spot 4 (walk_sheet seam score
    # 1.41 -> 0.64, same step) but left haze bands at the focus edge (1.33 / 1.31 on the new walk's spots 2-3);
    # releasing the sides at 0.75 too lets them follow the new band (0.82 / 0.80 on those steps) and is faster.
    "sharp": {"release": "graded", "side": 0.75, "back": 0.9, "focus": 1.0, "feather": 0.25},
    "fresh": {"release": "fresh"},
}
FLUX_STEP_RECIPE = "sharp"   # the user's pick after the Task 2b A/B sheets (2026-10-24): final


def check_recipe(name: str) -> dict:
    """The recipe called `name`, checked: an unknown name or a malformed recipe raises ValueError (at import for
    FLUX_STEP_RECIPE, so a typo stops the game at startup, not at its first walk step)."""
    if name not in FLUX_STEP_RECIPES:
        raise ValueError(f"unknown FLUX step recipe {name!r}; known: {', '.join(FLUX_STEP_RECIPES)}")
    rec = FLUX_STEP_RECIPES[name]
    release = rec.get("release")
    if release == "fresh":
        return rec
    if release not in ("linear", "graded"):
        raise ValueError(f"FLUX step recipe {name!r}: release {release!r} is not linear, graded or fresh")
    missing = {"side", "back", "focus", "feather"} - set(rec)
    if missing:
        raise ValueError(f"FLUX step recipe {name!r} lacks {', '.join(sorted(missing))}")
    if not all(0.0 <= rec[k] <= 1.0 for k in ("side", "back", "focus")) or not 0.0 < rec["feather"] < 0.5:
        raise ValueError(f"FLUX step recipe {name!r}: strengths must lie in 0..1 and the feather in (0, 0.5)")
    if release == "linear" and rec["focus"] != 1.0:
        raise ValueError(f"FLUX step recipe {name!r}: a linear release paints its focus from scratch (focus 1.0)")
    return rec


check_recipe(FLUX_STEP_RECIPE)
FLUX_STYLE = "dreamlike landscape painting, soft atmospheric light, wordless scenery"   # "wordless": short prompts drew title lettering
