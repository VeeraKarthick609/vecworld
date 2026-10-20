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

# Dream Roam pacing: the player walks up to WALK_LIMIT from a spot's centre before its edge, and the next spot is
# dollied by z = R / (R - walked) (R: the parent's median radius ahead), clamped to WALK_Z_MIN..WALK_Z_MAX.
# Measured 2026-10-19 on the RTX 3050 Ti: a spot builds in ~5.5 s; at the walking speed (stride 1.1 per 0.958 s
# cycle = 1.15 units/s) a radius of 6.5 is 5.7 s rim to rim. Real forest spots: scenery standing taller than
# OBSTACLE_H above the ground starts 7-10 units out in most directions, 4-6 where a bush or mound is near.
WALK_LIMIT = 6.5
MIN_WALK = 2.2             # never less than the spike's 0.55 * NEAR, whatever the depth says
WALL_FRACTION = 0.85       # stop this fraction of the way to scenery standing in the walking direction
OBSTACLE_H = 1.0           # scenery this far above the ground plane (spot units; the camera eye is ~3.2 up) blocks
WALL_HALF = 15.0           # degrees either side of a direction whose scenery bounds the walk that way
LIMIT_BINS = 36            # the walkable radius is sampled every 360 / LIMIT_BINS degrees, 0 = the spot's forward
WALK_Z_MIN = 1.15
# Past ~2.5 the depth re-read of the zoomed-in view stops agreeing with the parent's (scale clamped at MIN_STEP).
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
FLUX_BATCH = 4             # windows per transformer pass (peak 1.4 GB allocated on AC; Task 6 re-batches)
FLUX_BG_STRENGTH = 0.15    # a walk step releases its background at this sigma (0.25: SSIM 0.58 outside the focus, 0.15: 0.67)
FLUX_STYLE = "dreamlike digital painting, soft atmospheric light"
