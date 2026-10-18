from pathlib import Path

HOTSPOT_MIN = 0.3          # CLIPSeg peaks below this were wrong objects (gorilla 0.22 on an elephant)
# A walk step schedules STEP_STEPS timesteps; the background is plain img2img at BG_STRENGTH of them.
STEP_STEPS = 4
# Each step is pulled this far back toward the reference panorama's LAB palette (reset by theme/mood).
# 0 bands the sky yellow/blue over a few steps, 1.0 washes everything grey; 0.5 holds it and stays lively.
PALETTE_STRENGTH = 0.5
SESSION_TTL = 3600     # seconds idle before a session (and its scene image) is dropped
MAX_SESSIONS = 50
STYLE = "impressionist oil painting, visible brushstrokes, dreamlike, soft volumetric light"

SCENE_DIR = Path("var/scenes")

# Dream Walk: seamless cylindrical panorama
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
KEEP_OFFSET = 0.17
KEEP_MAX_W = 0.1           # a label wider (fraction of the panorama) or taller than this is scenery, not an object
KEEP_MAX_H = 0.6
MAX_KEEPS = 2              # consecutive steps that keep what you walk toward, then it is walked past
QUIET_EVERY = 3
DRIFT_STEPS = 3
PREFETCH_TOLERANCE = 45.0
MOODS = ("foggy", "night", "rain", "snow", "eerie", "warm")
MAX_THEME_CHARS = 120
