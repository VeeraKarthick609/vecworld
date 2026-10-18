from pathlib import Path

LAYER = 16
MAX_STEP_ANGLE = 0.35
N_NEIGHBOURS = 3           # neighbour hotspots offered (plus the pinned centre)
N_CANDIDATES = 6           # neighbours CLIPSeg is asked about; only located ones are offered
N_PROMPT_NEIGHBOURS = 5    # neighbours named in the scene prompt
# img2img runs int(STEP_STEPS * STEP_STRENGTH) denoising steps: 1 dissolves the scene over a few
# clicks, 3+ fries it, 2 keeps it painterly. A gentle zoom avoids radial streaks from compounding.
STEP_STRENGTH = 0.5
STEP_STEPS = 4
ZOOM_FRAC = 0.85
# Each step scene is pulled this far back toward the jump scene's LAB palette. Over 7 steps, 0 bands the
# sky yellow/blue, 0.3 still bands, 1.0 washes everything grey; 0.5 holds the palette and stays lively.
PALETTE_STRENGTH = 0.5
HOTSPOT_MIN = 0.3          # peaks below this were wrong objects (gorilla 0.22 on an elephant)
MAX_HOTSPOT_AREA = 0.5     # a box covering more of the frame is the whole scene, not an object
MAX_HOTSPOT_IOU = 0.6      # a box this similar to one already offered would hide it
CENTRE_BOX = (0.35, 0.35, 0.3, 0.3)  # the pinned centre's box when CLIPSeg can't place it
MAX_TEXT_CHARS = 200
HISTORY = 10               # snapshots kept for going back (S / down arrow)
RECENT = 4                 # the last few centres are left out of the neighbours, so walks don't loop
SESSION_TTL = 3600     # seconds idle before a session (and its scene image) is dropped
MAX_SESSIONS = 50
STYLE = "impressionist oil painting, visible brushstrokes, dreamlike, soft volumetric light"

DATA_DIR = Path("data")
SCENE_DIR = Path("var/scenes")

# Dream Walk: seamless cylindrical panorama
PANO_W = 2048
PANO_H = 512
WIN = 64                   # latent window size
STRIDE = 32
VAE_TILE = 64
VAE_MARGIN = 8
DOLLY = 1.15
BG_STRENGTH = 0.6
FOCUS_HALF = 0.08
FOCUS_FEATHER = 0.06
KEEP_OFFSET = 0.17
QUIET_EVERY = 3
DRIFT_STEPS = 3
PREFETCH_TOLERANCE = 45.0
MOODS = ("foggy", "night", "rain", "snow", "eerie", "warm")
MAX_THEME_CHARS = 120
