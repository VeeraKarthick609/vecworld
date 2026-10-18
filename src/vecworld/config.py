from pathlib import Path

LAYER = 16
MAX_STEP_ANGLE = 0.35
N_NEIGHBOURS = 3
# img2img runs int(STEP_STEPS * STEP_STRENGTH) denoising steps: 1 dissolves the scene over a few
# clicks, 3+ fries it, 2 keeps it painterly. A gentle zoom avoids radial streaks from compounding.
STEP_STRENGTH = 0.5
STEP_STEPS = 4
ZOOM_FRAC = 0.85
HOTSPOT_MIN = 0.2
MAX_TEXT_CHARS = 200
SESSION_TTL = 3600     # seconds idle before a session (and its scene image) is dropped
MAX_SESSIONS = 50
STYLE = "dreamlike painterly landscape, soft volumetric light, oil painting, cohesive palette"

DATA_DIR = Path("data")
SCENE_DIR = Path("var/scenes")
