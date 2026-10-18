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
HOTSPOT_MIN = 0.2
MAX_HOTSPOT_AREA = 0.5     # a box covering more of the frame is the whole scene, not an object
MAX_HOTSPOT_IOU = 0.6      # a box this similar to one already offered would hide it
CENTRE_BOX = (0.35, 0.35, 0.3, 0.3)  # the pinned centre's box when CLIPSeg can't place it
MAX_TEXT_CHARS = 200
SESSION_TTL = 3600     # seconds idle before a session (and its scene image) is dropped
MAX_SESSIONS = 50
STYLE = "dreamlike painterly landscape, soft volumetric light, oil painting, cohesive palette"

DATA_DIR = Path("data")
SCENE_DIR = Path("var/scenes")
