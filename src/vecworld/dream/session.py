from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from vecworld.dream.labels import Label
from vecworld.scenes.palette import Stats


@dataclass
class View:
    view_id: str
    image_path: Path
    labels: list[Label]
    theme: str
    moods: list[str]
    steps: int


@dataclass
class DreamSession:
    id: str
    seed: int
    theme: str = ""
    moods: list[str] = field(default_factory=list)
    pano: Image.Image | None = None      # the current spot, centred on the direction last walked
    ref: Stats | None = None             # LAB palette every step is pulled toward
    labels: list[Label] = field(default_factory=list)
    steps: int = 0
    ideas: deque[str] = field(default_factory=deque)
    drift: tuple[str, int] | None = None  # (old theme, drift steps done)
    version: int = 0                      # bumped on theme/mood change; a prefetch of another version is stale
    prefetch: tuple[float, int, Future] | None = None  # (yaw, version, future)
    busy: bool = False
    last_used: float = 0.0
    ref_pending: bool = False             # the next spot becomes the palette reference (mood change)
    ideas_theme: str = ""                 # the theme the queued ideas were dreamt for
    refilling: str | None = None          # the theme an ideas refill is running for
