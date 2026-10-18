from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from vecworld.scenes.hotspots import Hotspot


@dataclass
class Session:
    id: str
    seed: int
    position: np.ndarray | None = None
    image: Image.Image | None = None
    hotspots: list[Hotspot] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)   # last few centres, oldest first
    busy: bool = False
    last_used: float = 0.0


@dataclass
class Scene:
    scene_id: str
    image_path: Path
    hotspots: list[Hotspot]
    me: tuple[float, float]
    trail: list[tuple[float, float]]
    neighbours: list[tuple[str, float, float]]
    reach_radius: float
