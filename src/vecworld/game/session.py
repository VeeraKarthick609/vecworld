from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from vecworld.scenes.hotspots import Hotspot


@dataclass
class Scene:
    scene_id: str
    image_path: Path
    hotspots: list[Hotspot]
    me: tuple[float, float]
    trail: list[tuple[float, float]]
    neighbours: list[tuple[str, float, float]]
    reach_radius: float
    centre: str


@dataclass(frozen=True)
class Snapshot:
    """Everything needed to stand where you stood before a move (the image is reloaded from disk)."""
    scene: Scene
    position: np.ndarray
    heading: np.ndarray | None
    recent: list[str]


@dataclass
class Session:
    id: str
    seed: int
    position: np.ndarray | None = None
    image: Image.Image | None = None
    hotspots: list[Hotspot] = field(default_factory=list)
    trail: list[tuple[float, float]] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)   # last few centres, oldest first
    heading: np.ndarray | None = None                 # unit tangent of the last move at `position`
    scene: Scene | None = None
    history: list[Snapshot] = field(default_factory=list)
    busy: bool = False
    last_used: float = 0.0
