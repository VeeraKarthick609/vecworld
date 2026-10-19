"""Dream Roam session state and the view of one spot sent to the player."""
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from vecworld.roam.places import Journey, Neighbour
from vecworld.roam.spot import Spot
from vecworld.scenes.palette import Stats


@dataclass
class SpotView:
    spot_id: str
    image_path: Path        # JPEG panorama
    depth_path: Path        # 8-bit PNG 360-degree depth, 255 = near
    heading: float          # world bearing of the spot's forward column
    scale: float            # parent-to-child ratio (1 for the first spot)
    distance: float         # how far ahead of the parent's centre, in the parent's frame (0 for the first)
    place: str
    neighbours: list[Neighbour]
    arrived: str | None     # the place this spot arrived at, if it did


@dataclass(frozen=True)
class Visit:
    """A spot the session can step back into, with the journey state it had when it was last the current spot."""
    spot: Spot
    spot_stats: Stats
    journey: Journey
    ref: Stats
    paths: tuple[Path, Path]
    parent_id: str | None       # the spot it was built from (None for the first)


@dataclass(frozen=True)
class Predream:
    """One pre-dreamed direction: built from `spot_id` in world `version` toward `heading`."""
    heading: float
    version: int
    spot_id: str
    future: Future


@dataclass
class RoamSession:
    id: str
    seed: int
    moods: list[str] = field(default_factory=list)
    spot: Spot | None = None                 # the spot the player is in
    spot_stats: Stats | None = None          # its LAB palette
    journey: Journey | None = None           # never mutated in place: builds step a copy, edits replace it
    ref: Stats | None = None                 # the place's palette reference
    ref_pending: bool = False                # mood change: the next spot is painted fresh and becomes the ref
    steps: int = 0
    version: int = 0                         # bumped on mood/theme/neighbour changes; older pre-dreams are stale
    hint: float = 0.0                        # the heading the set is pre-dreamed around
    hints: int = 0                           # heading hints received, so a move doesn't override a newer one
    prefetch: dict[str, Predream] = field(default_factory=dict)   # the pre-dreamed set, by queue key
    pending: list[Callable[[Journey], Journey]] = field(default_factory=list)  # journey edits during a move
    visited: list[str] = field(default_factory=list)
    names: tuple[str, dict] | None = None   # (place, fallback slot -> LLM place): re-applied to every committed journey
    typed: set[str] = field(default_factory=set)   # places the player typed (lowercase): never renamed by the LLM
    places_for: str | None = None            # the place whose neighbours the LLM has answered (or failed) for
    refilling: str | None = None             # the place a places refill is running for
    busy: bool = False
    last_used: float = 0.0
    files: list[tuple[Path, Path]] = field(default_factory=list)  # (jpg, png) per spot still on disk, oldest first
    visits: dict[str, Visit] = field(default_factory=dict)   # by spot id: the spots whose files are still on disk
