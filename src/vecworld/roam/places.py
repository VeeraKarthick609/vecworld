"""Places for Dream Roam: compass bearings assigned by code, drift toward a neighbour as you walk.

The LLM only names neighbours and in-between scenery; where they lie is decided here.
Bearings are degrees, 0 = north (the world's initial forward), clockwise, in [0, 360).
"""
from __future__ import annotations

from dataclasses import dataclass

DRIFT_SPOTS = 3  # consecutive spots walked toward a neighbour before you arrive there
CONE_DEG = 45.0  # heading within this of a bearing counts as walking toward it
MIN_SEPARATION = 60.0
_STEP_WEIGHT = 0.25  # neighbour's share of the prompt blend per spot of drift (arrival at DRIFT_SPOTS)

FALLBACK_PLACES: dict[str, list[str]] = {
    "forest": ["lake", "village", "mountain", "meadow"],
    "village": ["forest", "farm", "river", "old mill"],
    "city": ["harbour", "market", "park", "suburb"],
    "lake": ["forest", "marsh", "village", "hills"],
    "desert": ["oasis", "canyon", "ruins", "dunes"],
    "mountain": ["pine forest", "glacier", "valley", "cave"],
    "coast": ["lighthouse", "fishing village", "cliffs", "dunes"],
    "ruins": ["forest", "hills", "river", "village"],
    "meadow": ["forest", "farm", "stream", "hills"],
    "river": ["village", "marsh", "bridge", "forest"],
    "farm": ["village", "meadow", "orchard", "road"],
    "marsh": ["lake", "forest", "village", "river"],
    "cave": ["mountain", "underground lake", "forest", "ruins"],
    "harbour": ["city", "coast", "lighthouse", "market"],
    "valley": ["mountain", "river", "village", "forest"],
    "island": ["coast", "lagoon", "ruins", "reef"],
    "castle": ["village", "forest", "hills", "river"],
    "market": ["city", "village", "farm", "road"],
}
GENERIC_NEIGHBOURS = ["forest", "village", "river", "hills"]
GENERIC_SCENERY = ("a winding path", "a quiet clearing", "a weathered signpost")


def fallback_neighbours(place: str) -> list[str]:
    key = place.strip().lower()
    if key in FALLBACK_PLACES:
        return list(FALLBACK_PLACES[key])
    for seed, names in FALLBACK_PLACES.items():  # "old forest" -> forest
        if seed in key.split():
            return list(names)
    return [n for n in GENERIC_NEIGHBOURS if n != key]


@dataclass(frozen=True)
class Neighbour:
    name: str
    bearing: float
    scenery: tuple[str, ...]


@dataclass(frozen=True)
class DriftState:
    weights: list[tuple[str, float]]  # prompt weights: [(current, w), (neighbour, 1 - w)]
    scenery: str  # in-between description, "" when not drifting
    arrived: str | None


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def assign_bearings(names: list[str], came_from: tuple[str, float] | None, rng,
                    scenery: dict[str, tuple[str, ...]] | None = None) -> list[Neighbour]:
    """Spread `names` around the compass at least 60 degrees apart. The came-from place (not in `names`)
    sits exactly at its given bearing, behind where you arrived. `scenery` maps name -> phrases;
    names without any get generic in-between scenery."""
    scenery = scenery or {}

    def make(name: str, bearing: float) -> Neighbour:
        sc = tuple(scenery.get(name, ())[:3])
        sc += GENERIC_SCENERY[len(sc):3]
        return Neighbour(name, round(bearing % 360, 6), sc)

    out: list[Neighbour] = []
    k = len(names)
    if came_from is not None:
        base = came_from[1] % 360
        out.append(make(came_from[0], base))
        step = 360 / (k + 1)
        # One shared jitter keeps spacing at `step`; limited so no gap, incl. to the came-from, drops under 60.
        slack = max(0.0, (step - MIN_SEPARATION) / 2)
        jitter = rng.uniform(-slack, slack)
        out += [make(n, base + step * (i + 1) + jitter) for i, n in enumerate(names)]
    elif k:
        step = 360 / k
        start = rng.uniform(0, 360)
        out = [make(n, start + step * i) for i, n in enumerate(names)]
    return out


class Journey:
    def __init__(self, place: str, neighbours: list[Neighbour], came_from: str | None = None):
        self.place = place
        self.neighbours = list(neighbours)
        self.came_from = came_from
        self.drift: dict[str, int] = {n.name: 0 for n in self.neighbours}

    def step(self, heading: float) -> DriftState:
        """One spot walked on `heading`. Drift +1 for the neighbour within 45 degrees, others reset."""
        toward = None
        for n in self.neighbours:
            if toward is None and _angle_diff(heading, n.bearing) <= CONE_DEG:
                toward = n
                self.drift[n.name] += 1
            else:
                self.drift[n.name] = 0
        if toward is None:
            return DriftState([(self.place, 1.0)], "", None)
        d = self.drift[toward.name]
        if d >= DRIFT_SPOTS:
            return DriftState([(toward.name, 1.0)], "", toward.name)
        w = _STEP_WEIGHT * d
        return DriftState([(self.place, 1 - w), (toward.name, w)], toward.scenery[min(d, len(toward.scenery)) - 1], None)

    def arrive(self, name: str, new_names: list[str], rng,
               scenery: dict[str, tuple[str, ...]] | None = None) -> None:
        """Move to `name`; the place you left sits behind your arrival direction."""
        old = self.place
        heading_bearing = next(n.bearing for n in self.neighbours if n.name == name)
        self.place, self.came_from = name, old
        self.neighbours = assign_bearings([n for n in new_names if n not in (old, name)],
                                          (old, (heading_bearing + 180) % 360), rng, scenery)
        self.drift = {n.name: 0 for n in self.neighbours}
