"""Spots for Dream Roam: one 360-degree painted panorama with depth, chained to its parent by heading and scale.

Headings are world bearings in degrees, 0 = the first spot's forward, clockwise; a spot's heading is
the bearing of its forward column (u = 0.5). Scale and distance are relative to the parent only, never
chained: depth is relative and re-normalised per panorama, so a product of ratios would drift along a
journey (shrinking the scenery around a fixed-size character).
"""
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from vecworld.config import (DOLLY, FAR, FOCUS_HALF, FOCUS_HALF_MAX, FOCUS_MAG, LIMIT_BINS, MIN_WALK, NEAR,
                             OBSTACLE_H, PALETTE_STRENGTH, WALK_LIMIT, WALK_Z_MAX, WALK_Z_MIN, WALL_FRACTION,
                             WALL_HALF)
from vecworld.dream.gpu import check_stop
from vecworld.dream.painter import Painter, Prompt
from vecworld.dream.pano import dolly, dolly_src, roll_to, yaw_to_u
from vecworld.roam.depth import DepthEstimator
from vecworld.scenes.palette import Stats, match_palette

VIEW_HALF = 30.0                    # degrees either side of the shared view direction compared by align_scale
MIN_STEP, MAX_STEP = 0.25, 4.0      # a parent-to-child rescale outside this is a bad depth read, not geometry
TYPICAL_RADIUS = 1 / (0.5 * (1 / NEAR - 1 / FAR) + 1 / FAR)   # the viewer radius of mid depth (0.5)
GRID = (128, 512)                   # the viewer's depth mesh grid (rows, columns): walk limits are judged on it
GROUND_ROWS = 12 / 128              # the bottom fraction of rows the viewer fits its ground plane from
SNAP_FROM = 0.8                     # the viewer slides points this close to the ground plane onto it


@dataclass(frozen=True)
class Spot:
    """One spot of the chain.

    scale: the parent-to-child ratio, i.e. the multiplier for this spot's radii (from its depth map) that
    lines it up with its parent's frame (1 for the first spot). distance: how far ahead of the parent's centre,
    along heading and in the parent's frame, this spot's centre sits. The viewer re-roots on every spot: during
    the transition it places the child at distance with its radii × scale, then treats the child as scale 1.
    limits: the walkable radius around this spot's centre (its own units) every 360 / LIMIT_BINS degrees from its
    forward, clockwise (see walk_limits); None when the spot was made by hand (spot_limits works it out).
    """

    id: str
    image: Image.Image
    depth: np.ndarray       # (H, W) float 0..1, 1 = near
    heading: float
    scale: float
    distance: float = 0.0   # 0 for the first spot
    limits: np.ndarray | None = field(default=None, compare=False)


def radius(d) -> np.ndarray:
    """The viewer's radius for 0..1 inverse depth (NEAR at 1, FAR at 0)."""
    return 1 / (np.asarray(d, np.float64) * (1 / NEAR - 1 / FAR) + 1 / FAR)


def wrap180(deg: float) -> float:
    """Degrees normalised to (-180, 180]."""
    r = deg % 360.0
    return r - 360.0 if r > 180.0 else r


def _median_radius(depth: np.ndarray, u: float, half_u: float, half_rows: float = 0.5) -> float:
    """Median radius over columns within half_u of fraction u (wrapping) and rows within half_rows·H of the horizon."""
    h, w = depth.shape
    cu = (np.arange(w) + 0.5) / w
    d = np.abs(cu - u) % 1.0
    cols = np.minimum(d, 1.0 - d) <= half_u
    y = (np.arange(h) + 0.5) / h - 0.5
    rows = np.abs(y) <= half_rows
    return float(np.median(radius(depth[np.ix_(rows, cols)])))


def _parent_view(parent: Spot, rel_heading: float, z: float) -> float:
    """Median radius (parent's own units) of what the child's forward ±VIEW_HALF window shows after the dolly:
    the parent's columns that dolly into it, and its rows within 1/z of the horizon (the dolly zooms ×z ahead)."""
    half_u = 0.5 - float(dolly_src(0.5 - VIEW_HALF / 360, z))
    return _median_radius(parent.depth, yaw_to_u(rel_heading), half_u, 0.5 / z)


def dolly_distance(z: float = DOLLY, radius: float = TYPICAL_RADIUS) -> float:
    """World distance the dolly walks: what was `radius` ahead looks z times bigger, i.e. is now radius/z away."""
    return radius * (1 - 1 / z)


def walk_z(r: float, walked: float) -> float:
    """The dolly that walks `walked` toward scenery `r` away: it is then r - walked away, i.e. z = r / (r - walked)
    times bigger. Clamped to WALK_Z_MIN..WALK_Z_MAX (WALK_Z_MAX when walked reaches r)."""
    if not (np.isfinite(r) and r > 0 and np.isfinite(walked)):
        return WALK_Z_MIN
    if walked >= r:
        return WALK_Z_MAX
    return float(np.clip(r / (r - walked), WALK_Z_MIN, WALK_Z_MAX))


def _dolly_for(parent: Spot, rel_heading: float, walked: float) -> tuple[float, float]:
    """(z, R): the dolly that walks `walked` toward rel_heading and the parent's median radius R that way. R is read
    over what the child's forward view shows after the dolly, which depends on z, so the two are iterated together."""
    z = DOLLY
    for _ in range(6):
        nz = walk_z(_parent_view(parent, rel_heading, z), walked)
        if abs(nz - z) < 1e-4:
            break
        z = nz
    return z, _parent_view(parent, rel_heading, z)


def focus_half(z: float) -> float:
    """Half-width (fraction of the panorama) of the band ahead repainted from scratch after a dolly by z: out to
    where the dolly magnifies by FOCUS_MAG (mag(θ) = 1 / (1 − (1 − 1/z)·cos θ)), within
    FOCUS_HALF..FOCUS_HALF_MAX."""
    c = (1 - 1 / FOCUS_MAG) / (1 - 1 / z) if z > 1 else 2.0
    if c >= 1:
        return FOCUS_HALF
    return float(np.clip(np.degrees(np.arccos(c)) / 360, FOCUS_HALF, FOCUS_HALF_MAX))


def row_angles(h: int) -> np.ndarray:
    """The height factor (0.5 − v)·π/2 of each of h rows on the unit cylinder (v at the row's centre)."""
    return (0.5 - (np.arange(h) + 0.5) / h) * (np.pi / 2)


def _grid(depth: np.ndarray) -> np.ndarray:
    """depth sampled down to at most the viewer's mesh grid (nearest rows and columns)."""
    h, w = depth.shape
    rows = ((np.arange(min(h, GRID[0])) + 0.5) * h / min(h, GRID[0])).astype(int)
    cols = ((np.arange(min(w, GRID[1])) + 0.5) * w / min(w, GRID[1])).astype(int)
    return np.asarray(depth, np.float64)[np.ix_(rows, cols)]


def walk_limits(depth: np.ndarray) -> np.ndarray:
    """How far the player may walk from the spot's centre in each of LIMIT_BINS directions (0 = forward,
    clockwise, in the spot's own units): WALK_LIMIT, but stopping WALL_FRACTION of the way to scenery standing in
    that direction (the WALL_HALF-degree window's lower quartile of the nearest point per column more than
    OBSTACLE_H above the ground, as the viewer's mesh puts it), and never further than a WALK_Z_MAX dolly can
    represent that way (so the next spot lands where the player left); at least MIN_WALK."""
    d = np.nan_to_num(_grid(depth), nan=0.5)
    h, w = d.shape
    r, a = radius(d), row_angles(h)[:, None]
    ground = float(np.median((a * r)[h - max(1, round(h * GROUND_ROWS)):]))
    # The viewer slides ground-level points down their ray onto its ground plane (spotmesh.worker.js): so here.
    rg = np.where(a < 0, np.minimum(FAR, ground / np.minimum(a, -1e-9)), r)     # rows above: left as they are
    f = r / rg
    t = np.clip((f - SNAP_FROM) / (1 - SNAP_FROM), 0, 1)
    r = np.where(f >= 1, rg, r + (rg - r) * t * t * (3 - 2 * t) * (f > SNAP_FROM))
    wall = np.where(a * r - ground > OBSTACLE_H, r, FAR).min(axis=0)
    cu = (np.arange(w) + 0.5) / w
    half_u = 0.5 - float(dolly_src(0.5 - VIEW_HALF / 360, WALK_Z_MAX))   # as _parent_view at WALK_Z_MAX
    out = np.empty(LIMIT_BINS)
    for k in range(LIMIT_BINS):
        rel = k * 360.0 / LIMIT_BINS
        du = np.abs(cu - yaw_to_u(rel)) % 1.0
        near_wall = np.percentile(wall[np.minimum(du, 1 - du) <= WALL_HALF / 360], 25)
        reach = dolly_distance(WALK_Z_MAX, _median_radius(d, yaw_to_u(rel), half_u, 0.5 / WALK_Z_MAX))
        lim = min(WALK_LIMIT, WALL_FRACTION * near_wall, reach)
        out[k] = np.clip(lim, MIN_WALK, WALK_LIMIT) if np.isfinite(lim) else MIN_WALK
    return out


def limit_at(limits: np.ndarray, rel_heading: float) -> float:
    """The walkable radius toward rel_heading (degrees from the spot's forward), linear between the samples."""
    n = len(limits)
    x = (rel_heading % 360.0) / (360.0 / n)
    i = int(np.floor(x)) % n
    t = x - np.floor(x)
    return float((1 - t) * limits[i] + t * limits[(i + 1) % n])


def spot_limits(spot: Spot) -> np.ndarray:
    return spot.limits if spot.limits is not None else walk_limits(spot.depth)


def align_scale(parent: Spot, child: Spot, rel_heading: float, z: float = DOLLY) -> float:
    """The parent-to-child ratio: the multiplier for the child's radii so its forward view meets the parent's
    view toward rel_heading, in the parent's frame (parent.scale does not enter it).

    The parent sees that view at median radius R; after the dolly the child stands dolly_distance(z, R) further
    on, so the same surfaces are R/z away. The child's forward median radius times the ratio equals that.
    Clamped to [MIN_STEP, MAX_STEP]; 1 when either view has no finite depth.
    """
    ratio = _parent_view(parent, rel_heading, z) / (z * _median_radius(child.depth, 0.5, VIEW_HALF / 360))
    return float(np.clip(ratio, MIN_STEP, MAX_STEP)) if np.isfinite(ratio) else 1.0


def first_spot(painter: Painter, depth: DepthEstimator, prompt: Prompt, seed: int) -> Spot:
    img = painter.panorama(prompt, seed)
    d = depth.depth360(img)
    return Spot(id=uuid.uuid4().hex[:12], image=img, depth=d, heading=0.0, scale=1.0, limits=walk_limits(d))


def next_spot(painter: Painter, depth: DepthEstimator, parent: Spot, exit_heading: float, base: Prompt,
              focus: Prompt | None, seed: int, ref: Stats | None, bg_strength: float | None = None,
              walked: float | None = None) -> Spot:
    """Dream the spot the player reaches by walking `walked` from parent's centre toward exit_heading (parent's
    units; None = to the parent's walkable radius that way, where a pre-dream expects the exit): roll, dolly by
    z = R / (R − walked) (walk_z, clamped), repaint ahead, palette, depth. Its distance is the walk the clamped
    dolly represents, which the viewer places it at."""
    rel = wrap180(exit_heading - parent.heading)
    if walked is None or not np.isfinite(walked):
        walked = limit_at(spot_limits(parent), rel)
    z, r = _dolly_for(parent, rel, max(0.0, float(walked)))
    prev = dolly(roll_to(parent.image, yaw_to_u(rel)), z)
    img = painter.step(prev, base, focus, 0.5, seed, bg_strength=bg_strength, focus_half=focus_half(z))
    check_stop()        # a pre-dream nobody wants any more ends here, not after the depth pass
    if ref is not None:
        img = match_palette(img, ref, PALETTE_STRENGTH)
    d = np.clip(np.nan_to_num(np.asarray(depth.depth360(img), np.float32), nan=0.5, posinf=1.0, neginf=0.0), 0, 1)
    heading = (parent.heading + rel) % 360.0
    child = Spot(id=uuid.uuid4().hex[:12], image=img, depth=d, heading=heading, scale=1.0)
    scale = align_scale(parent, child, rel, z)
    distance = dolly_distance(z, r)
    if not np.isfinite(distance):
        distance = dolly_distance(z)
    return Spot(id=child.id, image=img, depth=d, heading=heading, scale=scale, distance=distance,
                limits=walk_limits(d))
