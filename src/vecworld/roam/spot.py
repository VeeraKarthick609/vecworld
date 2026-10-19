"""Spots for Dream Roam: one 360-degree painted panorama with depth, chained to its parent by heading and scale.

Headings are world bearings in degrees, 0 = the first spot's forward, clockwise; a spot's heading is
the bearing of its forward column (u = 0.5). Scale and distance are relative to the parent only, never
chained: depth is relative and re-normalised per panorama, so a product of ratios would drift along a
journey (shrinking the scenery around a fixed-size character).
"""
import uuid
from dataclasses import dataclass

import numpy as np
from PIL import Image

from vecworld.config import DOLLY, FAR, NEAR, PALETTE_STRENGTH
from vecworld.dream.painter import Painter, Prompt
from vecworld.dream.pano import dolly, dolly_src, roll_to, yaw_to_u
from vecworld.roam.depth import DepthEstimator
from vecworld.scenes.palette import Stats, match_palette

VIEW_HALF = 30.0                    # degrees either side of the shared view direction compared by align_scale
MIN_STEP, MAX_STEP = 0.25, 4.0      # a parent-to-child rescale outside this is a bad depth read, not geometry
TYPICAL_RADIUS = 1 / (0.5 * (1 / NEAR - 1 / FAR) + 1 / FAR)   # the viewer radius of mid depth (0.5)


@dataclass(frozen=True)
class Spot:
    """One spot of the chain.

    scale: the parent-to-child ratio, i.e. the multiplier for this spot's radii (from its depth map) that
    lines it up with its parent's frame (1 for the first spot). distance: how far ahead of the parent's centre,
    along heading and in the parent's frame, this spot's centre sits. The viewer re-roots on every spot: during
    the transition it places the child at distance with its radii × scale, then treats the child as scale 1.
    """

    id: str
    image: Image.Image
    depth: np.ndarray       # (H, W) float 0..1, 1 = near
    heading: float
    scale: float
    distance: float = 0.0   # 0 for the first spot


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
    return Spot(id=uuid.uuid4().hex[:12], image=img, depth=depth.depth360(img), heading=0.0, scale=1.0)


def next_spot(painter: Painter, depth: DepthEstimator, parent: Spot, exit_heading: float, base: Prompt,
              focus: Prompt | None, seed: int, ref: Stats | None, bg_strength: float | None = None) -> Spot:
    """Dream the spot one dolly step from parent toward exit_heading: roll, dolly, repaint ahead, palette, depth."""
    rel = wrap180(exit_heading - parent.heading)
    prev = dolly(roll_to(parent.image, yaw_to_u(rel)))
    img = painter.step(prev, base, focus, 0.5, seed, bg_strength=bg_strength)
    if ref is not None:
        img = match_palette(img, ref, PALETTE_STRENGTH)
    d = np.clip(np.nan_to_num(np.asarray(depth.depth360(img), np.float32), nan=0.5, posinf=1.0, neginf=0.0), 0, 1)
    heading = (parent.heading + rel) % 360.0
    child = Spot(id=uuid.uuid4().hex[:12], image=img, depth=d, heading=heading, scale=1.0)
    scale = align_scale(parent, child, rel)
    distance = dolly_distance(DOLLY, _parent_view(parent, rel, DOLLY))
    if not np.isfinite(distance):
        distance = dolly_distance(DOLLY)
    return Spot(id=child.id, image=img, depth=d, heading=heading, scale=scale, distance=distance)
