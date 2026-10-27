"""2D geometry helpers shared by the synthetic sensor, the gaze policy and the planner.

Obstacles are circles (pillars: dicts with x, y, r) or axis-aligned rectangles (wall blocks: dicts with
x0, x1, y0, y1). Pure maths, no simulator access.
"""
from __future__ import annotations

import numpy as np


def segment_hits_circle(p, q, c, r: float) -> bool:
    """True if the segment p->q passes within r of centre c (exact segment-against-circle test)."""
    p, q, c = np.asarray(p, float), np.asarray(q, float), np.asarray(c, float)
    d = q - p
    L2 = float(d @ d)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - c)) < r
    t = float(np.clip(((c - p) @ d) / L2, 0.0, 1.0))
    return float(np.linalg.norm(p + t * d - c)) < r


def is_rect(o) -> bool:
    return "x0" in o


def obstacle_sdf(o, X, Y):
    """Distance from points (X, Y) (scalars or arrays) to the obstacle's surface; negative inside."""
    if is_rect(o):
        cx, cy = 0.5 * (o["x0"] + o["x1"]), 0.5 * (o["y0"] + o["y1"])
        hx, hy = 0.5 * (o["x1"] - o["x0"]), 0.5 * (o["y1"] - o["y0"])
        qx, qy = np.abs(np.asarray(X) - cx) - hx, np.abs(np.asarray(Y) - cy) - hy
        return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
    return np.hypot(np.asarray(X) - o["x"], np.asarray(Y) - o["y"]) - o["r"]


def segment_hits_rect(p, q, o) -> bool:
    """True if the segment p->q passes through the rectangle (slab test)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t0, t1 = 0.0, 1.0
    for k, (lo, hi) in enumerate(((o["x0"], o["x1"]), (o["y0"], o["y1"]))):
        if abs(d[k]) < 1e-12:
            if p[k] < lo or p[k] > hi:
                return False
        else:
            ta, tb = (lo - p[k]) / d[k], (hi - p[k]) / d[k]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return False
    return True


def segment_blocked(p, q, obstacles) -> bool:
    return any(segment_hits_rect(p, q, o) if is_rect(o) else segment_hits_circle(p, q, (o["x"], o["y"]), o["r"])
               for o in obstacles)


def marker_fully_visible(cam_xy, marker: dict, obstacles, half_width: float = 0.15) -> bool:
    """No obstacle blocks the line of sight to the marker centre or to either edge of its black square.

    ArUco detection needs the whole square, so a pillar covering one edge counts as occluded.
    """
    if not obstacles:
        return True
    n = marker["normal_angle"]
    tangent = np.array([-np.sin(n), np.cos(n)])
    centre = np.array([marker["x"], marker["y"]])
    return not any(segment_blocked(cam_xy, centre + s * half_width * tangent, obstacles) for s in (-1.0, 0.0, 1.0))
