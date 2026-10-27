"""The obstacle-sensing interface and the ground-plane geometry shared by every front-end.

A front-end (camera floor segmentation, a learned segmenter, the synthetic ray-caster used in headless
studies) turns one camera frame into an `ObstacleScan`: for each sampled direction, the floor point where free
floor ends, in the ROBOT BASE frame, and whether that point is an obstacle ("hit") or just the limit of what
could be ranged ("no hit"). The occupancy grid only sees scans, so front-ends are interchangeable.

Never reads simulator ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from activenav.sensing.camera import camera_model, pixel_rays
from activenav.sensing.camera import camera_to_base


@dataclass(frozen=True)
class ObstacleScan:
    origin: np.ndarray                 # camera position in the base frame, (2,) [m]
    points: np.ndarray                 # (n, 2) end of the seen free floor, base frame [m]
    hit: np.ndarray                    # (n,) bool: True = an obstacle is at that point
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (n,) range 1-sigma [m]


class GroundGeometry:
    """Pixel <-> floor point for the pan-tilt camera, from the head geometry and the camera intrinsics (and lens
    distortion, for a real-camera profile)."""

    def __init__(self, cfg: dict):
        self.camera = cam = camera_model(cfg)
        self.W, self.H = cam.width, cam.height
        self.K = cam.K
        self.Kinv = np.linalg.inv(self.K)
        self.head = cam.head
        self.floor_z = -float(cfg["head"].get("base_height", 0.0))      # floor height in the base frame
        self.vfov = cam.vfov
        self.focal = float(self.K[1, 1])

    def cam_height(self) -> float:
        return float(camera_to_base(self.head, 0.0, 0.0)[2, 3] - self.floor_z)

    def pose(self, pan: float, tilt: float):
        T = camera_to_base(self.head, pan, tilt)
        return T[:3, :3], T[:3, 3]

    def pixel_to_floor(self, u, v, pan: float, tilt: float):
        """Base-frame (x, y) of the floor point under pixel(s) (u, v), and a validity mask (ray points down)."""
        R, o = self.pose(pan, tilt)
        u, v = np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float))
        if self.camera.dist is None:
            d = (R @ (self.Kinv @ np.vstack([u, v, np.ones_like(u)]))).T      # (n, 3) ray directions, base frame
        else:
            d = (R @ pixel_rays(self.camera, u, v).T).T                       # lens distortion removed first
        ok = d[:, 2] < -1e-6
        s = np.where(ok, (self.floor_z - o[2]) / np.where(ok, d[:, 2], -1.0), 0.0)
        pts = o[None, :2] + s[:, None] * d[:, :2]
        return pts, ok

    def range_sigma(self, r, pixel_sigma: float):
        """Ground-plane ranging error: a boundary row error of `pixel_sigma` px moves the floor point by about
        r^2 * pixel_sigma / (h * f)."""
        return np.maximum(0.01, pixel_sigma * np.asarray(r) ** 2 / (self.cam_height() * self.focal))

    def near_far(self, tilt: float, max_range: float):
        """Horizontal distance (from the camera) where floor becomes visible at the bottom image row, and where
        it stops being resolvable at the top (horizon). Used by the synthetic front-end."""
        h = self.cam_height()
        lo = tilt + self.vfov / 2
        near = h / np.tan(lo) if lo > 1e-3 else max_range
        hi = tilt - self.vfov / 2
        far = h / np.tan(hi) if hi > 1e-3 else np.inf
        return float(near), float(min(far, max_range))
