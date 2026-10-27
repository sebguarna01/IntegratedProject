"""Frames that arrive late (a real camera's latency): move what was measured in them into the robot's current base frame.

A frame captured at t_cap is measured in the base frame of t_cap. By the time it is fused the robot has moved; its own
wheel odometry says how much, so the measurements are carried over to the base frame of now (standard motion
compensation of delayed measurements). Uses the robot's odometry only, never simulator truth.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from activenav.kinematics import wrap_angle


class OdometryHistory:
    """Recent dead-reckoned poses (t, x, y, theta) in an arbitrary odometry frame, one per control step."""

    def __init__(self, horizon_s: float = 2.0, dt: float = 0.02):
        self._poses = deque(maxlen=int(np.ceil(horizon_s / dt)) + 2)
        self._x = np.zeros(3)

    def reset(self, t: float):
        self._poses.clear()
        self._x = np.zeros(3)
        self._poses.append((float(t), self._x.copy()))

    def add(self, t: float, ds: float, dtheta: float):
        x, y, th = self._x
        mid = th + 0.5 * dtheta
        self._x = np.array([x + ds * np.cos(mid), y + ds * np.sin(mid), wrap_angle(th + dtheta)])
        self._poses.append((float(t), self._x.copy()))

    def motion_since(self, t: float) -> np.ndarray:
        """Pose of the robot now, in its base frame at time t: (dx, dy, dtheta). Zero if t is not before the history."""
        if not self._poses:
            return np.zeros(3)
        then = min(self._poses, key=lambda p: abs(p[0] - t))[1]
        now = self._poses[-1][1]
        c, s = np.cos(then[2]), np.sin(then[2])
        d = now[:2] - then[:2]
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], wrap_angle(now[2] - then[2])])


def _to_now(p: np.ndarray, motion) -> np.ndarray:
    """Points (n, 2) in the base frame at capture time -> base frame now (`motion` = now in the capture frame)."""
    dx, dy, dth = motion
    c, s = np.cos(dth), np.sin(dth)
    q = np.atleast_2d(p) - np.array([dx, dy])
    return np.column_stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1]])


def marker_to_now(z, motion):
    """A MarkerMeasurement taken at capture time, expressed in the base frame of now."""
    p = _to_now(np.array([z.range * np.cos(z.bearing), z.range * np.sin(z.bearing)]), motion)[0]
    return replace(z, range=float(np.hypot(*p)), bearing=float(np.arctan2(p[1], p[0])),
                   normal_angle=float(wrap_angle(z.normal_angle - motion[2])))


def scan_to_now(scan, motion):
    """An ObstacleScan taken at capture time, expressed in the base frame of now."""
    pts = scan.points if len(scan.points) == 0 else _to_now(scan.points, motion)
    return replace(scan, origin=_to_now(scan.origin, motion)[0], points=np.asarray(pts, float).reshape(-1, 2))
