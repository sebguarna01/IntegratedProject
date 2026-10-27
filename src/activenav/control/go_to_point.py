"""Simple go-to-point controller used for the Week 1 odometry baseline.

It only ever sees the pose it is given (an estimate). It is replaced by the LQR
tracker later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activenav.kinematics import wrap_angle


@dataclass(frozen=True)
class GoToPointGains:
    k_v: float = 1.0       # speed per metre of distance
    k_w: float = 2.0       # yaw rate per radian of heading error
    v_max: float = 0.2     # m/s
    w_max: float = 1.0     # rad/s
    pos_tol: float = 0.05  # m


def go_to_point(pose, goal, gains: GoToPointGains = GoToPointGains()):
    """Return (v, w, reached). Turns toward the goal and slows when misaligned."""
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    dist = float(np.hypot(dx, dy))
    if dist < gains.pos_tol:
        return 0.0, 0.0, True

    err = float(wrap_angle(np.arctan2(dy, dx) - pose[2]))
    w = float(np.clip(gains.k_w * err, -gains.w_max, gains.w_max))
    v = float(np.clip(gains.k_v * dist, 0.0, gains.v_max)) * max(0.0, np.cos(err))
    return v, w, False
