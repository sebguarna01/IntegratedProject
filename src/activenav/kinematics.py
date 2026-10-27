"""Differential-drive kinematics shared by the simulator wrapper, odometry and EKF.

Conventions: x forward, y left, theta counter-clockwise. A positive wheel angle
rolls the wheel forward. With the left wheel on +y: w > 0 means the right wheel
is faster than the left (turn left).
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def wheel_speeds_from_cmd(v: float, w: float, wheel_radius: float, track_width: float):
    """(v, w) -> (left, right) wheel angular speeds [rad/s]."""
    wl = (v - w * track_width / 2.0) / wheel_radius
    wr = (v + w * track_width / 2.0) / wheel_radius
    return wl, wr


def cmd_from_wheel_speeds(wl: float, wr: float, wheel_radius: float, track_width: float):
    """(left, right) wheel angular speeds [rad/s] -> (v, w)."""
    v = wheel_radius * (wl + wr) / 2.0
    w = wheel_radius * (wr - wl) / track_width
    return v, w


def unicycle_step(pose, v: float, w: float, dt: float):
    """One midpoint-integration step of the unicycle model. pose = (x, y, theta)."""
    x, y, th = pose
    th_mid = th + 0.5 * w * dt
    return np.array([
        x + v * np.cos(th_mid) * dt,
        y + v * np.sin(th_mid) * dt,
        float(wrap_angle(th + w * dt)),
    ])
