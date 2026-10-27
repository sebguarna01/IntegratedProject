"""Vertical visibility of a wall marker: does the whole black square fit in the camera's vertical field of view?

The camera sits ~0.19 m above the floor and looks along the head tilt (+ = down); markers hang at different heights.
A marker too high for the current tilt is invisible when the robot is close to it, however well the robot points
its head sideways. Pure geometry; shared by the dock controller, the C-info gaze and the synthetic sensor.
"""
from __future__ import annotations

import numpy as np

HALF_SQUARE = 0.15                 # the ArUco square is 0.30 m: its edges are +-0.15 m from the plate centre
TILT_MIN, TILT_MAX = -0.55, 0.95   # head tilt limits with a little room inside the joint range [rad]


def elevations(h_cam: float, z, r):
    """(bottom, top) elevation angles [rad, + = up] of the marker square seen from the camera at horizontal distance r."""
    r = np.maximum(np.asarray(r, float), 1e-3)
    return np.arctan2(np.asarray(z, float) - HALF_SQUARE - h_cam, r), np.arctan2(np.asarray(z, float) + HALF_SQUARE - h_cam, r)


def vertically_visible(h_cam: float, tilt, r, z, vfov: float, margin: float = 0.05):
    """True where the whole square is inside the vertical view (looking `tilt` rad down), `margin` rad from the edges."""
    e_bot, e_top = elevations(h_cam, z, r)
    e0 = -np.asarray(tilt, float)
    return (e_bot >= e0 - vfov / 2 + margin) & (e_top <= e0 + vfov / 2 - margin)


def tilt_interval(h_cam: float, r: float, z: float, vfov: float, margin: float = 0.05):
    """(lo, hi) head tilts that keep the whole square in view at distance r; lo > hi if it cannot fit."""
    e_bot, e_top = elevations(h_cam, z, r)
    return float(-e_bot - vfov / 2 + margin), float(vfov / 2 - margin - e_top)


def tilt_for_marker(h_cam: float, r: float, z: float, vfov: float, prefer: float, centre: bool = False,
                    margin: float = 0.05) -> float:
    """Head tilt that keeps the marker in view: `prefer` if that works, else the nearest tilt that does; with
    centre=True, the tilt that puts the marker in the middle of the image."""
    e_bot, e_top = elevations(h_cam, z, r)
    if centre:
        t = -0.5 * float(e_bot + e_top)
    else:
        lo, hi = tilt_interval(h_cam, r, z, vfov, margin)
        t = -0.5 * float(e_bot + e_top) if lo > hi else float(np.clip(prefer, lo, hi))
    return float(np.clip(t, TILT_MIN, TILT_MAX))
