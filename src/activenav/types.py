"""Data types passed between modules. Keep these small and explicit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SensorBundle:
    """Everything the robot is allowed to sense in one control step."""
    t: float                       # simulation time [s]
    enc_left: float                # cumulative left wheel angle, noisy [rad]
    enc_right: float               # cumulative right wheel angle, noisy [rad]
    pan: float                     # pan joint angle, noisy [rad]
    tilt: float                    # tilt joint angle, noisy [rad] (+ = look down)
    image: Optional[np.ndarray] = None   # RGB uint8 (H, W, 3) or None
    # A camera with a delay (real-camera profiles) delivers a frame captured earlier: when it was captured, and the head
    # encoder readings at that moment (the driver time-stamps frames, so they can be matched). None: same as above.
    image_t: Optional[float] = None
    image_pan: Optional[float] = None
    image_tilt: Optional[float] = None

    @property
    def view_pan(self) -> float:
        """Pan angle the image was taken at."""
        return self.pan if self.image_pan is None else self.image_pan

    @property
    def view_tilt(self) -> float:
        """Tilt angle the image was taken at."""
        return self.tilt if self.image_tilt is None else self.image_tilt


@dataclass(frozen=True)
class Command:
    """What the robot is asked to do for the next control step."""
    v: float = 0.0                 # forward speed [m/s]
    w: float = 0.0                 # yaw rate [rad/s], + = turn left
    pan_target: float = 0.0        # [rad]
    tilt_target: float = 0.0       # [rad]


@dataclass(frozen=True)
class MarkerDetection:
    """One ArUco marker found in an image, with its pose in the OpenCV camera frame
    (x right, y down, z forward). The marker frame is x right, y up, z out toward the viewer."""
    marker_id: int
    corners: np.ndarray        # (4, 2) pixel corners
    R_cm: np.ndarray           # (3, 3) rotation, marker frame -> camera frame
    t_cm: np.ndarray           # (3,) marker origin in the camera frame [m]


@dataclass(frozen=True)
class MarkerMeasurement:
    """A marker observation expressed in the robot base frame (planar)."""
    marker_id: int
    range: float               # distance base -> marker centre [m]
    bearing: float             # direction to the marker, relative to robot heading [rad], + = left
    normal_angle: float        # direction of the marker's outward normal, relative to robot heading [rad]
    height: Optional[float] = None   # marker centre height above the floor [m] (None if the sensor does not give it)
