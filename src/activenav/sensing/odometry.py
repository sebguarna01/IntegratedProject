"""Wheel odometry: the encoder model (what the wheels report, with the per-trial systematic errors) and dead reckoning from it."""
from __future__ import annotations

import numpy as np

from activenav.kinematics import wrap_angle


# -------------------- sensing/odometry.py --------------------
# Wheel-encoder error model (pure numpy, no simulator dependency).
class EncoderErrorModel:
    def __init__(self, noise_cfg: dict, rng: np.random.Generator):
        self.mult_std = float(noise_cfg.get("encoder_mult_std", 0.0))
        self.add_std = float(noise_cfg.get("encoder_add_std", 0.0))
        self.ticks = int(noise_cfg.get("encoder_ticks_per_rev", 0))   # 0 = no quantisation
        self.scale_std = float(noise_cfg.get("wheel_scale_std", 0.0))
        self.track_std = float(noise_cfg.get("track_width_error_std", 0.0))
        self.rng = rng
        self.reset(np.zeros(2))

    def reset(self, true_angles, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.scale = 1.0 + self.rng.normal(0.0, self.scale_std, 2)     # [left, right]
        self.track_err = float(self.rng.normal(0.0, self.track_std))
        self._prev = np.array(true_angles, dtype=float)
        self._cont = self._prev.copy()

    def measure(self, true_angles) -> np.ndarray:
        """True cumulative wheel angles [rad] -> reported cumulative angles [rad]."""
        true = np.asarray(true_angles, dtype=float)
        inc = (true - self._prev) * self.scale
        self._prev = true.copy()

        common = inc.mean()                                            # forward component
        diff = 0.5 * (inc[1] - inc[0]) / (1.0 + self.track_err)       # yaw component
        inc = np.array([common - diff, common + diff])

        inc = inc * (1.0 + self.rng.normal(0.0, self.mult_std, 2)) \
            + self.rng.normal(0.0, self.add_std, 2)
        self._cont = self._cont + inc

        if self.ticks > 0:
            q = 2.0 * np.pi / self.ticks
            return np.round(self._cont / q) * q
        return self._cont.copy()

    def errors(self) -> dict:
        """The hidden per-trial systematic errors. For evaluation and reporting only."""
        return {"scale_left": float(self.scale[0] - 1.0),
                "scale_right": float(self.scale[1] - 1.0),
                "track_error": self.track_err}


# -------------------- sensing/odometry.py --------------------
# Wheel odometry (dead reckoning) from encoder angles.
class WheelOdometry:
    """Integrates cumulative wheel angles into a pose estimate (x, y, theta)."""

    def __init__(self, wheel_radius: float, track_width: float):
        self.R = wheel_radius
        self.L = track_width
        self.pose = np.zeros(3)
        self._prev = None
        self.last_ds = 0.0
        self.last_dtheta = 0.0

    def reset(self, pose, enc_left: float, enc_right: float) -> None:
        self.pose = np.array(pose, dtype=float)
        self._prev = (enc_left, enc_right)
        self.last_ds = 0.0
        self.last_dtheta = 0.0

    def update(self, enc_left: float, enc_right: float) -> np.ndarray:
        if self._prev is None:
            raise RuntimeError("Call reset() with the initial pose and encoders first.")
        dl = (enc_left - self._prev[0]) * self.R     # left wheel arc length [m]
        dr = (enc_right - self._prev[1]) * self.R    # right wheel arc length [m]
        self._prev = (enc_left, enc_right)

        ds = 0.5 * (dl + dr)
        dtheta = (dr - dl) / self.L
        x, y, th = self.pose
        th_mid = th + 0.5 * dtheta
        self.pose = np.array([
            x + ds * np.cos(th_mid),
            y + ds * np.sin(th_mid),
            float(wrap_angle(th + dtheta)),
        ])
        self.last_ds, self.last_dtheta = ds, dtheta
        return self.pose.copy()
