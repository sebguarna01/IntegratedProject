"""Extended Kalman Filter for the planar pose [x, y, theta], with optional online odometry calibration.

State: [x, y, theta]                      (estimate_bias: false)
       [x, y, theta, c, s, k]             (estimate_bias: true)
  c = heading drift per metre driven [rad/m]   (left/right wheel-scale mismatch looks like this)
  s = distance scale error                     (common wheel-scale error)
  k = turn scale error                         (track-width error: spins read too large or small)
The corrected odometry increments are  ds' = ds (1 + s),  dtheta' = dtheta (1 + k) - c ds. A constant
per-trial bias is exactly what the characterised encoder model produces (configs `noise`), and a
filter that treats it as white noise lags it, becomes over-confident and starts gating out good
measurements. c and s are modelled as slow random walks in distance driven.

Prediction: odometry increments (ds, dtheta) drive the unicycle model; the process noise is
proportional to the distance and rotation travelled (see configs/default.yaml, section `ekf`).
Update: one ArUco marker at a time, measured as [range, bearing, marker-normal direction] in the
robot base frame, against the surveyed marker map.

Never reads simulator ground truth. It only sees odometry increments, marker measurements and
the known marker map.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activenav.kinematics import wrap_angle
from activenav.types import MarkerMeasurement


@dataclass(frozen=True)
class UpdateInfo:
    marker_id: int
    accepted: bool
    nis: float            # normalised innovation squared (chi-square, 3 dof if the normal is used)
    reason: str = ""      # why it was not used, if it was not


class EKF:
    def __init__(self, cfg: dict, markers: dict):
        e = cfg["ekf"]
        self.k_ds, self.k_dth, self.k_dth_ds = float(e["k_ds"]), float(e["k_dth"]), float(e["k_dth_ds"])
        self.r_a, self.r_b = (float(v) for v in e["range_sigma"])
        self.bias_c = float(e["range_bias_coeff"])
        self.sig_bearing = float(e["bearing_sigma_rad"])
        self.n_a, self.n_b = (np.radians(float(v)) for v in e["normal_sigma_deg"])
        self.max_range, self.min_range = float(e["max_range"]), float(e["min_range"])
        self.use_normal = bool(e["use_normal"])
        self.gate = float(e["gate_chi2"])
        # Successive frames of the same marker from a nearly unchanged viewpoint carry nearly the same
        # error (corner quantisation, pose-estimation bias), so they are not independent. Fusing all of them
        # makes the filter over-confident. 0 = fuse every frame (the original behaviour).
        self.min_update_dt = float(e.get("min_update_interval_s", 0.0))
        # What to do with a throttled frame. "skip" drops it. "bearing_only" still fuses the bearing: measured
        # on a parked robot (scripts/measurement_correlation.py), bearing errors are independent frame to frame (ICC ~0.1) while range
        # (ICC 0.84-0.99) and marker-normal errors repeat, so only those two are throttled.
        self.throttle_mode = str(e.get("throttle_mode", "bearing_only"))
        self._last_update = {}
        self.estimate_bias = bool(e.get("estimate_bias", False))
        self.n = 6 if self.estimate_bias else 3
        self.init_std = np.array(e["init_std"], dtype=float)
        self.bias_init_std = np.array(e.get("bias_init_std", [0.03, 0.01, 0.02]), dtype=float)   # [c rad/m, s, k]
        self.bias_walk = np.array(e.get("bias_walk", [0.002, 0.001, 0.002]), dtype=float)    # c, s per sqrt(m); k per sqrt(rad)
        self.markers = markers
        self.reset(np.zeros(3))

    def reset(self, pose, init_std=None) -> None:
        std = self.init_std if init_std is None else np.asarray(init_std, dtype=float)
        self._last_update = {}
        self.x = np.zeros(self.n)
        self.x[:3] = pose
        stds = np.concatenate([std, self.bias_init_std])[: self.n]
        self.P = np.diag(stds ** 2)

    @property
    def pose(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def bias(self) -> np.ndarray:
        """[c, s, k] (zeros when bias estimation is off)."""
        return self.x[3:6].copy() if self.estimate_bias else np.zeros(3)

    @property
    def P_pose(self) -> np.ndarray:
        return self.P[:3, :3]

    # ---------------------------------------------------------------- prediction
    def _motion(self, x, ds, dtheta):
        """Deterministic motion model on the (up to) 5-state vector."""
        c, s, k = x[3:6] if self.estimate_bias else (0.0, 0.0, 0.0)
        ds_c = ds * (1.0 + s)
        dth_c = dtheta * (1.0 + k) - c * ds
        tm = x[2] + 0.5 * dth_c
        out = x.copy()
        out[0] = x[0] + ds_c * np.cos(tm)
        out[1] = x[1] + ds_c * np.sin(tm)
        out[2] = wrap_angle(x[2] + dth_c)
        return out

    def predict(self, ds: float, dtheta: float) -> None:
        self.x, F, Q = self._predict_parts(ds, dtheta)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _predict_parts(self, ds: float, dtheta: float):
        """New robot-block state, its Jacobian F and process noise Q (size n x n). The SLAM filter reuses
        this and applies it to the robot block of a larger covariance."""
        x = self.x
        c, s, k = x[3:6] if self.estimate_bias else (0.0, 0.0, 0.0)
        ds_c = ds * (1.0 + s)
        tm = x[2] + 0.5 * (dtheta * (1.0 + k) - c * ds)
        sn, cs = np.sin(tm), np.cos(tm)

        F = np.eye(self.n)
        F[0, 2], F[1, 2] = -ds_c * sn, ds_c * cs
        # input Jacobian wrt the odometry increments (ds, dtheta)
        G = np.array([[(1 + s) * cs + 0.5 * c * ds_c * sn, -0.5 * (1 + k) * ds_c * sn],
                      [(1 + s) * sn - 0.5 * c * ds_c * cs, 0.5 * (1 + k) * ds_c * cs],
                      [-c, 1.0 + k]])
        sig_ds = self.k_ds * abs(ds)
        sig_dth = self.k_dth * abs(dtheta) + self.k_dth_ds * abs(ds)
        Q = np.zeros((self.n, self.n))
        Q[:3, :3] = G @ np.diag([sig_ds ** 2, sig_dth ** 2]) @ G.T
        if self.estimate_bias:
            F[0, 3], F[1, 3], F[2, 3] = 0.5 * ds * ds_c * sn, -0.5 * ds * ds_c * cs, -ds
            F[0, 4], F[1, 4] = ds * cs, ds * sn
            F[0, 5], F[1, 5], F[2, 5] = -0.5 * ds_c * dtheta * sn, 0.5 * ds_c * dtheta * cs, dtheta
            Q[3, 3] = (self.bias_walk[0] ** 2) * abs(ds)
            Q[4, 4] = (self.bias_walk[1] ** 2) * abs(ds)
            Q[5, 5] = (self.bias_walk[2] ** 2) * abs(dtheta)

        return self._motion(x, ds, dtheta), F, Q

    # -------------------------------------------------------------------- update
    def measurement_noise(self, r: float) -> np.ndarray:
        """Diagonal R for [range, bearing, normal] at true-ish range r."""
        return np.diag([(self.r_a + self.r_b * r ** 2) ** 2,
                        self.sig_bearing ** 2,
                        (self.n_a + self.n_b * r ** 3) ** 2])

    def predict_measurement(self, marker_id: int, pose=None):
        """Expected [range, bearing, normal] and Jacobian H for a marker at the given pose."""
        x, y, th = self.x[:3] if pose is None else pose
        m = self.markers[marker_id]
        dx, dy = m["x"] - x, m["y"] - y
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        h = np.array([r, wrap_angle(np.arctan2(dy, dx) - th), wrap_angle(m["normal_angle"] - th)])
        H = np.array([[-dx / r, -dy / r, 0.0],
                      [dy / q, -dx / q, -1.0],
                      [0.0, 0.0, -1.0]])
        return h, H

    def update_marker(self, z: MarkerMeasurement, t: float | None = None) -> UpdateInfo:
        if z.marker_id not in self.markers:
            return UpdateInfo(z.marker_id, False, float("nan"), "unknown id")
        if not (self.min_range <= z.range <= self.max_range):
            return UpdateInfo(z.marker_id, False, float("nan"), "range out of bounds")

        if t is not None and self.min_update_dt > 0.0:
            if t - self._last_update.get(z.marker_id, -1e9) < self.min_update_dt:
                if self.throttle_mode == "bearing_only":
                    return self._update_bearing_only(z)
                return UpdateInfo(z.marker_id, False, float("nan"), "throttled")
        h, H3 = self.predict_measurement(z.marker_id)
        H = np.zeros((3, self.n))
        H[:, :3] = H3                                           # bias states are not observed directly
        r_meas = z.range - self.bias_c * z.range ** 2          # remove the characterised range bias
        zv = np.array([r_meas, z.bearing, z.normal_angle])
        R = self.measurement_noise(z.range)
        if not self.use_normal:
            h, H, zv, R = h[:2], H[:2], zv[:2], R[:2, :2]

        nu = zv - h
        nu[1:] = wrap_angle(nu[1:])
        S = H @ self.P @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > self.gate:
            return UpdateInfo(z.marker_id, False, nis, "gated")

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = float(wrap_angle(self.x[2]))
        I_KH = np.eye(self.n) - K @ H                                # Joseph form keeps P symmetric positive definite
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        if t is not None:
            self._last_update[z.marker_id] = t
        return UpdateInfo(z.marker_id, True, nis)

    def _update_bearing_only(self, z: MarkerMeasurement) -> UpdateInfo:
        h, H3 = self.predict_measurement(z.marker_id)
        H = np.zeros((1, self.n))
        H[0, :3] = H3[1]
        R = self.sig_bearing ** 2
        nu = float(wrap_angle(z.bearing - h[1]))
        S = float((H @ self.P @ H.T)[0, 0]) + R
        nis = nu * nu / S
        if nis > 6.635:                                         # chi-square, 1 dof, 99%
            return UpdateInfo(z.marker_id, False, nis, "gated bearing")
        K = (self.P @ H.T) / S
        self.x = self.x + (K * nu).ravel()
        self.x[2] = float(wrap_angle(self.x[2]))
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + R * (K @ K.T)
        self.P = 0.5 * (self.P + self.P.T)
        return UpdateInfo(z.marker_id, True, nis, "bearing only")

    def sigma(self) -> np.ndarray:
        """1-sigma uncertainty of [x, y, theta]."""
        return np.sqrt(np.diag(self.P)[:3])
