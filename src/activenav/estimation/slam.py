"""EKF-SLAM with ArUco markers as landmarks: the robot knows no map.

State: [robot block | landmark 1 | landmark 2 | ...]
  robot block  [x, y, theta] or [x, y, theta, c, s, k]  exactly as in estimation/ekf.py (odometry calibration)
  landmark     [px, py, psi]   marker centre and the direction of its outward normal, in the filter's frame

The filter's frame is defined by where the robot starts: `reset(pose)` pins the robot at `pose` with a tiny
covariance (the search task uses pose = (0, 0, 0)). Nothing about the room is known in advance. A marker
becomes a landmark the first time it is seen (initialised from the current pose estimate and the measurement,
with the correct joint covariance) and is refined every time it is seen again. Marker ids come from the ArUco
dictionary, so there is no data-association problem. Re-seeing an old landmark after driving away is what
pulls the pose estimate (and the odometry calibration) back, exactly as in the known-map filter.

Predict, range-bias correction, noise model, chi-square gating and the correlated-frame throttle are shared
with EKF (same configuration block `ekf`). Never reads simulator ground truth.
"""
from __future__ import annotations

import numpy as np

from activenav.estimation.ekf import EKF, UpdateInfo
from activenav.kinematics import wrap_angle
from activenav.types import MarkerMeasurement


class SlamEKF(EKF):
    def __init__(self, cfg: dict, init_std=(1e-3, 1e-3, 1e-3)):
        self._init_std_slam = np.asarray(init_std, dtype=float)
        self.lm_ids: list[int] = []
        self.lm_index: dict[int, int] = {}
        self._gated_run: dict[int, int] = {}                  # consecutive gated full updates per landmark
        self.reinit_after = 8                                 # ... after which a close view replaces the landmark
        self.reinit_range = 2.0
        super().__init__(cfg, {})
        self.markers = {}

    # `markers` mirrors the landmark estimates in the same dict form the known-map EKF uses, so display and gaze
    # code can read either filter. It is rebuilt on access and cannot be assigned to.
    @property
    def markers(self) -> dict:
        return {i: dict(x=float(self.x[j]), y=float(self.x[j + 1]), normal_angle=float(self.x[j + 2]))
                for i, j in self.lm_index.items()}

    @markers.setter
    def markers(self, _):
        pass

    @property
    def dim(self) -> int:
        return len(self.x)

    def reset(self, pose, init_std=None) -> None:
        super().reset(pose, self._init_std_slam if init_std is None else init_std)
        self.lm_ids, self.lm_index = [], {}
        self._gated_run = {}

    def landmark_cov(self, marker_id: int) -> np.ndarray:
        j = self.lm_index[marker_id]
        return self.P[j:j + 3, j:j + 3].copy()

    # ---------------------------------------------------------------- prediction
    def predict(self, ds: float, dtheta: float) -> None:
        n, full = self.n, self.x
        self.x = full[:n].copy()
        x_r, F, Q = self._predict_parts(ds, dtheta)       # robot block only; landmarks do not move
        self.x = full
        self.x[:n] = x_r
        P = self.P
        P[:n, :n] = F @ P[:n, :n] @ F.T + Q
        if self.dim > n:
            cross = F @ P[:n, n:]
            P[:n, n:] = cross
            P[n:, :n] = cross.T
        self.P = 0.5 * (P + P.T)

    # -------------------------------------------------------------------- update
    def _expected(self, marker_id: int):
        """Expected [range, bearing, normal] and the Jacobian over the whole state."""
        x, y, th = self.x[:3]
        j = self.lm_index[marker_id]
        dx, dy = self.x[j] - x, self.x[j + 1] - y
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        h = np.array([r, wrap_angle(np.arctan2(dy, dx) - th), wrap_angle(self.x[j + 2] - th)])
        H = np.zeros((3, self.dim))
        H[:, :3] = [[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0], [0.0, 0.0, -1.0]]
        H[:, j:j + 3] = [[dx / r, dy / r, 0.0], [-dy / q, dx / q, 0.0], [0.0, 0.0, 1.0]]
        return h, H

    def predict_measurement(self, marker_id: int, pose=None):
        h, H = self._expected(marker_id)
        return h, H[:, :3]

    def _initialise(self, z: MarkerMeasurement) -> None:
        x, y, th = self.x[:3]
        r = z.range - self.bias_c * z.range ** 2
        a = th + z.bearing
        ca, sa = np.cos(a), np.sin(a)
        Jx = np.array([[1.0, 0.0, -r * sa], [0.0, 1.0, r * ca], [0.0, 0.0, 1.0]])
        Jz = np.array([[ca, -r * sa, 0.0], [sa, r * ca, 0.0], [0.0, 0.0, 1.0]])
        R = self.measurement_noise(z.range)
        rows = Jx @ self.P[:3, :]
        Pll = Jx @ self.P[:3, :3] @ Jx.T + Jz @ R @ Jz.T
        d = self.dim
        P = np.zeros((d + 3, d + 3))
        P[:d, :d] = self.P
        P[d:, :d] = rows
        P[:d, d:] = rows.T
        P[d:, d:] = Pll
        self.P = 0.5 * (P + P.T)
        self.x = np.concatenate([self.x, [x + r * ca, y + r * sa, wrap_angle(th + z.normal_angle)]])
        self.lm_index[z.marker_id] = d
        self.lm_ids.append(z.marker_id)

    def _reinitialise(self, z: MarkerMeasurement) -> None:
        """Replace one landmark by a fresh estimate from this (close, accurate) view. Its old cross-covariances are
        dropped; the robot pose block is untouched. Recovers from a landmark born from a poor long-range view."""
        i = self.lm_index[z.marker_id]
        x, y, th = self.x[:3]
        r = z.range - self.bias_c * z.range ** 2
        a = th + z.bearing
        ca, sa = np.cos(a), np.sin(a)
        Jx = np.array([[1.0, 0.0, -r * sa], [0.0, 1.0, r * ca], [0.0, 0.0, 1.0]])
        Jz = np.array([[ca, -r * sa, 0.0], [sa, r * ca, 0.0], [0.0, 0.0, 1.0]])
        R = self.measurement_noise(z.range)
        rows = Jx @ self.P[:3, :]
        P = self.P.copy()
        P[i:i + 3, :] = rows
        P[:, i:i + 3] = rows.T
        P[i:i + 3, i:i + 3] = Jx @ self.P[:3, :3] @ Jx.T + Jz @ R @ Jz.T
        self.P = 0.5 * (P + P.T)
        self.x[i:i + 3] = [x + r * ca, y + r * sa, wrap_angle(th + z.normal_angle)]

    def _apply(self, H, nu, R, gate):
        S = H @ self.P @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > gate:
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = float(wrap_angle(self.x[2]))
        for i in self.lm_index.values():
            self.x[i + 2] = float(wrap_angle(self.x[i + 2]))
        I_KH = np.eye(self.dim) - K @ H                          # Joseph form
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True, nis

    def update_marker(self, z: MarkerMeasurement, t: float | None = None) -> UpdateInfo:
        if not (self.min_range <= z.range <= self.max_range):
            return UpdateInfo(z.marker_id, False, float("nan"), "range out of bounds")
        if z.marker_id not in self.lm_index:
            self._initialise(z)
            if t is not None:
                self._last_update[z.marker_id] = t
            return UpdateInfo(z.marker_id, True, 0.0, "new landmark")

        h, H = self._expected(z.marker_id)
        throttled = t is not None and self.min_update_dt > 0.0 and \
            t - self._last_update.get(z.marker_id, -1e9) < self.min_update_dt
        if throttled and self.throttle_mode != "bearing_only":
            return UpdateInfo(z.marker_id, False, float("nan"), "throttled")
        if throttled:                                            # bearing errors are independent frame to frame
            nu = np.array([wrap_angle(z.bearing - h[1])])
            ok, nis = self._apply(H[1:2], nu, np.array([[self.sig_bearing ** 2]]), 6.635)
            return UpdateInfo(z.marker_id, ok, nis, "bearing only" if ok else "gated bearing")

        zv = np.array([z.range - self.bias_c * z.range ** 2, z.bearing, z.normal_angle])
        R = self.measurement_noise(z.range)
        if not self.use_normal:
            h, H, zv, R = h[:2], H[:2], zv[:2], R[:2, :2]
        nu = zv - h
        nu[1:] = wrap_angle(nu[1:])
        ok, nis = self._apply(H, nu, R, self.gate)
        if not ok:
            n = self._gated_run.get(z.marker_id, 0) + 1
            self._gated_run[z.marker_id] = n
            if n >= self.reinit_after and z.range <= self.reinit_range:
                self._reinitialise(z)
                self._gated_run[z.marker_id] = 0
                if t is not None:
                    self._last_update[z.marker_id] = t
                return UpdateInfo(z.marker_id, True, nis, "reinitialised")
            return UpdateInfo(z.marker_id, False, nis, "gated")
        self._gated_run[z.marker_id] = 0
        if t is not None:
            self._last_update[z.marker_id] = t
        return UpdateInfo(z.marker_id, True, nis)
