"""Trial sessions: the recorder that logs true vs estimated pose, and the odometry-only and EKF crossing sessions."""
from __future__ import annotations

import numpy as np

from activenav.control.go_to_point import GoToPointGains, go_to_point
from activenav.kinematics import wrap_angle
from activenav.sensing.odometry import WheelOdometry
from activenav.types import Command


# -------------------- evaluation/trials.py --------------------
# Trial recording and error metrics. The only place that reads simulator ground truth.
class TrialRecorder:
    def __init__(self, env):
        self._env = env
        self.t, self.est, self.gt = [], [], []
        self.collided = False
        self.min_clearance = float("inf")

    def log(self, t: float, est_pose) -> None:
        self.t.append(t)
        self.est.append(np.asarray(est_pose, dtype=float))
        self.gt.append(self._env.ground_truth())
        self.collided = self.collided or self._env.wall_contact()
        self.min_clearance = min(self.min_clearance, self._env.min_clearance())

    def arrays(self) -> dict:
        return {"t": np.array(self.t), "est": np.array(self.est), "gt": np.array(self.gt)}

    def position_error(self) -> np.ndarray:
        a = self.arrays()
        return np.hypot(*(a["est"][:, :2] - a["gt"][:, :2]).T)

    def heading_error(self) -> np.ndarray:
        a = self.arrays()
        return wrap_angle(a["est"][:, 2] - a["gt"][:, 2])

    def path_length(self) -> float:
        gt = self.arrays()["gt"]
        return float(np.sum(np.hypot(*np.diff(gt[:, :2], axis=0).T)))

    def summary(self) -> dict:
        pe, he = self.position_error(), self.heading_error()
        return {
            "duration_s": float(self.t[-1] - self.t[0]),
            "path_length_m": self.path_length(),
            "collided": self.collided,
            "min_clearance_m": float(self.min_clearance),     # base origin to nearest wall/pillar surface
            "final_position_error_m": float(pe[-1]),
            "max_position_error_m": float(pe.max()),
            "position_rmse_m": float(np.sqrt(np.mean(pe ** 2))),
            "final_heading_error_rad": float(he[-1]),
            "heading_rmse_rad": float(np.sqrt(np.mean(he ** 2))),
        }

    def save(self, path) -> None:
        np.savez(path, **self.arrays())


# -------------------- evaluation/trials.py --------------------
# Shared trial runner for all configurations.
class TrialSession:
    """One tour / crossing, advanced by `step()`.

    `goal` is one (x, y) point or a list of waypoints visited in order. If `final_yaw` is given the robot
    rotates in place to that heading after the last waypoint (docking alignment). With `ekf=None` the
    session is odometry-only (Config A). Otherwise `sensor.measure(sb)` returns marker measurements and
    `gaze_fn(t, ekf) -> (pan, tilt)` picks where the head looks (default: fixed forward).
    For EKF sessions odometry dead reckoning runs alongside on the same encoders, for comparison.
    """

    def __init__(self, env, seed, start_pose, goal, sensor=None, ekf=None, v_max=0.2, w_max=1.0,
                 max_time=300.0, camera_rate_hz=10.0, gaze_fn=None, final_yaw=None, yaw_tol=0.03,
                 via_tol=None):
        self.env, self.sensor, self.ekf, self.gaze_fn = env, sensor, ekf, gaze_fn
        self.waypoints = [goal] if np.ndim(goal) == 1 else list(goal)
        self.w_max, self.max_time = w_max, max_time
        self.final_yaw, self.yaw_tol = final_yaw, yaw_tol
        self.gains = GoToPointGains(v_max=v_max, w_max=w_max)
        self.gains_via = self.gains if via_tol is None else GoToPointGains(v_max=v_max, w_max=w_max, pos_tol=via_tol)
        self.start_pose = tuple(start_pose)

        self.s = env.reset(seed=seed, start_pose=self.start_pose)
        self.odo = WheelOdometry(env.wheel_radius, env.track_width)
        self.odo.reset(self.start_pose, self.s.enc_left, self.s.enc_right)
        if ekf is not None:
            ekf.reset(self.start_pose)
        if gaze_fn is not None and hasattr(gaze_fn, "reset"):
            gaze_fn.reset()
        self.rec = TrialRecorder(env)
        self.rec.log(self.s.t, self.pose)

        self.cam_dt = 1.0 / camera_rate_hz
        self.next_cam = 0.0
        self.pan = self.tilt = 0.0
        self.odo_log, self.nees, self.sig_log, self.pan_log = [self.odo.pose.copy()], [], [], [0.0]
        if ekf is not None:
            self.sig_log.append(ekf.sigma())
        self.n_used = self.n_gated = self.n_seen = self.n_cam = self.n_cam_hit = 0
        self.idx = 0
        self.done = False
        self.finished = False
        self.events = []             # [(MarkerMeasurement, UpdateInfo)] from the most recent camera frame
        self.last_image = None       # the image the robot most recently processed (None for synthetic sensors)
        self.last_cmd = (0.0, 0.0)

    # ------------------------------------------------------------------ helpers
    @property
    def pose(self) -> np.ndarray:
        """The estimate the controller is using."""
        return self.ekf.pose if self.ekf is not None else self.odo.pose

    @property
    def t(self) -> float:
        return self.s.t

    def _command(self):
        """Next (v, w), or None when the trial is over (sets `done` if it finished normally)."""
        while self.s.t < self.max_time:
            if self.idx < len(self.waypoints):
                last = self.idx == len(self.waypoints) - 1
                v, w, reached = go_to_point(self.pose, self.waypoints[self.idx],
                                            self.gains if last else self.gains_via)
                if reached:
                    self.idx += 1
                    continue
                return v, w
            if self.final_yaw is None:
                self.done = True
                return None
            err = float(wrap_angle(self.final_yaw - self.pose[2]))
            if abs(err) < self.yaw_tol:
                self.done = True
                return None
            return 0.0, float(np.clip(2.0 * err, -self.w_max, self.w_max))
        return None

    # --------------------------------------------------------------------- step
    def step(self) -> bool:
        """Advance one control step. Returns False once the trial has finished or timed out."""
        if self.finished:
            return False
        cmd = self._command()
        if cmd is None:
            self.finished = True
            return False
        v, w = cmd
        self.last_cmd = cmd
        env, ekf, s = self.env, self.ekf, self.s
        if ekf is not None and self.gaze_fn is not None:
            self.pan, self.tilt = self.gaze_fn(s.t, ekf)
        due = ekf is not None and s.t >= self.next_cam
        s = self.s = env.step(Command(v=v, w=w, pan_target=self.pan, tilt_target=self.tilt),
                              render=bool(due and self.sensor.needs_image))
        self.odo.update(s.enc_left, s.enc_right)
        if ekf is None:
            self.rec.log(s.t, self.odo.pose)
            return True

        ekf.predict(self.odo.last_ds, self.odo.last_dtheta)
        if due:
            self.next_cam = s.t + self.cam_dt
            self.n_cam += 1
            self.events = []
            self.last_image = s.image
            usable = False
            for z in self.sensor.measure(s):
                info = ekf.update_marker(z, t=s.t)
                if info.reason == "unknown id":
                    continue                      # the camera sees every marker; the filter only knows its map
                self.events.append((z, info))
                if info.reason in ("throttled", "bearing only", "gated bearing"):   # not full updates: do not count them
                    usable = True
                    continue
                self.n_seen += 1
                self.n_used += info.accepted
                self.n_gated += info.reason == "gated"
                usable |= info.reason != "range out of bounds"
            self.n_cam_hit += usable
        self.rec.log(s.t, ekf.pose)
        self.odo_log.append(self.odo.pose.copy())
        self.sig_log.append(ekf.sigma())
        self.pan_log.append(s.pan)
        e = env.ground_truth() - ekf.pose                           # NEES: needs ground truth
        e[2] = wrap_angle(e[2])
        self.nees.append(float(e @ np.linalg.solve(ekf.P_pose, e)))
        return True

    # ------------------------------------------------------------------- result
    def result(self) -> dict:
        env, rec, ekf = self.env, self.rec, self.ekf
        out = rec.summary()
        arr = rec.arrays()
        clearance = float(rec.min_clearance - env.cfg["robot"]["radius"])     # body clearance, < 0 = overlap
        if ekf is None:
            out.update(completed=self.done, errors=env.hidden_encoder_errors(), t=arr["t"], est=arr["est"],
                       gt=arr["gt"], pos_err=rec.position_error(), min_clearance_m=clearance)
            return out
        odo_arr = np.array(self.odo_log)
        odo_err = np.hypot(*(odo_arr[:, :2] - arr["gt"][:, :2]).T)
        gt_end = arr["gt"][-1]
        out.update(completed=self.done, errors=env.hidden_encoder_errors(),
                   t=arr["t"], est=arr["est"], gt=arr["gt"], pos_err=rec.position_error(),
                   odo=odo_arr, odo_pos_err=odo_err, sigma=np.array(self.sig_log), nees=np.array(self.nees),
                   pan=np.array(self.pan_log),
                   n_seen=self.n_seen, n_used=self.n_used, n_gated=self.n_gated, bias_est=ekf.bias,
                   frames_with_marker=self.n_cam_hit / max(self.n_cam, 1),
                   min_clearance_m=clearance,
                   final_miss=float(np.hypot(*(gt_end[:2] - np.asarray(self.waypoints[-1], dtype=float)))),
                   final_yaw_err=(float(wrap_angle(gt_end[2] - self.final_yaw)) if self.final_yaw is not None
                                  else float("nan")))
        return out

def run_odometry_trial(env, seed, start_pose, waypoints, v_max=0.2, w_max=1.0, max_time=300.0,
                       final_yaw=None, yaw_tol=0.03):
    """Config A: odometry only. The controller sees only the odometry estimate."""
    session = TrialSession(env, seed, start_pose, waypoints, v_max=v_max, w_max=w_max, max_time=max_time,
                           final_yaw=final_yaw, yaw_tol=yaw_tol)
    while session.step():
        pass
    return session.result()

def run_ekf_trial(env, seed, start_pose, goal, sensor, ekf, v_max=0.2, w_max=1.0, max_time=300.0,
                  camera_rate_hz=10.0, gaze_fn=None, final_yaw=None, yaw_tol=0.03, via_tol=None):
    """Config B / C: EKF localisation (pass a gaze_fn for Config C). See TrialSession."""
    session = TrialSession(env, seed, start_pose, goal, sensor=sensor, ekf=ekf, v_max=v_max, w_max=w_max,
                           max_time=max_time, camera_rate_hz=camera_rate_hz, gaze_fn=gaze_fn,
                           final_yaw=final_yaw, yaw_tol=yaw_tol, via_tol=via_tol)
    while session.step():
        pass
    return session.result()
