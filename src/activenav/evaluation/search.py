"""Search-and-dock trials: the robot is dropped into an unknown room and has to find marker 10 and dock.

`SearchSession` steps one trial (shared by the batch study and the live viewer). The agent (autonomy/) sees only
odometry, marker measurements and obstacle scans. Everything here that reads the simulator (true pose, true marker
and obstacle layout) is for SCORING: pose and map error are measured in the robot's own start frame, because that is
the frame the robot's map lives in.
"""
from __future__ import annotations

import copy

import numpy as np

from activenav.autonomy.agent import SearchAgent
from activenav.autonomy.occupancy import FREE, OCC
from activenav.autonomy.search_gaze import FixedSearchGaze, InfoSearchGaze, SweepSearchGaze
from activenav.evaluation.worlds import ScenarioError, available_levels, get_world
from activenav.evaluation.synthetic import SyntheticMarkerSensor
from activenav.evaluation.synthetic import SyntheticObstacleSensor
from activenav.geometry import obstacle_sdf
from activenav.kinematics import wrap_angle
from activenav.sensing.delay import OdometryHistory, marker_to_now, scan_to_now
from activenav.sensing.odometry import WheelOdometry
from activenav.types import Command

SEARCH_CONFIGS = ["B", "C-sweep", "C-info"]       # A cannot perceive anything: it cannot search, so it is not run


def to_frame(origin, pose):
    """World pose -> the frame whose origin is `origin` (the robot's start pose)."""
    d = np.asarray(pose[:2]) - np.asarray(origin[:2])
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], wrap_angle(pose[2] - origin[2])])


def make_gaze(cfg, conf, env):
    if conf == "B":
        return FixedSearchGaze(cfg)
    if conf == "C-sweep":
        return SweepSearchGaze(cfg)
    if conf == "C-info":
        return InfoSearchGaze(cfg, env.pan_range)
    raise ScenarioError(f"config '{conf}' cannot search (use {', '.join(SEARCH_CONFIGS)}); A has no way to perceive")


def pick_start(ctx, seed, clear=0.6):
    """A random start pose with the body well clear of walls and pillars. Evaluation-side (reads the true world)."""
    rng = np.random.default_rng([seed, 777])
    x0, x1, y0, y1 = ctx.env.room_bounds()
    for _ in range(10000):
        p = (rng.uniform(x0 + clear, x1 - clear), rng.uniform(y0 + clear, y1 - clear), rng.uniform(-np.pi, np.pi))
        if all(obstacle_sdf(o, p[0], p[1]) > clear for o in ctx.obstacles):
            return p
    raise ScenarioError("could not find a free start pose")


def gaze_metrics(t, est, head, v_min: float = 0.05):
    """Behavioural descriptors of the head, from the logs (commanded head angles, estimated pose). Scored after the run.
    gaze (world) = body heading + pan. Moving = the body advanced more than v_min m/s.
      gaze_switch_per_min   reversals of the pan direction per minute (rate above 0.15 rad/s counts as moving)
      gaze_dwell_med_s      median time the world-frame gaze stays still (under 0.15 rad/s) between movements
      gaze_toward_travel    share of moving time with the gaze within 0.6 rad of the direction of travel
      gaze_lead_s           lag [s] at which the pan best predicts the body's future heading change (0..4 s; nan if the
                            body hardly turns). Positive: the head turns before the body does
      pan_std_rad           spread of the pan angle"""
    t, est, head = np.asarray(t, float), np.asarray(est, float), np.asarray(head, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    pan = head[:, 0]
    gaze = np.unwrap(est[:, 2]) + pan
    win = max(int(round(0.5 / dt)), 1)                                   # 0.5 s differences, so quantisation noise does not count
    dgaze = np.zeros_like(gaze)
    dgaze[win:] = (gaze[win:] - gaze[:-win]) / (win * dt)
    dpan = np.zeros_like(pan)
    dpan[win:] = (pan[win:] - pan[:-win]) / (win * dt)
    sp = np.hypot(*(est[win:, :2] - est[:-win, :2]).T) / (win * dt)
    speed = np.concatenate([np.zeros(win), sp])
    moving = speed > v_min
    out = {}
    sg = np.sign(dpan) * (np.abs(dpan) > 0.15)
    sg = sg[sg != 0]
    out["gaze_switch_per_min"] = float((np.diff(sg) != 0).sum() / max((t[-1] - t[0]) / 60.0, 1e-9)) if len(sg) > 1 else 0.0
    still = np.abs(dgaze) < 0.15
    runs, n = [], 0
    for k in still:
        if k:
            n += 1
        elif n:
            runs.append(n * dt); n = 0
    if n:
        runs.append(n * dt)
    out["gaze_dwell_med_s"] = float(np.median(runs)) if runs else float("nan")
    if moving.any():
        trav = np.arctan2(*(est[win:, :2] - est[:-win, :2]).T[::-1])
        trav = np.concatenate([np.full(win, trav[0]), trav])
        off = np.abs(np.arctan2(np.sin(gaze - trav), np.cos(gaze - trav)))
        out["gaze_toward_travel"] = float((off[moving] < 0.6).mean())
    else:
        out["gaze_toward_travel"] = float("nan")
    th = np.unwrap(est[:, 2])
    lags = np.arange(0, int(4.0 / dt) + 1, max(int(round(0.25 / dt)), 1))
    dth = np.diff(th, prepend=th[0])
    if np.abs(th[-1] - th[0]) < 1e-6 and np.std(dth) < 1e-9:
        out["gaze_lead_s"] = float("nan")
    else:
        best, bl = -np.inf, float("nan")
        for L in lags:
            if L >= len(pan) - 10:
                break
            a, b = pan[: len(pan) - L], th[L:] - th[: len(th) - L]          # pan now vs heading change over the next L steps... lead score
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best:
                best, bl = c, L * dt
        out["gaze_lead_s"] = bl
    out["pan_std_rad"] = float(np.std(pan))
    return out


class SearchSession:
    def __init__(self, ctx, cfg, conf, seed, start_pose, marker_sensor, obstacle_front, max_time=None, settle_steps=15):
        self.ctx, self.cfg, self.conf, self.seed = ctx, cfg, conf, seed
        self.env = env = ctx.env
        self.start_pose = tuple(float(v) for v in start_pose)
        self.max_time = float(cfg["search"]["max_time"] if max_time is None else max_time)
        self.marker_sensor, self.front = marker_sensor, obstacle_front
        self.agent = SearchAgent(cfg, make_gaze(cfg, conf, env), env.control_dt)
        self.s = env.reset(seed=seed, start_pose=self.start_pose)
        self.odo = WheelOdometry(env.wheel_radius, env.track_width)
        self.odo.reset((0.0, 0.0, 0.0), self.s.enc_left, self.s.enc_right)
        self.odo_hist = OdometryHistory(dt=env.control_dt)          # for frames that arrive late (real-camera profiles)
        self.odo_hist.reset(self.s.t)
        self.agent.reset()
        self.cam_dt = 1.0 / cfg["camera_rate_hz"]
        self.next_cam = 0.0
        self.needs_image = bool(getattr(marker_sensor, "needs_image", False) or getattr(obstacle_front, "needs_image", False))
        self.finished, self.settle, self.settle_steps = False, 0, settle_steps
        self.collided = False
        self.person_contact = False
        self.person_gap_log = []
        self.min_clearance = np.inf
        self.t_log, self.est, self.gt, self.nees_log = [], [], [], []
        self.sig_log, self.head_log, self.clr_log, self.cov_log = [], [], [], []
        self.next_cov = 0.0
        self.n_cam = 0
        self.last_image = None
        self.last_cmd = (0.0, 0.0)
        self.t_first_contact = None
        self._log()

    @property
    def t(self):
        return self.s.t

    @property
    def pose(self):
        return self.agent.pose

    def _log(self):
        gt = to_frame(self.start_pose, self.env.ground_truth())
        self.t_log.append(self.s.t)
        self.est.append(self.agent.pose.copy())
        self.gt.append(gt)
        pc = self.env.person_contact()
        self.person_contact = self.person_contact or pc
        c = self.env.wall_contact() or pc
        if c and not self.collided:
            self.t_first_contact = self.s.t
        self.collided = self.collided or c
        self.min_clearance = min(self.min_clearance, self.env.min_clearance())
        # scoring-only series (ground truth is read here, never by the agent)
        e = gt - self.agent.pose
        e[2] = wrap_angle(e[2])
        P = self.agent.slam.P[:3, :3]
        self.nees_log.append(float(e @ np.linalg.solve(P + 1e-12 * np.eye(3), e)))
        self.sig_log.append(float(np.sqrt(P[0, 0] + P[1, 1])))
        self.head_log.append((self.agent.pan_cmd, self.agent.tilt_cmd))      # commanded: the encoders add noise
        self.clr_log.append(self.env.min_clearance() - self.cfg["robot"]["radius"])
        self.person_gap_log.append(self.env.person_gap())
        if self.s.t >= self.next_cov:
            self.next_cov = self.s.t + 10.0
            self.cov_log.append((self.s.t, self.map_scores()[0]))

    def step(self) -> bool:
        if self.finished:
            return False
        a = self.agent
        if a.state in ("DONE", "FAILED") or self.s.t >= self.max_time or self.collided:
            self.settle += 1
            if self.settle > self.settle_steps or a.state == "FAILED" or self.collided or self.s.t >= self.max_time:
                self.finished = True
                return False
        v, w, pan, tilt = a.command(self.s.t)
        if a.state in ("DONE", "FAILED") or self.collided:
            v = w = 0.0
        self.last_cmd = (v, w)
        due = self.s.t >= self.next_cam
        self.s = self.env.step(Command(v=v, w=w, pan_target=pan, tilt_target=tilt), render=bool(due and self.needs_image))
        self.odo.update(self.s.enc_left, self.s.enc_right)
        self.odo_hist.add(self.s.t, self.odo.last_ds, self.odo.last_dtheta)
        a.predict(self.odo.last_ds, self.odo.last_dtheta)
        if due:
            self.next_cam = self.s.t + self.cam_dt
            self.n_cam += 1
            self.last_image = self.s.image
            zs, scan = self.marker_sensor.measure(self.s), self.front.scan_sb(self.s)
            if self.s.image_t is not None:                           # a late frame: carry it into the base frame of now
                motion = self.odo_hist.motion_since(self.s.image_t)
                zs, scan = [marker_to_now(z, motion) for z in zs], scan_to_now(scan, motion)
            a.on_frame(self.s.t, zs, scan)
        self._log()
        return True

    # ------------------------------------------------------------------ scoring
    def true_marker(self, mid):
        m = self.env.marker_map()[mid]
        return to_frame(self.start_pose, (m["x"], m["y"], m["normal_angle"]))

    def map_scores(self):
        """(coverage of the true floor marked free, fraction of occupied cells that are really obstacles)."""
        g = self.agent.grid
        st = g.state()
        x0, x1, y0, y1 = self.env.room_bounds()
        c, s = np.cos(self.start_pose[2]), np.sin(self.start_pose[2])
        # cells known free/occupied -> world coordinates
        def world(iy, ix):
            px, py = g.to_world(iy, ix)
            return self.start_pose[0] + c * px - s * py, self.start_pose[1] + s * px + c * py
        iy, ix = np.nonzero(st == FREE)
        fx, fy = world(iy, ix)
        # true free floor area: room minus obstacles, sampled on a 0.1 m lattice
        gx, gy = np.meshgrid(np.arange(x0 + 0.05, x1, 0.1), np.arange(y0 + 0.05, y1, 0.1))
        floor = np.ones(gx.shape, bool)
        for o in self.ctx.obstacles:
            floor &= obstacle_sdf(o, gx, gy) > 0.0
        known = np.zeros(gx.shape, bool)
        ci = np.clip(((fx - x0) / 0.1).astype(int), 0, gx.shape[1] - 1)
        cj = np.clip(((fy - y0) / 0.1).astype(int), 0, gx.shape[0] - 1)
        known[cj, ci] = True
        coverage = float((known & floor).sum() / max(1, floor.sum()))
        oy, ox = np.nonzero(st == OCC)
        if len(ox) == 0:
            return coverage, float("nan")
        wx, wy = world(oy, ox)
        d = np.minimum.reduce([wx - x0, x1 - wx, wy - y0, y1 - wy] +
                              [obstacle_sdf(o, wx, wy) for o in self.ctx.obstacles])
        return coverage, float((np.abs(d) < 0.20).mean())

    def result(self) -> dict:
        env, a, cfg = self.env, self.agent, self.cfg
        est, gt = np.array(self.est), np.array(self.gt)
        t = np.array(self.t_log)
        dock_id = a.dock_id
        mk = env.marker_map()[dock_id]
        standoff = float(cfg["search"]["standoff"])
        n = np.array([np.cos(mk["normal_angle"]), np.sin(mk["normal_angle"])])
        D = np.array([mk["x"], mk["y"]]) + standoff * n
        gw = env.ground_truth()
        miss = float(np.hypot(*(gw[:2] - D)))
        yaw = float(abs(np.degrees(wrap_angle(gw[2] - (mk["normal_angle"] + np.pi)))))
        claimed = a.state == "DONE"
        docked = bool(claimed and not self.collided and miss <= 0.10 and yaw <= 5.0)
        if docked:
            outcome = "docked"
        elif self.collided:
            outcome = "contact"        # a wall, a pillar or a person; `person_contact` says which
        elif claimed:
            outcome = "false dock"
        elif a.state == "FAILED":
            outcome = "gave up"
        else:
            outcome = "timeout"
        lm_err = []
        real = self.env.marker_map()
        n_spurious = 0
        for mid, lm in a.slam.markers.items():
            if mid not in real:                      # a mis-read ArUco id: a landmark that does not exist; it cannot be scored
                n_spurious += 1
                continue
            tm = self.true_marker(mid)
            lm_err.append(float(np.hypot(lm["x"] - tm[0], lm["y"] - tm[1])))
        pe = np.hypot(*(est[:, :2] - gt[:, :2]).T)
        coverage, map_precision = self.map_scores()
        path = float(np.sum(np.hypot(*np.diff(gt[:, :2], axis=0).T)))
        dt = np.diff(t, prepend=t[0])
        head = np.array(self.head_log)
        nees = np.array(self.nees_log)
        cov = np.array(self.cov_log)
        cov_at = lambda tt: float(cov[cov[:, 0] <= tt][-1, 1]) if len(cov) and (cov[:, 0] <= tt).any() else float("nan")
        frames = max(1, a.n_frames)
        t_sight = a.t_first_sight
        extra = dict(
            dock_height_m=float(mk["z"]),
            approach_s=float(t[-1] - t_sight) if t_sight is not None else float("nan"),
            head_pan_travel_rad=float(np.abs(np.diff(head[:, 0])).sum()),
            head_tilt_travel_rad=float(np.abs(np.diff(head[:, 1])).sum()),
            tilt_mean_rad=float(head[:, 1].mean()), tilt_std_rad=float(head[:, 1].std()),
            nees_mean=float(nees.mean()),                                # 3 dof: about 3 when the filter is honest
            nees_consistent=float(((nees > 0.216) & (nees < 9.348)).mean()),   # share of time inside the 95% chi-square band
            sigma_xy_mean_m=float(np.mean(self.sig_log)), sigma_xy_final_m=float(self.sig_log[-1]),
            final_pos_err_m=float(pe[-1]),
            coverage_30s=cov_at(30.0), coverage_60s=cov_at(60.0), coverage_90s=cov_at(90.0),
            near_time_s=float(dt[np.array(self.clr_log) < 0.10].sum()),  # time with the robot's edge under 0.10 m from an obstacle
            min_person_gap_m=float(np.min(self.person_gap_log)) if np.isfinite(np.min(self.person_gap_log)) else float("nan"),
            person_near_s=float(dt[np.array(self.person_gap_log) < 0.30].sum()),     # time with a person closer than 0.30 m (surface to surface)
            n_recover=int(a.n_recover), blocked_s=float(a.t_blocked), n_replans=int(a.n_replans),
            dock_seen_frac=float(a.n_dock_frames / frames) if t_sight is not None else float("nan"),
            **gaze_metrics(t, est, head))
        return dict(**extra, outcome=outcome, docked=docked, claimed=claimed, collided=self.collided, person_contact=self.person_contact, miss_m=miss, yaw_err_deg=yaw,
                    duration_s=float(t[-1]), path_length_m=path, time_to_first_sight_s=a.t_first_sight,
                    min_clearance_m=float(self.min_clearance - cfg["robot"]["radius"]),
                    pos_rmse_m=float(np.sqrt(np.mean(pe ** 2))),
                    n_landmarks=len(a.slam.markers), n_spurious_landmarks=n_spurious, landmark_err_m=float(np.mean(lm_err)) if lm_err else float("nan"),
                    coverage=coverage, map_precision=map_precision, reason=a.reason, t=t, est=est, gt=gt)


def make_search_session(cfg, cache, conf, seed, level=None, start=None, sensor="synthetic", camera_sensor=None,
                        random_markers=False, random_dock=False, max_time=None, heights=False, dock_height=None):
    """Build a search trial. `level` names a preset in `levels:` (default the large room with six pillars)."""
    levels = cfg["levels"]
    level = level or "large_6_pillars"
    if level not in levels:
        raise ScenarioError(f"unknown level '{level}'. Available: {', '.join(available_levels(cfg))}")
    if not levels[level].get("available", True):
        raise ScenarioError(f"level '{level}' is planned but not available yet: {levels[level].get('description', '')}")
    c = cfg
    if levels[level].get("set"):                     # a level may change settings (e.g. noisier odometry): `set: ["a.b=1", ...]`
        from activenav.config import apply_overrides
        c = apply_overrides(copy.deepcopy(cfg), levels[level]["set"])
    from activenav.sensing.camera import camera_settings
    if camera_settings(cfg):                         # so may a real-camera profile (e.g. the EKF range-bias fitted to it)
        from activenav.config import apply_overrides
        c = apply_overrides(copy.deepcopy(c), camera_settings(cfg))
    if random_dock:
        c = copy.deepcopy(c)
        c["random_markers"]["keep_ids"] = []
    layout_seed = None
    if random_markers or random_dock:
        layout_seed = seed + (1_000_000 if random_dock else 0)
    ctx = get_world(c, cache, levels[level]["world"], layout_seed, height_seed=seed,
                    heights=(cfg["marker_heights"] if heights else levels[level].get("marker_heights")),
                    dock_height=dock_height)
    if start is None or start == "random":
        start = pick_start(ctx, seed)
    start = tuple(float(v) for v in start)
    x0, x1, y0, y1 = ctx.env.room_bounds()
    if not (x0 + 0.3 < start[0] < x1 - 0.3 and y0 + 0.3 < start[1] < y1 - 0.3):
        raise ScenarioError(f"start ({start[0]:.2f}, {start[1]:.2f}) is outside the usable room "
                            f"(x {x0 + 0.3:.2f}..{x1 - 0.3:.2f}, y {y0 + 0.3:.2f}..{y1 - 0.3:.2f})")
    for o in ctx.obstacles:
        if obstacle_sdf(o, start[0], start[1]) < 0.3:
            raise ScenarioError("start is too close to an obstacle")
    env = ctx.env
    if sensor == "camera":
        from activenav.sensing.floor import FloorRanger
        marker_sensor, front = camera_sensor, FloorRanger(cfg)
    else:
        marker_sensor = SyntheticMarkerSensor(env, np.random.default_rng(1000 + seed), cfg=cfg)
        front = SyntheticObstacleSensor(env, np.random.default_rng(2000 + seed), cfg)
    return SearchSession(ctx, c, conf, seed, start, marker_sensor, front, max_time), ctx
