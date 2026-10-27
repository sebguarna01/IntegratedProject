"""The search-and-dock agent: explore an unknown room, find marker 10, avoid obstacles, dock in front of it.

It is given NOTHING about the world. Its inputs are wheel odometry increments, marker measurements and obstacle
scans from the camera; from them it builds a landmark map (SLAM), an occupancy map and a plan. States:

  SCAN      initial look-around (body turn for Config B, head sweep for C)
  EXPLORE   drive to the best frontier (edge of known free floor and unknown space) through known-free cells
  APPROACH  marker 10 is in the map: plan to a staging point in front of it
  DOCK      close to the marker: servo on the live measurements of it to the dock point, facing it
  DONE / FAILED

The head is steered by a gaze policy while exploring (autonomy/search_gaze.py) and points at marker 10 once known.
Never reads simulator ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from activenav.autonomy import explore
from activenav.autonomy.marker_view import tilt_for_marker
from activenav.autonomy.obstacle_sensor import GroundGeometry
from activenav.autonomy.occupancy import OccupancyGrid
from activenav.control.go_to_point import GoToPointGains, go_to_point
from activenav.control.lqr import LqrTracker, fillet_path
from activenav.estimation.slam import SlamEKF
from activenav.kinematics import wrap_angle

PAN_LIM = 2.9


class SearchAgent:
    def __init__(self, cfg: dict, gaze, control_dt: float = 0.02):
        s = cfg["search"]
        self.cfg, self.s, self.gaze, self.dt = cfg, s, gaze, control_dt
        self.dock_id = int(s["dock_marker"])
        self.standoff, self.stage_extra = float(s["standoff"]), float(s["stage_extra"])
        self.v_max, self.w_max = float(s["v_max"]), float(s["w_max"])
        self.tilt_search, self.tilt_dock = float(s["tilt_search"]), float(s["tilt_dock"])
        ft = s.get("fixed_tilt")                  # B and C-sweep: the head tilt is locked at this angle for the whole run
        self.tilt_hold = float(ft) if (ft is not None and gaze.name != "C-info") else None
        self.tilt_base = self.tilt_search if self.tilt_hold is None else self.tilt_hold
        fp = s.get("fixed_pan")
        self.pan_hold = float(fp) if (fp is not None and gaze.name == "B") else None
        self.slew = float(s["gaze"]["slew"])
        self.slam = SlamEKF(cfg)
        self.grid = OccupancyGrid(cfg)
        geo = GroundGeometry(cfg)
        self.cam_h, self.vfov = geo.cam_height(), geo.vfov           # known from the head geometry (not from the world)
        self.body_scan = gaze.name == "B"
        self.reset()

    def reset(self):
        self.slam.reset(np.zeros(3))
        self.grid = OccupancyGrid(self.cfg)
        if hasattr(self.gaze, "reset"):
            self.gaze.reset()
        self.state = "SCAN" if self.s["initial_scan"] else "EXPLORE"
        self.path, self.target, self.cm = [], None, None
        self._lqr_src, self._lqr, self._lqr_vs = None, None, None
        self.t_plan = -1e9
        self.t_state = 0.0
        self.rot_acc = 0.0
        self.pan_cmd = 0.0
        self.tilt_cmd = self.tilt_base
        self.lm_z = {}                           # marker id -> recent measured centre heights [m]
        self.dock_meas = None                    # (t, MarkerMeasurement) most recent view of the dock marker
        self.t_first_sight = None
        self.n_frames = self.n_dock_frames = self.n_replans = 0      # bookkeeping for the study metrics
        self.t_blocked = 0.0                     # seconds the safety reflex held the robot back
        self.last_scan = None
        self.events = []
        self.reason = ""
        self.t_dock_lost = None
        self.dock_hist = []
        self.no_frontier_scans = 0
        self.clutter = False
        self.stage = None
        self.scan_lock = False                     # a forced look-around that marker detection must not cut short
        self.t_nofrontier = -1e9
        self.visited = []                          # frontier targets already reached: not worth going back to
        self.t_still = 0.0                         # last time the robot was moving (or had nowhere to go)
        self.recover_until, self.recover_cmd, self.n_recover = -1e9, (0.0, 0.0), 0

    # ----------------------------------------------------------------- inputs
    def predict(self, ds: float, dtheta: float) -> None:
        self.slam.predict(ds, dtheta)
        self.rot_acc += abs(dtheta)

    def on_frame(self, t: float, measurements, scan) -> None:
        self.events = []
        self.n_frames += 1
        if self.t_first_sight is not None and any(z.marker_id == self.dock_id for z in measurements):
            self.n_dock_frames += 1
        for z in measurements:
            if z.height is not None and z.range < 3.5:
                self.lm_z.setdefault(z.marker_id, []).append(z.height)
                self.lm_z[z.marker_id] = self.lm_z[z.marker_id][-15:]
            info = self.slam.update_marker(z, t=t)
            self.events.append((z, info))
            if z.marker_id == self.dock_id and info.accepted or (z.marker_id == self.dock_id and info.reason == "bearing only"):
                self.dock_meas = (t, z)
                if self.t_first_sight is None:
                    self.t_first_sight = t
        if scan is not None:
            self.last_scan = scan
            self.grid.integrate(self.slam.pose, scan)
        self.grid.mark_footprint(self.slam.pose)

    # ------------------------------------------------------------------ helpers
    @property
    def pose(self):
        return self.slam.pose

    def lm_height(self, marker_id):
        """Measured centre height of a marker, or None if no height has been measured yet."""
        v = self.lm_z.get(marker_id)
        return float(np.median(v)) if v else None

    def dock_landmark(self):
        return self.slam.markers.get(self.dock_id)

    def _set(self, state, t, reason=""):
        if state != self.state:
            self.state, self.t_state, self.path, self.t_plan = state, t, [], -1e9
        self.reason = reason or self.reason

    def _margin(self) -> float:
        sg = float(np.hypot(*self.slam.sigma()[:2]))
        return float(min(self.s["plan_margin"] + self.s["margin_sigma_k"] * sg, self.s["margin_max"]))

    def _look_at(self, bearing: float):
        return float(np.clip(bearing, -PAN_LIM, PAN_LIM))

    # ------------------------------------------------------------------- command
    def command(self, t: float):
        """(v, w, pan_target, tilt_target) for this control step."""
        v = w = 0.0
        pan, tilt = 0.0, self.tilt_base
        lm = self.dock_landmark()
        if lm is not None and self.state in ("SCAN", "EXPLORE") and not self.scan_lock:
            self._set("APPROACH", t)
        if self.state == "SCAN":
            v, w, pan = self._scan(t)
        elif self.state == "EXPLORE":
            v, w = self._explore(t)
            pan, tilt = self.gaze(self, t)
            if self.clutter and v > 0.0 and self.gaze.name == "C-info":
                lim = np.radians(37.6) - 0.15                     # keep the direction of travel inside the view
                pan = float(np.clip(pan, self._travel_bearing() - lim, self._travel_bearing() + lim))
        elif self.state == "APPROACH":
            v, w = self._approach(t)
            pan, tilt = self._head_on_dock(t)
        elif self.state == "DOCK":
            v, w = self._dock(t)
            pan, tilt = self._head_on_dock(t, dock=True)
        if self.pan_hold is not None:                                # B: a fixed head never pans, whatever the state
            pan = self.pan_hold
        v_wanted = v
        v = self._safety(v)
        if v_wanted > 0.0 and v == 0.0:
            self.t_blocked += self.dt
        v, w = self._recover(t, v, w)
        # the head moves at a limited rate, whatever the policy asks for
        step = self.slew * self.dt
        self.pan_cmd += float(np.clip(pan - self.pan_cmd, -step, step))
        self.tilt_cmd += float(np.clip(tilt - self.tilt_cmd, -step, step))
        return v, w, self.pan_cmd, self.tilt_cmd

    def _recover(self, t: float, v: float, w: float):
        """Deadlock breaker. If the robot has a plan but has not moved for `stuck_s`, drop that target, back off a little
        if the map shows clear floor behind it (else turn on the spot), and plan again."""
        if t < self.recover_until:
            return self.recover_cmd
        want = self.state in ("EXPLORE", "APPROACH") and bool(self.path)
        if not want or abs(v) > 0.01 or abs(w) > 0.05:
            self.t_still = t
            return v, w
        if t - self.t_still < float(self.s.get("stuck_s", 8.0)):
            return v, w
        self.t_still = t
        self.n_recover += 1
        if self.target is not None:
            self.visited.append(tuple(self.target))
        self.path, self.t_plan = [], -1e9
        x, y, th = self.pose
        clear_behind = all(self._free_of_obstacles(x - d * np.cos(th), y - d * np.sin(th), self.cfg["robot"]["radius"] + 0.05)
                           for d in (0.25, 0.4))
        self.recover_cmd = (-0.1, 0.0) if clear_behind else (0.0, 0.6)
        self.recover_until = t + 3.0
        return self.recover_cmd

    def _free_of_obstacles(self, px: float, py: float, r: float) -> bool:
        """The map shows floor at (px, py) and no mapped obstacle within r of it."""
        g = self.grid
        iy, ix = g.to_cell(px, py)
        if not (0 <= iy < g.n and 0 <= ix < g.n) or g.state()[iy, ix] != 1:
            return False
        k = int(np.ceil((r + 0.05) / g.res))
        y0, y1, x0, x1 = max(0, iy - k), min(g.n, iy + k + 1), max(0, ix - k), min(g.n, ix + k + 1)
        oy, ox = np.nonzero(g.state()[y0:y1, x0:x1] == 2)
        if len(ox) == 0:
            return True
        wx, wy = g.to_world(oy + y0, ox + x0)
        return bool(np.min(np.hypot(wx - px, wy - py)) >= r)

    def _safety(self, v: float) -> float:
        """Last-resort reflex: something close in front in the latest scan -> do not drive forward."""
        if v > 0 and self.state != "DOCK" and self._map_blocks_ahead():
            return 0.0
        sc = self.last_scan
        if sc is None or v <= 0 or self.state == "DOCK":
            return v
        p = sc.points[sc.hit]
        if len(p) and np.any((p[:, 0] > 0.0) & (p[:, 0] < 0.38) & (np.abs(p[:, 1]) < 0.28)):
            return 0.0
        return v

    def _map_blocks_ahead(self) -> bool:
        """The body (radius) would overlap a mapped obstacle 0.15 m further along the current heading."""
        g = self.grid
        x, y, th = self.pose
        px, py = x + 0.15 * np.cos(th), y + 0.15 * np.sin(th)
        k = int(np.ceil(0.9 / g.res))
        cy, cx = g.to_cell(px, py)
        y0, y1, x0, x1 = max(0, cy - k), min(g.n, cy + k + 1), max(0, cx - k), min(g.n, cx + k + 1)
        oy, ox = np.nonzero(g.state()[y0:y1, x0:x1] == 2)
        if len(ox) == 0:
            return False
        wx, wy = g.to_world(oy + y0, ox + x0)
        return bool(np.min(np.hypot(wx - px, wy - py)) < self.cfg["robot"]["radius"])     # planner floor is radius + 0.03: no deadlock on the edge

    # --------------------------------------------------------------------- SCAN
    def _scan(self, t):
        if self.body_scan:
            if self.rot_acc >= 2 * np.pi:
                self.scan_lock = False
                self._set("EXPLORE", t)
                return 0.0, 0.0, 0.0
            return 0.0, 0.6, 0.0
        tt = t - self.t_state                              # head sweep: 0 -> +PAN_LIM -> -PAN_LIM -> 0 at 1 rad/s
        a = PAN_LIM
        T = 4 * a
        if tt >= T + 0.3:
            self._set("EXPLORE", t)
            return 0.0, 0.0, 0.0
        if tt < a:
            pan = tt
        elif tt < 3 * a:
            pan = a - (tt - a)
        else:
            pan = -a + (tt - 3 * a)
        return 0.0, 0.0, float(np.clip(pan, -a, a))

    # ------------------------------------------------------------------ EXPLORE
    def _plan_view(self):
        cm = explore.build_coarse(self.grid, self.pose, self._margin(), self.s["coarse_res"])
        self.cm = cm
        return cm

    def _explore(self, t):
        if t - self.t_plan >= self.s["replan_s"] or not self.path:
            self.t_plan = t
            self._replan_explore(t)
        if self.state != "EXPLORE":
            return 0.0, 0.0
        return self._follow(t)

    def _replan_explore(self, t, goal_bias=None):
        self.n_replans += 1
        cm = self._plan_view()
        self.path, self.target = [], None
        if cm is None:
            return
        dist, parent = explore.dijkstra(cm, self.pose[:2])
        # after one fruitless look-around, accept even the smallest frontier (a doorway seen at an angle leaves very few cells)
        min_cells = int(self.s["frontier_min_cells"]) if self.no_frontier_scans == 0 else 1
        clusters = explore.frontier_clusters(cm, min_cells)
        best, best_cost = None, np.inf
        for cells in clusters:
            if self.visited:
                cx, cy = cm.to_world(cells[:, 0], cells[:, 1])
                keep = np.ones(len(cells), bool)
                for vx, vy in self.visited:
                    keep &= np.hypot(cx - vx, cy - vy) > 0.6
                cells = cells[keep]
                if len(cells) < min_cells:
                    continue
            d = dist[cells[:, 0], cells[:, 1]]
            if not np.isfinite(d).any():
                continue
            k = int(np.argmin(d))
            cost = float(d[k]) - 0.05 * min(len(cells), 20) * cm.res
            if goal_bias is not None:
                wx, wy = cm.to_world(*cells[k])
                cost += float(np.hypot(wx - goal_bias[0], wy - goal_bias[1]))
            if cost < best_cost:
                best, best_cost = tuple(cells[k]), cost
        if best is None:
            self._no_frontier(t)
            return
        self.target = cm.to_world(*best)
        self.path = explore.path_to(cm, parent, self.pose[:2], best)
        self._update_clutter(cm)

    def _no_frontier(self, t):
        """Nothing left to explore: look round once more (the map may have gaps), then give up."""
        if t - self.t_nofrontier < 3.0:
            return
        self.t_nofrontier = t
        self.no_frontier_scans += 1
        if self.no_frontier_scans <= 2 or (self.state == "APPROACH" and self.no_frontier_scans <= 4):
            self.state, self.t_state, self.rot_acc, self.path = "SCAN", t, 0.0, []
            self.body_scan, self.scan_lock = True, True
            return
        self._set("FAILED", t, "no frontier left and marker not found")

    def _update_clutter(self, cm):
        """True when the next metre of the path runs close to a KNOWN obstacle or wall. There the head must look where
        the robot is going, whatever the gaze policy would prefer. Unknown space does not count: a path that ends at a
        frontier always touches it, and that would pin the head forward for the whole exploration."""
        self.clutter = False
        if cm is None or not self.path:
            return
        bad = cv2.dilate(((~cm.trav) & (cm.unknown < 0.5)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)      # +-0.4 m
        pts, last, acc = [], self.pose[:2], 0.0
        for p in self.path:
            seg = np.hypot(p[0] - last[0], p[1] - last[1])
            n = max(1, int(seg / cm.res))
            for k in range(1, n + 1):
                pts.append(last + (np.array(p) - last) * k / n)
            acc += seg
            last = np.array(p)
            if acc > 1.0:
                break
        for q in pts:
            iy, ix = cm.to_cell(*q)
            if cm.inside(iy, ix) and bad[iy, ix]:
                self.clutter = True
                return

    def _travel_bearing(self):
        """Bearing of the next path point relative to the body (what lies ahead), or 0 with no path."""
        if not self.path:
            return 0.0
        x, y, th = self.pose
        return float(wrap_angle(np.arctan2(self.path[0][1] - y, self.path[0][0] - x) - th))

    def _follow(self, t, v_max=None):
        if not self.path:
            return 0.0, 0.0
        if self.s.get("follow", "lqr") == "lqr":
            return self._follow_lqr(v_max)
        gains = GoToPointGains(v_max=v_max or self.v_max, w_max=self.w_max,
                               pos_tol=0.06 if len(self.path) == 1 else 0.15)
        v, w, reached = go_to_point(self.pose, self.path[0], gains)
        if reached:
            self._pop_waypoint()
            return 0.0, 0.0
        return v, w

    def _pop_waypoint(self):
        self.path.pop(0)
        if not self.path:
            self.t_plan = -1e9
            if self.state in ("EXPLORE", "APPROACH") and self.stage is None and self.target is not None:
                self.visited.append(tuple(self.target))

    def _follow_lqr(self, v_max=None):
        """Track the planner's polyline with the LQR tracker on a filleted copy of it. The reference is built once per plan
        (so the cross-track error is measured against the planned line, not the pose at the last tick); waypoints behind
        the robot are dropped from self.path, which the gaze and clutter logic read."""
        if self._lqr_src is not self.path:
            ref, vs = fillet_path([self.pose[:2], *self.path], radius=float(self.s.get("fillet_radius", 0.3)))
            self._lqr = LqrTracker(ref, v_ref=self.v_max, w_max=self.w_max, rotate_above=0.8)
            self._lqr_src, self._lqr_vs = self.path, vs
        trk = self._lqr
        v, w = trk.command(self.pose)
        remaining = trk.path.length - float(trk.path.s[trk.i])
        if remaining < 0.06 or np.hypot(*(np.asarray(self.path[-1]) - self.pose[:2])) < 0.06:
            self.path.clear()
            self.t_plan = -1e9
            if self.state in ("EXPLORE", "APPROACH") and self.stage is None and self.target is not None:
                self.visited.append(tuple(self.target))
            return 0.0, 0.0
        s_now = float(trk.path.s[trk.i])
        while len(self.path) > 1 and self._lqr_vs[len(self._lqr_vs) - len(self.path)] <= s_now:
            self.path.pop(0)
        return min(v, v_max or v, remaining), w

    # ----------------------------------------------------------------- APPROACH
    def _stage_goal(self, cm, dist):
        if self.stage is not None:                          # keep the staging point we were already heading for (no flip-flopping)
            iy, ix = cm.to_cell(*self.stage)
            if cm.inside(iy, ix) and np.isfinite(dist[iy, ix]):
                return (iy, ix)
        lm = self.dock_landmark()
        n = np.array([np.cos(lm["normal_angle"]), np.sin(lm["normal_angle"])])
        L = np.array([lm["x"], lm["y"]])
        for extra in (0.0, 0.2, 0.5, 0.8):
            pt = L + (self.standoff + self.stage_extra + extra) * n
            c = explore.nearest_traversable(cm, dist, pt, 0.3)
            if c is not None:
                return c
        return None

    def _approach(self, t):
        lm = self.dock_landmark()
        m = self.dock_meas
        fresh = m is not None and t - m[0] < 0.6
        if fresh and m[1].range <= 1.3 and abs(m[1].bearing) < 0.7:
            self._set("DOCK", t)
            return 0.0, 0.0
        if self.stage is not None and np.hypot(*(np.array(self.stage) - self.pose[:2])) < 0.25 and self.t_state < t - 1.0:
            self._set("DOCK", t)                              # at the staging point: servo on the marker
            return 0.0, 0.0
        if t - self.t_plan >= self.s["replan_s"] or not self.path:
            self.t_plan = t
            cm = self._plan_view()
            self.path = []
            if cm is not None:
                dist, parent = explore.dijkstra(cm, self.pose[:2])
                goal = self._stage_goal(cm, dist)
                self.stage = None
                if goal is not None:
                    self.target = self.stage = cm.to_world(*goal)
                    self.path = explore.path_to(cm, parent, self.pose[:2], goal)
                    self._update_clutter(cm)
                    if not self.path and np.hypot(*(np.array(self.stage) - self.pose[:2])) < 0.25:
                        self._set("DOCK", t)
                        return 0.0, 0.0
                else:                                       # staging point not known to be reachable yet: explore toward it
                    self._replan_explore(t, goal_bias=(lm["x"], lm["y"]))
                    if self.state == "FAILED":
                        return 0.0, 0.0
        return self._follow(t)

    # --------------------------------------------------------------------- DOCK
    def _head_on_dock(self, t, dock=False):
        lm = self.dock_landmark()
        m = self.dock_meas
        fresh = m is not None and t - m[0] < 0.6
        if lm is None:
            return self.gaze(self, t)[0], self.tilt_base
        x, y, th = self.pose
        r = m[1].range if fresh else float(np.hypot(lm["x"] - x, lm["y"] - y))
        close = dock or r < 1.5
        z = self.lm_height(self.dock_id)
        if self.tilt_hold is not None:                              # B and C-sweep with a locked tilt: never moves
            tilt = self.tilt_hold
        elif (self.gaze.name != "C-info" and not self.s.get("adaptive_dock_tilt", False)) or z is None:
            tilt = self.tilt_dock if close else self.tilt_search    # fixed tilts (dock tilt close in, search tilt otherwise)
        else:                                                       # keep the whole marker in view, centre it when close
            tilt = tilt_for_marker(self.cam_h, r, z, self.vfov, self.tilt_search, centre=close)
        if fresh:
            return self._look_at(m[1].bearing), tilt                # keep it centred (bearing is relative to the body)
        return self._look_at(float(wrap_angle(np.arctan2(lm["y"] - y, lm["x"] - x) - th))), tilt

    def _dock(self, t):
        m = self.dock_meas
        if m is None or t - m[0] > 0.8:
            lm = self.dock_landmark()
            if self.pan_hold is not None and lm is not None:   # the head cannot turn to look: turn the body towards the
                x, y, th = self.pose                           # SLAM estimate of the marker until it comes into view
                b = float(wrap_angle(np.arctan2(lm["y"] - y, lm["x"] - x) - th))
                if abs(b) > 0.15:
                    self.t_dock_lost = None
                    return 0.0, float(np.clip(1.5 * b, -0.6, 0.6))
                if np.hypot(lm["x"] - x, lm["y"] - y) < self.standoff - 0.05:     # facing it but too close to see all of it: back off
                    return -0.05, float(np.clip(1.5 * b, -0.3, 0.3))
            if self.t_dock_lost is None:
                self.t_dock_lost = t
            if t - self.t_dock_lost > 4.0:                  # lost it: go back to planning
                self.t_dock_lost = None
                self._set("APPROACH", t)
            return 0.0, 0.0
        self.t_dock_lost = None
        z = m[1]
        if not self.dock_hist or self.dock_hist[-1][0] != m[0]:
            self.dock_hist.append((m[0], z.range, z.bearing, z.normal_angle))
            self.dock_hist = self.dock_hist[-5:]
        r, b, a = z.range, z.bearing, z.normal_angle
        dist = self._dock_error(r, b, a)
        if dist < 0.15 and len(self.dock_hist) >= 3:       # close in: average the last frames (slow, so the lag is small)
            h = np.array(self.dock_hist)
            r, b = h[:, 1].mean(), h[:, 2].mean()
            a = float(np.arctan2(np.sin(h[:, 3]).mean(), np.cos(h[:, 3]).mean()))
            dist = self._dock_error(r, b, a)
        Dx, Dy = self._dock_point(r, b, a)
        if dist <= 0.04 and abs(b) < 0.02 and abs(wrap_angle(a - np.pi)) < np.radians(2.0) and len(self.dock_hist) >= 5:
            self._set("DONE", t, "docked")
            return 0.0, 0.0
        fixed = self.pan_hold is not None
        misaligned = abs(wrap_angle(a - np.pi)) >= np.radians(2.0)
        if dist > 0.04 or (fixed and misaligned and abs(b) < 0.05 and dist > 0.015):
            err = float(np.arctan2(Dy, Dx))
            if fixed:
                # Fixed head: the marker must stay in view (half field of view ~0.52 rad). Turn towards the dock point by at most
                # 0.4 rad, and creep so the lateral error is worked off in small arcs; this also covers the case of sitting inside
                # the 0.04 m gate with the marker centred but the approach angle a few degrees off, which needs a sideways shift.
                if abs(err) > np.pi / 2 and dist < 0.30:      # dock point behind: back up, marker centred
                    return -float(np.clip(dist, 0.03, 0.06)), float(np.clip(2.0 * b, -0.4, 0.4))
                ec = float(np.clip(err, -0.4, 0.4))
                return float(np.clip(1.0 * dist, 0.03, 0.12)) * max(0.0, np.cos(ec)), float(np.clip(2.0 * ec, -self.w_max, self.w_max))
            w = float(np.clip(2.0 * err, -self.w_max, self.w_max))
            v = float(np.clip(1.0 * dist, 0.0, 0.12)) * max(0.0, np.cos(err))
            return v, w
        return 0.0, float(np.clip(2.0 * b, -0.4, 0.4))

    def _dock_point(self, r, b, a):
        """Dock point in the robot frame: standoff in front of the marker along its normal."""
        return (r * np.cos(b) + self.standoff * np.cos(a), r * np.sin(b) + self.standoff * np.sin(a))

    def _dock_error(self, r, b, a) -> float:
        return float(np.hypot(*self._dock_point(r, b, a)))
