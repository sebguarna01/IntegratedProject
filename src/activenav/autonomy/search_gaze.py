"""Head policies for the search task. Each is a callable  policy(agent, t) -> (pan, tilt)  that sees only the
agent's own belief (SLAM pose, occupancy map, landmarks). Used while exploring; the agent itself points the head
at marker 10 once it has been found.

  FixedSearchGaze   Config B: head straight ahead (the body has to turn to look anywhere else)
  SweepSearchGaze   Config C-sweep: head swings left-right on a timer, ignoring what it knows
  InfoSearchGaze    Config C-info: picks the pan with the best expected gain: unknown map in view (this is also
                    where marker 10 could be) plus known landmarks in view when position uncertainty is high

All of them can run an initial look-around by head sweep (C) or leave it to the body (B).
"""
from __future__ import annotations

import numpy as np

from activenav.autonomy.marker_view import vertically_visible
from activenav.autonomy.obstacle_sensor import GroundGeometry
from activenav.autonomy.occupancy import UNKNOWN
from activenav.kinematics import wrap_angle


class FixedSearchGaze:
    name = "B"

    def __init__(self, cfg):
        ft = cfg["search"].get("fixed_tilt")            # null: the tilt is free (search tilt while exploring, dock tilt close in)
        self.tilt = float(cfg["search"]["tilt_search"] if ft is None else ft)

    def reset(self):
        pass

    def __call__(self, agent, t):
        return 0.0, self.tilt


class SweepSearchGaze:
    """Constant-rate pan back and forth between -amplitude and +amplitude (triangle wave), ignoring everything the
    robot knows. Amplitude [rad] and period [s] come from search.gaze.sweep_amp / sweep_period."""
    name = "C-sweep"

    def __init__(self, cfg, amplitude: float | None = None, period: float | None = None):
        g = cfg["search"]["gaze"]
        ft = cfg["search"].get("fixed_tilt")            # null: the tilt is free (search tilt while exploring, dock tilt close in)
        self.tilt = float(cfg["search"]["tilt_search"] if ft is None else ft)
        self.amp = float(g.get("sweep_amp", 1.2) if amplitude is None else amplitude)
        self.period = float(g.get("sweep_period", 8.0) if period is None else period)

    def reset(self):
        pass

    def __call__(self, agent, t):
        ph = (t / self.period) % 1.0                          # 0..1 over one full cycle
        tri = 4.0 * ph if ph < 0.25 else (2.0 - 4.0 * ph if ph < 0.75 else 4.0 * ph - 4.0)     # 0 -> +1 -> -1 -> 0
        return float(self.amp * tri), self.tilt


class InfoSearchGaze:
    """Chooses pan AND tilt every `replan_s`. A candidate (pan, tilt) is scored by
      floor    unknown map in view whose floor is actually visible at that tilt (looking up hides the near floor),
      marker   space in view, out to marker-detection range, that could hold a marker (unknown or a known wall) and
               has not been SEARCHED yet, weighted by how much of the plausible height band would fit the vertical view
               (a high marker is invisible from close by at the default tilt). The head keeps a coarse memory of what
               it has looked at, because the floor is only mapped to 3 m while markers are detected out to ~5 m,
               so that ring never turns into 'known' map by itself,
      landmark known landmarks in view, with their measured height, when position uncertainty is high,
    minus a cost for moving the head."""
    name = "C-info"
    CELL = 5                                   # search-memory cell = 5 map cells (0.25 m)

    def __init__(self, cfg, pan_range=(-3.0, 3.0), hfov_deg: float | None = None):
        if hfov_deg is None:                       # the camera profile's horizontal field of view
            from activenav.sensing.camera import hfov_deg as camera_hfov_deg
            hfov_deg = camera_hfov_deg(cfg)
        g = cfg["search"]["gaze"]
        self.tilt = float(cfg["search"]["tilt_search"])
        self.tilts = [float(v) for v in g.get("tilt_cands", [self.tilt])]
        if self.tilt not in self.tilts:
            self.tilts.append(self.tilt)
        self.n_cand, self.slew, self.dt = int(g["n_cand"]), float(g["slew"]), float(g["replan_s"])
        self.switch_cost, self.w_unk, self.w_lm = float(g["switch_cost"]), float(g["w_unknown"]), float(g["w_landmark"])
        self.w_marker = float(g.get("w_marker", 1.0))
        self.switch_cost_tilt = float(g.get("switch_cost_tilt", 3.0))
        self.sigma_ref = float(g["sigma_ref"])
        self.max_range = float(cfg["obstacles"]["max_range"])
        self.marker_range = float(g.get("marker_range", 4.5))
        self.near_limit = float(g.get("near_limit", 0.45))     # nearest floor [m] that may be out of view near obstacles
        self.prior_z = np.asarray(cfg.get("marker_heights", [0.2, 0.3, 0.45, 0.6]), float)   # what the robot may assume
        self.geo = GroundGeometry(cfg)
        self.half = np.radians(hfov_deg) / 2.0 - 0.05
        self.lo, self.hi = pan_range[0] + 0.1, pan_range[1] - 0.1
        self.reset()

    def reset(self):
        self.cmd, self.cmd_tilt, self.t_next = 0.0, self.tilt, -1.0
        self.searched = None
        az = np.linspace(-self.half, self.half, 9)
        # floor fan: only the range the floor can be mapped to
        self.fan = np.array([(a, d) for a in az for d in np.linspace(0.5, self.max_range, 6)])
        # marker fan: out to marker-detection range
        self.mfan = np.array([(a, d) for a in np.linspace(-self.half, self.half, 13)
                              for d in np.linspace(0.5, self.marker_range, 9)])
        self._cov = {}

    # ----------------------------------------------------------------- terms
    def cov(self, tilt, fan):
        """Share of the plausible marker heights that fit the vertical view, per fan sample, for this tilt."""
        key = (round(float(tilt), 4), len(fan))
        if key not in self._cov:
            h = self.geo.cam_height()
            self._cov[key] = np.mean([vertically_visible(h, tilt, fan[:, 1], z, self.geo.vfov, 0.05) for z in self.prior_z], axis=0)
        return self._cov[key]

    def _cells(self, grid, pose, pan, fan):
        x, y, th = pose
        ang = th + pan + fan[:, 0]
        iy, ix = grid.to_cell(x + fan[:, 1] * np.cos(ang), y + fan[:, 1] * np.sin(ang))
        ok = (iy >= 0) & (iy < grid.n) & (ix >= 0) & (ix < grid.n)
        return iy, ix, ok

    def unknown_mask(self, grid, pose, pan, state=None):
        st = grid.state() if state is None else state
        iy, ix, ok = self._cells(grid, pose, pan, self.fan)
        unk = np.zeros(len(self.fan), bool)
        unk[ok] = st[iy[ok], ix[ok]] == UNKNOWN
        return unk

    def unknown_in_view(self, grid, pose, pan) -> float:
        """Unknown map samples within the obstacle-ranging range (kept for the pan-only use and the tests)."""
        return float(self.unknown_mask(grid, pose, pan).sum())

    def floor_gain(self, unk, tilt) -> float:
        d = self.fan[:, 1]
        near, far = self.geo.near_far(tilt, self.max_range)
        return float((unk & (d >= near) & (d <= far)).sum())

    def marker_gain(self, grid, state, pose, pan, tilt) -> float:
        if self.searched is None:
            return 0.0
        iy, ix, ok = self._cells(grid, pose, pan, self.mfan)
        st = np.full(len(self.mfan), -1)
        st[ok] = state[iy[ok], ix[ok]]
        could = (st == UNKNOWN) | (st == 2)                    # unknown space or a wall: a marker could hang there
        seen = np.zeros(len(self.mfan))
        seen[ok] = self.searched[iy[ok] // self.CELL, ix[ok] // self.CELL]
        return float((could * np.maximum(0.0, self.cov(tilt, self.mfan) - seen)).sum())

    def mark_searched(self, grid, pose, pan, tilt):
        if self.searched is None:
            m = grid.n // self.CELL + 1
            self.searched = np.zeros((m, m), np.float32)
        iy, ix, ok = self._cells(grid, pose, pan, self.mfan)
        c = self.cov(tilt, self.mfan)
        for a, b, k, cv in zip(iy[ok] // self.CELL, ix[ok] // self.CELL, np.nonzero(ok)[0], c[ok]):
            if cv > self.searched[a, b]:
                self.searched[a, b] = cv

    def landmark_gain(self, slam, pan, tilt=None, heights=None) -> float:
        sig = float(np.hypot(*slam.sigma()[:2]))
        if sig <= self.sigma_ref:
            return 0.0
        x, y, th = slam.pose
        g = 0.0
        h = self.geo.cam_height()
        for mid, m in slam.markers.items():
            dx, dy = m["x"] - x, m["y"] - y
            r = np.hypot(dx, dy)
            if 0.5 <= r <= 4.5 and abs(wrap_angle(np.arctan2(dy, dx) - th - pan)) < self.half:
                if tilt is not None:
                    z = (heights or {}).get(mid)
                    z = float(np.mean(self.prior_z)) if z is None else z
                    if not bool(vertically_visible(h, tilt, r, z, self.geo.vfov, 0.05)):
                        continue
                g += min(1.0, sig / self.sigma_ref - 1.0)
        return g

    def __call__(self, agent, t):
        if t >= self.t_next:
            self.t_next = t + self.dt
            pose, grid = agent.slam.pose, agent.grid
            self.mark_searched(grid, pose, agent.pan_cmd, agent.tilt_cmd)        # what the head has just been looking at
            state = grid.state()
            cands = np.linspace(self.lo, self.hi, self.n_cand)
            heights = {mid: agent.lm_height(mid) for mid in agent.slam.markers}
            best, best_s = (self.cmd, self.cmd_tilt), -np.inf
            tilts = self.tilts
            if agent.clutter:                      # near known obstacles the floor right in front must stay in view
                tilts = [tl for tl in self.tilts if self.geo.near_far(tl, self.max_range)[0] <= self.near_limit] or [self.tilt]
            for p in cands:
                unk = self.unknown_mask(grid, pose, p, state)
                for tl in tilts:
                    sc = (self.w_unk * self.floor_gain(unk, tl) + self.w_marker * self.marker_gain(grid, state, pose, p, tl)
                          + self.w_lm * self.landmark_gain(agent.slam, p, tl, heights)
                          - self.switch_cost * abs(p - self.cmd) * self.w_unk * 10
                          - self.switch_cost_tilt * abs(tl - self.cmd_tilt))
                    if sc > best_s:
                        best, best_s = (float(p), float(tl)), sc
            self.cmd, self.cmd_tilt = best
        return self.cmd, self.cmd_tilt
