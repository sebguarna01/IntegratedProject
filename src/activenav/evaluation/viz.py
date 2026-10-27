"""Live viewer for search-and-dock runs (evaluation/: it reads ground truth for DISPLAY only).

Shows what the robot knows: the occupancy map it is building (grey = unknown, white = free, dark = obstacle) in
its own start frame, the landmarks it has found (red = marker 10), its path and frontier target, its belief
(orange, with a 2-sigma ellipse), and the head's field of view. Press `t` to overlay the TRUE room and robot so
you can see how close the map and pose are. Left: the head camera (or a notice if rendering is unavailable).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from activenav.autonomy.occupancy import FREE, OCC
from activenav.evaluation.worlds import ScenarioError, available_levels
from activenav.evaluation.search import SEARCH_CONFIGS, make_search_session, to_frame

BLUE, ORANGE, GREY = (200, 120, 30), (30, 140, 255), (150, 150, 150)
GREEN, RED, YELLOW, MAGENTA, WHITE = (60, 190, 60), (60, 60, 230), (0, 200, 230), (200, 60, 200), (255, 255, 255)
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CAM_W, CAM_H, MAP_W, MAP_H, HUD_H = 640, 480, 760, 480, 118


class MapView:
    """World (metres, y up) to pixel mapping that fits a room into a w x h panel."""

    def __init__(self, bounds, w, h, pad=0.35):
        x0, x1, y0, y1 = bounds
        self.s = min(w / (x1 - x0 + 2 * pad), h / (y1 - y0 + 2 * pad))
        self.ox = (w - self.s * (x1 - x0)) / 2 - self.s * x0
        self.oy = (h - self.s * (y1 - y0)) / 2 + self.s * y1

    def p(self, x, y):
        return int(round(self.ox + self.s * x)), int(round(self.oy - self.s * y))

    def r(self, metres):
        return max(1, int(round(self.s * metres)))


def put(img, text, org, scale=0.48, color=WHITE, thick=1):
    # outline = the same text in black at eight 1-px offsets (a thicker stroke would also change the glyph spacing
    # and leave dark "ghost" text sticking out past the end of the line)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def dashed(img, pts, color, thick=1, dash=7, gap=5):
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        L = np.linalg.norm(b - a)
        if L < 1:
            continue
        d = (b - a) / L
        k = 0.0
        while k < L:
            cv2.line(img, tuple((a + d * k).astype(int)), tuple((a + d * min(k + dash, L)).astype(int)), color, thick, cv2.LINE_AA)
            k += dash + gap


STATE_COL = {"SCAN": YELLOW, "EXPLORE": WHITE, "APPROACH": ORANGE, "DOCK": GREEN, "DONE": GREEN, "FAILED": RED}


class SearchWatcher:
    def __init__(self, cfg, conf="C-info", seed=None, level=None, start=None, sensor="synthetic", speed=1.0,
                 random_markers=False, random_dock=False, seeds=None, heights=False):
        self.cfg, self.sensor_kind, self.cache = cfg, sensor, {}
        self.camera_sensor = None
        if sensor == "camera":
            from activenav.sensing.markers import MarkerSensor
            self.camera_sensor = MarkerSensor(cfg)
        self.conf, self.level, self.start = conf, level or "large_6_pillars", start
        self.random_markers, self.random_dock, self.heights = random_markers, random_dock, heights
        self.seeds = list(seeds) if seeds else None
        self.seed = (self.seeds[0] if self.seeds else cfg["seed"]) if seed is None else seed
        self.speed, self.paused, self.writer, self.actual_speed = speed, False, None, speed
        self.show_truth, self.notice, self.runs, self._good = False, "", [], None
        self.restart()

    # --------------------------------------------------------------- lifecycle
    def restart(self):
        args = dict(conf=self.conf, seed=self.seed, level=self.level, start=self.start, sensor=self.sensor_kind,
                    camera_sensor=self.camera_sensor, random_markers=self.random_markers, random_dock=self.random_dock,
                    heights=self.heights)
        try:
            self.session, self.ctx = make_search_session(self.cfg, self.cache, **args)
        except ScenarioError as e:
            if self._good is None:
                raise
            self.notice = str(e)
            print("cannot run that:", e, flush=True)
            for k, v in self._good.items():
                setattr(self, k, v)
            args.update(conf=self.conf, seed=self.seed, level=self.level, start=self.start,
                        random_markers=self.random_markers, random_dock=self.random_dock, heights=self.heights)
            self.session, self.ctx = make_search_session(self.cfg, self.cache, **args)
        else:
            self.notice = ""
        self._good = dict(conf=self.conf, seed=self.seed, level=self.level, start=self.start,
                          random_markers=self.random_markers, random_dock=self.random_dock, heights=self.heights)
        self.env = self.session.env
        self.trail_est, self.trail_gt = [], []
        self.cam_img, self.render_ok = None, True
        self.next_display, self.budget, self.result, self._n = 0.0, 0.0, None, 0
        self._sample()

    def _sample(self):
        s = self.session
        self.trail_est.append(s.agent.pose[:2].copy())
        self.trail_gt.append(to_frame(s.start_pose, self.env.ground_truth())[:2])

    def _finish(self):
        r = self.session.result()
        self.result = r
        self.runs.append(dict(level=self.level, config=self.conf, seed=self.seed, outcome=r["outcome"], docked=int(r["docked"]),
                              duration_s=round(r["duration_s"], 1), miss_m=round(r["miss_m"], 3), yaw_err_deg=round(r["yaw_err_deg"], 1),
                              min_clearance_m=round(r["min_clearance_m"], 2), coverage=round(r["coverage"], 2)))
        print(f"{self.level} {self.conf} seed {self.seed}: {r['outcome'].upper()}  miss {r['miss_m']:.3f} m  yaw {r['yaw_err_deg']:.1f} deg  "
              f"{r['duration_s']:.0f} s  clearance {r['min_clearance_m']:+.2f} m", flush=True)

    # ---------------------------------------------------------------- stepping
    def _update_camera(self):
        s = self.session
        if self.camera_sensor is not None:
            if s.last_image is not None:
                self.cam_img = s.last_image
            return
        if s.t >= self.next_display and self.render_ok:
            self.next_display = s.t + 0.1
            try:
                self.cam_img = self.env.render(noise=False)
            except Exception as e:
                self.render_ok = False
                print("camera panel disabled (could not render):", str(e).splitlines()[0][:100])

    def advance_steps(self, n):
        for _ in range(n):
            if self.result is not None:
                return
            if not self.session.step():
                self._finish()
                return
            self._n += 1
            if self._n % 5 == 0:
                self._sample()
            self._update_camera()

    def advance(self, wall_dt):
        if self.paused or self.result is not None:
            return
        self.budget += min(wall_dt, 0.1) * self.speed
        n = int(self.budget / self.env.control_dt)
        self.budget -= n * self.env.control_dt
        t0, sim0 = time.perf_counter(), self.session.t
        self.advance_steps(min(n, 400))
        if n:
            self.actual_speed = 0.8 * self.actual_speed + 0.2 * ((self.session.t - sim0) / max(time.perf_counter() - t0, 1e-6))

    # -------------------------------------------------------------------- keys
    def handle_key(self, k):
        if k in (ord("q"), 27):
            return False
        c = chr(k) if 0 < k < 128 else ""
        if c == " ":
            self.paused = not self.paused
        elif c in "+=":
            self.speed = SPEEDS[min(SPEEDS.index(self.speed) + 1, len(SPEEDS) - 1)] if self.speed in SPEEDS else 1.0
        elif c in "-_":
            self.speed = SPEEDS[max(SPEEDS.index(self.speed) - 1, 0)] if self.speed in SPEEDS else 1.0
        elif c == "r":
            self.restart()
        elif c in "np":
            d = 1 if c == "n" else -1
            if self.seeds:
                i = self.seeds.index(self.seed) if self.seed in self.seeds else -1
                self.seed = self.seeds[(i + d) % len(self.seeds)]
            else:
                self.seed = max(0, self.seed + d)
            self.restart()
        elif c in "123":
            self.conf = SEARCH_CONFIGS[int(c) - 1]
            self.restart()
        elif c == "l":
            lv = available_levels(self.cfg)
            self.level = lv[(lv.index(self.level) + 1) % len(lv)] if self.level in lv else lv[0]
            self.restart()
        elif c == "m":
            self.random_markers = not self.random_markers
            self.restart()
        elif c == "d":
            self.random_dock = not self.random_dock
            self.restart()
        elif c == "h":
            self.heights = not self.heights
            self.restart()
        elif c == "t":
            self.show_truth = not self.show_truth
        elif c == "s":
            self.screenshot()
        elif c == "v":
            self.toggle_recording()
        return True

    def _out(self, sub):
        p = Path(self.cfg["_root"]) / "results" / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def screenshot(self):
        p = self._out("screenshots") / time.strftime(f"search_{self.conf}_s{self.seed}_%H%M%S.png")
        cv2.imwrite(str(p), self.compose())
        print("saved", p)

    def toggle_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("recording stopped")
            return
        p = self._out("videos") / time.strftime(f"search_{self.conf}_s{self.seed}_%H%M%S.mp4")
        self.writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (CAM_W + MAP_W, CAM_H + HUD_H))
        print("recording to", p)

    def save_log(self):
        if not self.runs:
            return None
        import csv
        p = self._out("") / time.strftime("watch_search_%Y%m%d_%H%M%S.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.runs[0]))
            w.writeheader()
            w.writerows(self.runs)
        print(f"\n{'level':22s} {'config':8s} runs docked   median time [s]")
        for key in sorted({(r["level"], r["config"]) for r in self.runs}):
            rs = [r for r in self.runs if (r["level"], r["config"]) == key]
            print(f"{key[0]:22s} {key[1]:8s} {len(rs):4d} {sum(r['docked'] for r in rs):6d}   {np.median([r['duration_s'] for r in rs]):8.0f}")
        print("saved", p)
        return p

    # ----------------------------------------------------------------- drawing
    def _view(self):
        a = self.session.agent
        ext = a.grid.extent()
        x, y = a.pose[:2]
        x0, x1, y0, y1 = (x - 2, x + 2, y - 1.5, y + 1.5) if ext is None else ext
        x0, x1, y0, y1 = min(x0, x - 1), max(x1, x + 1), min(y0, y - 1), max(y1, y + 1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, 6.0), max(y1 - y0, 4.0 * MAP_H / MAP_W * 1.5)
        return MapView((cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), MAP_W, MAP_H, pad=0.3)

    def draw_map(self):
        s, a, v = self.session, self.session.agent, self._view()
        img = np.full((MAP_H, MAP_W, 3), 238, np.uint8)
        g = a.grid
        st = g.state()
        ys, xs = np.nonzero(st)
        if len(xs):
            iy0, iy1, ix0, ix1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            col = np.full((iy1 - iy0, ix1 - ix0, 3), 200, np.uint8)
            sub = st[iy0:iy1, ix0:ix1]
            col[sub == FREE] = (255, 255, 255)
            col[sub == OCC] = (70, 60, 55)
            wx0, wy1 = g.x0 + ix0 * g.res, g.y0 + iy1 * g.res
            p0, p1 = v.p(wx0, wy1), v.p(g.x0 + ix1 * g.res, g.y0 + iy0 * g.res)
            wpx, hpx = max(1, p1[0] - p0[0]), max(1, p1[1] - p0[1])
            patch = cv2.resize(col[::-1], (wpx, hpx), interpolation=cv2.INTER_NEAREST)
            xa, ya = max(0, p0[0]), max(0, p0[1])
            xb, yb = min(MAP_W, p0[0] + wpx), min(MAP_H, p0[1] + hpx)
            if xb > xa and yb > ya:
                img[ya:yb, xa:xb] = patch[ya - p0[1]:yb - p0[1], xa - p0[0]:xb - p0[0]]
        if self.show_truth:                                                   # display only: the true room in the start frame
            x0, x1, y0, y1 = self.env.room_bounds()
            pts = [to_frame(s.start_pose, (px, py, 0))[:2] for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            cv2.polylines(img, [np.array([v.p(*p) for p in pts], np.int32)], True, (0, 150, 0), 1, cv2.LINE_AA)
            for o in self.ctx.obstacles:
                if "x0" in o:
                    pts = [to_frame(s.start_pose, (px, py, 0))[:2] for px, py in
                           ((o["x0"], o["y0"]), (o["x1"], o["y0"]), (o["x1"], o["y1"]), (o["x0"], o["y1"]))]
                    cv2.polylines(img, [np.array([v.p(*p) for p in pts], np.int32)], True, (0, 150, 0), 1, cv2.LINE_AA)
                    continue
                c = to_frame(s.start_pose, (o["x"], o["y"], 0))[:2]
                cv2.circle(img, v.p(*c), v.r(o["r"]), (0, 150, 0), 1, cv2.LINE_AA)
            for pc in self.env.people_state():                                # walking people (display only)
                c = to_frame(s.start_pose, (pc["x"], pc["y"], 0))[:2]
                cv2.circle(img, v.p(*c), v.r(pc["r"]), (30, 120, 230), -1, cv2.LINE_AA)
            for mid, m in self.env.marker_map().items():
                c = to_frame(s.start_pose, (m["x"], m["y"], 0))[:2]
                cv2.circle(img, v.p(*c), 4, (0, 150, 0), 1, cv2.LINE_AA)
        for trail, colr, th in ((self.trail_gt, BLUE, 1), (self.trail_est, ORANGE, 2)):
            if self.show_truth or trail is self.trail_est:
                if len(trail) > 1:
                    cv2.polylines(img, [np.array([v.p(*p) for p in trail], np.int32)], False, colr, th, cv2.LINE_AA)
        for mid, m in a.slam.markers.items():                                  # landmarks the robot found
            n = m["normal_angle"]
            tx, ty = -np.sin(n), np.cos(n)
            cv2.line(img, v.p(m["x"] - 0.2 * tx, m["y"] - 0.2 * ty), v.p(m["x"] + 0.2 * tx, m["y"] + 0.2 * ty),
                     (0, 0, 220) if mid == a.dock_id else (40, 40, 40), 4)
            lx, ly = v.p(m["x"] + 0.4 * np.cos(n), m["y"] + 0.4 * np.sin(n))
            put(img, str(mid), (lx - 6, ly + 5), 0.45, (0, 0, 220) if mid == a.dock_id else (60, 60, 60))
        if a.path:
            dashed(img, [v.p(*a.pose[:2])] + [v.p(*p) for p in a.path], (170, 140, 90), 1)
        if a.target is not None:
            cv2.drawMarker(img, v.p(*a.target), RED if a.state == "APPROACH" else MAGENTA, cv2.MARKER_STAR, 12, 2)
        x, y, th = a.pose
        half = np.arctan(np.tan(np.radians(self.cfg["camera_model"]["fovy_deg"]) / 2) * 640 / 480)
        yaw = th + s.s.pan
        R = self.cfg["obstacles"]["max_range"]
        wedge = [v.p(x, y)] + [v.p(x + R * np.cos(yaw + q), y + R * np.sin(yaw + q)) for q in np.linspace(-half, half, 16)]
        ov = img.copy()
        cv2.fillPoly(ov, [np.array(wedge, np.int32)], GREEN)
        img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
        cv2.circle(img, v.p(x, y), v.r(self.cfg["robot"]["radius"]), ORANGE, 2, cv2.LINE_AA)
        cv2.line(img, v.p(x, y), v.p(x + 0.3 * np.cos(th), y + 0.3 * np.sin(th)), ORANGE, 2, cv2.LINE_AA)
        vals, vecs = np.linalg.eigh(a.slam.P[:2, :2])
        vals = np.sqrt(np.maximum(vals, 0)) * 2.0
        ang = -np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        cv2.ellipse(img, v.p(x, y), (max(3, v.r(vals[1])), max(3, v.r(vals[0]))), ang, 0, 360, ORANGE, 1, cv2.LINE_AA)
        if self.notice:
            cv2.rectangle(img, (6, 6), (MAP_W - 6, 52), (230, 230, 255), -1)
            for i, ln in enumerate([self.notice[j:j + 85] for j in range(0, min(len(self.notice), 170), 85)]):
                cv2.putText(img, ln, (12, 24 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
        if self.result is not None:
            r = self.result
            msg, col = ("DOCKED", (0, 150, 0)) if r["docked"] else (r["outcome"].upper(), (0, 0, 200))
            cv2.rectangle(img, (MAP_W // 2 - 200, MAP_H - 70), (MAP_W // 2 + 200, MAP_H - 12), (255, 255, 255), -1)
            cv2.rectangle(img, (MAP_W // 2 - 200, MAP_H - 70), (MAP_W // 2 + 200, MAP_H - 12), col, 2)
            cv2.putText(img, msg, (MAP_W // 2 - 190, MAP_H - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            cv2.putText(img, f"miss {100 * r['miss_m']:.1f} cm   yaw {r['yaw_err_deg']:.1f} deg   {r['duration_s']:.0f} s",
                        (MAP_W // 2 - 190, MAP_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        return img

    def draw_camera(self):
        img = np.full((CAM_H, CAM_W, 3), 60, np.uint8)
        if self.cam_img is not None:
            img = cv2.cvtColor(cv2.resize(self.cam_img, (CAM_W, CAM_H)), cv2.COLOR_RGB2BGR)
            for z, info in self.session.agent.events:
                put(img, f"marker {z.marker_id}  {z.range:.1f} m", (10, 20 + 18 * int(z.marker_id % 8)), 0.5,
                    GREEN if z.marker_id == self.session.agent.dock_id else WHITE)
        else:
            put(img, "no camera image (GL unavailable)" if not self.render_ok else "rendering...", (CAM_W // 2 - 170, CAM_H // 2), 0.6)
        put(img, f"Config {self.conf}   pan {np.degrees(self.session.s.pan):+.0f} deg", (10, CAM_H - 12), 0.5, YELLOW)
        return img

    def draw_hud(self):
        s, a = self.session, self.session.agent
        img = np.full((HUD_H, CAM_W + MAP_W, 3), 28, np.uint8)
        sg = a.slam.sigma()
        gt = to_frame(s.start_pose, self.env.ground_truth())
        err = float(np.hypot(*(gt[:2] - a.pose[:2])))
        lm = a.dock_landmark()
        lines = [
            f"search_dock [{self.level}{'  random markers' if self.random_markers else ''}{'  random dock' if self.random_dock else ''}{'  marker heights' if self.heights else ''}]   "
            f"config {self.conf}   seed {self.seed}   speed x{self.speed:g} (actual x{self.actual_speed:.1f}){'   PAUSED' if self.paused else ''}",
            f"t {s.t:6.1f} s   state {a.state}   landmarks found {len(a.slam.markers)}   marker 10 {'FOUND' if lm else 'not found yet'}"
            f"{'' if a.t_first_sight is None else f' (first seen at {a.t_first_sight:.0f} s)'}",
            f"position error {err:.3f} m   sigma_xy {np.hypot(sg[0], sg[1]):.3f} m   clearance "
            f"{self.env.min_clearance() - self.cfg['robot']['radius']:+.2f} m{'   CONTACT' if s.collided else ''}   "
            f"command v {s.last_cmd[0]:+.2f} w {s.last_cmd[1]:+.2f}",
            f"map known: {int((a.grid.state() != 0).sum() * a.grid.res ** 2):d} m2    {a.reason}",
            "keys: space pause | + / - speed | r restart | n / p seed | 1-3 config B C-sweep C-info | l level | m random markers | "
            "d random dock | h marker heights | t show truth | v record | s screenshot | q quit",
        ]
        for i, ln in enumerate(lines):
            put(img, ln, (10, 20 + 22 * i), 0.5, (230, 230, 230) if i < 4 else (160, 160, 160))
        put(img, f"state: {a.state}", (1150, 20), 0.5, STATE_COL.get(a.state, WHITE))
        if s.collided:
            cv2.rectangle(img, (0, 0), (CAM_W + MAP_W - 1, HUD_H - 1), (0, 0, 220), 3)
        return img

    def compose(self):
        return np.vstack([np.hstack([self.draw_camera(), self.draw_map()]), self.draw_hud()])
