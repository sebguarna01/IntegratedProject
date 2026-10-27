"""Live viewer for search-and-dock runs (evaluation/: it reads ground truth for DISPLAY only).

Shows what the robot knows: the occupancy map it is building (grey = unknown, white = free, dark = obstacle) in
its own start frame, the landmarks it has found (red = marker 10), its path and frontier target, its belief
(orange, with a 2-sigma ellipse), and the head's field of view. Press `t` to overlay the TRUE room and robot so
you can see how close the map and pose are. Also: the head camera (or a notice if rendering is unavailable).

3D view (free=True, the large panel on the left; draw_free): the TRUE scene rendered by MuJoCo from a camera the
mouse moves, with the true path and robot (blue), the head camera's field of view (green) and the marker ids drawn on
top. It starts straight above the room, framed from the walls of whichever room is loaded. Mouse: left drag rotates,
right drag (or shift + left drag) moves, the wheel (or middle drag, or z / x) zooms, a double-click (or o) goes back
to the view from above; f follows the robot. free=False keeps the original layout: head camera | map, status below.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from activenav.autonomy.occupancy import FREE, OCC
from activenav.evaluation.worlds import ScenarioError, available_levels
from activenav.evaluation.search import SEARCH_CONFIGS, make_search_session, to_frame

BLUE, ORANGE, GREY = (200, 120, 30), (30, 140, 255), (150, 150, 150)
GREEN, RED, YELLOW, MAGENTA, WHITE = (60, 190, 60), (60, 60, 230), (0, 200, 230), (200, 60, 200), (255, 255, 255)
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
CAM_W, CAM_H, MAP_W, MAP_H, HUD_H = 640, 480, 760, 480, 118     # original layout: camera | map, status bar below
# Layout with the 3D view (1340 x 774, fits a 1536 x 864 screen): 3D view over the status panel on the left,
# head camera over the map on the right. Each panel is drawn at its final size, so the text stays sharp.
FREE_W, FREE_H, INFO_H = 780, 585, 189
SIDE_W, SIDE_CAM_H, SIDE_MAP_H = 560, 420, 354
FREE_FOVY = 30.0               # 3D view field of view [deg]: narrower than MuJoCo's 45 so tall walls take less of the top view
FREE_PAD = 0.25                # margin around the room in the view from above [m]
FOLLOW_HALF = 2.5              # f (follow): floor visible above and below the robot when looking straight down [m]
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEYS = ("keys: space pause | + / - speed | r restart | n / p seed | 1-3 config B C-sweep C-info | l level | m random markers | "
        "d random dock | h marker heights | t show truth | v record | s screenshot | q quit")
MOUSE = ("3D view: left drag rotate | right drag (or shift + left drag) move | wheel (or z / x) zoom | "
         "double-click (or o) view from above | f follow the robot")


def project(cam, pts, w, h, depth=None):
    """World points (N, 3) [m] to pixels in a w x h image rendered with `cam` (an mjvGLCamera).
    Returns (uv int32 (N, 2), ok (N,) bool). ok is False for points behind the camera and, given the depth image of
    that render, for points hidden behind something (a wall, a pillar, a person)."""
    pos, fwd, up = np.array(cam.pos, float), np.array(cam.forward, float), np.array(cam.up, float)
    d = np.atleast_2d(np.asarray(pts, float)) - pos
    zc = d @ fwd                                                       # depth along the view axis
    ok = zc > cam.frustum_near
    zc = np.where(ok, zc, 1.0)
    xn = (d @ np.cross(fwd, up)) * cam.frustum_near / zc               # on the near plane
    yn = (d @ up) * cam.frustum_near / zc
    top, bot = cam.frustum_top, cam.frustum_bottom
    half_w = 0.5 * (top - bot) * w / h                                 # MuJoCo matches the frustum width to the image aspect
    u = w / 2 + (xn - cam.frustum_center) / half_w * w / 2
    v = h * (top - yn) / (top - bot)
    uv = np.clip(np.round(np.stack([u, v], axis=1)), -1e5, 1e5).astype(np.int32)
    if depth is not None:                                              # occlusion: something nearer at that pixel
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        seen = depth[np.clip(uv[:, 1], 0, h - 1), np.clip(uv[:, 0], 0, w - 1)]
        ok &= ~inside | (zc <= seen + 0.02 + 0.005 * zc)
    return uv, ok


def surface_height(cam, xs, ys, depth, w, h):
    """World height [m] of what the render shows at pixels (xs, ys), from its depth image."""
    pos, fwd, up = np.array(cam.pos, float), np.array(cam.forward, float), np.array(cam.up, float)
    top, bot, near = cam.frustum_top, cam.frustum_bottom, cam.frustum_near
    xn = (xs + 0.5 - w / 2) / (w / 2) * (0.5 * (top - bot) * w / h) + cam.frustum_center
    yn = top - (ys + 0.5) / h * (top - bot)
    dz = fwd[2] + np.cross(fwd, up)[2] * xn / near + up[2] * yn / near   # height change per metre of depth
    return pos[2] + depth[ys, xs] * dz


def polyline_ok(img, uv, ok, color, thick=1, closed=False):
    """Draw the runs of consecutive points in front of the camera (a point behind it breaks the line)."""
    if closed and ok.all():
        cv2.polylines(img, [uv], True, color, thick, cv2.LINE_AA)
        return
    run = []
    for p, k in zip(uv, ok):
        if k:
            run.append(p)
            continue
        if len(run) > 1:
            cv2.polylines(img, [np.array(run, np.int32)], False, color, thick, cv2.LINE_AA)
        run = []
    if len(run) > 1:
        cv2.polylines(img, [np.array(run, np.int32)], False, color, thick, cv2.LINE_AA)


def wrap(text, width, scale=0.5):
    """Split a status line into lines no wider than `width` pixels."""
    out, cur = [], ""
    for word in text.split(" "):
        trial = word if not cur else cur + " " + word
        if cur and cv2.getTextSize(trial, FONT, scale, 1)[0][0] > width:
            out.append(cur.rstrip())
            cur = word
        else:
            cur = trial
    out.append(cur.rstrip())
    return out


def wheel_delta(flags):
    """Mouse-wheel notches from an EVENT_MOUSEWHEEL `flags` (signed 16 bits at the top; 120 per notch, + = away from you)."""
    d = (int(flags) >> 16) & 0xFFFF
    return (d - 0x10000 if d >= 0x8000 else d) / 120.0


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
                 random_markers=False, random_dock=False, seeds=None, heights=False, free=False):
        self.cfg, self.sensor_kind, self.cache = cfg, sensor, {}
        # 3D view (draw_free): camera state, whether it follows the robot, the room it was framed on, mouse drag
        self.free, self.follow, self.free_ok = free, False, True
        self.view, self._view_env, self._drag, self._mouse = None, None, None, (-1, -1)
        self._moved = False                       # the user moved the 3D view (a window resize then keeps it as it is)
        # panel sizes of the 3D-view layout (set_size lays them out over the window; these are the native ones)
        self.fw, self.fh, self.ih, self.sw, self.sch, self.smh = FREE_W, FREE_H, INFO_H, SIDE_W, SIDE_CAM_H, SIDE_MAP_H
        self.writer_size = None
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
        self.trail_est, self.trail_gt, self.trail_world, self._room = [], [], [], None
        self.cam_img, self.render_ok = None, True
        self.next_display, self.budget, self.result, self._n = 0.0, 0.0, None, 0
        self._sample()
        if self.free and self.env is not self._view_env:                     # another room: frame the 3D view on it
            self._view_env, follow = self.env, self.follow
            self.reset_view()
            if follow:
                self.start_follow()

    def _sample(self):
        s = self.session
        gt = self.env.ground_truth()
        self.trail_est.append(s.agent.pose[:2].copy())
        self.trail_gt.append(to_frame(s.start_pose, gt)[:2])
        self.trail_world.append(gt[:2].copy())                              # same, in the room frame (3D view)

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
        elif self.free and c == "f":
            self.reset_view() if self.follow else self.start_follow()
        elif self.free and c == "o":
            self.reset_view()
        elif self.free and c in ("z", "x"):
            self.zoom(1 if c == "z" else -1)
        return True

    # ------------------------------------------------------------- 3D view camera
    def reset_view(self):
        """3D view straight above the whole room, its long side across the panel; stops following the robot."""
        room = self._room_geometry()
        x0, x1, y0, y1 = room["box"]
        t = np.tan(np.radians(FREE_FOVY) / 2)
        wide = x1 - x0 >= y1 - y0
        across, along = (x1 - x0, y1 - y0) if wide else (y1 - y0, x1 - x0)
        # the wall tops are the closest to the camera, so they set the framing
        d = room["height"] + max((across / 2 + FREE_PAD) * self.fh / self.fw, along / 2 + FREE_PAD) / t
        self.view = dict(lookat=np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]), distance=float(d),
                         azimuth=90.0 if wide else 180.0, elevation=-90.0)   # 90: +x right, +y up; 180: +y right, +x down
        self.follow, self._moved = False, False

    def set_size(self, width: int, height: int):
        """Lay the 3D-view layout out over a window of width x height pixels, with the native proportions (3D view 780 of
        1340 px wide and 585 of 774 px high, head camera 420 of 774 px high). Every panel is then drawn at its own size,
        so a larger window shows more, sharply (no stretching). The 3D view is re-framed unless the user moved it."""
        W, H = max(int(width), 640), max(int(height), 400)
        self.fw = int(round(W * FREE_W / (FREE_W + SIDE_W)))
        self.sw = W - self.fw
        self.fh = int(round(H * FREE_H / (FREE_H + INFO_H)))
        self.ih = H - self.fh
        self.sch = int(round(H * SIDE_CAM_H / (SIDE_CAM_H + SIDE_MAP_H)))
        self.smh = H - self.sch
        if self.view is not None and not self._moved:
            follow = self.follow
            self.reset_view()
            if follow:
                self.start_follow()

    def start_follow(self):
        """Keep the robot in the centre and come closer; the viewing angle stays as it is."""
        self.follow = True
        self.view["distance"] = max(FOLLOW_HALF / np.tan(np.radians(FREE_FOVY) / 2), self._room_geometry()["height"] + 1.0)

    def zoom(self, notches):
        self.view["distance"] = float(np.clip(self.view["distance"] * 0.85 ** notches, 0.6, 300.0))
        self._moved = True

    def drag(self, mode, dx, dy):
        """Mouse drag of (dx, dy) pixels on the 3D view."""
        v = self.view
        self._moved = True
        if mode == "rotate":                       # orbit the look-at point (MuJoCo viewer convention)
            v["azimuth"] = (v["azimuth"] - 0.3 * dx) % 360.0
            v["elevation"] = float(np.clip(v["elevation"] - 0.3 * dy, -90.0, -5.0))
        elif mode == "move":                       # slide over the floor: the scene follows the pointer
            mpp = 2.0 * self._clear_distance(v) * np.tan(np.radians(FREE_FOVY) / 2) / self.fh  # metres per pixel at the look-at point
            a = np.radians(v["azimuth"])
            ahead, right = np.array([np.cos(a), np.sin(a), 0.0]), np.array([np.sin(a), -np.cos(a), 0.0])
            v["lookat"] = v["lookat"] + (dy * ahead - dx * right) * mpp
            self.follow = False
        elif mode == "zoom":
            self.zoom(-dy / 40.0)

    def mouse(self, event, x, y, flags):
        """Mouse event at (x, y) in frame pixels (the window's scale already undone). Acts on the 3D view only."""
        if not self.free or self.view is None:
            return
        inside = 0 <= x < self.fw and 0 <= y < self.fh
        if event == cv2.EVENT_MOUSEMOVE:
            if self._drag:
                self.drag(self._drag, x - self._mouse[0], y - self._mouse[1])
            self._mouse = (x, y)
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self._mouse = (x, y)
            if inside:
                shift = flags & cv2.EVENT_FLAG_SHIFTKEY
                self._drag = {cv2.EVENT_LBUTTONDOWN: "move" if shift else "rotate",
                              cv2.EVENT_RBUTTONDOWN: "move", cv2.EVENT_MBUTTONDOWN: "zoom"}[event]
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_LBUTTONDBLCLK and inside:
            self._drag = None
            self.reset_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            mx, my = self._mouse                   # Windows reports the wheel in screen coordinates: use the last pointer position
            if 0 <= mx < self.fw and 0 <= my < self.fh:
                self.zoom(wheel_delta(flags))

    # ------------------------------------------------------------------ output
    def _out(self, sub):
        p = Path(self.cfg["_root"]) / "results" / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def frame_size(self):
        """(width, height) of compose()."""
        return (self.fw + self.sw, self.fh + self.ih) if self.free else (CAM_W + MAP_W, CAM_H + HUD_H)

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
        self.writer_size = self.frame_size()                     # a video keeps its size if the window is resized meanwhile
        self.writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, self.writer_size)
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
    def _view(self, pw=MAP_W, ph=MAP_H):
        a = self.session.agent
        ext = a.grid.extent()
        x, y = a.pose[:2]
        x0, x1, y0, y1 = (x - 2, x + 2, y - 1.5, y + 1.5) if ext is None else ext
        x0, x1, y0, y1 = min(x0, x - 1), max(x1, x + 1), min(y0, y - 1), max(y1, y + 1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = max(x1 - x0, 6.0), max(y1 - y0, 4.0 * ph / pw * 1.5)
        return MapView((cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), pw, ph, pad=0.3)

    def draw_map(self, pw=MAP_W, ph=MAP_H):
        s, a, v = self.session, self.session.agent, self._view(pw, ph)
        img = np.full((ph, pw, 3), 238, np.uint8)
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
            xb, yb = min(pw, p0[0] + wpx), min(ph, p0[1] + hpx)
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
        half = self.env.camera.hfov / 2                                       # horizontal field of view of the camera in use
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
            cv2.rectangle(img, (6, 6), (pw - 6, 52), (230, 230, 255), -1)
            for i, ln in enumerate([self.notice[j:j + 85] for j in range(0, min(len(self.notice), 170), 85)]):
                cv2.putText(img, ln, (12, 24 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
        if self.result is not None:
            r = self.result
            msg, col = ("DOCKED", (0, 150, 0)) if r["docked"] else (r["outcome"].upper(), (0, 0, 200))
            cv2.rectangle(img, (pw // 2 - 200, ph - 70), (pw // 2 + 200, ph - 12), (255, 255, 255), -1)
            cv2.rectangle(img, (pw // 2 - 200, ph - 70), (pw // 2 + 200, ph - 12), col, 2)
            cv2.putText(img, msg, (pw // 2 - 190, ph - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            cv2.putText(img, f"miss {100 * r['miss_m']:.1f} cm   yaw {r['yaw_err_deg']:.1f} deg   {r['duration_s']:.0f} s",
                        (pw // 2 - 190, ph - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        return img

    def draw_camera(self, pw=CAM_W, ph=CAM_H):
        img = np.full((ph, pw, 3), 60, np.uint8)
        if self.cam_img is not None:
            h, w = self.cam_img.shape[:2]
            if abs(w / h - pw / ph) < 1e-3:
                img = cv2.cvtColor(cv2.resize(self.cam_img, (pw, ph)), cv2.COLOR_RGB2BGR)
            else:                                                       # another aspect ratio (D435i: 16:9): fit, no stretch
                s = min(pw / w, ph / h)
                fw, fh = int(round(w * s)), int(round(h * s))
                x0, y0 = (pw - fw) // 2, (ph - fh) // 2
                img[y0:y0 + fh, x0:x0 + fw] = cv2.cvtColor(cv2.resize(self.cam_img, (fw, fh), interpolation=cv2.INTER_AREA),
                                                           cv2.COLOR_RGB2BGR)
            for z, info in self.session.agent.events:
                put(img, f"marker {z.marker_id}  {z.range:.1f} m", (10, 20 + 18 * int(z.marker_id % 8)), 0.5,
                    GREEN if z.marker_id == self.session.agent.dock_id else WHITE)
        else:
            put(img, "no camera image (GL unavailable)" if not self.render_ok else "rendering...", (pw // 2 - 170, ph // 2), 0.6)
        cam = self.env.camera
        put(img, f"Config {self.conf}   pan {np.degrees(self.session.s.pan):+.0f} deg"
            + ("" if cam.name == "ideal" else f"   camera {cam.name} {cam.width}x{cam.height}"), (10, ph - 12), 0.5, YELLOW)
        return img

    def _status(self):
        """The four status lines (run, search state, estimation and safety, map)."""
        s, a = self.session, self.session.agent
        sg = a.slam.sigma()
        gt = to_frame(s.start_pose, self.env.ground_truth())
        err = float(np.hypot(*(gt[:2] - a.pose[:2])))
        lm = a.dock_landmark()
        return [
            f"search_dock [{self.level}{'  random markers' if self.random_markers else ''}{'  random dock' if self.random_dock else ''}{'  marker heights' if self.heights else ''}]   "
            f"config {self.conf}   seed {self.seed}   speed x{self.speed:g} (actual x{self.actual_speed:.1f}){'   PAUSED' if self.paused else ''}",
            f"t {s.t:6.1f} s   state {a.state}   landmarks found {len(a.slam.markers)}   marker 10 {'FOUND' if lm else 'not found yet'}"
            f"{'' if a.t_first_sight is None else f' (first seen at {a.t_first_sight:.0f} s)'}",
            f"position error {err:.3f} m   sigma_xy {np.hypot(sg[0], sg[1]):.3f} m   clearance "
            f"{self.env.min_clearance() - self.cfg['robot']['radius']:+.2f} m{'   CONTACT' if s.collided else ''}   "
            f"command v {s.last_cmd[0]:+.2f} w {s.last_cmd[1]:+.2f}",
            f"map known: {int((a.grid.state() != 0).sum() * a.grid.res ** 2):d} m2    {a.reason}",
        ]

    def draw_hud(self):
        """Status bar of the original layout (below camera | map)."""
        s, a = self.session, self.session.agent
        img = np.full((HUD_H, CAM_W + MAP_W, 3), 28, np.uint8)
        for i, ln in enumerate(self._status() + [KEYS]):
            put(img, ln, (10, 20 + 22 * i), 0.5, (230, 230, 230) if i < 4 else (160, 160, 160))
        put(img, f"state: {a.state}", (1150, 20), 0.5, STATE_COL.get(a.state, WHITE))
        if s.collided:
            cv2.rectangle(img, (0, 0), (CAM_W + MAP_W - 1, HUD_H - 1), (0, 0, 220), 3)
        return img

    def draw_info(self, pw=None, ph=None):
        """Status panel of the 3D-view layout: the same lines as the status bar, wrapped to the panel, plus the mouse."""
        pw, ph = pw or self.fw, ph or self.ih
        s, a = self.session, self.session.agent
        img = np.full((ph, pw, 3), 28, np.uint8)
        lines = []
        for i, ln in enumerate(self._status()):                           # the first line leaves room for the state label
            lines += [(t, (230, 230, 230)) for t in wrap(ln, pw - (150 if i == 0 else 20))]
        for ln in (KEYS, MOUSE):
            lines += [(t, (160, 160, 160)) for t in wrap(ln, pw - 20)]
        for i, (t, col) in enumerate(lines):
            put(img, t, (10, 18 + 20 * i), 0.5, col)
        put(img, f"state: {a.state}", (pw - 135, 18), 0.5, STATE_COL.get(a.state, WHITE))
        if s.collided:
            cv2.rectangle(img, (0, 0), (pw - 1, ph - 1), (0, 0, 220), 3)
        return img

    def compose(self):
        if not self.free:
            return np.vstack([np.hstack([self.draw_camera(), self.draw_map()]), self.draw_hud()])
        left = np.vstack([self.draw_free(), self.draw_info()])
        right = np.vstack([self.draw_camera(self.sw, self.sch), self.draw_map(self.sw, self.smh)])
        return np.hstack([left, right])

    # ---------------------------------------------------------------- 3D view
    def _room_geometry(self):
        """Outer extent of the walls (x0, x1, y0, y1), wall top height and marker layout of the loaded room, read from
        the model so the 3D view frames any layout. Cached until the next restart (a level change rebuilds it)."""
        if self._room is None:
            m = self.env.model
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for g in range(m.ngeom):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if name.startswith("wall_"):                                # outer walls and wall blocks: axis-aligned boxes
                    lo = np.minimum(lo, m.geom_pos[g] - m.geom_size[g])
                    hi = np.maximum(hi, m.geom_pos[g] + m.geom_size[g])
            self._room = dict(box=(lo[0], hi[0], lo[1], hi[1]), height=float(hi[2]), markers=self.env.marker_map())
        return self._room

    def _free_overlay(self, img, depth, cam):
        """True path, robot, head field of view and marker ids, drawn only where the 3D render shows them (depth test;
        depth None: no test, used when looking straight down, where nothing on the floor hides behind a wall)."""
        s, a, room = self.session, self.session.agent, self._room_geometry()
        P = lambda pts: project(cam, pts, self.fw, self.fh, depth)
        x, y, yaw = self.env.ground_truth()
        # head camera field of view on the floor: the same wedge as on the map, at the TRUE pose, cut at the walls;
        # only floor pixels are tinted, so pillars, people and walls in front of it stay as they are
        half = self.env.camera.hfov / 2
        R = self.cfg["obstacles"]["max_range"]
        fx0, fx1, fy0, fy1 = self.env.room_bounds()
        pts = [(x, y, 0.0)]
        for b in yaw + s.s.pan + np.linspace(-half, half, 24):
            c, sn = np.cos(b), np.sin(b)
            reach = min(R, (fx1 - x) / c if c > 1e-9 else np.inf, (fx0 - x) / c if c < -1e-9 else np.inf,
                        (fy1 - y) / sn if sn > 1e-9 else np.inf, (fy0 - y) / sn if sn < -1e-9 else np.inf)
            pts.append((x + reach * c, y + reach * sn, 0.0))
        uv, ok = project(cam, pts, self.fw, self.fh)                         # no depth test: the wedge is a floor area
        if ok.all():
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mask, [uv], 1)
            ys, xs = np.nonzero(mask)
            if depth is not None:
                floor = np.abs(surface_height(cam, xs, ys, depth, self.fw, self.fh)) < 0.03
                ys, xs = ys[floor], xs[floor]
            img[ys, xs] = (0.82 * img[ys, xs] + 0.18 * np.array(GREEN)).astype(np.uint8)
        if len(self.trail_world) > 1:                                       # true path
            polyline_ok(img, *P([(px, py, 0.01) for px, py in self.trail_world]), BLUE, 2)
        r = self.cfg["robot"]["radius"]                                    # true robot: footprint ring + heading, on the floor
        polyline_ok(img, *P([(x + r * np.cos(q), y + r * np.sin(q), 0.02) for q in np.linspace(0, 2 * np.pi, 33)]), BLUE, 2)
        polyline_ok(img, *P([(x + k * np.cos(yaw), y + k * np.sin(yaw), 0.02) for k in np.linspace(0, 0.45, 10)]), BLUE, 2)
        for mid, m in room["markers"].items():                              # marker ids on each plate
            n = m["normal_angle"]
            uv, ok = P([(m["x"] + 0.05 * np.cos(n), m["y"] + 0.05 * np.sin(n), m["z"])])
            if ok[0]:
                put(img, str(mid), (int(uv[0, 0]) - 6, int(uv[0, 1]) + 5), 0.45, RED if mid == a.dock_id else WHITE)
        return img

    def _clear_distance(self, v):
        """Camera distance to draw with: pulled in along the view ray when the camera would sit outside the room below
        the wall tops (it would only see the back of a wall). The distance set with the mouse is kept for later."""
        d, (lx, ly, lz) = v["distance"], v["lookat"]
        x0, x1, y0, y1 = self.env.room_bounds()
        m = 0.15
        if not (x0 + m < lx < x1 - m and y0 + m < ly < y1 - m):
            return d                                                        # looking at a point outside the room
        a, e = np.radians(v["azimuth"]), np.radians(v["elevation"])
        bx, by = -np.cos(e) * np.cos(a), -np.cos(e) * np.sin(a)          # camera position = look-at + d * (bx, by, -sin e)
        cx, cy, cz = lx + d * bx, ly + d * by, lz - d * np.sin(e)
        if cz >= self._room_geometry()["height"] + 0.1 or (x0 + m <= cx <= x1 - m and y0 + m <= cy <= y1 - m):
            return d
        for b, lo, hi, l in ((bx, x0 + m, x1 - m, lx), (by, y0 + m, y1 - m, ly)):
            if b > 1e-9:
                d = min(d, (hi - l) / b)
            elif b < -1e-9:
                d = min(d, (lo - l) / b)
        return max(d, 0.6)

    def draw_free(self):
        """3D view: MuJoCo's render of the TRUE scene from the mouse-driven camera (display only), with the overlays."""
        img = np.full((self.fh, self.fw, 3), 60, np.uint8)
        if self.free_ok:
            v = self.view
            if self.follow:
                v["lookat"][:2] = self.env.ground_truth()[:2]
            try:
                rgb, depth, cam = self.env.render_free(v["lookat"], self._clear_distance(v), v["azimuth"], v["elevation"],
                                                       self.fw, self.fh, fovy=FREE_FOVY, depth=v["elevation"] > -80.0)
            except Exception as e:
                self.free_ok = False
                print("3D view disabled (could not render):", str(e).splitlines()[0][:100])
            else:
                img = self._free_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth, cam)
        if not self.free_ok:
            put(img, "no 3D view (GL unavailable)", (self.fw // 2 - 140, self.fh // 2), 0.6)
        put(img, f"3D view (simulator truth)   {self.level}{'   following the robot' if self.follow else ''}", (10, 20), 0.5)
        if self.result is not None:
            msg, col = ("DOCKED", GREEN) if self.result["docked"] else (self.result["outcome"].upper(), RED)
            (tw, _), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
            put(img, msg, (self.fw - tw - 12, 26), 0.7, col, 2)
        return img
