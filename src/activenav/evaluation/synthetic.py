"""Synthetic sensors: marker measurements and obstacle scans drawn from the measured error models (no rendering), for fast studies and tests."""
from __future__ import annotations

import numpy as np

from activenav.autonomy.marker_view import vertically_visible
from activenav.autonomy.obstacle_sensor import GroundGeometry
from activenav.autonomy.obstacle_sensor import ObstacleScan
from activenav.geometry import marker_fully_visible
from activenav.kinematics import wrap_angle
from activenav.types import MarkerMeasurement, SensorBundle


# -------------------- evaluation/synthetic.py --------------------
# Stand-in marker sensor that needs no rendering (for fast studies and headless machines).
_NORMAL_R = [1.0, 2.0, 3.0, 4.0, 5.25]            # bin centres of the characterisation table [m]

_NORMAL_STD_DEG = [0.19, 0.30, 0.91, 1.45, 6.6]

class SyntheticMarkerSensor:
    needs_image = False

    def __init__(self, env, rng: np.random.Generator, hfov_deg: float = 75.2,
                 max_range: float = 5.0, max_view_deg: float = 75.0, marker_ids=None, occlusion: bool = True,
                 cfg: dict | None = None):
        self.env, self.rng = env, rng
        self.obstacles = env.obstacle_map() if occlusion else []      # pillars hide markers (exact 2D line-of-sight test)
        self.markers = {k: v for k, v in env.marker_map().items() if marker_ids is None or k in marker_ids}
        self.half_fov = np.radians(hfov_deg) / 2.0
        self.max_range, self.max_view = max_range, np.radians(max_view_deg)
        self.geo = GroundGeometry(cfg) if cfg is not None else None       # camera height and vertical field of view

    def measure(self, sb: SensorBundle) -> list[MarkerMeasurement]:
        x, y, th = self.env.ground_truth()
        out = []
        for mid, m in self.markers.items():
            dx, dy = m["x"] - x, m["y"] - y
            r = float(np.hypot(dx, dy))
            bearing = float(wrap_angle(np.arctan2(dy, dx) - th))
            view = abs(wrap_angle(np.arctan2(-dy, -dx) - m["normal_angle"]))
            if r > self.max_range or r < 0.3 or view > self.max_view:
                continue
            if abs(wrap_angle(bearing - sb.pan)) > self.half_fov - 0.05:     # camera yaw = pan
                continue
            if not marker_fully_visible((x, y), m, self.obstacles + self.env.people_state()):
                continue
            if self.geo is not None and not bool(vertically_visible(self.geo.cam_height(), sb.tilt, r, m["z"], self.geo.vfov, 0.03)):
                continue                                                   # too high or too low for the current tilt
            sn = np.radians(np.interp(r, _NORMAL_R, _NORMAL_STD_DEG))
            out.append(MarkerMeasurement(
                marker_id=mid,
                range=r + 0.0023 * r ** 2 + self.rng.normal(0.0, 0.0012 * r ** 2),
                bearing=float(bearing + self.rng.normal(0.0, np.radians(0.12))),
                normal_angle=float(wrap_angle(m["normal_angle"] - th + self.rng.normal(0.0, sn))),
                height=float(m["z"] + self.rng.normal(0.0, 0.004 * r))))
        return out


# -------------------- evaluation/synthetic.py --------------------
# Stand-in obstacle front-end for headless runs: ray-casts the TRUE walls and pillars from the camera pose and
# degrades the result the way ground-plane ranging does (limited near/far visibility, range error growing as r^2,
# an obstacle too close to see its base is reported at the nearest visible floor). It is a proxy for
# sensing/floor.py (the real camera path); it reads ground truth, so it lives in evaluation/.
def _ray_circle(o, d, c, r):
    f = o - c
    b = f @ d
    disc = b * b - (f @ f - r * r)
    if disc < 0:
        return np.inf
    t = -b - np.sqrt(disc)
    return t if t > 0 else np.inf

def _ray_box_inside(o, d, b):
    """Distance along d from a point inside the rectangle b=(x0,x1,y0,y1) to its boundary."""
    ts = []
    for k, (lo, hi) in enumerate(((b[0], b[1]), (b[2], b[3]))):
        if d[k] > 1e-12:
            ts.append((hi - o[k]) / d[k])
        elif d[k] < -1e-12:
            ts.append((lo - o[k]) / d[k])
    return min(ts) if ts else np.inf

def _ray_rect(o, d, r):
    """Distance along d from a point outside the rectangle to where the ray enters it (inf if it misses)."""
    t0, t1 = 0.0, np.inf
    for k, (lo, hi) in enumerate(((r["x0"], r["x1"]), (r["y0"], r["y1"]))):
        if abs(d[k]) < 1e-12:
            if o[k] < lo or o[k] > hi:
                return np.inf
        else:
            ta, tb = (lo - o[k]) / d[k], (hi - o[k]) / d[k]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return np.inf
    return t0 if t0 > 0 else np.inf

def _ray_obstacle(o, d, p):
    return _ray_rect(o, d, p) if "x0" in p else _ray_circle(o, d, np.array([p["x"], p["y"]]), p["r"])

class SyntheticObstacleSensor:
    needs_image = False

    def __init__(self, env, rng: np.random.Generator, cfg: dict, hfov_deg: float = 75.2):
        self.env, self.rng = env, rng
        self.geo = GroundGeometry(cfg)
        o = cfg["obstacles"]
        self.max_range, self.px_sigma, self.n_rays = float(o["max_range"]), float(o["pixel_sigma"]), int(o["n_rays"])
        self.half = np.radians(hfov_deg) / 2.0
        self.bounds = env.room_bounds()
        self.pillars = env.obstacle_map()                           # static; walking people are added at every scan

    def cast(self, o_w, ang) -> float:
        """True horizontal distance from world point o_w along world azimuth ang to the nearest wall or pillar."""
        d = np.array([np.cos(ang), np.sin(ang)])
        return float(min([_ray_box_inside(o_w, d, self.bounds)] +
                         [_ray_obstacle(o_w, d, p) for p in self.pillars + self.env.people_state()]))

    def scan(self, sb) -> ObstacleScan:
        x, y, th = self.env.ground_truth()
        R, cam = self.geo.pose(sb.pan, sb.tilt)
        c, s = np.cos(th), np.sin(th)
        o_w = np.array([x, y]) + np.array([[c, -s], [s, c]]) @ cam[:2]
        near, far = self.geo.near_far(sb.tilt, self.max_range)
        pts, hits, sig = [], [], []
        solid = self.pillars + self.env.people_state()
        for a in np.linspace(-self.half * 0.97, self.half * 0.97, self.n_rays):
            ang = th + sb.pan + a
            d = np.array([np.cos(ang), np.sin(ang)])
            t = min([_ray_box_inside(o_w, d, self.bounds)] + [_ray_obstacle(o_w, d, p) for p in solid])
            hit = t <= far
            rng_m = max(t, near) if hit else far                      # base hidden below the image: report the nearest floor
            sg = float(self.geo.range_sigma(rng_m, self.px_sigma))
            rng_n = max(near, rng_m + self.rng.normal(0.0, sg))
            ang_n = ang + self.rng.normal(0.0, 0.003)
            end_w = o_w + rng_n * np.array([np.cos(ang_n), np.sin(ang_n)])
            rel = end_w - np.array([x, y])
            pts.append([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])      # back to the base frame
            hits.append(hit)
            sig.append(sg)
        return ObstacleScan(cam[:2].copy(), np.array(pts), np.array(hits), np.array(sig))

    scan_sb = scan
