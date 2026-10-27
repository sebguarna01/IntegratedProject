"""Log-odds occupancy grid built only from obstacle scans and the filter's pose estimate.

The grid lives in the SLAM frame (origin = where the robot started, since nothing about the room is known) and
is big enough for any room; unknown cells stay unknown until a ray crosses them. Free space is marked along each
ray out to the end of the seen floor, an obstacle at the end if it was a hit.

Never reads simulator ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from activenav.autonomy.obstacle_sensor import ObstacleScan

UNKNOWN, FREE, OCC = 0, 1, 2


class OccupancyGrid:
    def __init__(self, cfg: dict):
        o = cfg["obstacles"]
        self.res = float(o["grid_res"])
        n = int(round(float(o["grid_size"]) / self.res))
        self.n = n
        self.x0 = self.y0 = -n * self.res / 2.0
        self.l_occ, self.l_free, self.clamp = float(o["l_occ"]), float(o["l_free"]), float(o["l_clamp"])
        self.lo = np.zeros((n, n), np.float32)
        self.seen = np.zeros((n, n), bool)
        self.robot_radius = float(cfg["robot"]["radius"])

    # ---------------------------------------------------------------- helpers
    def to_cell(self, x, y):
        return (np.floor((np.asarray(y) - self.y0) / self.res).astype(int),
                np.floor((np.asarray(x) - self.x0) / self.res).astype(int))

    def to_world(self, iy, ix):
        return self.x0 + (np.asarray(ix) + 0.5) * self.res, self.y0 + (np.asarray(iy) + 0.5) * self.res

    # ----------------------------------------------------------------- update
    def integrate(self, pose, scan: ObstacleScan) -> None:
        if len(scan.points) == 0:
            return
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        o = np.array([x, y]) + R @ scan.origin
        ends = np.array([x, y]) + scan.points @ R.T
        free_idx = []
        sig = scan.sigma if len(scan.sigma) == len(ends) else np.zeros(len(ends))
        far = []                                          # end of the free part of each ray (None if it has none)
        for e, sg in zip(ends, sig):
            L = float(np.hypot(*(e - o)))
            # free space stops 2 sigma short of the measured end: with range noise, freeing right up to the end would
            # keep erasing the obstacle surface the same rays (and rays from other views) just marked occupied
            Lf = max(0.0, L - 2.0 * float(sg) - self.res)
            if Lf <= 0.0:
                far.append(None)
                continue
            far.append(o + (e - o) * (Lf / L))
            k = max(2, int(np.ceil(Lf / (self.res * 0.7))))
            t = np.linspace(0.0, Lf / L, k)
            pts = o[None] + t[:, None] * (e - o)[None]
            iy, ix = self.to_cell(pts[:, 0], pts[:, 1])
            free_idx.append(iy * self.n + ix)
        if not free_idx:
            return
        free_idx.append(self._wedges(o, far))
        free = np.unique(np.concatenate(free_idx))
        free = free[(free >= 0) & (free < self.n * self.n)]
        flat = self.lo.ravel()
        flat[free] = np.maximum(flat[free] + self.l_free, -self.clamp)
        self.seen.ravel()[free] = True
        hit_pts = ends[scan.hit]
        if len(hit_pts):
            iy, ix = self.to_cell(hit_pts[:, 0], hit_pts[:, 1])
            ok = (iy >= 0) & (iy < self.n) & (ix >= 0) & (ix < self.n)
            idx = np.unique(iy[ok] * self.n + ix[ok])
            flat[idx] = np.minimum(flat[idx] + self.l_occ, self.clamp)
            self.seen.ravel()[idx] = True

    def _wedges(self, o, far, max_gap: float = 0.4) -> np.ndarray:
        """Cells inside the triangle (camera, end of ray i, end of ray i+1) for neighbouring rays whose free ends are
        close together. Far from the camera the rays are wider apart than a cell, and the stripes between them would
        stay unknown for ever (a doorway seen from across the room would look like a wall)."""
        tris = [(a, b) for a, b in zip(far[:-1], far[1:]) if a is not None and b is not None
                and float(np.hypot(*(a - b))) <= max_gap]
        if not tris:
            return np.zeros(0, np.int64)
        pts = np.array([[o, a, b] for a, b in tris])                       # (m, 3, 2)
        px = (pts[..., 0] - self.x0) / self.res
        py = (pts[..., 1] - self.y0) / self.res
        xa, xb = int(max(0, np.floor(px.min()) - 1)), int(min(self.n, np.ceil(px.max()) + 2))
        ya, yb = int(max(0, np.floor(py.min()) - 1)), int(min(self.n, np.ceil(py.max()) + 2))
        if xb <= xa or yb <= ya:
            return np.zeros(0, np.int64)
        mask = np.zeros((yb - ya, xb - xa), np.uint8)
        for k in range(len(tris)):
            tri = np.stack([px[k] - xa, py[k] - ya], axis=1)
            cv2.fillConvexPoly(mask, np.round(tri).astype(np.int32), 1)
        yy, xx = np.nonzero(mask)
        return ((yy + ya) * self.n + (xx + xa)).astype(np.int64)

    def mark_footprint(self, pose, radius: float | None = None) -> None:
        """The robot is standing on this floor, so the cells under its centre are free. Not the whole footprint: cells
        between 0.12 m and the body edge that the camera has not seen could still be an obstacle's flank."""
        r = 0.12 if radius is None else radius          # just the floor under the robot; the rest must be seen to count as free
        k = int(np.ceil(r / self.res))
        cy, cx = self.to_cell(pose[0], pose[1])
        y0, y1, x0, x1 = max(0, cy - k), min(self.n, cy + k + 1), max(0, cx - k), min(self.n, cx + k + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= k * k
        sub = self.lo[y0:y1, x0:x1]
        m &= ~(self.seen[y0:y1, x0:x1] & (sub > 0.5))         # never erase an obstacle the camera saw: the robot may be too close to it
        sub[m] = np.minimum(sub[m], -1.0)
        self.seen[y0:y1, x0:x1] |= m

    # ---------------------------------------------------------------- queries
    def state(self) -> np.ndarray:
        st = np.full(self.lo.shape, UNKNOWN, np.uint8)
        st[self.seen & (self.lo < -0.3)] = FREE
        st[self.seen & (self.lo > 0.5)] = OCC
        return st

    def inflated_occupied(self, margin: float) -> np.ndarray:
        """Cells the robot's centre must not enter: occupied cells grown by robot radius + margin."""
        occ = (self.state() == OCC).astype(np.uint8)
        r = int(np.ceil((self.robot_radius + margin) / self.res))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(occ, k).astype(bool)

    def obstacle_distance(self) -> np.ndarray:
        """Distance [m] from every cell to the nearest occupied cell (large where nothing is mapped)."""
        free = (self.state() != OCC).astype(np.uint8)
        if free.all():
            return np.full(self.lo.shape, 99.0, np.float32)
        return cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.res

    def extent(self):
        """(x_min, x_max, y_min, y_max) of everything seen so far, or None."""
        ys, xs = np.nonzero(self.seen)
        if len(xs) == 0:
            return None
        x0, y0 = self.to_world(ys.min(), xs.min())
        x1, y1 = self.to_world(ys.max(), xs.max())
        return float(x0), float(x1), float(y0), float(y1)
