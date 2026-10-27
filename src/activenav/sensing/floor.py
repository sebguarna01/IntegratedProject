"""Floor ranging: pixel to floor point from the head geometry, for camera-only obstacle and wall detection."""
from __future__ import annotations

import cv2
import numpy as np

from activenav.autonomy.obstacle_sensor import GroundGeometry, ObstacleScan


# -------------------- sensing/floor.py --------------------
# Camera-only obstacle sensing by ground-plane ranging (OpenCV, no learning).
class FloorRanger:
    needs_image = True

    def __init__(self, cfg: dict, thresh: float = 3.2, min_std: float = 3.0, run: int = 4, strip_rows: int = 40,
                 l_edge: float = 16.0):
        self.geo = GroundGeometry(cfg)
        o = cfg["obstacles"]
        self.max_range, self.px_sigma, self.n_rays = float(o["max_range"]), float(o["pixel_sigma"]), int(o["n_rays"])
        self.thresh, self.min_std, self.run, self.strip_rows, self.l_edge = thresh, min_std, run, strip_rows, l_edge
        self.mean = None            # floor colour model in Lab
        self.std = None
        self.last_mask = None       # (H, n_rays) bool, for the diagnostic overlay
        self.last_cols = None

    # ------------------------------------------------------------- floor model
    def _lab(self, img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    def _init_model(self, lab):
        # chroma only: lighting changes the floor's lightness a lot (the lamp is a point source), but not its colour
        strip = lab[-self.strip_rows:, :, 1:].reshape(-1, 2)
        self.mean = np.median(strip, axis=0)

    def floor_mask_columns(self, lab, cols):
        """Floor = same chroma as the floor model AND no sudden lightness step from the pixels just below."""
        band = np.stack([lab[:, max(0, c - 2):c + 3].mean(axis=1) for c in cols], axis=1)      # (H, n, 3)
        d = np.sqrt(((band[:, :, 1:] - self.mean) ** 2).sum(axis=2))
        m = d < self.thresh
        L = band[:, :, 0]
        Ls = np.vstack([L[:1], (L[:-2] + L[1:-1] + L[2:]) / 3.0, L[-1:]])                  # 3-row smoothing
        step = np.zeros_like(L, bool)
        step[:-3] = np.abs(Ls[:-3] - Ls[3:]) > self.l_edge                                  # step against 3 rows below
        return m & ~step

    # -------------------------------------------------------------------- scan
    def scan(self, image_rgb, pan: float, tilt: float) -> ObstacleScan:
        geo = self.geo
        lab = self._lab(image_rgb)
        if self.mean is None:
            self._init_model(lab)
        H, W = geo.H, geo.W
        cols = np.linspace(W * 0.03, W * 0.97, self.n_rays).astype(int)
        mask = self.floor_mask_columns(lab, cols)
        self.last_mask, self.last_cols = mask, cols
        # rows between the bottom of the image and the row where the floor point reaches the range limit
        rows = np.arange(H - 1, -1, -1)
        cx = np.full(len(rows), geo.K[0, 2])
        pts_c, ok_c = geo.pixel_to_floor(cx, rows.astype(float), pan, tilt)
        _, cam = geo.pose(pan, tilt)
        dist = np.hypot(pts_c[:, 0] - cam[0], pts_c[:, 1] - cam[1])
        usable = ok_c & (dist <= self.max_range * 1.0)
        v_top = int(rows[usable][-1]) if usable.any() else H - 1            # highest row still inside the range limit
        pts, hits, sig = [], [], []
        floor_samples = []
        for j, u in enumerate(cols):
            m = mask[:, j]
            v_b, hit = None, False
            nonfloor_run = 0
            for v in range(H - 1, v_top - 1, -1):
                if m[v]:
                    nonfloor_run = 0
                else:
                    nonfloor_run += 1
                    if nonfloor_run >= self.run:
                        v_b, hit = v + self.run - 1, True               # last floor row is just below the run
                        break
            if not hit:
                v_b = v_top
            else:
                v_b = min(H - 1, v_b)
            p, ok = geo.pixel_to_floor([u], [float(v_b)], pan, tilt)
            if not ok[0]:                                               # boundary at or above the horizon: nothing to range
                continue
            r = float(np.hypot(p[0, 0] - cam[0], p[0, 1] - cam[1]))
            pts.append(p[0])
            hits.append(hit and r <= self.max_range)
            sig.append(float(geo.range_sigma(r, self.px_sigma)))
            if hit and v_b < H - 12:
                floor_samples.append(lab[v_b + 8:min(H, v_b + 20), max(0, u - 2):u + 3].reshape(-1, 3))
        if floor_samples:                                               # slow adaptation to the floor actually seen
            fs = np.concatenate(floor_samples)
            self.mean = 0.98 * self.mean + 0.02 * np.median(fs[:, 1:], axis=0)
        return ObstacleScan(cam[:2].copy(), np.array(pts).reshape(-1, 2), np.array(hits, bool), np.array(sig))

    def scan_sb(self, sb) -> ObstacleScan:
        """Scan the image of a SensorBundle with the head angles measured when the image was taken."""
        return self.scan(sb.image, sb.view_pan, sb.view_tilt)
