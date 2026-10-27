"""Planning on the occupancy map: coarse traversable grid, frontier detection, Dijkstra distances, paths.

Unknown space is NOT traversable: the robot only plans through floor it has seen. Frontiers are traversable cells
next to unknown space. Never reads simulator ground truth.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import cv2
import numpy as np

from activenav.autonomy.occupancy import FREE, UNKNOWN, OccupancyGrid

_MOVES = [(dy, dx, float(np.hypot(dy, dx))) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


@dataclass
class CoarseMap:
    res: float
    x0: float
    y0: float
    trav: np.ndarray          # (ny, nx) bool, the robot centre may be here
    frontier: np.ndarray      # (ny, nx) bool
    unknown: np.ndarray       # (ny, nx) float, fraction of unknown fine cells

    def to_cell(self, x, y):
        return int(np.floor((y - self.y0) / self.res)), int(np.floor((x - self.x0) / self.res))

    def to_world(self, iy, ix):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def inside(self, iy, ix):
        return 0 <= iy < self.trav.shape[0] and 0 <= ix < self.trav.shape[1]


def build_coarse(grid: OccupancyGrid, pose, margin: float, res: float = 0.1, min_free: float = 0.7,
                 footprint: float = 0.15) -> CoarseMap | None:
    ext = grid.extent()
    if ext is None:
        return None
    f = max(1, int(round(res / grid.res)))
    pad = 0.6
    ix0 = max(0, int(np.floor((ext[0] - pad - grid.x0) / grid.res)) // f * f)
    iy0 = max(0, int(np.floor((ext[2] - pad - grid.y0) / grid.res)) // f * f)
    ix1 = min(grid.n, int(np.ceil((ext[1] + pad - grid.x0) / grid.res)) // f * f + f)
    iy1 = min(grid.n, int(np.ceil((ext[3] + pad - grid.y0) / grid.res)) // f * f + f)
    st = grid.state()[iy0:iy1, ix0:ix1]
    od_full = grid.obstacle_distance()
    od = od_full[iy0:iy1, ix0:ix1]
    # Required clearance = robot radius + margin. If the robot already stands closer than that (uncertainty, and so the
    # margin, grew while it was there), it must still be able to leave: never demand more than where it stands now,
    # but never less than 12 cm.
    req = grid.robot_radius + margin
    ry, rx = grid.to_cell(pose[0], pose[1])
    here = float(od_full[min(max(int(ry), 0), grid.n - 1), min(max(int(rx), 0), grid.n - 1)])
    floor = grid.robot_radius + 0.03             # never plan closer than this to a mapped obstacle ...
    if here >= floor:                            # ... (stops the allowance below from ratcheting the robot into it)
        thr = min(req, max(here - 0.04, floor))
    else:                                        # already inside it (drift): leave by whatever is open
        thr = max(here - 0.04, 0.12)
    infl = od < thr
    # Unseen cells can be an obstacle's flank (a surface seen edge-on leaves free-looking cells beside it), so keep the
    # body clear of unknown space too. Less than from known obstacles, or nothing could ever be explored.
    unk_clear = 0.8 * grid.robot_radius
    du_full = cv2.distanceTransform((grid.state()[iy0:iy1, ix0:ix1] != UNKNOWN).astype(np.uint8), cv2.DIST_L2, 5) * grid.res
    ury, urx = ry - iy0, rx - ix0
    here_u = float(du_full[min(max(int(ury), 0), du_full.shape[0] - 1), min(max(int(urx), 0), du_full.shape[1] - 1)])
    infl = infl | (du_full < min(unk_clear, max(here_u - 0.04, 0.05)))
    # the robot stands on free floor: its own footprint is never an obstacle
    px, py = pose[0], pose[1]
    yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
    wx, wy = grid.to_world(yy, xx)
    near = (wx - px) ** 2 + (wy - py) ** 2 < footprint ** 2
    infl = infl & ~near
    h, w = (iy1 - iy0) // f * f, (ix1 - ix0) // f * f
    st, infl, near = st[:h, :w], infl[:h, :w], near[:h, :w]
    free_frac = cv2.resize(((st == FREE) | near).astype(np.float32), (w // f, h // f), interpolation=cv2.INTER_AREA)
    unk = cv2.resize((st == UNKNOWN).astype(np.float32), (w // f, h // f), interpolation=cv2.INTER_AREA)
    blocked = cv2.resize(infl.astype(np.float32), (w // f, h // f), interpolation=cv2.INTER_AREA) > 0.05
    trav = (free_frac >= min_free) & ~blocked
    k = int(round(0.35 / (f * grid.res))) * 2 + 1          # frontier band: reachable cells within ~0.35 m of unknown space
    near_unk = cv2.dilate((unk > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    fr = trav & near_unk
    x0, y0 = grid.x0 + ix0 * grid.res, grid.y0 + iy0 * grid.res
    return CoarseMap(res=f * grid.res, x0=x0, y0=y0, trav=trav, frontier=fr, unknown=unk)


def dijkstra(cm: CoarseMap, start_xy):
    """Cost-to-reach every traversable cell from the start, and parent pointers. Start cell is always allowed."""
    ny, nx = cm.trav.shape
    sy, sx = cm.to_cell(*start_xy)
    dist = np.full((ny, nx), np.inf)
    parent = {}
    if not cm.inside(sy, sx):
        return dist, parent
    dist[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy, dx, c in _MOVES:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < ny and 0 <= xx < nx) or not cm.trav[yy, xx]:
                continue
            if dy and dx and not (cm.trav[y, xx] and cm.trav[yy, x]):          # no corner cutting
                continue
            nd = d + c * cm.res
            if nd < dist[yy, xx]:
                dist[yy, xx] = nd
                parent[(yy, xx)] = (y, x)
                heapq.heappush(pq, (nd, yy, xx))
    return dist, parent


def path_to(cm: CoarseMap, parent, start_xy, goal_cell, smooth: bool = True):
    cells = [goal_cell]
    sy, sx = cm.to_cell(*start_xy)
    while cells[-1] != (sy, sx):
        if cells[-1] not in parent:
            return []
        cells.append(parent[cells[-1]])
    pts = [cm.to_world(*c) for c in reversed(cells)]
    if not smooth or len(pts) < 3:
        return pts[1:] if len(pts) > 1 else pts
    out, i = [], 0
    while i < len(pts) - 1:                          # greedy line-of-sight shortcuts
        j = len(pts) - 1
        while j > i + 1 and not _line_free(cm, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _line_free(cm: CoarseMap, p, q) -> bool:
    n = max(2, int(np.ceil(np.hypot(q[0] - p[0], q[1] - p[1]) / (cm.res * 0.5))))
    for k in range(n + 1):
        iy, ix = cm.to_cell(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n)
        if not cm.inside(iy, ix) or not cm.trav[iy, ix]:
            return False
    return True


def frontier_clusters(cm: CoarseMap, min_cells: int):
    """[(cells (n, 2) array of (iy, ix))] of connected frontier cells with at least min_cells."""
    f = cm.frontier.astype(np.uint8)
    # a narrow doorway gives a short, ragged frontier: join cells up to 2 apart before counting them
    n, lab = cv2.connectedComponents(cv2.dilate(f, np.ones((5, 5), np.uint8)), connectivity=8)
    out = []
    for k in range(1, n):
        cells = np.argwhere((lab == k) & (f > 0))
        if len(cells) >= min_cells:
            out.append(cells)
    return out


def nearest_traversable(cm: CoarseMap, dist, xy, max_dist: float = 0.6):
    """Reachable cell closest to the world point xy (within max_dist), or None."""
    cy, cx = cm.to_cell(*xy)
    r = int(np.ceil(max_dist / cm.res))
    best, bd = None, np.inf
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y, x = cy + dy, cx + dx
            if cm.inside(y, x) and np.isfinite(dist[y, x]):
                d = np.hypot(dy, dx) * cm.res
                if d < bd and d <= max_dist:
                    best, bd = (y, x), d
    return best
