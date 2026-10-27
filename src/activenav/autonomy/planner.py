"""Planning: the occupancy-grid helpers and A* route planning through known-free cells."""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from activenav.geometry import obstacle_sdf


# -------------------- autonomy/planner.py --------------------
# Occupancy grid built from the SURVEYED map (room bounds + pillars). No simulator access.
@dataclass
class Grid:
    res: float
    x0: float
    y0: float
    blocked: np.ndarray            # [ny, nx] bool, inflated

    @property
    def shape(self):
        return self.blocked.shape

    def to_cell(self, x, y):
        return int(np.floor((y - self.y0) / self.res)), int(np.floor((x - self.x0) / self.res))

    def to_world(self, iy, ix):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def in_bounds(self, iy, ix):
        return 0 <= iy < self.shape[0] and 0 <= ix < self.shape[1]

    def is_free(self, x, y) -> bool:
        iy, ix = self.to_cell(x, y)
        return self.in_bounds(iy, ix) and not self.blocked[iy, ix]

    def line_free(self, p, q) -> bool:
        p, q = np.asarray(p, float), np.asarray(q, float)
        n = max(2, int(np.ceil(np.linalg.norm(q - p) / (self.res * 0.5))))
        return all(self.is_free(*(p + (q - p) * k / n)) for k in range(n + 1))

def build_grid(bounds, obstacles, res: float = 0.05, inflate: float = 0.30) -> Grid:
    x0, x1, y0, y1 = bounds
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    blocked = (X < x0 + inflate) | (X > x1 - inflate) | (Y < y0 + inflate) | (Y > y1 - inflate)
    for o in obstacles:
        blocked |= obstacle_sdf(o, X, Y) < inflate
    return Grid(res, x0, y0, blocked)

def reachable(grid: Grid, start) -> np.ndarray:
    """Boolean map of free cells connected (8-neighbour) to `start`."""
    iy, ix = grid.to_cell(*start)
    seen = np.zeros_like(grid.blocked)
    if not grid.in_bounds(iy, ix) or grid.blocked[iy, ix]:
        return seen
    stack = [(iy, ix)]
    seen[iy, ix] = True
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny_, nx_ = y + dy, x + dx
                if (dy or dx) and grid.in_bounds(ny_, nx_) and not grid.blocked[ny_, nx_] and not seen[ny_, nx_]:
                    seen[ny_, nx_] = True
                    stack.append((ny_, nx_))
    return seen


# -------------------- autonomy/planner.py --------------------
# A* on the inflated occupancy grid, then line-of-sight shortcutting.
_MOVES = [(dy, dx, float(np.hypot(dy, dx))) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

class NoPath(Exception):
    pass

def astar_cells(grid: Grid, start, goal, start_clear: float = 0.2):
    """Cell path from start to goal (world xy). Cells within `start_clear` of the start are treated as free,
    so a robot standing close to a wall can still leave."""
    blocked = grid.blocked.copy()
    sy, sx = grid.to_cell(*start)
    gy, gx = grid.to_cell(*goal)
    r = int(np.ceil(start_clear / grid.res))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r and grid.in_bounds(sy + dy, sx + dx):
                blocked[sy + dy, sx + dx] = False
    if not grid.in_bounds(gy, gx) or blocked[gy, gx]:
        raise NoPath(f"goal {tuple(goal)} is not in free space")
    h = lambda y, x: max(abs(y - gy), abs(x - gx)) + (np.sqrt(2) - 1) * min(abs(y - gy), abs(x - gx))
    open_ = [(h(sy, sx), 0.0, sy, sx)]
    g = {(sy, sx): 0.0}
    parent = {}
    while open_:
        _, cost, y, x = heapq.heappop(open_)
        if (y, x) == (gy, gx):
            path = [(y, x)]
            while (y, x) in parent:
                y, x = parent[(y, x)]
                path.append((y, x))
            return path[::-1]
        if cost > g[(y, x)]:
            continue
        for dy, dx, c in _MOVES:
            ny_, nx_ = y + dy, x + dx
            if not grid.in_bounds(ny_, nx_) or blocked[ny_, nx_]:
                continue
            if dy and dx and (blocked[y + dy, x] or blocked[y, x + dx]):     # no corner cutting
                continue
            nc = cost + c
            if nc < g.get((ny_, nx_), np.inf):
                g[(ny_, nx_)] = nc
                parent[(ny_, nx_)] = (y, x)
                heapq.heappush(open_, (nc + h(ny_, nx_), nc, ny_, nx_))
    raise NoPath("no route between start and goal")

def plan_path(grid: Grid, start, goal, start_clear: float = 0.2):
    """Smoothed waypoint list [(x, y), ...] from start (excluded) to goal (included, exact)."""
    cells = astar_cells(grid, start, goal, start_clear)
    pts = [tuple(start)] + [grid.to_world(*c) for c in cells[1:-1]] + [tuple(goal)]
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _visible(grid, pts[i], pts[j], start_clear if i == 0 else 0.0):
            j -= 1
        out.append(pts[j])
        i = j
    return out[1:]

def _visible(grid: Grid, p, q, start_clear):
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = max(2, int(np.ceil(np.linalg.norm(q - p) / (grid.res * 0.5))))
    for k in range(n + 1):
        pt = p + (q - p) * k / n
        if start_clear and np.linalg.norm(pt - p) <= start_clear:
            continue
        if not grid.is_free(*pt):
            return False
    return True

def plan_route(grid: Grid, start_xy, goals, start_clear: float = 0.2):
    """Plan through a list of goals in order; returns the concatenated waypoint list."""
    route, cur = [], tuple(start_xy)
    for g in goals:
        route += plan_path(grid, cur, tuple(g), start_clear)
        cur = tuple(g)
    return route
