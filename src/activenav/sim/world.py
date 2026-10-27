"""World specification (room, markers, pillars) read from configs/default.yaml.

The same spec generates the scene XML (rendering, collisions), the surveyed obstacle map used by
the planner and gaze policy, and the occlusion test in the synthetic sensor. Layouts are
deterministic: nothing here is random.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from activenav.geometry import obstacle_sdf
from activenav.autonomy.planner import build_grid, reachable


class WorldError(Exception):
    pass


@dataclass
class WorldSpec:
    name: str
    width: float                 # x extent: the room spans x in [0, width]
    depth: float                 # y extent: the room spans y in [-depth/2, depth/2]
    wall_height: float
    markers: list = field(default_factory=list)      # dicts: id, wall (east/west/north/south), s, z
    pillars: list = field(default_factory=list)      # dicts: x, y, r, h
    blocks: list = field(default_factory=list)       # full-height wall blocks: dicts x0, x1, y0, y1 (partitions, L-room corners)
    people: dict | None = None                       # walking people (sim/people.py settings); None = nobody

    @property
    def bounds(self):
        return 0.0, self.width, -self.depth / 2, self.depth / 2

    @property
    def obstacles(self):
        return [dict(x=p["x"], y=p["y"], r=p["r"]) for p in self.pillars] + [dict(b) for b in self.blocks]


def load_world(cfg: dict, name: str) -> WorldSpec:
    w = cfg["worlds"][name]
    W, D = (float(v) for v in w["size"])
    h = float(w["wall_height"])
    markers = [dict(id=int(m["id"]), wall=m["wall"], s=float(m["s"]), z=float(m.get("z", 0.25)))
               for m in w["markers"]]
    pillars = []
    p = w.get("pillars")
    if p:
        nx, ny = int(p["nx"]), int(p["ny"])
        sx, sy = (float(v) for v in p["spacing"])
        cx, cy = (float(v) for v in p.get("center", [W / 2, 0.0]))
        for j in range(ny):
            for i in range(nx):
                pillars.append(dict(x=cx + (i - (nx - 1) / 2) * sx, y=cy + (j - (ny - 1) / 2) * sy,
                                    r=float(p["radius"]), h=float(p.get("height", h))))
    blocks = []
    for b in w.get("blocks") or []:
        x0, y0, x1, y1 = (float(v) for v in b)
        blocks.append(dict(x0=min(x0, x1), x1=max(x0, x1), y0=min(y0, y1), y1=max(y0, y1)))
    from activenav.sim.people import people_settings
    return WorldSpec(name, W, D, h, markers, pillars, blocks, people_settings(w.get("people")))


def wall_free_intervals(spec: "WorldSpec", wall: str, margin: float = 0.6) -> list:
    """Stretches (lo, hi) of an outer wall, in marker coordinate s, that are not covered by a wall block
    (which touches the wall) and are at least `margin` away from one."""
    W, D = spec.width, spec.depth
    lo, hi = {"east": (-D / 2, D / 2), "west": (-D / 2, D / 2), "north": (0.0, W), "south": (0.0, W)}[wall]
    cut = []
    for b in spec.blocks:
        touches = {"east": b["x1"] >= W - 0.05, "west": b["x0"] <= 0.05,
                   "north": b["y1"] >= D / 2 - 0.05, "south": b["y0"] <= -D / 2 + 0.05}[wall]
        if touches:
            a, c = (b["y0"], b["y1"]) if wall in ("east", "west") else (b["x0"], b["x1"])
            cut.append((a - margin, c + margin))
    out, cur = [], lo
    for a, c in sorted(cut):
        if a > cur:
            out.append((cur, min(a, hi)))
        cur = max(cur, c)
    if cur < hi:
        out.append((cur, hi))
    return [(a, c) for a, c in out if c - a > 1e-6]


def validate_world(spec: WorldSpec, cfg: dict, starts=(), goals=()) -> dict:
    """Check the layout rules; raise WorldError listing every violation. Returns a report dict."""
    rule = cfg.get("layout_rules", {})
    min_wall = float(rule.get("min_pillar_wall_clearance", 1.0))      # pillar surface to nearest wall
    min_gap = float(rule.get("min_pillar_gap", 0.8))                  # surface to surface
    radius = float(cfg["robot"]["radius"])
    margin = float(cfg["robot"]["plan_margin"])
    x0, x1, y0, y1 = spec.bounds
    problems = []

    wall_clear = []
    for i, p in enumerate(spec.pillars):
        c = min(p["x"] - x0, x1 - p["x"], p["y"] - y0, y1 - p["y"]) - p["r"]
        for b in spec.blocks:
            c = min(c, float(obstacle_sdf(b, p["x"], p["y"])) - p["r"])
        wall_clear.append(c)
        if c < min_wall - 1e-9:
            problems.append(f"pillar {i} is {c:.2f} m from the nearest wall (rule: >= {min_wall} m)")
    gaps = [float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]) - a["r"] - b["r"])
            for i, a in enumerate(spec.pillars) for b in spec.pillars[i + 1:]]
    if gaps and min(gaps) < min_gap - 1e-9:
        problems.append(f"two pillars are {min(gaps):.2f} m apart (rule: >= {min_gap} m)")
    for m in spec.markers:
        if m["wall"] not in ("east", "west", "north", "south"):
            problems.append(f"marker {m['id']}: unknown wall '{m['wall']}'")
            continue
        if not any(a - 1e-9 <= m["s"] <= c + 1e-9 for a, c in wall_free_intervals(spec, m["wall"], 0.3)):
            problems.append(f"marker {m['id']} on the {m['wall']} wall at s={m['s']:g} is hidden by a wall block")
        if not (0.2 - 1e-9 <= m["z"] <= spec.wall_height - 0.2 + 1e-9):
            problems.append(f"marker {m['id']}: centre height {m['z']:g} m puts the 0.40 m plate through the floor or ceiling")
    ids = [m["id"] for m in spec.markers]
    if len(ids) != len(set(ids)):
        problems.append("duplicate marker ids")

    grid = build_grid(spec.bounds, spec.obstacles, res=0.05, inflate=radius + margin)
    pts = [("start", s) for s in starts] + [("goal", g) for g in goals]
    for label, q in pts:
        if not grid.is_free(q[0], q[1]):
            problems.append(f"{label} point {tuple(round(float(v), 2) for v in q[:2])} is not in free space "
                            f"(robot radius {radius} m + margin {margin} m)")
    free_frac = reach_frac = 1.0
    if not problems:
        seed = starts[0][:2] if len(starts) else (x0 + 1.0, y0 + 1.0)
        seen = reachable(grid, seed)
        free = ~grid.blocked
        free_frac = float(free.mean())
        reach_frac = float((seen & free).sum() / max(free.sum(), 1))
        if reach_frac < 1.0 - 1e-9:
            problems.append(f"only {100 * reach_frac:.1f}% of free space is reachable from the start (routes blocked)")
        for label, q in pts:
            iy, ix = grid.to_cell(q[0], q[1])
            if grid.in_bounds(iy, ix) and not seen[iy, ix]:
                problems.append(f"{label} point {tuple(round(float(v), 2) for v in q[:2])} is unreachable")
    if problems:
        raise WorldError(f"world '{spec.name}' violates layout rules:\n  - " + "\n  - ".join(problems))
    return dict(min_wall_clearance=min(wall_clear) if wall_clear else float("inf"),
                min_pillar_gap=min(gaps) if gaps else float("inf"),
                free_fraction=free_frac, reachable_fraction=reach_frac)


# ---------------------------------------------------------------------------------------------------
# Random marker layouts and coverage summaries
_NORMALS = {"east": np.pi, "west": 0.0, "north": -np.pi / 2, "south": np.pi / 2}


def marker_pose(spec: WorldSpec, m: dict):
    """(x, y, normal_angle) of a marker spec, matching what scene_builder generates."""
    W, D = spec.width, spec.depth
    x, y = {"east": (W, m["s"]), "west": (0.0, m["s"]), "north": (m["s"], D / 2), "south": (m["s"], -D / 2)}[m["wall"]]
    return x, y, _NORMALS[m["wall"]]


def randomize_markers(spec: WorldSpec, cfg: dict, seed: int) -> WorldSpec:
    """Same marker ids, new positions: every marker except `keep_ids` is re-placed on a random wall, weighted by
    the length of wall left free of blocks, with at least `min_spacing` between markers on a wall and
    `corner_margin` from the corners and from wall blocks. Deterministic in (seed): the same seed always gives
    the same layout, so paired comparisons stay paired."""
    rc = cfg.get("random_markers", {})
    keep = set(rc.get("keep_ids", [10]))
    gap, corner = float(rc.get("min_spacing", 1.0)), float(rc.get("corner_margin", 0.8))
    rng = np.random.default_rng([int(seed), 4242])
    free = {w: [(a + (corner if a == _wall_lo(spec, w) else 0.0), c - (corner if c == _wall_hi(spec, w) else 0.0))
                for a, c in wall_free_intervals(spec, w, corner)] for w in ("east", "west", "north", "south")}
    free = {w: [(a, c) for a, c in iv if c - a > 1e-6] for w, iv in free.items()}
    walls = [w for w in free if free[w]]
    p = np.array([sum(c - a for a, c in free[w]) for w in walls], float)
    p /= p.sum()
    placed = [dict(m) for m in spec.markers if m["id"] in keep]
    for m in spec.markers:
        if m["id"] in keep:
            continue
        for _ in range(2000):
            wall = walls[int(rng.choice(len(walls), p=p))]
            iv = free[wall]
            w_len = np.array([c - a for a, c in iv])
            a, c = iv[int(rng.choice(len(iv), p=w_len / w_len.sum()))]
            s = float(rng.uniform(a, c))
            if all(abs(o["s"] - s) >= gap for o in placed if o["wall"] == wall):
                placed.append(dict(id=m["id"], wall=wall, s=round(s, 3), z=m.get("z", 0.25)))
                break
        else:
            raise WorldError(f"could not place marker {m['id']} with spacing {gap} m")
    order = {m["id"]: i for i, m in enumerate(spec.markers)}
    return replace(spec, markers=sorted(placed, key=lambda m: order[m["id"]]))


def _wall_lo(spec, wall):
    return -spec.depth / 2 if wall in ("east", "west") else 0.0


def _wall_hi(spec, wall):
    return spec.depth / 2 if wall in ("east", "west") else spec.width


def assign_marker_heights(spec: WorldSpec, heights, seed: int) -> WorldSpec:
    """Every marker (the dock marker included) gets a centre height drawn from `heights`, per seed."""
    rng = np.random.default_rng([int(seed), 9191])
    ms = [dict(m, z=float(heights[int(rng.integers(len(heights)))])) for m in spec.markers]
    return replace(spec, markers=ms)


def coverage_summary(spec: WorldSpec, cfg: dict, max_range: float = 4.5, max_view_deg: float = 70.0, res: float = 0.25) -> dict:
    """Share of reachable floor from which >= 1 / >= 2 markers are usable (range, viewing angle and pillar
    occlusion; the head is assumed free to pan). Coarse grid: a quick sanity number for a layout."""
    from activenav.geometry import marker_fully_visible
    from activenav.kinematics import wrap_angle
    grid = build_grid(spec.bounds, spec.obstacles, res, float(cfg["robot"]["radius"]) + float(cfg["robot"]["plan_margin"]))
    mk = []
    for m in spec.markers:
        x, y, n = marker_pose(spec, m)
        mk.append(dict(x=x, y=y, normal_angle=n))
    obs, lim, counts = spec.obstacles, np.radians(max_view_deg), []
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            if grid.blocked[iy, ix]:
                continue
            x, y = grid.to_world(iy, ix)
            c = 0
            for m in mk:
                dx, dy = m["x"] - x, m["y"] - y
                r = float(np.hypot(dx, dy))
                if 0.3 <= r <= max_range and abs(wrap_angle(np.arctan2(-dy, -dx) - m["normal_angle"])) <= lim \
                        and marker_fully_visible((x, y), m, obs):
                    c += 1
            counts.append(c)
    counts = np.array(counts)
    return dict(none=float(np.mean(counts == 0)), ge1=float(np.mean(counts >= 1)), ge2=float(np.mean(counts >= 2)))
