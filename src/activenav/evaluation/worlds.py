"""Worlds: build the environment, surveyed obstacle map and planner grid for a named world (`worlds:` / `levels:` in the config).

The search-and-dock study, the live viewer and the diagnostics all get their world from `get_world`, so they cannot
disagree. The robot never sees any of this: it is the experimenter's side (ground truth lives here and in sim/).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

from activenav.autonomy.planner import Grid, build_grid
from activenav.sim.env import DockingEnv
from activenav.sim.world import assign_marker_heights, coverage_summary, load_world, randomize_markers

_MAX_RANDOM_LAYOUTS_CACHED = 4


class ScenarioError(Exception):
    """The requested combination cannot run; the message says why."""


@dataclass
class WorldContext:
    name: str
    cfg: dict
    env: DockingEnv
    obstacles: list
    grid: Grid
    spec: object = None
    layout_seed: int | None = None
    coverage: dict = field(default_factory=dict)


def available_levels(cfg: dict) -> list:
    return [k for k, v in cfg.get("levels", {}).items() if v.get("available", True)]


def get_world(cfg: dict, cache: dict, world: str, layout_seed: int | None = None, height_seed: int | None = None,
              heights=None, dock_height: float | None = None) -> WorldContext:
    """Environment, surveyed maps and planner grid for one world. The plain 6x4 room is the hand-written
    regression scene; every other world (and any random marker layout) is generated from the config."""
    hk = (height_seed, tuple(heights)) if heights else None
    key = (world, layout_seed) if hk is None else (world, layout_seed, hk)
    if dock_height is not None:
        key = key + (("dock", float(dock_height)),)
    if key not in cache:
        c = copy.deepcopy(cfg)
        spec = load_world(cfg, world)
        tag = ""
        if layout_seed is not None:
            spec = randomize_markers(spec, cfg, layout_seed)
            tag += f"_m{layout_seed}"
        if heights:
            spec = assign_marker_heights(spec, heights, 0 if height_seed is None else height_seed)
            tag += f"_h{0 if height_seed is None else height_seed}"
        if dock_height is not None:                                  # only the dock marker: decoys keep their own heights
            dock_id = int(cfg["search"]["dock_marker"])
            if not (0.2 - 1e-9 <= dock_height <= spec.wall_height - 0.2 + 1e-9):
                raise ScenarioError(f"dock height {dock_height:g} m must be between 0.2 m and {spec.wall_height - 0.2:g} m "
                                    f"(the 0.40 m plate has to fit between floor and ceiling)")
            spec = replace(spec, markers=[dict(m, z=float(dock_height)) if m["id"] == dock_id else m for m in spec.markers])
            tag += f"_d{dock_height:g}"
        if tag:
            c["world_markers"], c["world_tag"] = spec.markers, tag
        c["world"] = None if (world == "room_6x4" and layout_seed is None) else world
        env = DockingEnv(c)
        obstacles = env.obstacle_map()
        grid = build_grid(env.room_bounds(), obstacles, 0.05, c["robot"]["radius"] + c["robot"]["plan_margin"])
        cache[key] = WorldContext(world, c, env, obstacles, grid, spec, layout_seed, coverage_summary(spec, cfg))
        random_keys = [k for k in cache if k[1] is not None or len(k) > 2]
        while len(random_keys) > _MAX_RANDOM_LAYOUTS_CACHED:         # random layouts are per seed: keep memory bounded
            cache.pop(random_keys.pop(0)).env.close()
    return cache[key]
