import copy

import mujoco
import numpy as np
import pytest

from activenav.config import apply_overrides, load_config
from activenav.estimation.ekf import EKF
from activenav.evaluation.synthetic import SyntheticMarkerSensor
from activenav.geometry import marker_fully_visible, segment_hits_circle
from activenav.autonomy.planner import NoPath, plan_route
from activenav.autonomy.planner import build_grid
from activenav.sim.env import DockingEnv
from activenav.sim.world import WorldError, load_world, validate_world
from activenav.types import Command, MarkerMeasurement, SensorBundle


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env_v2(cfg):
    c = copy.deepcopy(cfg)
    c["world"] = "room_10x6"
    return DockingEnv(c)


def test_generated_6x4_is_identical_to_the_handwritten_scene(cfg):
    """The regression scene: generating room_6x4 from config must change nothing."""
    e0 = DockingEnv(cfg)
    c1 = copy.deepcopy(cfg)
    c1["world"] = "room_6x4"
    e1 = DockingEnv(c1)

    def geoms(m):
        out = {}
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if n and n.startswith(("wall_", "marker_", "floor")):
                out[n] = np.concatenate([m.geom_pos[g], m.geom_size[g], m.geom_quat[g]])
        return out

    a, b = geoms(e0.model), geoms(e1.model)
    assert set(a) == set(b)
    assert all(np.allclose(a[k], b[k], atol=1e-12) for k in a)
    for e in (e0, e1):
        e.reset(seed=0, start_pose=(0.5, -1.0, 0.0))
    for _ in range(200):
        e0.step(Command(v=0.2, w=0.3))
        e1.step(Command(v=0.2, w=0.3))
    assert np.allclose(e0.ground_truth(), e1.ground_truth(), atol=1e-12)


def test_v2_pillar_layout_follows_the_rules(cfg):
    spec = load_world(cfg, "room_10x6")
    assert (spec.width, spec.depth, spec.wall_height) == (10.0, 6.0, 2.4)
    assert len(spec.pillars) == 6
    assert sorted({p["x"] for p in spec.pillars}) == [2.0, 5.0, 8.0]      # regular 3 x 2 grid, centred
    assert sorted({p["y"] for p in spec.pillars}) == [-1.2, 1.2]
    assert all(p["r"] == 0.3 and p["h"] == 2.4 for p in spec.pillars)
    rep = validate_world(spec, cfg, starts=[cfg["worlds"]["room_10x6"]["start"]], goals=[(9.5, 1.0)])
    assert rep["min_wall_clearance"] >= 1.0 and rep["reachable_fraction"] == 1.0


def test_layout_rule_violations_are_reported(cfg):
    bad = copy.deepcopy(cfg)
    bad["worlds"]["room_10x6"]["pillars"]["spacing"] = [3.9, 3.4]        # pushes pillars within 1 m of the walls
    with pytest.raises(WorldError, match="nearest wall"):
        validate_world(load_world(bad, "room_10x6"), bad, starts=[[0.6, -2.2, 0.0]])
    blocked = copy.deepcopy(cfg)
    with pytest.raises(WorldError, match="not in free space"):
        validate_world(load_world(blocked, "room_10x6"), blocked, starts=[[2.0, -1.2, 0.0]])   # inside a pillar


def test_env_obstacle_map_and_bounds_match_the_spec(cfg, env_v2):
    spec = load_world(cfg, "room_10x6")
    got = sorted((round(o["x"], 6), round(o["y"], 6), round(o["r"], 6)) for o in env_v2.obstacle_map())
    assert got == sorted((p["x"], p["y"], p["r"]) for p in spec.pillars)
    assert np.allclose(env_v2.room_bounds(), spec.bounds)
    assert DockingEnv(cfg).obstacle_map() == []                          # the 6x4 room has no pillars


def test_pillar_contact_counts_as_a_collision(env_v2):
    env_v2.reset(seed=0, start_pose=(1.2, -1.2, 0.0))                    # pillar 0 at (2.0, -1.2), driving straight at it
    hit = False
    for _ in range(400):
        env_v2.step(Command(v=0.2, w=0.0))
        hit = hit or env_v2.wall_contact()
    assert hit


def test_segment_circle_and_marker_occlusion():
    assert segment_hits_circle((0, 0), (4, 0), (2, 0.2), 0.3)
    assert not segment_hits_circle((0, 0), (4, 0), (2, 0.5), 0.3)
    assert not segment_hits_circle((0, 0), (1, 0), (2, 0.0), 0.3)       # circle beyond the end of the segment
    marker = dict(x=5.0, y=3.0, normal_angle=-np.pi / 2)
    pillar = [dict(x=5.0, y=1.2, r=0.3)]
    assert not marker_fully_visible((5.0, -1.0), marker, pillar)        # directly behind a pillar
    assert marker_fully_visible((7.0, -1.0), marker, pillar)
    # a pillar covering only one edge of the black square still counts as occluded
    edge_pillar = [dict(x=5.08, y=1.5, r=0.05)]                          # on the sight line to the right edge only
    assert not segment_hits_circle((5.0, 0.0), (5.0, 3.0), (5.08, 1.5), 0.05)       # the centre ray is clear
    assert not marker_fully_visible((5.0, 0.0), marker, edge_pillar)


def test_synthetic_sensor_hides_markers_behind_pillars(env_v2):
    env_v2.reset(seed=0, start_pose=(2.0, -0.3, np.pi / 2))             # marker 2 (north wall, x = 2.5) is behind pillar (2, 1.2)
    sb = SensorBundle(t=0.0, enc_left=0.0, enc_right=0.0, pan=0.0, tilt=0.0, image=None)
    rng = np.random.default_rng(0)
    seen = {z.marker_id for z in SyntheticMarkerSensor(env_v2, rng, occlusion=True).measure(sb)}
    open_ = {z.marker_id for z in SyntheticMarkerSensor(env_v2, rng, occlusion=False).measure(sb)}
    assert 2 in open_ and 2 not in seen


def test_planner_routes_around_pillars_with_clearance(cfg, env_v2):
    inflate = cfg["robot"]["radius"] + cfg["robot"]["plan_margin"]
    grid = build_grid(env_v2.room_bounds(), env_v2.obstacle_map(), 0.05, inflate)
    start = np.array(cfg["worlds"]["room_10x6"]["start"][:2])
    route = plan_route(grid, start, [(9.5, 1.0)])
    assert route[-1] == (9.5, 1.0)
    pts = [start] + [np.array(p) for p in route]
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 200):
            q = a + (b - a) * t
            assert all(np.hypot(*(q - [o["x"], o["y"]])) >= o["r"] + inflate - 0.06 for o in env_v2.obstacle_map())
    with pytest.raises(NoPath):
        plan_route(grid, start, [(2.0, 1.2)])                           # goal inside a pillar


def test_ekf_throttles_repeated_looks_at_the_same_marker(cfg):
    c = copy.deepcopy(cfg)
    c["ekf"]["min_update_interval_s"] = 0.5
    c["ekf"]["throttle_mode"] = "skip"
    markers = {1: dict(x=6.0, y=1.0, z=0.25, normal_angle=np.pi, size=0.3)}
    e = EKF(c, markers)
    e.reset((2.0, 0.0, 0.0))
    dx, dy = 4.0, 1.0
    z = MarkerMeasurement(1, float(np.hypot(dx, dy)), float(np.arctan2(dy, dx)), float(np.pi))
    assert e.update_marker(z, t=1.0).accepted
    assert e.update_marker(z, t=1.2).reason == "throttled"
    assert e.update_marker(z, t=1.6).accepted
    assert EKF(cfg, markers).min_update_dt == 0.0                       # off by default: earlier results unchanged


def test_config_overrides():
    cfg = load_config()
    apply_overrides(cfg, ["ekf.min_update_interval_s=0.5", "world=room_10x6", "gaze.n_candidates=11"])
    assert cfg["ekf"]["min_update_interval_s"] == 0.5 and cfg["world"] == "room_10x6" and cfg["gaze"]["n_candidates"] == 11
