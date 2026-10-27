"""Walking people: placement, motion rules, determinism, sensing and the contact metrics."""
import numpy as np
import pytest

from activenav.config import load_config
from activenav.evaluation.worlds import get_world
from activenav.evaluation.search import make_search_session
from activenav.geometry import obstacle_sdf
from activenav.sim.people import PeopleModel, people_settings
from activenav.sim.world import load_world, validate_world

cfg = load_config()
LEVELS = ["people_constant", "people_random"]


def _run(level, seed, steps, conf="B"):
    s, ctx = make_search_session(cfg, {}, conf, cfg["seed"] + seed, level=level, max_time=600)
    traj = [np.array([[c["x"], c["y"]] for c in s.env.people_state()])]
    for _ in range(steps):
        if not s.step():
            break
        traj.append(np.array([[c["x"], c["y"]] for c in s.env.people_state()]))
    return s, ctx, np.array(traj)


def test_people_settings_validate_and_default():
    assert people_settings(None) is None and people_settings({}) is None
    assert people_settings({"count": 0}) is None
    p = people_settings({"count": 2, "mode": "random"})
    assert p["radius"] == 0.2 and p["yield_gap"] == 0.35
    with pytest.raises(ValueError):
        people_settings({"count": 2, "mode": "teleport"})
    with pytest.raises(ValueError):
        people_settings({"count": 2, "speed": -1.0})


@pytest.mark.parametrize("level", LEVELS)
def test_people_level_builds_valid_world_with_people_bodies(level):
    w = cfg["levels"][level]["world"]
    spec = load_world(cfg, w)
    assert spec.people and spec.people["count"] == 3
    assert validate_world(spec, cfg, starts=[cfg["worlds"][w]["start"]])["reachable_fraction"] == 1.0
    ctx = get_world(cfg, {}, w)
    assert ctx.env.people is not None and len(ctx.env.people_state()) == 3
    # people are NOT part of the surveyed static obstacle map
    assert len(ctx.obstacles) == len(spec.pillars) + len(spec.blocks)


def test_rooms_without_people_have_none():
    ctx = get_world(cfg, {}, cfg["levels"]["large_4_pillars"]["world"])
    assert ctx.env.people is None and ctx.env.people_state() == [] and ctx.env.person_gap() == float("inf")
    assert ctx.env.person_contact() is False


@pytest.mark.parametrize("level", LEVELS)
def test_people_stay_in_free_space_and_apart(level):
    s, ctx, traj = _run(level, 3, 1500)
    x0, x1, y0, y1 = s.env.room_bounds()
    r = 0.2
    for P in traj[::10]:
        for x, y in P:
            assert x0 + r - 1e-6 <= x <= x1 - r + 1e-6 and y0 + r - 1e-6 <= y <= y1 - r + 1e-6
            assert all(float(obstacle_sdf(o, x, y)) >= r - 1e-6 for o in ctx.obstacles)
        d = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
        assert d[np.triu_indices(3, 1)].min() >= 2 * r - 1e-6
    assert np.abs(np.diff(traj, axis=0)).sum() > 3.0              # they actually walk


def test_constant_people_keep_their_speed_and_random_people_vary_it():
    _, _, tc = _run("people_constant", 5, 800)
    _, _, tr = _run("people_random", 5, 800)
    dt = cfg["sim"]["control_dt"]
    sc = np.hypot(*np.diff(tc, axis=0).transpose(2, 0, 1)) / dt
    sr = np.hypot(*np.diff(tr, axis=0).transpose(2, 0, 1)) / dt
    moving = sc[sc > 1e-6]
    assert np.allclose(moving, 0.4, atol=1e-3)
    assert sr[sr > 1e-6].std() > 0.03                              # speeds drift


def test_same_seed_same_people_for_every_config_and_seeds_differ():
    _, _, a = _run("people_random", 7, 400, "B")
    _, _, b = _run("people_random", 7, 400, "C-info")
    _, _, c = _run("people_random", 8, 400, "B")
    assert np.allclose(a[0], b[0])                                 # same start positions (paired comparison)
    assert not np.allclose(a[0], c[0])


def test_a_person_waits_then_walks_away_from_a_robot_that_stays():
    p = people_settings({"count": 1, "mode": "constant", "speed": 0.4, "yield_gap": 0.35, "yield_wait": 3.0})
    m = PeopleModel(p, (0.0, 10.0, -3.0, 3.0), [], 0.25, 0.02)
    m.reset(1, (9.0, 0.0))
    here = m.pos[0].copy()
    robot = here + np.array([-0.5, 0.0]) if here[0] > 2 else here + np.array([0.5, 0.0])    # 0.5 m away: inside 0.2 + 0.25 + 0.35
    for _ in range(100):                                           # 2 s: still waiting
        m.step(robot)
    assert np.allclose(m.pos[0], here)
    for _ in range(250):                                           # 7 s in all: it has walked away from the robot
        m.step(robot)
    assert np.hypot(*(m.pos[0] - robot)) > np.hypot(*(here - robot)) + 0.3
    far = m.pos[0].copy()
    for _ in range(50):                                            # robot gone: keeps walking (no freezing)
        m.step((-50.0, 0.0))
    assert not np.allclose(m.pos[0], far)


def test_a_person_cornered_against_a_wall_slides_along_it_instead_of_staying():
    p = people_settings({"count": 1, "mode": "constant", "speed": 0.4, "yield_gap": 0.35, "yield_wait": 1.0})
    m = PeopleModel(p, (0.0, 10.0, -3.0, 3.0), [], 0.25, 0.02)
    m.reset(1, (0.0, 0.0))
    m.pos[0] = np.array([9.8, 0.0])                                # against the east wall (centre at wall - radius)
    robot = np.array([9.0, 0.0])                                   # directly west of it: "away" points into the wall
    start = m.pos[0].copy()
    for _ in range(400):
        m.step(robot)
    assert np.hypot(*(m.pos[0] - start)) > 0.8                     # it got out of the corner region
    assert m.pos[0][0] <= 9.8 + 1e-6


def test_synthetic_scan_sees_a_person_and_a_person_hides_a_marker():
    s, ctx = make_search_session(cfg, {}, "B", cfg["seed"], level="people_constant", start=(1.0, 0.0, 0.0), max_time=5)
    env = s.env
    env.people.pos[0] = np.array([2.2, 0.0])                       # right in front of the robot
    env.people.pos[1:] = np.array([[8.0, 2.5], [8.0, -2.5]])
    env._place_people()
    front = s.front
    assert front.cast(np.array([1.0, 0.0]), 0.0) == pytest.approx(2.2 - 0.2 - 1.0, abs=1e-6)
    tilt = 0.0
    from activenav.types import SensorBundle
    sb = SensorBundle(t=0.0, enc_left=0.0, enc_right=0.0, pan=0.0, tilt=tilt, image=None)
    seen_with = {mm.marker_id for mm in s.marker_sensor.measure(sb)}
    env.people.pos[0] = np.array([5.0, 8.0])                       # out of the way (clamped by nothing: just a position)
    env._place_people()
    seen_without = {mm.marker_id for mm in s.marker_sensor.measure(sb)}
    assert seen_with <= seen_without


def test_person_contact_is_detected_and_ends_the_trial_as_contact():
    s, ctx = make_search_session(cfg, {}, "B", cfg["seed"], level="people_constant", start=(1.0, 0.0, 0.0), max_time=30)
    env = s.env
    env.people.p["yield_gap"] = 0.0
    env.people.pos[0] = np.array([1.6, 0.0])                       # 0.1 m clear of the robot's front edge
    env.people.vel[0] = np.array([-0.4, 0.0])                      # walks into the robot
    env.people.speed[0] = 0.4
    env.people.pos[1:] = np.array([[8.0, 2.5], [8.0, -2.5]])
    env._place_people()
    for _ in range(600):
        if not s.step():
            break
    r = s.result()
    assert r["person_contact"] and r["outcome"] == "contact" and r["collided"]
    assert r["min_person_gap_m"] <= 0.0 + 1e-6
