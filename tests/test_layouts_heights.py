"""Wall blocks (L-room, partitions), the new levels, marker heights and the vertical-view geometry."""
import copy

import numpy as np
import pytest

from activenav.autonomy import explore
from activenav.autonomy.marker_view import elevations, tilt_for_marker, tilt_interval, vertically_visible
from activenav.autonomy.search_gaze import InfoSearchGaze
from activenav.config import load_config
from activenav.evaluation.search import make_search_session
from activenav.geometry import obstacle_sdf, segment_blocked
from activenav.autonomy.planner import build_grid
from activenav.sim.world import (WorldError, assign_marker_heights, load_world, randomize_markers, validate_world,
                                 wall_free_intervals)

cfg = load_config()
NEW = ["room_10x10_p9", "room_l", "room_two_rooms", "room_zigzag"]


def test_sdf_of_rectangle_and_circle():
    r = dict(x0=0.0, x1=2.0, y0=0.0, y1=1.0)
    assert obstacle_sdf(r, 1.0, 0.5) == pytest.approx(-0.5)            # inside: distance to the nearest face, negative
    assert obstacle_sdf(r, 3.0, 0.5) == pytest.approx(1.0)
    assert obstacle_sdf(r, 3.0, 2.0) == pytest.approx(np.hypot(1.0, 1.0))     # corner region: Euclidean
    assert obstacle_sdf(dict(x=0.0, y=0.0, r=0.3), 1.0, 0.0) == pytest.approx(0.7)


def test_line_of_sight_is_blocked_by_a_rectangle():
    r = [dict(x0=4.9, x1=5.1, y0=-3.0, y1=0.5)]
    assert segment_blocked((1, 0), (9, 0), r)
    assert not segment_blocked((1, 2), (9, 2), r)                       # passes through the gap above it


@pytest.mark.parametrize("name", NEW)
def test_new_worlds_obey_the_layout_rules_and_are_fully_reachable(name):
    spec = load_world(cfg, name)
    rep = validate_world(spec, cfg, starts=[cfg["worlds"][name]["start"]])
    assert rep["reachable_fraction"] == pytest.approx(1.0)


def test_planner_grid_blocks_the_wall_blocks():
    spec = load_world(cfg, "room_two_rooms")
    g = build_grid(spec.bounds, spec.obstacles, 0.05, 0.27)
    assert not g.is_free(6.0, -2.0) and not g.is_free(6.0, 2.0)          # inside the partition
    assert g.is_free(6.0, 0.2)                                           # the doorway is passable


def test_markers_cannot_hang_where_a_block_covers_the_wall():
    spec = load_world(cfg, "room_l")
    assert wall_free_intervals(spec, "north")[0][1] < 5.0                # north wall: only the left arm is free
    bad = copy.deepcopy(spec)
    bad.markers[0] = dict(id=0, wall="north", s=7.5, z=0.25)             # x = 7.5 is behind the filled-in quarter
    with pytest.raises(WorldError, match="hidden by a wall block"):
        validate_world(bad, cfg, starts=[cfg["worlds"]["room_l"]["start"]])


def test_random_markers_avoid_blocks_and_stay_on_free_wall():
    spec = load_world(cfg, "room_two_rooms")
    for seed in range(20):
        sp = randomize_markers(spec, cfg, seed)
        validate_world(sp, cfg, starts=[cfg["worlds"]["room_two_rooms"]["start"]])


def test_marker_height_must_keep_the_plate_inside_floor_and_ceiling():
    spec = load_world(cfg, "room_10x6")
    spec.markers[0]["z"] = 0.1
    with pytest.raises(WorldError, match="height"):
        validate_world(spec, cfg, starts=[cfg["worlds"]["room_10x6"]["start"]])


def test_heights_are_drawn_per_seed_from_the_list_and_are_reproducible():
    spec = load_world(cfg, "room_10x6")
    hs = cfg["marker_heights"]
    a, b = assign_marker_heights(spec, hs, 3), assign_marker_heights(spec, hs, 3)
    c = assign_marker_heights(spec, hs, 4)
    assert [m["z"] for m in a.markers] == [m["z"] for m in b.markers]
    assert [m["z"] for m in a.markers] != [m["z"] for m in c.markers]
    assert {m["z"] for m in a.markers} <= set(hs)


def test_vertical_view_geometry():
    v = np.radians(60)
    h = 0.19
    assert bool(vertically_visible(h, 0.2, 3.0, 0.45, v))                # far away, everything fits
    assert not bool(vertically_visible(h, 0.2, 0.8, 0.6, v))             # tall marker, close, looking down: cut off
    lo, hi = tilt_interval(h, 0.8, 0.6, v)
    t = tilt_for_marker(h, 0.8, 0.6, v, prefer=0.2)
    assert lo <= t <= hi and t < 0.2                                     # it has to look up
    e_bot, e_top = elevations(h, 0.6, 0.8)
    assert tilt_for_marker(h, 0.8, 0.6, v, 0.2, centre=True) == pytest.approx(-0.5 * (e_bot + e_top))


def test_synthetic_sensor_loses_a_tall_marker_that_is_too_close_and_reports_height():
    ses, _ = make_search_session(cfg, {}, "B", 0, level="large_room", sensor="synthetic")
    sens, env = ses.marker_sensor, ses.env
    m10 = env.marker_map()[10]
    z = m10["z"]
    # stand 1.0 m in front of marker 10, facing it, head looking down 0.2 rad: a 0.25 m marker is fine
    env.reset(0, start_pose=(m10["x"] - 1.0, m10["y"], 0.0))
    sb = env.step(__import__("activenav.types", fromlist=["Command"]).Command(0, 0, 0, 0.2), render=False)
    got = {q.marker_id: q for q in sens.measure(sb)}
    assert 10 in got and got[10].height == pytest.approx(z, abs=0.05)
    sens.markers[10] = dict(m10, z=0.6)                                  # now the same marker hangs high
    assert 10 not in {q.marker_id for q in sens.measure(sb)}


def test_doorway_frontier_cells_are_merged_into_one_cluster():
    class CM:                                                           # only .frontier is used
        frontier = np.zeros((10, 20), bool)
    CM.frontier[5, 3:5] = True                                           # 2 cells ... gap of 2 ... 3 cells
    CM.frontier[6, 7:10] = True
    assert len(explore.frontier_clusters(CM, 4)) == 1


def test_info_gaze_remembers_what_it_searched():
    from activenav.autonomy.occupancy import OccupancyGrid
    g = InfoSearchGaze(cfg)
    grid = OccupancyGrid(cfg)
    pose = np.zeros(3)
    state = grid.state()
    before = g.marker_gain(grid, state, pose, 0.0, 0.2)
    g.mark_searched(grid, pose, 0.0, 0.2)
    after = g.marker_gain(grid, state, pose, 0.0, 0.2)
    assert before == 0.0 and after == 0.0                                # no memory yet / nothing left to gain
    g.reset()
    g.mark_searched(grid, pose, 1.0, 0.2)                                # look away, then compare the two directions
    assert g.marker_gain(grid, state, pose, 0.0, 0.2) > g.marker_gain(grid, state, pose, 1.0, 0.2)


@pytest.mark.parametrize("level", ["l_room", "two_rooms"])
def test_search_docks_in_a_room_with_wall_blocks(level):
    s, _ = make_search_session(cfg, {}, "C-info", 1, level=level, max_time=400)
    while s.step():
        pass
    r = s.result()
    assert r["outcome"] == "docked" and r["min_clearance_m"] > 0.0


def test_camera_near_clip_is_constant_across_rooms():
    """MuJoCo's near clip scales with the model extent; in big rooms it clipped the nearest floor at steep tilt."""
    from activenav.config import load_config
    from activenav.evaluation.worlds import get_world
    cfg = load_config()
    for w in ["room_10x6", "room_10x10_p9", "room_l", "room_two_rooms"]:
        m = get_world(cfg, {}, w).env.model
        assert abs(m.vis.map.znear * m.stat.extent - 0.17) < 1e-3, w


def test_b_and_c_sweep_tilt_is_locked_for_the_whole_run():
    """B and C-sweep keep the head at exactly search.fixed_tilt (0 deg) from start to dock, whatever the marker height."""
    from activenav.config import load_config
    from activenav.evaluation.search import make_search_session
    cfg = load_config()
    for conf in ("B", "C-sweep"):
        s, _ = make_search_session(cfg, {}, conf, cfg["seed"] + 1, level="two_rooms", max_time=150.0, heights=True)
        tilts = []
        while s.step():
            tilts.append(s.agent.tilt_cmd)
        assert np.allclose(tilts, float(cfg["search"]["fixed_tilt"])), conf


def test_dock_height_option_and_new_metrics():
    from activenav.config import load_config
    from activenav.evaluation.worlds import ScenarioError
    from activenav.evaluation.search import make_search_session
    cfg = load_config()
    s, _ = make_search_session(cfg, {}, "C-info", cfg["seed"], level="two_rooms", max_time=6.0, dock_height=0.6)
    while s.step():
        pass
    r = s.result()
    assert abs(r["dock_height_m"] - 0.6) < 1e-9
    for k in ("nees_mean", "nees_consistent", "sigma_xy_mean_m", "coverage_60s", "head_pan_travel_rad", "head_tilt_travel_rad",
              "near_time_s", "blocked_s", "n_recover", "n_replans", "dock_seen_frac", "approach_s"):
        assert k in r, k
    assert r["nees_mean"] >= 0.0 and 0.0 <= r["nees_consistent"] <= 1.0
    with pytest.raises(ScenarioError):
        make_search_session(cfg, {}, "B", cfg["seed"], level="two_rooms", dock_height=2.3)


def test_b_head_never_pans():
    cfg = load_config()
    s, _ = make_search_session(cfg, {}, "B", cfg["seed"] + 1, level="large_room", max_time=60, dock_height=0.25)
    pans = []
    while s.step():
        pans.append(s.agent.pan_cmd)
    assert np.allclose(pans, 0.0)


def test_b_turns_its_body_to_the_dock_when_the_head_cannot():
    """Seed 4 used to loop APPROACH -> DOCK (4 s idle) -> APPROACH forever: the locked head never saw the marker at the staging point."""
    cfg = load_config()
    s, _ = make_search_session(cfg, {}, "B", cfg["seed"] + 4, level="large_room", max_time=100, dock_height=0.3)
    while s.step():
        pass
    assert s.result()["docked"]


def test_result_survives_a_misread_marker_id():
    """A camera mis-read can put a landmark with a non-existent id into the SLAM map; scoring must skip it, not crash."""
    cfg = load_config()
    s, _ = make_search_session(cfg, {}, "C-info", cfg["seed"] + 1, level="large_room", max_time=20)
    while s.step():
        pass
    sl = s.agent.slam
    sl.lm_index[999] = next(iter(sl.lm_index.values()))          # a landmark id that is not in the world
    r = s.result()
    assert r["n_spurious_landmarks"] == 1 and r["landmark_err_m"] == r["landmark_err_m"]


@pytest.mark.parametrize("level", ["corridor", "alcove", "cluttered_14", "sparse_noisy_odom"])
def test_new_levels_build_and_are_valid(level):
    from activenav.evaluation.worlds import get_world
    from activenav.sim.world import validate_world
    cfg = load_config()
    w = cfg["levels"][level]["world"]
    ctx = get_world(cfg, {}, w)
    rep = validate_world(ctx.spec, cfg, starts=[cfg["worlds"][w]["start"]])
    assert rep["reachable_fraction"] == 1.0
    assert 10 in ctx.env.marker_map()


def test_level_settings_override_reaches_the_environment():
    cfg = load_config()
    s, ctx = make_search_session(cfg, {}, "B", cfg["seed"], level="sparse_noisy_odom", max_time=2)
    assert ctx.cfg["noise"]["wheel_scale_std"] == 0.02 and cfg["noise"]["wheel_scale_std"] != 0.02
