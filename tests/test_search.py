import numpy as np
import pytest

from activenav.config import load_config
from activenav.evaluation.worlds import ScenarioError
from activenav.evaluation.search import make_search_session, to_frame

cfg = load_config()


def _run(conf, level, seed, **kw):
    s, ctx = make_search_session(cfg, {}, conf, seed, level=level, **kw)
    while s.step():
        pass
    return s, s.result()


@pytest.mark.parametrize("conf", ["B", "C-sweep", "C-info"])
def test_robot_with_no_map_finds_marker_10_and_docks_in_the_large_room(conf):
    s, r = _run(conf, "large_room", 1)
    assert r["outcome"] == "docked", r["outcome"]
    assert r["miss_m"] <= 0.10 and r["yaw_err_deg"] <= 5.0
    assert r["n_landmarks"] >= 1


def test_search_in_the_pillar_room_with_random_markers_docks_without_contact():
    s, r = _run("C-info", "large_6_pillars", 2, random_markers=True, random_dock=True)
    assert r["outcome"] == "docked" and not r["collided"]
    assert r["coverage"] > 0.1                       # it mapped something on the way
    assert r["landmark_err_m"] < 0.25


def test_config_a_and_unavailable_levels_are_refused_with_a_message():
    with pytest.raises(ScenarioError):
        make_search_session(cfg, {}, "A", 0)
    with pytest.raises(ScenarioError):
        make_search_session(cfg, {}, "B", 0, level="closing_door")
    with pytest.raises(ScenarioError):
        make_search_session(cfg, {}, "B", 0, start=(0.1, 0.1, 0.0))


def test_agent_only_sees_odometry_markers_and_scans():
    """The agent's belief frame is the start frame: its first pose is the origin whatever the true start is."""
    s, _ = make_search_session(cfg, {}, "B", 3, level="large_room", start=(7.0, 1.0, 2.0))
    assert np.allclose(s.agent.pose, 0.0)
    assert np.allclose(to_frame(s.start_pose, s.start_pose), 0.0)


def test_sweep_gaze_is_a_constant_rate_triangle_between_the_limits():
    import numpy as np
    from activenav.autonomy.search_gaze import SweepSearchGaze
    g = SweepSearchGaze(load_config())
    t = np.arange(0.0, 16.0, 0.05)
    pan = np.array([g(None, x)[0] for x in t])
    assert abs(pan.max() - g.amp) < 0.03 and abs(pan.min() + g.amp) < 0.03
    rate = np.abs(np.diff(pan)) / 0.05
    assert np.allclose(rate, 4 * g.amp / g.period, atol=1e-6)


def test_sweep_pan_is_not_clamped_by_clutter_in_the_agent():
    import numpy as np
    s, _ = make_search_session(load_config(), {}, "C-sweep", 3, sensor="synthetic")
    pans = []
    while s.step() and s.env.data.time < 60:
        if s.agent.state == "EXPLORE":
            pans.append(abs(s.agent.pan_cmd))
    assert max(pans) > np.radians(60)


def test_gaze_metrics_tell_a_fixed_head_from_a_sweep():
    import numpy as np
    from activenav.evaluation.search import gaze_metrics
    t = np.arange(0, 80, 0.02)
    est = np.column_stack([0.2 * t, np.zeros_like(t), np.zeros_like(t)])            # driving straight, heading 0
    fixed = gaze_metrics(t, est, np.zeros((len(t), 2)))
    ph = (t / 8.0) % 1.0
    tri = np.where(ph < 0.25, 4 * ph, np.where(ph < 0.75, 2 - 4 * ph, 4 * ph - 4))
    sweep = gaze_metrics(t, est, np.column_stack([1.2 * tri, np.zeros_like(t)]))
    assert fixed["gaze_switch_per_min"] == 0 and fixed["pan_std_rad"] == 0
    assert fixed["gaze_toward_travel"] == 1.0
    assert abs(sweep["gaze_switch_per_min"] - 15.0) < 2.0                              # two reversals per 8 s cycle
    assert sweep["pan_std_rad"] > 0.6 and sweep["gaze_toward_travel"] < 0.9
