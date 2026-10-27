import numpy as np
from pathlib import Path

from activenav.config import load_config
from activenav.estimation.slam import SlamEKF
from activenav.kinematics import wrap_angle
from activenav.types import MarkerMeasurement

cfg = load_config()


def _meas(pose, m):
    x, y, th = pose
    dx, dy = m[0] - x, m[1] - y
    return MarkerMeasurement(0, float(np.hypot(dx, dy)), float(wrap_angle(np.arctan2(dy, dx) - th)),
                             float(wrap_angle(m[2] - th)))


def _filter():
    f = SlamEKF(cfg)
    f.reset(np.array([0.0, 0.0, 0.0]))
    return f


def test_jacobian_matches_finite_differences():
    f = _filter()
    f.update_marker(_meas((0, 0, 0), (2.0, 1.0, 3.0)))
    f.x[:3] = [0.3, -0.2, 0.4]
    h0, H = f._expected(0)
    for k in list(range(3)) + [f.lm_index[0] + i for i in range(3)]:
        e = 1e-6
        x0 = f.x.copy()
        f.x[k] += e
        h1, _ = f._expected(0)
        f.x = x0
        d = h1 - h0
        d[1:] = wrap_angle(d[1:])
        assert np.allclose(d / e, H[:, k], atol=1e-4), k


def test_new_landmark_lands_on_truth_and_covariance_is_valid():
    f = _filter()
    m = (2.0, 1.0, 3.0)
    info = f.update_marker(_meas((0, 0, 0), m))
    assert info.accepted and info.reason == "new landmark"
    mk = f.markers[0]
    # range bias correction is tiny; position should be within a few mm, normal exact
    assert abs(mk["x"] - m[0]) < 0.02 and abs(mk["y"] - m[1]) < 0.02
    assert abs(wrap_angle(mk["normal_angle"] - m[2])) < 1e-9
    assert np.all(np.linalg.eigvalsh(f.P) > -1e-12)


def test_loop_closure_reduces_pose_uncertainty_and_corrects_drift():
    f = _filter()
    m = (2.0, 0.0, np.pi)
    f.update_marker(_meas((0, 0, 0), m), t=0.0)
    sig0 = f.sigma()[0]
    # drive away with a systematic heading error, then see the marker again from the true place
    for _ in range(200):
        f.predict(0.005, 0.0)                         # 1 m forward, no change in belief about drift
    true = (1.0, 0.0, 0.0)
    f.x[:3] = [1.0, 0.05, 0.03]                       # belief has drifted
    grew = f.sigma()[1]
    for k in range(5):
        f.update_marker(_meas(true, m), t=10.0 + k)
    assert f.sigma()[1] < grew
    assert abs(f.x[1]) < 0.05 and abs(f.x[2]) < 0.03   # pulled back toward the truth
    assert sig0 < grew


def test_slam_matches_known_map_filter_on_a_crossing():
    import numpy as np
    from activenav.estimation.ekf import EKF
    from activenav.evaluation.synthetic import SyntheticMarkerSensor
    from activenav.evaluation.trials import run_ekf_trial
    from activenav.sim.env import DockingEnv
    cr = cfg["crossing"]
    env = DockingEnv(cfg)
    out = {}
    for slam in (False, True):
        ekf = SlamEKF(cfg) if slam else EKF(cfg, env.marker_map())
        r = run_ekf_trial(env, 0, cr["start_pose"], [cr["goal"]], SyntheticMarkerSensor(env, np.random.default_rng(1000)), ekf,
                          max_time=120.0)
        out[slam] = r["position_rmse_m"]
        assert not r["collided"]
    env.close()
    assert out[True] < 0.08, out            # no map to anchor to: a few cm is expected (known map: ~1 cm)


def test_robot_side_code_never_reads_maps_or_ground_truth():
    src = Path(__file__).resolve().parents[1] / "src" / "activenav"
    banned = ["ground_truth", "marker_map", "obstacle_map", "room_bounds", "activenav.sim", "activenav.evaluation",
              "load_world", "WorldSpec"]
    robot_side = [src / "estimation" / "slam.py"] + list((src / "autonomy").glob("*.py")) \
        if (src / "autonomy").exists() else [src / "estimation" / "slam.py"]
    bad = [(f.name, w) for f in robot_side for w in banned if w in f.read_text()]
    assert not bad, bad


def test_landmark_born_from_a_bad_view_is_replaced_by_a_close_view():
    """A first sighting with a wrong normal (long-range pose flip) must not lock the landmark for ever."""
    f = _filter()
    truth = (3.0, 1.0, 2.0)
    g = _meas((0, 0, 0), truth)
    bad = MarkerMeasurement(0, g.range, g.bearing, float(wrap_angle(g.normal_angle + 1.2)))
    f.update_marker(bad)                                           # lands with a wrong normal
    f.x[:3] = [1.6, 0.6, 0.0]                                      # robot now close; filter is confident in its pose
    f.P[:3, :3] = np.eye(3) * 1e-6
    infos = [f.update_marker(_meas((1.6, 0.6, 0.0), truth), t=0.1 * k) for k in range(12)]
    assert any(i.reason == "reinitialised" for i in infos)
    assert infos[-1].accepted
    m = f.markers[0]
    assert abs(wrap_angle(m["normal_angle"] - truth[2])) < 0.05
