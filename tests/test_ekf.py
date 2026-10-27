import copy

import numpy as np
import pytest

from activenav.config import load_config
from activenav.estimation.ekf import EKF
from activenav.kinematics import unicycle_step, wrap_angle
from activenav.types import MarkerMeasurement

MARKERS = {1: dict(x=6.0, y=1.0, z=0.25, normal_angle=np.pi, size=0.3),
           2: dict(x=3.0, y=2.0, z=0.25, normal_angle=-np.pi / 2, size=0.3)}


def make(bias=False, **over):
    cfg = copy.deepcopy(load_config())
    cfg["ekf"]["estimate_bias"] = bias
    cfg["ekf"].update(over)
    return EKF(cfg, MARKERS)


def true_meas(pose, mid):
    m = MARKERS[mid]
    dx, dy = m["x"] - pose[0], m["y"] - pose[1]
    return MarkerMeasurement(mid, float(np.hypot(dx, dy)), float(wrap_angle(np.arctan2(dy, dx) - pose[2])),
                             float(wrap_angle(m["normal_angle"] - pose[2])))


def test_predict_matches_unicycle_without_bias():
    e = make()
    e.reset((1.0, 0.5, 0.3))
    ref = np.array([1.0, 0.5, 0.3])
    for _ in range(50):
        e.predict(0.004, 0.01)
        ref = unicycle_step(ref, 0.004 / 0.02, 0.01 / 0.02, 0.02)
    assert np.allclose(e.pose, ref, atol=1e-9)


def test_state_jacobian_matches_finite_differences():
    e = make(bias=True, k_ds=0.0, k_dth=0.0, k_dth_ds=0.0, bias_walk=[0.0, 0.0, 0.0])
    x0 = np.array([1.0, 0.5, 0.7, 0.03, 0.01, -0.02])
    ds, dth, h = 0.004, 0.01, 1e-6
    Fn = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = h
        Fn[:, i] = (e._motion(x0 + d, ds, dth) - e._motion(x0 - d, ds, dth)) / (2 * h)
    e.x, e.P = x0.copy(), np.eye(6)
    e.predict(ds, dth)
    assert np.allclose(e.P, Fn @ Fn.T, atol=1e-8)          # P' = F I F^T when Q = 0


@pytest.mark.parametrize("col", [0, 1])
def test_input_jacobian_matches_finite_differences(col):
    over = dict(k_ds=0.5, k_dth=0.0, k_dth_ds=0.0) if col == 0 else dict(k_ds=0.0, k_dth=1.0, k_dth_ds=0.0)
    e = make(bias=True, bias_walk=[0.0, 0.0, 0.0], **over)
    x0 = np.array([1.0, 0.5, 0.7, 0.03, 0.01, -0.02])
    ds, dth, h = 0.004, 0.01, 1e-7
    e.x, e.P = x0.copy(), np.zeros((6, 6))
    e.predict(ds, dth)
    inp = np.array([ds, dth])
    d = np.zeros(2)
    d[col] = h
    g = (e._motion(x0, *(inp + d)) - e._motion(x0, *(inp - d)))[:3] / (2 * h)
    g[2] = wrap_angle(g[2])
    sigma = (0.5 * ds) if col == 0 else dth
    assert np.allclose(e.P[:3, :3], sigma ** 2 * np.outer(g, g), atol=1e-12)


def test_measurement_jacobian_matches_finite_differences():
    e = make()
    pose = np.array([2.0, -0.5, 0.4])
    _, H = e.predict_measurement(1, pose)
    Hn = np.zeros((3, 3))
    h = 1e-6
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        diff = e.predict_measurement(1, pose + d)[0] - e.predict_measurement(1, pose - d)[0]
        diff[1:] = wrap_angle(diff[1:])
        Hn[:, i] = diff / (2 * h)
    assert np.allclose(H, Hn, atol=1e-6)


def test_perfect_measurement_leaves_state_and_shrinks_covariance():
    e = make()
    pose = (2.0, -0.5, 0.4)
    e.reset(pose)
    e.bias_c = 0.0                                          # no bias correction for an ideal measurement
    p0 = e.P.copy()
    info = e.update_marker(true_meas(pose, 1))
    assert info.accepted and np.allclose(e.pose, pose, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(e.P) > 0) and np.allclose(e.P, e.P.T)
    assert np.trace(e.P) < np.trace(p0)


def test_update_pulls_estimate_toward_truth():
    e = make()
    truth = np.array([2.0, -0.5, 0.4])
    e.reset(truth + [0.05, -0.04, 0.03], init_std=[0.1, 0.1, 0.1])
    e.bias_c = 0.0
    before = np.linalg.norm(e.pose[:2] - truth[:2])
    for mid in (1, 2, 1, 2):
        e.update_marker(true_meas(truth, mid))
    assert np.linalg.norm(e.pose[:2] - truth[:2]) < 0.3 * before
    assert abs(wrap_angle(e.pose[2] - truth[2])) < 0.01


def test_gate_rejects_gross_outlier_and_leaves_state():
    e = make()
    pose = (2.0, -0.5, 0.4)
    e.reset(pose)
    z = true_meas(pose, 1)
    bad = MarkerMeasurement(1, z.range + 0.3, z.bearing, z.normal_angle)   # 30 cm range error: far outside R
    x0, P0 = e.x.copy(), e.P.copy()
    info = e.update_marker(bad)
    assert not info.accepted and info.reason == "gated"
    assert np.array_equal(e.x, x0) and np.array_equal(e.P, P0)


def test_unknown_id_and_range_limits_are_ignored():
    e = make()
    e.reset((2.0, 0.0, 0.0))
    assert e.update_marker(MarkerMeasurement(99, 2.0, 0.0, 0.0)).reason == "unknown id"
    assert e.update_marker(MarkerMeasurement(1, 9.0, 0.0, 0.0)).reason == "range out of bounds"


def test_bias_states_recover_a_constant_heading_drift_and_stay_consistent():
    """Kinematic simulation (no MuJoCo): odometry has a constant drift of c_true rad/m."""
    rng = np.random.default_rng(3)
    c_true = 0.05                                               # odometry heading drift [rad/m]
    e = make(bias=True)
    e.bias_c = 0.0                                          # this toy sensor has no range bias to correct
    truth = np.array([0.5, -1.0, 0.0])
    e.reset(truth)
    nees = []
    ds = 0.004
    for k in range(1500):
        w_cmd = 0.0 if k < 800 else 0.4                        # some turning so heading changes too
        dth = w_cmd * 0.02
        truth = unicycle_step(truth, ds / 0.02, w_cmd, 0.02)
        # The filter corrects odometry as dth_true = dth_odo - c*ds, so odometry that reads
        # dth_odo = dth_true + c_true*ds is exactly a drift of c_true that the filter must find.
        e.predict(ds, dth + c_true * ds)
        if k % 5 == 0:
            for mid in (1, 2):
                m = true_meas(truth, mid)
                if abs(m.bearing) < 1.2 and m.range < 5.0:
                    e.update_marker(MarkerMeasurement(
                        mid, m.range + rng.normal(0, 0.004 + 0.002 * m.range ** 2), m.bearing + rng.normal(0, 0.003),
                        m.normal_angle + rng.normal(0, 0.01)))
        err = truth - e.pose
        err[2] = wrap_angle(err[2])
        nees.append(err @ np.linalg.solve(e.P_pose, err))
    assert abs(e.bias[0] - c_true) < 0.4 * abs(c_true)
    assert 0.2 < np.mean(nees) < 12.0


def test_bias_states_recover_a_turn_scale_error():
    """Spin-in-place readings are 3% too large; the filter must find k = -0.03 from marker bearings."""
    rng = np.random.default_rng(5)
    e = make(bias=True)
    e.bias_c = 0.0
    truth = np.array([2.0, 0.0, 0.0])
    e.reset(truth)
    k_true = -0.03                                         # dth_true = dth_odo * (1 + k)
    for step in range(900):
        w_cmd = 0.8 if (step // 150) % 2 == 0 else -0.8    # back-and-forth spins
        dth = w_cmd * 0.02
        truth = unicycle_step(truth, 0.0, w_cmd, 0.02)
        e.predict(0.0, dth / (1.0 + k_true))
        if step % 5 == 0:
            for mid in (1, 2):
                m = true_meas(truth, mid)
                if abs(m.bearing) < 1.2 and m.range < 5.0:
                    e.update_marker(MarkerMeasurement(
                        mid, m.range + rng.normal(0, 0.004), m.bearing + rng.normal(0, 0.003),
                        m.normal_angle + rng.normal(0, 0.01)))
    assert abs(e.bias[2] - k_true) < 0.4 * abs(k_true)


def test_bearing_only_mode_still_fuses_bearing_when_range_and_normal_are_throttled():
    markers = {1: dict(x=6.0, y=1.0, z=0.25, normal_angle=np.pi, size=0.3)}
    e = make(bias=False, min_update_interval_s=0.5, throttle_mode="bearing_only")
    e.markers = markers
    e.reset((2.0, 0.0, 0.0), init_std=[0.1, 0.1, 0.1])
    dx, dy = 4.0, 1.0
    full = MarkerMeasurement(1, float(np.hypot(dx, dy)), float(np.arctan2(dy, dx)), float(np.pi))
    assert e.update_marker(full, t=1.0).accepted
    e.x[2] += 0.004
    before = abs(e.x[2])
    info = e.update_marker(full, t=1.1)                             # inside the 0.5 s window
    assert info.reason == "bearing only" and info.accepted
    assert abs(e.x[2]) < before                                     # bearing information was used
    assert np.all(np.linalg.eigvalsh(e.P) > 0) and np.allclose(e.P, e.P.T)
    bad = MarkerMeasurement(1, full.range, full.bearing + 0.2, full.normal_angle)
    assert e.update_marker(bad, t=1.2).reason == "gated bearing"
    skip = make(bias=False, min_update_interval_s=0.5, throttle_mode="skip")
    skip.markers = markers
    skip.reset((2.0, 0.0, 0.0))
    skip.update_marker(full, t=1.0)
    assert skip.update_marker(full, t=1.1).reason == "throttled"
