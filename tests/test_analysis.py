import numpy as np

from activenav import analysis as an
from activenav.config import load_config

cfg = load_config()
M1 = {1: dict(x=4.0, y=1.0, normal_angle=np.pi)}
M2 = {**M1, 2: dict(x=2.0, y=-1.5, normal_angle=np.pi / 2)}
LM = [(4.0, 1.0, np.pi), (2.0, -1.5, np.pi / 2)]


def test_error_model_matches_finite_differences_of_the_nonlinear_model():
    for v, k in [(0.3, 0.0), (0.3, 0.5), (0.1, 2.0)]:
        A, B = an.jacobians(lambda e, u: an.error_dynamics(e, v, k, u[0]), [0.0, 0.0], [0.0])
        Ac, Bc = an.error_model(v, k)
        assert np.allclose(A, Ac, atol=1e-6) and np.allclose(B, Bc, atol=1e-6)


def test_path_error_model_is_controllable_only_while_moving():
    for k in (0.0, 0.5):
        assert np.isclose(np.linalg.det(an.ctrb(*an.error_model(0.2, k))), -0.2)       # det C = -v
        assert an.rank_and_sv(an.ctrb(*an.error_model(0.2, k)))[0] == 2
        assert an.rank_and_sv(an.ctrb(*an.error_model(0.0, k)))[0] == 1
    Ad, Bd = an.discretise(*an.error_model(0.2), cfg["sim"]["control_dt"])
    assert an.rank_and_sv(an.ctrb(Ad, Bd))[0] == 2
    assert np.allclose(np.abs(np.linalg.eigvals(Ad)), 1.0)                           # open loop: a double integrator on the unit circle


def test_pose_model_is_controllable_while_moving_but_not_at_rest():
    assert an.rank_and_sv(an.ctrb(*an.unicycle_linearisation(0.2, 0.7)))[0] == 3
    assert an.rank_and_sv(an.ctrb(*an.unicycle_linearisation(0.0, 0.7)))[0] == 2       # no sideways step: the nonholonomic constraint


def test_pose_observability_needs_the_marker_normal_with_one_marker():
    assert an.pose_observability(cfg, M1, (), False)[0] == 0                          # odometry only: nothing is observable
    assert an.pose_observability(cfg, M1, (0, 1), False)[0] == 2                      # range + bearing: rotation about the marker is left over
    assert an.pose_observability(cfg, M1, (0, 1, 2), False)[0] == 3                   # plus the marker normal: full pose
    assert an.pose_observability(cfg, M2, (0, 1), False)[0] == 3                      # two markers need no normal


def test_bias_states_are_observable_with_normal_or_two_markers():
    assert an.pose_observability(cfg, M1, (0, 1, 2), True)[0] == 6
    assert an.pose_observability(cfg, M2, (0, 1), True)[0] == 6
    assert an.pose_observability(cfg, M1, (0, 1), True)[0] < 6


def test_slam_has_a_three_dimensional_gauge_unless_the_start_pose_is_anchored():
    for k in (1, 2):
        r, d = an.slam_observability(cfg, LM[:k], anchored=False, bias=True)
        assert d - r == 3
        r, d = an.slam_observability(cfg, LM[:k], anchored=True, bias=True)
        assert d - r == 0


def test_uncertainty_grows_while_no_marker_is_in_view():
    s5, s10 = an.blind_growth(cfg, 5.0), an.blind_growth(cfg, 10.0)
    assert np.all(s10 > s5)
