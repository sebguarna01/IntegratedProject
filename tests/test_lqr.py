import numpy as np

from activenav.control.lqr import LqrTracker, LqrWeights, design_lqr, stadium
from activenav.kinematics import unicycle_step

DT = 0.02


def _drive(tracker, pose, seconds, bias=0.0):
    """Perfect-state simulation of the unicycle under the tracker; returns the true pose history."""
    out = []
    for _ in range(int(seconds / DT)):
        v, w = tracker.command(pose)
        pose = unicycle_step(pose, v, w + bias, DT)
        out.append(pose)
    return np.array(out)


def _ey(path, poses):
    return np.array([path.errors(p, path.nearest(p, None))[0] for p in poses])


def test_design_is_stable_and_has_the_right_shape():
    K, poles = design_lqr(0.2, DT)
    assert K.shape == (1, 3) and np.all(np.abs(poles) < 1.0) and K[0, 0] > 0 and K[0, 1] > 0
    K2, poles2 = design_lqr(0.2, DT, integral=False)
    assert K2.shape == (1, 2) and np.all(np.abs(poles2) < 1.0)


def test_heavier_effort_weight_gives_smaller_gains():
    K_hi, _ = design_lqr(0.2, DT, LqrWeights(dw_max=2.0))
    K_lo, _ = design_lqr(0.2, DT, LqrWeights(dw_max=0.2))
    assert np.all(K_hi > K_lo)                      # allowing more effort buys a harder controller


def test_stadium_geometry_and_error_signs():
    p = stadium()
    assert np.isclose(p.length, 4.0 + 2 * np.pi, atol=0.05)
    assert np.isclose(p.kappa[0], 0.0) and np.isclose(p.kappa.max(), 1.0)
    i = 10                                          # lower straight, heading east
    ey, eth = p.errors((p.p[i][0], p.p[i][1] + 0.1, 0.2), i)
    assert np.isclose(ey, 0.1) and np.isclose(eth, 0.2)      # left of the path is positive


def test_converges_from_an_offset_start():
    p = stadium()
    t = LqrTracker(p, 0.2, Ts=DT)
    poses = _drive(t, np.array([p.p[40][0], p.p[40][1] + 0.3, 0.3]), 20.0)
    e = abs(_ey(p, poses))
    assert e[int(5.0 / DT):].max() < 0.05           # inside 5 cm within 5 s
    assert e[-100:].max() < 0.005                    # and settled by 20 s


def test_integral_action_removes_the_offset_a_constant_disturbance_causes():
    p = stadium()
    laps_s = 2 * p.length / 0.2
    last = lambda poses: poses[-int(p.length / 0.2 / DT):]
    with_i = _drive(LqrTracker(p, 0.2, Ts=DT, integral=True), np.array([*p.p[0], p.psi[0]]), laps_s, bias=0.1)
    without = _drive(LqrTracker(p, 0.2, Ts=DT, integral=False), np.array([*p.p[0], p.psi[0]]), laps_s, bias=0.1)
    assert abs(_ey(p, last(with_i)).mean()) < 0.002
    assert abs(_ey(p, last(without)).mean()) > 0.005


def test_fillet_path_rounds_a_corner_and_keeps_the_ends():
    from activenav.control.lqr import fillet_path
    path, vs = fillet_path([(0, 0), (1, 0), (1, 1)], radius=0.3)
    assert np.allclose(path.p[0], (0, 0)) and np.allclose(path.p[-1], (1, 1))
    assert len(vs) == 3 and vs[0] == 0 and abs(vs[-1] - path.length) < 1e-9
    assert np.isclose(path.kappa.max(), 1 / 0.3, rtol=1e-6)
    assert np.hypot(*(path.p - np.array([1.0, 0.0])).T).min() > 0.05          # the corner itself is cut
