import numpy as np

from activenav.kinematics import (cmd_from_wheel_speeds, unicycle_step,
                             wheel_speeds_from_cmd, wrap_angle)

R, L = 0.033, 0.288


def test_wheel_mapping_roundtrip():
    for v, w in [(0.2, 0.0), (0.0, 1.0), (0.1, -0.5), (-0.05, 0.3)]:
        wl, wr = wheel_speeds_from_cmd(v, w, R, L)
        v2, w2 = cmd_from_wheel_speeds(wl, wr, R, L)
        assert np.allclose([v, w], [v2, w2])


def test_turn_left_means_right_wheel_faster():
    wl, wr = wheel_speeds_from_cmd(0.0, 1.0, R, L)
    assert wr > wl


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi / 2), np.pi / 2)


def test_unicycle_straight_and_arc():
    p = unicycle_step((0, 0, 0), 1.0, 0.0, 1.0)
    assert np.allclose(p, [1, 0, 0])
    p = (0.0, 0.0, 0.0)
    for _ in range(1000):                     # full circle: v=1, w=1 -> radius 1
        p = unicycle_step(p, 1.0, 1.0, 2 * np.pi / 1000)
    assert np.allclose(p[:2], [0, 0], atol=1e-2)
