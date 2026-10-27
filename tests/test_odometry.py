import numpy as np

from activenav.kinematics import unicycle_step, wheel_speeds_from_cmd
from activenav.sensing.odometry import WheelOdometry

R, L = 0.033, 0.288


def test_odometry_matches_unicycle_on_noise_free_arc():
    dt = 0.02
    odo = WheelOdometry(R, L)
    odo.reset((1.0, 0.5, 0.3), 0.0, 0.0)
    truth = np.array([1.0, 0.5, 0.3])
    el = er = 0.0
    for k in range(500):
        v, w = 0.15, 0.4 * np.sin(0.02 * k)
        wl, wr = wheel_speeds_from_cmd(v, w, R, L)
        el += wl * dt
        er += wr * dt
        odo.update(el, er)
        truth = unicycle_step(truth, v, w, dt)
    assert np.allclose(odo.pose[:2], truth[:2], atol=2e-3)
    assert abs(odo.pose[2] - truth[2]) < 2e-3
