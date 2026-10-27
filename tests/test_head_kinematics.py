import numpy as np

from activenav.sensing.camera import intrinsics_from_fovy
from activenav.sensing.camera import camera_to_base

HEAD = dict(pan_origin=[-0.049, 0.0, 0.1255], tilt_offset=[0.0, 0.0, 0.065],
            camera_offset=[0.022, 0.0, 0.0])


def test_camera_position_and_axes_at_zero():
    T = camera_to_base(HEAD, 0.0, 0.0)
    assert np.allclose(T[:3, 3], [-0.027, 0.0, 0.1905])
    assert np.allclose(T[:3, 2], [1, 0, 0])      # forward
    assert np.allclose(T[:3, 0], [0, -1, 0])     # image right = robot right
    assert np.allclose(T[:3, 1], [0, 0, -1])     # image down = down


def test_positive_tilt_looks_down():
    T = camera_to_base(HEAD, 0.0, 0.3)
    assert np.allclose(T[:3, 2], [np.cos(0.3), 0, -np.sin(0.3)])


def test_positive_pan_looks_left():
    T = camera_to_base(HEAD, np.pi / 2, 0.0)
    assert np.allclose(T[:3, 2], [0, 1, 0], atol=1e-12)


def test_intrinsics_from_fovy():
    K = intrinsics_from_fovy(60.0, 640, 480)
    assert np.isclose(K[0, 0], 240 / np.tan(np.radians(30)))
    assert np.isclose(K[0, 0], K[1, 1])
    assert np.allclose([K[0, 2], K[1, 2]], [319.5, 239.5])
