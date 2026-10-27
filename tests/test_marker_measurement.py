import numpy as np

from activenav.kinematics import wrap_angle
from activenav.sensing.markers import marker_world_pose, measure_marker
from activenav.sensing.camera import camera_to_base, hom, rot_z
from activenav.types import MarkerDetection

HEAD = dict(pan_origin=[-0.049, 0.0, 0.1255], tilt_offset=[0.0, 0.0, 0.065],
            camera_offset=[0.022, 0.0, 0.0])


def _synthetic_measurement(pose, pan, tilt, marker):
    """Build the marker pose in the camera frame from known world poses (no OpenCV)."""
    x, y, yaw = pose
    T_W_B = hom(R=rot_z(yaw), t=[x, y, 0.0])
    T_B_C = camera_to_base(HEAD, pan, tilt)
    T_W_M = marker_world_pose(*marker)
    T_C_M = np.linalg.inv(T_W_B @ T_B_C) @ T_W_M
    det = MarkerDetection(3, np.zeros((4, 2)), T_C_M[:3, :3], T_C_M[:3, 3])
    return measure_marker(T_B_C, det)


def test_range_bearing_normal_match_truth_for_random_poses():
    rng = np.random.default_rng(0)
    for _ in range(50):
        pose = (rng.uniform(0.5, 5.5), rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi))
        pan, tilt = rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.9)
        marker = (rng.uniform(0, 6), rng.uniform(-2, 2), 0.25, rng.uniform(-np.pi, np.pi))
        z = _synthetic_measurement(pose, pan, tilt, marker)

        dx, dy = marker[0] - pose[0], marker[1] - pose[1]
        assert np.isclose(z.range, np.hypot(dx, dy))
        assert np.isclose(wrap_angle(z.bearing - (np.arctan2(dy, dx) - pose[2])), 0.0, atol=1e-9)
        assert np.isclose(wrap_angle(z.normal_angle - (marker[3] - pose[2])), 0.0, atol=1e-9)
