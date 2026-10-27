"""Camera and head: pinhole intrinsics and the pan-tilt forward kinematics (base frame to camera frame)."""
from __future__ import annotations

import numpy as np



# -------------------- sensing/camera.py --------------------
# Camera intrinsics.
def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsic matrix K for a square-pixel camera with the given vertical field of view.

    This is the analytic value; scripts/calibrate_camera.py reproduces it from rendered checkerboard views to within 1 px.
    """
    f = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0],
                     [0.0, f, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


# -------------------- sensing/camera.py --------------------
# Kinematic chain  base -> pan joint -> tilt joint -> camera.
# Columns are the OpenCV camera axes (right, down, forward) expressed in the tilt-body frame.
R_TILT_CAM = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])

def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def hom(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T

def camera_to_base(head: dict, pan: float, tilt: float) -> np.ndarray:
    """4x4 transform T_B_C taking camera-frame points into the robot base frame.

    `head` holds the geometry (config section `head`): pan_origin, tilt_offset, camera_offset.
    """
    return (hom(t=head["pan_origin"]) @ hom(R=rot_z(pan))
            @ hom(t=head["tilt_offset"]) @ hom(R=rot_y(tilt))
            @ hom(t=head["camera_offset"]) @ hom(R=R_TILT_CAM))


# ----------------------------------------------------------------------------------------- calibration
def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    """Inner-corner grid of a checkerboard in its own frame (x along the columns, y along the rows, z = 0), [m]."""
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return g


def calibrate(obj_pts: list, img_pts: list, size: tuple, pinhole_only: bool = False):
    """OpenCV calibration from checkerboard corners. `size` = (width, height).
    pinhole_only fixes all distortion at zero (to test whether the distortion terms earn their place).
    Returns dict(K, dist, rms, per_image_rms, rvecs, tvecs); errors in pixels."""
    import cv2
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3) if pinhole_only else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None, flags=flags)
    per = []
    for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return dict(K=K, dist=dist.ravel(), rms=float(rms), per_image_rms=np.array(per), rvecs=rvecs, tvecs=tvecs)


def pose_difference(T_a: np.ndarray, T_b: np.ndarray):
    """(translation difference [m], rotation difference [deg]) between two 4x4 poses."""
    dT = np.linalg.inv(T_a) @ T_b
    ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    return float(np.linalg.norm(dT[:3, 3])), float(ang)
