"""Camera and head: intrinsics (the camera profile in use), and the pan-tilt forward kinematics (base frame to camera frame)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class CameraModel:
    """What the robot knows about its head camera: image size, intrinsics, lens distortion and where the lens sits on
    the head. For the `ideal` profile these come from `camera_model.fovy_deg` and `sim.camera`, exactly as before;
    for a real-camera profile (`cameras.<name>`) they are the values the device reports (its factory calibration)."""
    name: str
    width: int
    height: int
    K: np.ndarray
    dist: Optional[np.ndarray]     # OpenCV k1 k2 p1 p2 k3, or None for a distortion-free pinhole
    vfov: float                    # vertical field of view [rad]
    hfov: float                    # horizontal field of view [rad]
    head: dict                     # head geometry (config `head`, camera_offset = this camera's lens position)


def camera_profile(cfg: dict) -> str:
    """Name of the head camera profile in use: `ideal` (default) or a key of `cameras` (select with --set camera=d435i)."""
    name = cfg.get("camera") or "ideal"
    if name != "ideal" and name not in cfg.get("cameras", {}):
        raise ValueError(f"unknown camera profile '{name}'. Available: ideal, {', '.join(cfg.get('cameras', {}))}")
    return name


def camera_model(cfg: dict) -> CameraModel:
    """The head camera of the selected profile."""
    name = camera_profile(cfg)
    if name == "ideal":
        cam, fovy = cfg["sim"]["camera"], float(cfg["camera_model"]["fovy_deg"])
        W, H = int(cam["width"]), int(cam["height"])
        return CameraModel(name, W, H, intrinsics_from_fovy(fovy, W, H), None, float(np.radians(fovy)),
                           float(2.0 * np.arctan(np.tan(np.radians(fovy) / 2) * W / H)), cfg["head"])
    p = cfg["cameras"][name]
    W, H = int(p["width"]), int(p["height"])
    K = np.array([[p["fx"], 0.0, p["cx"]], [0.0, p["fy"], p["cy"]], [0.0, 0.0, 1.0]], float)
    head = dict(cfg["head"], camera_offset=[float(v) for v in p["lens_offset"]])
    return CameraModel(name, W, H, K, np.asarray(p["dist"], float), float(2.0 * np.arctan(H / (2.0 * p["fy"]))),
                       float(2.0 * np.arctan(W / (2.0 * p["fx"]))), head)


def camera_settings(cfg: dict) -> list:
    """Settings a real-camera profile changes elsewhere (`set: ["ekf.range_bias_coeff=...", ...]`, applied like --set
    when a trial is built), e.g. the EKF's range-bias correction fitted to that camera. None for the ideal camera."""
    name = camera_profile(cfg)
    return [] if name == "ideal" else list(cfg["cameras"][name].get("set") or [])


def hfov_deg(cfg: dict) -> float:
    """Horizontal field of view [deg] that the gaze policies and synthetic sensors assume: 75.2 for the ideal camera
    (the rounded value every earlier result was produced with), the profile's own value otherwise."""
    m = camera_model(cfg)
    return 75.2 if m.name == "ideal" else float(np.degrees(m.hfov))


def pixel_rays(model: CameraModel, u, v) -> np.ndarray:
    """Normalised camera rays (n, 3) [x, y, 1] (OpenCV camera frame) for pixels (u, v), lens distortion removed."""
    u, v = np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float))
    if model.dist is None:
        return (np.linalg.inv(model.K) @ np.vstack([u, v, np.ones_like(u)])).T
    import cv2
    xy = cv2.undistortPoints(np.stack([u, v], axis=1).reshape(-1, 1, 2), model.K, model.dist,
                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9)).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy))])


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
