"""Head camera profiles: the ideal camera is unchanged, and the simulated RealSense D435i colour camera forms its image
the way the robot's model of it says (intrinsics, lens distortion, colour, rolling shutter, latency)."""
import copy

import cv2
import numpy as np
import pytest

from activenav.autonomy.obstacle_sensor import GroundGeometry
from activenav.config import apply_overrides, load_config
from activenav.sensing.camera import camera_model, camera_to_base, hfov_deg, intrinsics_from_fovy
from activenav.sim.realsense import ColorCamera


def _cfg(camera="ideal"):
    return apply_overrides(load_config(), [f"camera={camera}"])


def _dot_image(cam: ColorCamera, ray, value=255):
    """Ideal render (black) with one bright pixel where the render camera sees the direction `ray`."""
    img = np.zeros((cam.render_h, cam.render_w, 3), np.uint8)
    Kr = intrinsics_from_fovy(cam.render_fovy, cam.render_w, cam.render_h)
    u = Kr[0, 0] * ray[0] / ray[2] + Kr[0, 2]
    v = Kr[1, 1] * ray[1] / ray[2] + Kr[1, 2]
    img[int(round(v)), int(round(u))] = value
    return img, (u, v)


def _peak(img):
    g = img.astype(float).sum(axis=2)
    v, u = np.unravel_index(np.argmax(g), g.shape)
    w = g[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    vv, uu = np.mgrid[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    return float((uu * w).sum() / w.sum()), float((vv * w).sum() / w.sum())


def test_ideal_profile_is_the_original_camera():
    cfg = load_config()
    m = camera_model(cfg)
    assert m.name == "ideal" and m.dist is None
    assert np.array_equal(m.K, intrinsics_from_fovy(cfg["camera_model"]["fovy_deg"], 640, 480))
    assert m.head is cfg["head"]
    assert hfov_deg(cfg) == 75.2


def test_d435i_field_of_view_matches_the_datasheet():
    m = camera_model(_cfg("d435i"))
    assert (m.width, m.height) == (848, 480)
    assert abs(np.degrees(m.hfov) - 69.0) <= 3.0 and abs(np.degrees(m.vfov) - 42.0) <= 3.0     # 69 x 42 deg +-3
    assert m.head["camera_offset"] == [0.0125, 0.0, 0.0]


def test_colour_survives_the_bayer_filter_and_demosaicing():
    cam = ColorCamera(_cfg("d435i")["cameras"]["d435i"])
    img = np.empty((cam.render_h, cam.render_w, 3), np.uint8)
    img[:] = (180, 120, 60)
    out = cam(img, (0.0, 0.0, 0.0), rng=None)
    centre = out[cam.H // 2 - 5:cam.H // 2 + 5, cam.W // 2 - 5:cam.W // 2 + 5].reshape(-1, 3).mean(axis=0)
    assert np.all(np.abs(centre - (180, 120, 60)) < 4)
    assert out[2, 2].mean() < centre.mean() - 10                                  # vignetting: corners darker


def test_lens_distortion_puts_points_where_opencv_projects_them():
    p = _cfg("d435i")["cameras"]["d435i"]
    cam = ColorCamera(p)
    K, dist = cam.K, cam.dist
    for ray in ([0.0, 0.0, 1.0], [0.55, 0.3, 1.0], [-0.6, -0.33, 1.0], [0.62, -0.05, 1.0]):
        img, _ = _dot_image(cam, ray)
        u, v = _peak(cam(img, (0.0, 0.0, 0.0), rng=None))
        exp, _ = cv2.projectPoints(np.array([ray], float), np.zeros(3), np.zeros(3), K, dist)
        assert np.hypot(u - exp[0, 0, 0], v - exp[0, 0, 1]) < 0.75, (ray, (u, v), exp.ravel())


def test_rolling_shutter_skews_and_blurs_a_turning_camera():
    p = _cfg("d435i")["cameras"]["d435i"]
    cam = ColorCamera(p)
    img = np.zeros((cam.render_h, cam.render_w, 3), np.uint8)
    img[:, cam.render_w // 2 - 1:cam.render_w // 2 + 1] = 255                   # a vertical line straight ahead
    wy = 1.5                                                                    # panning, rad/s (OpenCV y axis)
    still = cam(img, (0.0, 0.0, 0.0), rng=None).astype(float).sum(axis=2)
    moving = cam(img, (0.0, wy, 0.0), rng=None).astype(float).sum(axis=2)

    def col(row_img, r):
        x = row_img[r]
        return float((np.arange(len(x)) * x).sum() / x.sum())
    top, bottom = 40, cam.H - 41
    skew = col(moving, bottom) - col(moving, top)
    # +wy turns the camera to the right, so the scene slides left; the bottom rows are read later: further left
    expected = -cam.K[0, 0] * wy * p["readout_s"] * (bottom - top) / (cam.H - 1)
    assert abs(skew - expected) < 0.15 * abs(expected), (skew, expected)
    assert abs(col(still, bottom) - col(still, top)) < 1.5                      # lens distortion alone barely bends it
    width = lambda a: float((a[cam.H // 2] > 0.25 * a[cam.H // 2].max()).sum())
    assert width(moving) > width(still) + 0.5 * cam.K[0, 0] * wy * p["exposure_s"]   # motion blur


def test_floor_ranging_undoes_the_lens_distortion():
    cfg = _cfg("d435i")
    geo = GroundGeometry(cfg)
    m = camera_model(cfg)
    pan, tilt = 0.3, 0.35
    T = camera_to_base(m.head, pan, tilt)
    floor = np.array([[1.2, 0.4, 0.0], [0.9, -0.35, 0.0], [2.0, 0.9, 0.0]])
    pc = (np.linalg.inv(T) @ np.c_[floor, np.ones(3)].T).T[:, :3]               # floor points in the camera frame
    uv, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), m.K, m.dist)
    pts, ok = geo.pixel_to_floor(uv[:, 0, 0], uv[:, 0, 1], pan, tilt)
    assert ok.all() and np.abs(pts - floor[:, :2]).max() < 1e-3


def test_late_measurements_are_carried_into_the_current_base_frame():
    from activenav.autonomy.obstacle_sensor import ObstacleScan
    from activenav.sensing.delay import OdometryHistory, marker_to_now, scan_to_now
    from activenav.types import MarkerMeasurement
    h = OdometryHistory(dt=0.02)
    h.reset(0.0)
    for k in range(5):                                                          # forward while turning left
        h.add(0.02 * (k + 1), 0.012, 0.06)
    x, y, th = h._poses[-1][1]                                                  # pose now, in the frame of t = 0
    P = np.array([2.0, 0.5])                                                    # a landmark, seen at t = 0 from the origin
    z = marker_to_now(MarkerMeasurement(10, float(np.hypot(*P)), float(np.arctan2(P[1], P[0])), np.pi), h.motion_since(0.0))
    c, s = np.cos(th), np.sin(th)
    q = np.array([c * (P[0] - x) + s * (P[1] - y), -s * (P[0] - x) + c * (P[1] - y)])
    assert abs(z.range - np.hypot(*q)) < 1e-9 and abs(z.bearing - np.arctan2(q[1], q[0])) < 1e-9
    assert abs(z.normal_angle - (np.pi - th)) < 1e-9
    scan = scan_to_now(ObstacleScan(np.zeros(2), P[None, :], np.array([True]), np.array([0.01])), h.motion_since(0.0))
    assert np.allclose(scan.points[0], q) and np.allclose(scan.origin, [c * -x + s * -y, -s * -x + c * -y])


def test_d435i_on_the_robot_frames_arrive_late_with_their_head_angles():
    from activenav.sim.env import DockingEnv
    from activenav.types import Command
    cfg = copy.deepcopy(_cfg("d435i"))
    env = DockingEnv(cfg)
    p = cfg["cameras"]["d435i"]
    assert np.allclose(env.model.cam_pos[env._cam_id], p["lens_offset"])
    assert "d435i_body" in {env.model.geom(i).name for i in range(env.model.ngeom)}
    env.reset(seed=0, start_pose=(1.0, -0.5, 0.0))
    try:
        for k in range(40):
            sb = env.step(Command(pan_target=0.8 * (k > 20)), render=(k % 5 == 4))
            if sb.image is None:
                continue
            assert sb.image.shape == (480, 848, 3)
            assert sb.image_t <= sb.t - p["latency_s"] + 1e-6
            assert sb.image_t >= sb.t - p["latency_s"] - 1.0 / p["fps"] - env.control_dt
    except Exception as e:                                                      # no GL context on this machine
        if "GL" in str(e) or "context" in str(e).lower():
            pytest.skip(f"rendering unavailable: {e}")
        raise
    finally:
        env.close()
