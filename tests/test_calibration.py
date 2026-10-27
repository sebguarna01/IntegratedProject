import cv2
import numpy as np

from activenav.sensing.camera import board_points, calibrate, hom, intrinsics_from_fovy, pose_difference


def _views(K, dist, n=15, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    obj = board_points(9, 6, 0.05)
    objs, imgs = [], []
    while len(imgs) < n:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([-0.2, -0.12, 0.0]) + np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.6, 1.4)])
        pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        pts = pts.reshape(-1, 1, 2).astype(np.float32) + rng.normal(0, noise, (len(obj), 1, 2)).astype(np.float32)
        if pts.min() > 5 and pts[:, 0, 0].max() < 635 and pts[:, 0, 1].max() < 475:
            objs.append(obj)
            imgs.append(pts)
    return objs, imgs


def test_board_points_grid():
    g = board_points(9, 6, 0.05)
    assert g.shape == (54, 3) and np.allclose(g[1], [0.05, 0, 0]) and np.allclose(g[9], [0, 0.05, 0]) and np.allclose(g[:, 2], 0)


def test_calibration_recovers_known_intrinsics_and_distortion():
    K = intrinsics_from_fovy(60.0, 640, 480)
    dist = np.array([-0.12, 0.03, 0.0005, -0.0004, 0.0])
    objs, imgs = _views(K, dist)
    c = calibrate(objs, imgs, (640, 480))
    assert abs(c["K"][0, 0] - K[0, 0]) < 2.0 and abs(c["K"][1, 1] - K[1, 1]) < 2.0
    assert abs(c["K"][0, 2] - K[0, 2]) < 2.0 and abs(c["K"][1, 2] - K[1, 2]) < 2.0
    assert abs(c["dist"][0] - dist[0]) < 0.03
    assert c["rms"] < 0.15


def test_pinhole_only_fits_distorted_data_worse_and_clean_data_equally_well():
    K = intrinsics_from_fovy(60.0, 640, 480)
    for dist, worse in ((np.array([-0.2, 0.05, 0, 0, 0.0]), True), (np.zeros(5), False)):
        objs, imgs = _views(K, dist, seed=1)
        full, pin = calibrate(objs, imgs, (640, 480)), calibrate(objs, imgs, (640, 480), pinhole_only=True)
        assert (pin["rms"] > 3 * full["rms"]) == worse


def test_pose_difference():
    T = hom(np.eye(3), [1, 2, 3])
    R = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(10)]))[0]
    d, a = pose_difference(T, hom(R, [1, 2, 3.01]))
    assert abs(d - 0.01) < 1e-9 and abs(a - 10) < 1e-6
