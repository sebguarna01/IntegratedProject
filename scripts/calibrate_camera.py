"""Calibrate the head camera from rendered checkerboard views, and check the pan-tilt frame chain (Part B of the brief).

    python scripts/calibrate_camera.py [--views 30] [--seed 0]      (needs a GL context: MUJOCO_GL=glfw or egl)
    python scripts/calibrate_camera.py --set camera=d435i           # the RealSense D435i colour camera profile

1. Builds a small scene (floor + one checkerboard plate) around the real robot and renders the real head camera
   (same resolution, same image formation) from random robot poses and pan-tilt angles.
2. Finds the corners, runs cv2.calibrateCamera, and compares the result with what the robot assumes about its camera
   (sensing/camera.py: camera_model; for the ideal camera the analytic matrix from fovy_deg, for a real-camera profile
   its intrinsics and lens distortion). Reports focal lengths, principal point, distortion and reprojection error,
   for the full model and for a pinhole-only model (is distortion worth modelling?).
3. Frame chain: for every view, the camera pose three ways, in the world frame, all compared with MuJoCo's own camera pose:
   (a) the base pose and noisy head encoders pushed through camera_to_base (what the robot computes),
   (b) solvePnP on the detected board with the calibrated camera (what the image says).
Ground truth (true base pose, MuJoCo camera pose) is used only to score: this is an evaluation script.
Outputs: results/camera_calibration.csv, results/figures/camera_calibration.png (another profile: *_<profile>.csv / .png)
"""
import argparse
import copy
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from activenav.config import apply_overrides, load_config
from activenav.sensing.camera import board_points, calibrate, camera_model, camera_to_base, hom, pose_difference
from activenav.sim.env import DockingEnv
from activenav.types import Command

COLS, ROWS, SQUARE = 9, 6, 0.05                 # inner corners and square size [m]; the board has one extra square of white border
BOARD_C = np.array([3.0, 0.0, 0.30])            # plate centre, facing -x (towards the robot)
X_P, Y_P, Z_P = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([-1.0, 0.0, 0.0])   # plate axes (same convention as the marker plates)


def build_scene(cfg):
    root = cfg["_root"]
    px = 60
    img = np.full(((ROWS + 3) * px, (COLS + 3) * px), 255, np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                img[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
    # (COLS+1) x (ROWS+1) squares give COLS x ROWS inner corners; one white square of border on every side
    cv2.imwrite(str(root / "models" / "assets" / "_generated_calibration_board.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    hx, hy = (COLS + 3) * SQUARE / 2, (ROWS + 3) * SQUARE / 2
    assets = ('<texture name="board" type="2d" file="_generated_calibration_board.png"/>\n'
              '<material name="mat_board" texture="board" rgba="1 1 1 1" specular="0" shininess="0" reflectance="0"/>')
    world = ('<light pos="2 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>\n'
             '<geom name="floor" type="plane" pos="3 0 0" size="6 4 0.1" material="floor"/>\n'
             f'<geom name="board" type="box" size="{hx:g} {hy:g} 0.002" pos="{BOARD_C[0]:g} {BOARD_C[1]:g} {BOARD_C[2]:g}" '
             f'xyaxes="{X_P[0]:g} {X_P[1]:g} {X_P[2]:g} {Y_P[0]:g} {Y_P[1]:g} {Y_P[2]:g}" material="mat_board" contype="0" conaffinity="0"/>')
    xml = (root / "models" / "robot_template.xml").read_text().replace("<!--@SCENE_ASSETS@-->", assets).replace("<!--@SCENE_WORLD@-->", world)
    path = root / "models" / "_generated_calibration.xml"
    path.write_text(xml)
    return path


def T_world_board():
    """Board frame (x along columns, y along rows, origin at the first inner corner) in the world."""
    R_wp = np.column_stack([X_P, Y_P, Z_P])
    T_wp = hom(R_wp, BOARD_C)
    T_pb = hom(np.diag([1.0, -1.0, -1.0]), [-(COLS - 1) / 2 * SQUARE, (ROWS - 1) / 2 * SQUARE, 0.0])
    return T_wp @ T_pb


def true_camera_pose(env):
    """MuJoCo's own pose of head_cam as a 4x4 in the OpenCV camera convention (x right, y down, z forward)."""
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    R = env.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return hom(R, env.data.cam_xpos[cid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. camera=d435i")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(), args.set)
    cam = camera_model(cfg)
    suffix = "" if cam.name == "ideal" else f"_{cam.name}"
    c = copy.deepcopy(cfg)
    c["world"] = None
    c["sim"]["model"] = str(build_scene(cfg).relative_to(cfg["_root"]))
    env = DockingEnv(c)
    W, H = cam.width, cam.height
    rng = np.random.default_rng(args.seed)
    obj = board_points(COLS, ROWS, SQUARE)
    T_wb = T_world_board()

    obj_pts, img_pts, _chain_err, meta = [], [], [], []
    tries = 0
    while len(img_pts) < args.views and tries < 6 * args.views:
        tries += 1
        pose = (rng.uniform(1.2, 2.3), rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6))
        pan, tilt = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.4)
        s = env.reset(seed=int(rng.integers(1 << 30)), start_pose=pose)
        for _ in range(150):                                   # let the head settle on its targets
            s = env.step(Command(pan_target=pan, tilt_target=tilt))
        if cam.name == "ideal":
            img = env.render()
        else:                                                  # an image as the robot receives it (frame timing, encoders)
            s = env.step(Command(pan_target=pan, tilt_target=tilt), render=True)
            img = s.image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        corners = np.ascontiguousarray(corners, np.float32).reshape(-1, 1, 2)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
        if corners.min() < 8 or corners[:, 0, 0].max() > W - 8 or corners[:, 0, 1].max() > H - 8:
            continue
        obj_pts.append(obj)
        img_pts.append(corners)
        meta.append((env.ground_truth().copy(), s.view_pan, s.view_tilt, true_camera_pose(env)))
    n = len(img_pts)
    print(f"{n} views with all {COLS * ROWS} corners found ({tries} tried)")
    if n < 10:
        raise SystemExit("too few usable views: check the scene or widen the pose ranges")

    full = calibrate(obj_pts, img_pts, (W, H))
    pin = calibrate(obj_pts, img_pts, (W, H), pinhole_only=True)
    K0, d0 = cam.K, (np.zeros(5) if cam.dist is None else cam.dist)
    names = ["fx", "fy", "cx", "cy"]
    got = [full["K"][0, 0], full["K"][1, 1], full["K"][0, 2], full["K"][1, 2]]
    ref = [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]
    print(f"camera profile: {cam.name} ({W} x {H})")
    print(f"\n{'':8s}{'assumed':>16s}{'calibrated':>14s}{'difference':>12s}")
    for nm, a, b in zip(names, ref, got):
        print(f"{nm:8s}{a:16.2f}{b:14.2f}{b - a:+12.2f}  px")
    print(f"distortion (k1 k2 p1 p2 k3): assumed {np.array2string(d0, precision=4, suppress_small=True)}, "
          f"calibrated {np.array2string(full['dist'], precision=4, suppress_small=True)}")
    # what the differences mean in pixels: where the assumed camera puts the rays the calibrated one gives each pixel
    grid = np.stack(np.meshgrid(np.linspace(0, W - 1, 17), np.linspace(0, H - 1, 11)), -1).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(grid, full["K"], full["dist"]).reshape(-1, 2)
    back, _ = cv2.projectPoints(np.c_[rays, np.ones(len(rays))].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K0, d0)
    gap = np.linalg.norm(back.reshape(-1, 2) - grid.reshape(-1, 2), axis=1)
    print(f"assumed vs calibrated camera over the image: median {np.median(gap):.2f} px, max {gap.max():.2f} px")
    print(f"reprojection RMS: full model {full['rms']:.3f} px, pinhole only {pin['rms']:.3f} px "
          f"(worst view {full['per_image_rms'].max():.3f} px)")
    hfov = np.degrees(2 * np.arctan(W / (2 * full["K"][0, 0])))
    print(f"horizontal field of view from the calibration: {hfov:.2f} deg; vertical {np.degrees(2 * np.arctan(H / (2 * full['K'][1, 1]))):.2f} deg "
          f"(assumed {np.degrees(cam.hfov):.2f} x {np.degrees(cam.vfov):.2f} deg)")

    e_chain, e_pnp = [], []
    for (gt, pan, tilt, T_true), o, i in zip(meta, obj_pts, img_pts):
        # chain pose from the true base pose of THIS view (stored in gt) and the noisy encoders the robot read
        cth, sth = np.cos(gt[2]), np.sin(gt[2])
        T_wbase = hom(np.array([[cth, -sth, 0], [sth, cth, 0], [0, 0, 1.0]]), [gt[0], gt[1], cam.head["base_height"]])
        T_chain = T_wbase @ camera_to_base(cam.head, pan, tilt)
        okp, rv, tv = cv2.solvePnP(o, i, full["K"], full["dist"], flags=cv2.SOLVEPNP_ITERATIVE)
        T_cb = hom(cv2.Rodrigues(rv)[0], tv.ravel())
        T_pnp = T_wb @ np.linalg.inv(T_cb)
        e_chain.append(pose_difference(T_true, T_chain))
        e_pnp.append(pose_difference(T_true, T_pnp))
    e_chain, e_pnp = np.array(e_chain), np.array(e_pnp)
    print("\nCamera pose in the world vs MuJoCo's own camera pose:")
    print(f"  base pose + head encoders through camera_to_base: translation median {1e3 * np.median(e_chain[:, 0]):.1f} mm "
          f"(max {1e3 * e_chain[:, 0].max():.1f}), rotation median {np.median(e_chain[:, 1]):.2f} deg (max {e_chain[:, 1].max():.2f})")
    print(f"  checkerboard PnP with the calibrated camera:      translation median {1e3 * np.median(e_pnp[:, 0]):.1f} mm "
          f"(max {1e3 * e_pnp[:, 0].max():.1f}), rotation median {np.median(e_pnp[:, 1]):.2f} deg (max {e_pnp[:, 1].max():.2f})")
    print("(encoder noise 0.002 rad per joint is about 0.1 deg per joint; the PnP numbers include the corner noise from the image noise)")

    out = cfg["_root"] / "results"
    with open(out / f"camera_calibration{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "analytic" if cam.name == "ideal" else "assumed", "calibrated_full", "calibrated_pinhole_only"])
        for nm, a, b, p in zip(names, ref, got, [pin["K"][0, 0], pin["K"][1, 1], pin["K"][0, 2], pin["K"][1, 2]]):
            w.writerow([nm, f"{a:.3f}", f"{b:.3f}", f"{p:.3f}"])
        for nm, a, v in zip(["k1", "k2", "p1", "p2", "k3"], d0, full["dist"]):
            w.writerow([nm, 0 if cam.dist is None else f"{a:.6f}", f"{v:.6f}", 0])
        w.writerow(["reprojection_rms_px", "", f"{full['rms']:.4f}", f"{pin['rms']:.4f}"])
        w.writerow(["views", "", n, n])
        w.writerow(["chain_translation_median_mm", "", f"{1e3 * np.median(e_chain[:, 0]):.2f}", ""])
        w.writerow(["chain_rotation_median_deg", "", f"{np.median(e_chain[:, 1]):.3f}", ""])
        w.writerow(["pnp_translation_median_mm", "", f"{1e3 * np.median(e_pnp[:, 0]):.2f}", ""])
        w.writerow(["pnp_rotation_median_deg", "", f"{np.median(e_pnp[:, 1]):.3f}", ""])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    allc = np.concatenate([c.reshape(-1, 2) for c in img_pts])
    ax[0].plot(allc[:, 0], allc[:, 1], ".", ms=2, alpha=0.5)
    ax[0].set(xlim=(0, W), ylim=(H, 0), aspect="equal", title=f"Detected corners, {n} views", xlabel="u [px]", ylabel="v [px]")
    ax[1].bar(range(n), full["per_image_rms"])
    ax[1].axhline(full["rms"], color="k", ls="--", label=f"overall {full['rms']:.2f} px")
    ax[1].set(xlabel="view", ylabel="reprojection RMS [px]", title="Reprojection error per view")
    ax[1].legend()
    ax[2].scatter(1e3 * e_chain[:, 0], e_chain[:, 1], label="head chain (encoders)")
    ax[2].scatter(1e3 * e_pnp[:, 0], e_pnp[:, 1], label="board PnP", marker="x")
    ax[2].set(xlabel="camera position error [mm]", ylabel="camera rotation error [deg]", title="Camera pose vs MuJoCo")
    ax[2].legend()
    ax[2].grid(True)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"camera_calibration{suffix}.png", dpi=150)
    print(f"\nsaved results/camera_calibration{suffix}.csv and results/figures/camera_calibration{suffix}.png")
    env.close()


if __name__ == "__main__":
    main()
