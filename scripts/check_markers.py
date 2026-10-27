"""Step 4: render the head camera, detect ArUco markers, and compare the measurements
(range, bearing, normal direction) against the true values.

    python scripts/make_markers.py        # once
    python scripts/check_markers.py
    python scripts/check_markers.py --set camera=d435i      # the RealSense D435i colour camera profile

Ground truth is read here only to score the measurements (this is an evaluation script).
Outputs: results/figures/marker_check.png (another camera profile: marker_check_<profile>.png)
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from activenav.config import apply_overrides, load_config
from activenav.kinematics import wrap_angle
from activenav.sensing.markers import ArucoMarkerDetector
from activenav.sensing.markers import measure_marker
from activenav.sensing.camera import camera_model
from activenav.sensing.camera import camera_to_base
from activenav.sim.env import DockingEnv
from activenav.types import Command

# (robot pose x, y, yaw), pan, tilt, description
VIEWS = [
    ((3.0, 0.0, 0.0), 0.0, 0.0, "3 m from the east wall, looking east"),
    ((1.0, 0.0, 0.0), 0.0, 0.0, "5 m from the east wall (long range)"),
    ((3.0, 0.0, np.pi / 2), 0.0, 0.0, "facing north wall"),
    ((5.0, 0.0, 0.0), 0.5, 0.0, "near the dock, head panned left"),
    ((3.0, 0.5, 0.0), -0.6, 0.15, "head panned right, tilted down"),
]


def check_config_against_model(cfg, env):
    m = env.model
    h = camera_model(cfg).head                            # the camera profile's lens position
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    pairs = [("pan_origin", m.body("head_pan").pos), ("tilt_offset", m.body("head_tilt").pos),
             ("camera_offset", m.cam_pos[cam])]
    ok = True
    for key, model_val in pairs:
        if not np.allclose(h[key], model_val, atol=1e-6):
            ok = False
            print(f"CONFIG MISMATCH head.{key}: config {h[key]} vs model {list(model_val)}")
    fovy = cfg["camera_model"]["fovy_deg"] if env._color is None else env._color.render_fovy   # a real camera renders wider, then resamples
    if abs(fovy - float(m.cam_fovy[cam])) > 1e-6:
        ok = False
        print(f"CONFIG MISMATCH camera field of view: config {fovy} "
              f"vs model {float(m.cam_fovy[cam])}")
    print("config matches model" if ok else "FIX THE CONFIG MISMATCHES ABOVE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. camera=d435i")
    cfg = apply_overrides(load_config(), ap.parse_args().set)
    env = DockingEnv(cfg)
    check_config_against_model(cfg, env)
    markers = env.marker_map()
    print("markers in scene:", sorted(markers))

    cam = camera_model(cfg)
    det = ArucoMarkerDetector(cam.K, cfg["perception"]["marker_size_m"], dist_coeffs=cam.dist)

    fig, axes = plt.subplots(1, len(VIEWS), figsize=(4.2 * len(VIEWS), 3.6))
    rows, _any_found = [], False
    for ax, (pose, pan, tilt, desc) in zip(np.atleast_1d(axes), VIEWS):
        env.reset(start_pose=pose)
        s = None
        for _ in range(50):                               # let the head settle
            s = env.step(Command(pan_target=pan, tilt_target=tilt))
        s = env.step(Command(pan_target=pan, tilt_target=tilt), render=True)
        img = s.image
        dets = det.detect(img)
        T_B_C = camera_to_base(cam.head, s.view_pan, s.view_tilt)   # measured (noisy) head angles when the image was taken
        gt = env.ground_truth()

        ax.imshow(img)
        ax.set_title(desc, fontsize=8)
        ax.axis("off")
        if not dets:
            flipped = det.detect(cv2.flip(img, 1))
            msg = f"[{desc}] no markers detected"
            if flipped:
                msg += "  -> detected in the MIRRORED image: run  python scripts/make_markers.py --mirror"
            print(msg)
        for d in dets:
            c = d.corners
            ax.plot(*np.vstack([c, c[:1]]).T, "r-", lw=1.5)
            ax.text(c[:, 0].mean(), c[:, 1].mean(), str(d.marker_id), color="yellow", fontsize=10, ha="center")
            z = measure_marker(T_B_C, d)
            if d.marker_id not in markers:
                print(f"detected unknown marker id {d.marker_id}")
                continue
            mk = markers[d.marker_id]
            dx, dy = mk["x"] - gt[0], mk["y"] - gt[1]
            r_true = float(np.hypot(dx, dy))
            b_true = float(wrap_angle(np.arctan2(dy, dx) - gt[2]))
            n_true = float(wrap_angle(mk["normal_angle"] - gt[2]))
            rows.append((desc[:24], d.marker_id, r_true, z.range, b_true, z.bearing, n_true, z.normal_angle))
    env.close()

    fig.tight_layout()
    out = cfg["_root"] / "results" / "figures" / ("marker_check.png" if cam.name == "ideal" else f"marker_check_{cam.name}.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out.relative_to(cfg['_root'])}\n")

    if not rows:
        print("No detections at all: check that the marker PNGs exist and open the saved image.")
        return
    print(f"{'view':24s} {'id':>3s} | {'range true':>10s} {'meas':>6s} {'err':>7s} | "
          f"{'bear true':>9s} {'meas':>6s} {'err':>6s} | {'normal err':>10s}")
    for v, i, rt, rm, bt, bm, nt, nm in rows:
        print(f"{v:24s} {i:3d} | {rt:9.3f}m {rm:6.3f} {rm - rt:+7.3f} | "
              f"{np.degrees(bt):8.1f}° {np.degrees(bm):6.1f} {np.degrees(wrap_angle(bm - bt)):+6.2f} | "
              f"{np.degrees(wrap_angle(nm - nt)):+9.2f}°")
    arr = np.array([(r[3] - r[2], np.degrees(wrap_angle(r[5] - r[4])), np.degrees(wrap_angle(r[7] - r[6])))
                    for r in rows])
    print(f"\n{len(rows)} detections.  mean / std of error:")
    print(f"  range   {arr[:, 0].mean():+.3f} / {arr[:, 0].std():.3f} m")
    print(f"  bearing {arr[:, 1].mean():+.2f} / {arr[:, 1].std():.2f} deg")
    print(f"  normal  {arr[:, 2].mean():+.2f} / {arr[:, 2].std():.2f} deg")


if __name__ == "__main__":
    main()
