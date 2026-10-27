"""Step 5: characterise the marker measurement error against range and viewing angle.

    python scripts/characterise_markers.py [--samples 300] [--noise 2.0]

Random robot poses, with the head aimed roughly at a random marker. Every detection is
compared to the true range, bearing and marker-normal direction, with and without sub-pixel
corner refinement. This is the evidence for the EKF measurement noise R (Criterion B/C).
Ground truth is read only to score the measurements.

Outputs: results/marker_characterisation.csv, results/figures/marker_characterisation.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activenav.config import load_config
from activenav.kinematics import wrap_angle
from activenav.sensing.markers import ArucoMarkerDetector
from activenav.sensing.markers import measure_marker
from activenav.sensing.camera import intrinsics_from_fovy
from activenav.sensing.camera import camera_to_base
from activenav.sim.env import DockingEnv
from activenav.types import Command

R_BINS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0]
V_BINS = [0, 20, 40, 60, 90]


def table(rows, method, key, edges, unit):
    sel = [r for r in rows if r["method"] == method]
    print(f"\n[{method}] errors binned by {unit}")
    print(f"  {'bin':>12s} {'n':>4s} | {'range bias':>10s} {'std':>6s} {'rel %':>6s} | "
          f"{'bearing std':>11s} | {'normal bias':>11s} {'std':>6s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [r for r in sel if lo <= r[key] < hi]
        if len(b) < 3:
            print(f"  {lo:5.1f}-{hi:<5.1f} {len(b):4d} |  (too few)")
            continue
        er = np.array([r["e_range"] for r in b])
        eb = np.degrees([r["e_bearing"] for r in b])
        en = np.degrees([r["e_normal"] for r in b])
        rel = 100 * np.mean([r["e_range"] / r["r_true"] for r in b])
        print(f"  {lo:5.1f}-{hi:<5.1f} {len(b):4d} | {er.mean():+9.3f}m {er.std():6.3f} {rel:+6.1f} | "
              f"{eb.std():9.2f}° | {en.mean():+10.2f}° {en.std():6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=None, help="override camera_model.image_noise_std")
    args = ap.parse_args()

    cfg = load_config()
    if args.noise is not None:
        cfg["camera_model"]["image_noise_std"] = args.noise
    env = DockingEnv(cfg)
    markers = env.marker_map()
    ids = sorted(markers)
    cam = cfg["sim"]["camera"]
    K = intrinsics_from_fovy(cfg["camera_model"]["fovy_deg"], cam["width"], cam["height"])
    size = cfg["perception"]["marker_size_m"]
    detectors = {"plain": ArucoMarkerDetector(K, size, refine=False),
                 "refined": ArucoMarkerDetector(K, size, refine=True)}
    rng = np.random.default_rng(args.seed)
    print(f"image noise std = {cfg['camera_model']['image_noise_std']} gray levels")

    rows, views_with_detection = [], 0
    for k in range(args.samples):
        x, y, yaw = rng.uniform(0.8, 5.2), rng.uniform(-1.4, 1.4), rng.uniform(-np.pi, np.pi)
        target = markers[ids[rng.integers(len(ids))]]
        b_to_target = wrap_angle(np.arctan2(target["y"] - y, target["x"] - x) - yaw)
        pan = float(np.clip(b_to_target + rng.uniform(-0.4, 0.4), *env.pan_range))
        tilt = float(rng.uniform(0.0, 0.3))

        env.reset(seed=args.seed * 100000 + k, start_pose=(x, y, yaw))
        for _ in range(50):
            env.step(Command(pan_target=pan, tilt_target=tilt))
        s = env.step(Command(pan_target=pan, tilt_target=tilt), render=True)
        T_B_C = camera_to_base(cfg["head"], s.pan, s.tilt)
        gt = env.ground_truth()

        found = False
        for method, det in detectors.items():
            for d in det.detect(s.image):
                if d.marker_id not in markers:
                    continue
                found = True
                z = measure_marker(T_B_C, d)
                mk = markers[d.marker_id]
                dx, dy = mk["x"] - gt[0], mk["y"] - gt[1]
                r_true = float(np.hypot(dx, dy))
                view = abs(wrap_angle(np.arctan2(-dy, -dx) - mk["normal_angle"]))
                edges = np.linalg.norm(np.roll(d.corners, -1, axis=0) - d.corners, axis=1)
                rows.append(dict(
                    method=method, sample=k, id=d.marker_id, r_true=r_true, view_deg=float(np.degrees(view)),
                    px=float(edges.mean()),
                    e_range=z.range - r_true,
                    e_bearing=float(wrap_angle(z.bearing - (np.arctan2(dy, dx) - gt[2]))),
                    e_normal=float(wrap_angle(z.normal_angle - (mk["normal_angle"] - gt[2])))))
        views_with_detection += found
    env.close()

    print(f"{views_with_detection} of {args.samples} views had at least one detection; "
          f"{sum(r['method'] == 'refined' for r in rows)} refined detections")
    out = cfg["_root"] / "results"
    with open(out / "marker_characterisation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for method in ("plain", "refined"):
        table(rows, method, "r_true", R_BINS, "true range [m]")
    table(rows, "refined", "view_deg", V_BINS, "viewing angle [deg] (0 = head-on)")

    ref = [r for r in rows if r["method"] == "refined"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    r_t = np.array([r["r_true"] for r in ref])
    v_t = np.array([r["view_deg"] for r in ref])
    sc = ax[0, 0].scatter(r_t, [r["e_range"] for r in ref], c=v_t, s=12, cmap="viridis")
    ax[0, 0].axhline(0, color="k", lw=0.5)
    ax[0, 0].set_xlabel("true range [m]")
    ax[0, 0].set_ylabel("range error [m]")
    ax[0, 0].set_title("Range error (colour = viewing angle)")
    fig.colorbar(sc, ax=ax[0, 0], label="viewing angle [deg]")
    ax[0, 1].scatter(r_t, np.degrees([r["e_bearing"] for r in ref]), s=12)
    ax[0, 1].set_xlabel("true range [m]")
    ax[0, 1].set_ylabel("bearing error [deg]")
    ax[0, 1].set_title("Bearing error")
    ax[1, 0].scatter(r_t, np.degrees([r["e_normal"] for r in ref]), c=v_t, s=12, cmap="viridis")
    ax[1, 0].set_xlabel("true range [m]")
    ax[1, 0].set_ylabel("marker-normal error [deg]")
    ax[1, 0].set_title("Marker orientation error")
    ax[1, 1].scatter([r["px"] for r in ref], [abs(r["e_range"]) / r["r_true"] * 100 for r in ref], s=12)
    ax[1, 1].set_xlabel("marker size in image [px]")
    ax[1, 1].set_ylabel("|relative range error| [%]")
    ax[1, 1].set_title("Error vs apparent size")
    for a in ax.ravel():
        a.grid(True)
    fig.tight_layout()
    path = out / "figures" / "marker_characterisation.png"
    fig.savefig(path, dpi=140)
    print(f"\nsaved results/marker_characterisation.csv and {path.relative_to(cfg['_root'])}")


if __name__ == "__main__":
    main()
