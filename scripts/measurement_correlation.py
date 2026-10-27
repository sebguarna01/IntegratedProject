"""Are successive marker measurements from a parked robot independent?

    python scripts/measurement_correlation.py                  # real rendered camera
    python scripts/measurement_correlation.py --sensor synthetic   # sanity check, needs no GL

The EKF fuses a marker measurement every frame (10 Hz). That is only right if each frame's error is
independent. This parks the robot at many poses, takes --frames frames at each, and splits the error of
each quantity into
    within-pose  spread  (changes frame to frame: image noise)
    between-pose spread  (stays the same at a pose: corner quantisation, geometry/model bias)
and reports the intra-class correlation  ICC = between / (between + within).
ICC near 0: frames are independent, the filter is right to fuse all of them.
ICC near 1: frames repeat the same error; N frames are worth ~1 measurement, and a filter that fuses
all of them becomes over-confident (high NEES), most when the head stares at one marker (C-info).
In that case try  --set ekf.min_update_interval_s=0.5  in scripts/ekf_crossing.py and search_study.py.
Truth comes from the simulator (evaluation only). Writes results/measurement_correlation_<sensor>.csv.
"""
import argparse
import csv

import numpy as np

from activenav.config import load_config
from activenav.kinematics import wrap_angle
from activenav.sim.env import DockingEnv
from activenav.types import Command


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["camera", "synthetic"], default="camera")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    env = DockingEnv(cfg)
    markers = env.marker_map()
    x0, x1, y0, y1 = env.room_bounds()
    if args.sensor == "camera":
        from activenav.sensing.markers import MarkerSensor
        sensor = MarkerSensor(cfg)
    else:
        from activenav.evaluation.synthetic import SyntheticMarkerSensor
        sensor = SyntheticMarkerSensor(env, np.random.default_rng(0))

    rows = []          # one row per (pose, frame)
    pose_id = 0
    for mid, m in markers.items():
        for r in (1.0, 2.0, 3.0, 4.0, 4.8):
            for view in np.radians([0.0, 30.0, 50.0, -40.0]):
                ang = m["normal_angle"] + view
                px, py = m["x"] + r * np.cos(ang), m["y"] + r * np.sin(ang)
                if not (x0 + 0.3 < px < x1 - 0.3 and y0 + 0.3 < py < y1 - 0.3):
                    continue
                yaw = float(np.arctan2(m["y"] - py, m["x"] - px))
                env.reset(seed=pose_id, start_pose=(px, py, yaw))
                for _ in range(15):                                       # let the robot and head settle
                    env.step(Command(v=0.0, w=0.0))
                got = []
                for _ in range(args.frames):
                    sb = env.step(Command(v=0.0, w=0.0, pan_target=0.0, tilt_target=0.0), render=sensor.needs_image)
                    for z in sensor.measure(sb):
                        if z.marker_id == mid:
                            tx, ty, tyaw = env.ground_truth()
                            dx, dy = m["x"] - tx, m["y"] - ty
                            got.append((z.range - np.hypot(dx, dy),
                                        float(wrap_angle(z.bearing - (np.arctan2(dy, dx) - tyaw))),
                                        float(wrap_angle(z.normal_angle - (m["normal_angle"] - tyaw))),
                                        float(np.hypot(dx, dy))))
                if len(got) >= args.frames // 2:
                    for g in got:
                        rows.append((pose_id, mid, np.degrees(view), *g))
                    pose_id += 1
    env.close()
    a = np.array(rows)
    out = cfg["_root"] / "results" / f"measurement_correlation_{args.sensor}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pose", "marker", "view_deg", "e_range_m", "e_bearing_rad", "e_normal_rad", "range_m"])
        w.writerows(rows)

    print(f"{pose_id} poses, {len(rows)} frames ({args.sensor}). ICC = share of the error that repeats frame to frame.\n")
    print(f"{'range bin':>10s} {'poses':>6s} | " + " | ".join(f"{q:^28s}" for q in ("range [m]", "bearing [deg]", "normal [deg]")))
    print(f"{'':>10s} {'':>6s} | " + " | ".join(f"{'within':>8s} {'between':>8s} {'ICC':>7s}   " for _ in range(3)))
    scale = [1.0, 180 / np.pi, 180 / np.pi]
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 6.0)]:
        sel = (a[:, 6] >= lo) & (a[:, 6] < hi)
        ids = np.unique(a[sel, 0])
        if len(ids) < 3:
            continue
        cells = []
        for q in range(3):
            means, within = [], []
            for p in ids:
                e = a[sel & (a[:, 0] == p), 3 + q]
                means.append(e.mean())
                within.append(e.var(ddof=1))
            vb, vw = np.var(means, ddof=1), float(np.mean(within))
            icc = vb / (vb + vw) if vb + vw > 0 else float("nan")
            cells.append(f"{np.sqrt(vw) * scale[q]:8.3f} {np.sqrt(vb) * scale[q]:8.3f} {icc:7.2f}   ")
        print(f"{lo:4.1f}-{hi:4.1f} m {len(ids):6d} | " + " | ".join(cells))
    print("\nIf the ICC is high (say above 0.5) for bearing or normal, fusing 10 frames per second counts the same"
          "\nerror ten times. Compare NEES in scripts/search_study.py with and without --set ekf.min_update_interval_s=0.5.")
    print(f"saved {out.relative_to(cfg['_root'])}")


if __name__ == "__main__":
    main()
