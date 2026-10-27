"""Path tracking with the EKF estimate in the loop: discrete LQR (integral action) against the go-to-point law the agent uses.

    python scripts/lqr_tracking.py [--trials 12] [--laps 3] [--sensor synthetic|camera]

A counter-clockwise stadium loop (two 2 m straights, two semicircles of radius 1 m) in the 6 x 4 m room, driven at 0.2 m/s.
The robot localises with the known-map EKF (odometry + ArUco markers, head sweeping) and the CONTROLLER SEES ONLY THE
ESTIMATE. Every trial uses the same seed for both controllers, so they meet the same hidden wheel errors and the same
sensor noise. Metrics are measured on the last lap against TRUE pose (ground truth is used only to score):
    rms / mean true cross-track error (tracking and steady offset), RMS feedback steering effort w - v kappa (what the
    controller asks of the wheels beyond the path itself), NEES (is the filter honest while being steered?).
Reference limits taken from the course's path-following specification: RMS cross-track <= 0.08 m, |mean| <= 0.05 m,
RMS effort <= 0.30 rad/s, NEES inside the 95% band >= 90%.
Outputs: results/lqr_tracking.csv, results/figures/lqr_tracking.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activenav.config import load_config
from activenav.control.go_to_point import GoToPointGains, go_to_point
from activenav.control.lqr import LqrTracker, Path, stadium
from activenav.estimation.ekf import EKF
from activenav.evaluation.stats import bootstrap_ci
from activenav.evaluation.synthetic import SyntheticMarkerSensor
from activenav.kinematics import wrap_angle
from activenav.sensing.odometry import WheelOdometry
from activenav.sim.env import DockingEnv
from activenav.types import Command

V_REF = 0.2
CONTROLLERS = ["go_to_point", "lqr", "lqr_no_integral"]


class WaypointFollower:
    """The agent's `_follow`: go_to_point towards the next waypoint, popped within 0.15 m (the via tolerance)."""

    def __init__(self, path: Path, spacing: float = 0.4, w_max: float = 1.0):
        step = max(1, int(round(spacing / (path.s[1] - path.s[0]))))
        self.wps = [tuple(p) for p in path.p[::step]]
        self.gains = GoToPointGains(v_max=V_REF, w_max=w_max, pos_tol=0.15)
        self.path, self.i = path, 0
        self.dw = 0.0
        self.k = 0

    def reset(self):
        self.i = 0

    def command(self, pose):
        for _ in range(len(self.wps)):
            v, w, reached = go_to_point(pose, self.wps[self.i % len(self.wps)], self.gains)
            if not reached:
                break
            self.i += 1
        self.k = self.path.nearest(pose, self.k)
        self.dw = w - v * self.path.kappa[self.k]          # bookkeeping only: the effort beyond what the path needs
        return v, w


def true_offset(path, xy_theta, hint):
    i = path.nearest(xy_theta, hint, window=200)
    ey, eth = path.errors(xy_theta, i)
    return i, ey, eth


def run(cfg, path, name, seed, laps, sensor_kind, yaw_bias=0.0):
    env = DockingEnv(cfg)
    ctrl = {"go_to_point": lambda: WaypointFollower(path), "lqr": lambda: LqrTracker(path, V_REF, integral=True),
            "lqr_no_integral": lambda: LqrTracker(path, V_REF, integral=False)}[name]()
    start = (*path.p[0], path.psi[0])
    s = env.reset(seed=seed, start_pose=start)
    odo = WheelOdometry(env.wheel_radius, env.track_width)
    odo.reset(start, s.enc_left, s.enc_right)
    ekf = EKF(cfg, env.marker_map())
    ekf.reset(np.array(start))
    if sensor_kind == "camera":
        from activenav.sensing.markers import MarkerSensor
        sensor = MarkerSensor(cfg)
    else:
        sensor = SyntheticMarkerSensor(env, np.random.default_rng(1000 + seed))
    cam_dt, next_cam = 1.0 / cfg["camera_rate_hz"], 0.0
    max_t = 1.6 * laps * path.length / V_REF
    log = []
    it, progress, prev_i = None, 0.0, 0
    while s.t < max_t and progress < laps * path.length:
        v, w = ctrl.command(ekf.pose)
        pan = 1.2 * np.sin(2 * np.pi * s.t / 8.0)
        due = s.t >= next_cam
        s = env.step(Command(v=v, w=w + yaw_bias, pan_target=pan), render=bool(due and sensor.needs_image))
        odo.update(s.enc_left, s.enc_right)
        ekf.predict(odo.last_ds, odo.last_dtheta)
        if due:
            next_cam = s.t + cam_dt
            for z in sensor.measure(s):
                ekf.update_marker(z, t=s.t)
        gt = env.ground_truth()
        it, ey, eth = true_offset(path, gt, it)
        step_i = (it - prev_i + len(path) // 2) % len(path) - len(path) // 2              # progress along the loop, signed
        progress += step_i * (path.length / len(path))
        prev_i = it
        e = gt - ekf.pose
        e[2] = wrap_angle(e[2])
        nees = float(e @ np.linalg.solve(ekf.P_pose, e))
        log.append((s.t, progress, ey, eth, ctrl.dw, w, nees, np.hypot(*e[:2])))
    env.close()
    L = np.array(log)
    done = progress >= laps * path.length
    last = L[L[:, 1] >= (laps - 1) * path.length]
    if not done or len(last) < 50:
        return dict(done=False)
    ey, dw, nees = last[:, 2], last[:, 4], last[:, 6]
    return dict(done=True, rms_ey=float(np.sqrt(np.mean(ey ** 2))), mean_ey=float(ey.mean()), max_ey=float(np.abs(ey).max()),
                rms_effort=float(np.sqrt(np.mean(dw ** 2))), sat=float(np.mean(np.abs(last[:, 5]) >= 0.999)),
                nees_one_sided=float(np.mean(nees <= 7.81)), nees_band=float(np.mean((nees > 0.216) & (nees < 9.348))),
                est_rmse=float(np.sqrt(np.mean(last[:, 7] ** 2))), lap_time=float(last[-1, 0] - last[0, 0]), trace=L)


def table(res, args, bias):
    keys = [("rms_ey", "RMS cross-track error [m]", 0.08), ("mean_ey", "mean cross-track offset [m]", 0.05),
            ("rms_effort", "RMS steering effort w - v kappa [rad/s]", 0.30), ("nees_one_sided", "NEES <= 7.81 [share of samples]", 0.90),
            ("nees_band", "NEES in 95% two-sided band", None), ("est_rmse", "estimate position RMSE [m]", None),
            ("max_ey", "worst cross-track error [m]", None), ("sat", "share of time w saturated", None)]
    print(f"\n== yaw bias {bias:+.2f} rad/s (unmodelled, added to the commanded w); last of {args.laps} laps, {args.trials} seeds, "
          f"{args.sensor} sensor; median [min, max] ==")
    print(f"{'':44s}" + "".join(f"{c:>26s}" for c in args.controllers) + "   limit")
    for k, label, lim in keys:
        cells = []
        for c in args.controllers:
            v = np.array([r[k] for r in res[(bias, c)] if r["done"]])
            cells.append(f"{np.median(v):10.4f} [{v.min():.3f},{v.max():.3f}]" if len(v) else "n/a")
        print(f"{label:44s}" + "".join(f"{x:>26s}" for x in cells) + (f"   {'>=' if k.startswith('nees_one') else '<='} {lim}" if lim else ""))
    print("finished:", {c: f"{sum(r['done'] for r in res[(bias, c)])}/{args.trials}" for c in args.controllers})
    if "go_to_point" in args.controllers and "lqr" in args.controllers:
        for k, label, _ in keys[:3]:
            d = np.array([abs(a[k]) - abs(b[k]) for a, b in zip(res[(bias, "go_to_point")], res[(bias, "lqr")]) if a["done"] and b["done"]])
            if len(d) >= 3:
                lo, hi = bootstrap_ci(d)
                print(f"paired |{label}|, go_to_point - lqr: median {np.median(d):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] ({len(d)} pairs)")
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--sensor", choices=["synthetic", "camera"], default="synthetic")
    ap.add_argument("--controllers", nargs="+", default=CONTROLLERS, choices=CONTROLLERS)
    ap.add_argument("--yaw-bias", nargs="+", type=float, default=[0.0, 0.1], metavar="RAD_S",
                    help="unmodelled yaw-rate disturbance added to the command, one condition per value (a worn wheel)")
    args = ap.parse_args()
    cfg = load_config()
    path = stadium()
    res = {(b, c): [] for b in args.yaw_bias for c in args.controllers}
    for b in args.yaw_bias:
        for i in range(args.trials):
            seed = cfg["seed"] + i
            line = f"bias {b:+.2f} seed {seed:3d}: "
            for c in args.controllers:
                r = run(cfg, path, c, seed, args.laps, args.sensor, b)
                res[(b, c)].append(r)
                line += f"{c} " + (f"rms {r['rms_ey'] * 1e3:5.1f} mm  " if r["done"] else "DID NOT FINISH  ")
            print(line)
    for b in args.yaw_bias:
        keys = table(res, args, b)

    out = cfg["_root"] / "results"
    with open(out / "lqr_tracking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["yaw_bias_rad_s", "controller", "seed"] + [k for k, _, _ in keys] + ["lap_time_s", "done"])
        for (b, c), rs in res.items():
            for i, r in enumerate(rs):
                w.writerow([b, c, cfg["seed"] + i] + ([f"{r[k]:.5f}" for k, _, _ in keys] + [f"{r['lap_time']:.1f}"] if r["done"] else [""] * (len(keys) + 1)) + [int(r["done"])])

    nb = len(args.yaw_bias)
    fig, ax = plt.subplots(nb, 3, figsize=(15, 4.2 * nb), squeeze=False)
    col = dict(zip(CONTROLLERS, ["tab:gray", "tab:blue", "tab:orange"]))
    for row, b in enumerate(args.yaw_bias):
        for c in args.controllers:
            ok = [r for r in res[(b, c)] if r["done"]]
            if ok:
                tr = ok[0]["trace"]
                m = tr[:, 1] >= (args.laps - 1) * path.length
                ax[row, 0].plot(tr[m, 1] - (args.laps - 1) * path.length, 1e3 * tr[m, 2], color=col[c], lw=1, label=c)
        ax[row, 0].axhline(0, color="k", lw=0.5)
        ax[row, 0].set(xlabel="distance along the last lap [m]", ylabel="true cross-track error [mm]", title=f"One trial, yaw bias {b:+.2f} rad/s")
        ax[row, 0].legend()
        data = [[1e3 * r["rms_ey"] for r in res[(b, c)] if r["done"]] for c in args.controllers]
        ax[row, 1].boxplot(data, tick_labels=args.controllers, showfliers=True)
        ax[row, 1].set(ylabel="RMS true cross-track error [mm]", title="Tracking over seeds (spec 80 mm)")
        data = [[r["rms_effort"] for r in res[(b, c)] if r["done"]] for c in args.controllers]
        ax[row, 2].boxplot(data, tick_labels=args.controllers, showfliers=True)
        ax[row, 2].set(ylabel="RMS steering effort w - v kappa [rad/s]", title="Effort over seeds (spec 0.30)")
        for a in ax[row, 1:]:
            a.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out / "figures" / "lqr_tracking.png", dpi=150)
    print("\nsaved results/lqr_tracking.csv and results/figures/lqr_tracking.png")


if __name__ == "__main__":
    main()
