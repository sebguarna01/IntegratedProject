"""Odometry-only baselines (Config A): how far dead reckoning drifts, and what that does to docking.

    python scripts/odometry_baseline.py square   [--laps 2] [--seed 0]   # one run, drift plot
    python scripts/odometry_baseline.py study    [--trials 30] [--laps 4]  # many seeds, closed square
    python scripts/odometry_baseline.py crossing [--trials 30]            # many seeds, cross the room to the dock point

The controller only sees the odometry estimate; ground truth is read by the evaluation code to score drift.
Outputs: results/odometry_baseline.csv, results/crossing_baseline.csv and the matching figures.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activenav.config import load_config
from activenav.control.go_to_point import GoToPointGains, go_to_point
from activenav.evaluation.trials import TrialRecorder
from activenav.evaluation.stats import bootstrap_ci, wilson_interval
from activenav.evaluation.trials import run_odometry_trial
from activenav.sensing.odometry import WheelOdometry
from activenav.sim.env import DockingEnv
from activenav.types import Command


def plot_drift(rec, waypoints, path):
    a = rec.arrays()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(a["gt"][:, 0], a["gt"][:, 1], label="true path")
    ax[0].plot(a["est"][:, 0], a["est"][:, 1], "--", label="odometry estimate")
    wp = np.array(waypoints)
    ax[0].plot(wp[:, 0], wp[:, 1], "ko", label="waypoints")
    ax[0].set_aspect("equal")
    ax[0].set(xlabel="x [m]", ylabel="y [m]", title="Square drive on odometry only")
    ax[0].legend()
    ax[1].plot(a["t"], rec.position_error())
    ax[1].set(xlabel="time [s]", ylabel="position error [m]", title="Odometry drift vs ground truth")
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stats(x):
    x = np.asarray(x, dtype=float)
    return (f"mean {x.mean():.3f}  std {x.std(ddof=1):.3f}  median {np.median(x):.3f}  "
            f"p95 {np.percentile(x, 95):.3f}  max {x.max():.3f}")


def flags(r):
    return f"{'' if r['completed'] else '  (TIMED OUT)'}{'  (WALL CONTACT)' if r['collided'] else ''}"


def square(cfg, args):
    sq = cfg["square"]
    laps = args.laps or sq["laps"]
    seed = cfg["seed"] if args.seed is None else args.seed
    gains = GoToPointGains(v_max=sq["v_max"], w_max=sq["w_max"])
    env = DockingEnv(cfg)
    start = tuple(cfg["start_pose"])
    s = env.reset(seed=seed, start_pose=start)
    odo = WheelOdometry(env.wheel_radius, env.track_width)
    odo.reset(start, s.enc_left, s.enc_right)       # dead reckoning starts from the known start
    rec = TrialRecorder(env)
    rec.log(s.t, odo.pose)

    waypoints = sq["waypoints"] * laps
    idx = 0
    while idx < len(waypoints) and s.t < sq["max_time"]:
        v, w, reached = go_to_point(odo.pose, waypoints[idx], gains)   # estimate only
        if reached:
            idx += 1
            continue
        s = env.step(Command(v=v, w=w))
        odo.update(s.enc_left, s.enc_right)
        rec.log(s.t, odo.pose)
    print(f"{'Completed' if idx >= len(waypoints) else 'TIMED OUT'}: {idx}/{len(waypoints)} waypoints, {s.t:.1f} s")
    for k, v in rec.summary().items():
        print(f"  {k}: {v:.4f}")
    out = cfg["_root"] / "results" / "figures" / "odometry_drift.png"
    plot_drift(rec, sq["waypoints"], out)
    print(f"saved {out.relative_to(cfg['_root'])}")
    env.close()


def study(cfg, args):
    n = args.trials or cfg["study"]["trials"]
    laps = args.laps or cfg["study"]["laps"]
    sq = cfg["square"]
    env = DockingEnv(cfg)
    results = []
    for i in range(n):
        seed = cfg["seed"] + i
        r = run_odometry_trial(env, seed, cfg["start_pose"], sq["waypoints"] * laps, sq["v_max"], sq["w_max"], sq["max_time"])
        r["seed"] = seed
        results.append(r)
        print(f"seed {seed:3d}: path {r['path_length_m']:5.1f} m, final pos err {r['final_position_error_m']:.3f} m, "
              f"heading err {np.degrees(r['final_heading_error_rad']):+6.1f} deg{flags(r)}")
    env.close()

    fin = [r["final_position_error_m"] for r in results]
    head = [abs(np.degrees(r["final_heading_error_rad"])) for r in results]
    pct = [100 * r["final_position_error_m"] / r["path_length_m"] for r in results]
    lo, hi = bootstrap_ci(fin)
    print(f"\n== Odometry-only baseline (closed square): {n} trials, {laps} laps each ==")
    print("completed:", sum(r["completed"] for r in results), "/", n)
    print("trials with wall contact:", sum(r["collided"] for r in results), "/", n)
    print("path length [m]          :", stats([r["path_length_m"] for r in results]))
    print("final position error [m] :", stats(fin))
    print(f"   median 95% bootstrap CI: [{lo:.3f}, {hi:.3f}] m")
    print("final |heading err| [deg]:", stats(head))
    print("position RMSE [m]        :", stats([r["position_rmse_m"] for r in results]))
    print("final error / distance [%]:", stats(pct))

    out_dir = cfg["_root"] / "results"
    with open(out_dir / "odometry_baseline.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seed", "completed", "duration_s", "path_length_m", "final_position_error_m", "final_heading_error_deg",
                     "position_rmse_m", "wheel_scale_left", "wheel_scale_right", "track_error", "collided"])
        for r in results:
            e = r["errors"]
            wr.writerow([r["seed"], r["completed"], f"{r['duration_s']:.2f}", f"{r['path_length_m']:.3f}",
                         f"{r['final_position_error_m']:.4f}", f"{np.degrees(r['final_heading_error_rad']):.3f}",
                         f"{r['position_rmse_m']:.4f}", f"{e['scale_left']:.5f}", f"{e['scale_right']:.5f}",
                         f"{e['track_error']:.5f}", r["collided"]])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    t_max = min(r["t"][-1] for r in results)
    for r in results:
        m = r["t"] <= t_max
        ax[0].plot(r["t"][m], r["pos_err"][m], color="tab:blue", alpha=0.25, lw=1)
    grid = np.linspace(0, t_max, 200)
    ax[0].plot(grid, np.median([np.interp(grid, r["t"], r["pos_err"]) for r in results], axis=0), color="k", lw=2, label="median")
    ax[0].set(xlabel="time [s]", ylabel="position error [m]", title=f"Odometry drift, {n} seeds")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].hist(fin, bins=min(12, max(5, n // 3)), color="tab:blue", alpha=0.8)
    ax[1].axvline(np.median(fin), color="k", ls="--", label=f"median {np.median(fin):.2f} m")
    ax[1].set(xlabel="final position error [m]", ylabel="trials", title="Spread of final error")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "odometry_baseline.png", dpi=150)
    print("\nsaved results/odometry_baseline.csv and results/figures/odometry_baseline.png")


def crossing(cfg, args):
    cr = cfg["crossing"]
    n = args.trials or cfg["study"]["trials"]
    goal = np.array(cr["goal"], dtype=float)
    env = DockingEnv(cfg)
    results = []
    for i in range(n):
        seed = cfg["seed"] + i
        r = run_odometry_trial(env, seed, cr["start_pose"], [cr["goal"]], cr["v_max"], cr["w_max"], cr["max_time"])
        r["seed"] = seed
        r["miss"] = float(np.hypot(*(r["gt"][-1, :2] - goal)))      # ground truth only to score
        results.append(r)
        print(f"seed {seed:3d}: true miss {r['miss']:.3f} m, odometry err {r['final_position_error_m']:.3f} m, "
              f"heading err {np.degrees(r['final_heading_error_rad']):+6.1f} deg{flags(r)}")
    env.close()

    miss = np.array([r["miss"] for r in results])
    lo, hi = bootstrap_ci(miss)
    print(f"\n== Odometry-only room crossing: {n} trials, start {cr['start_pose']}, dock point {cr['goal']} ==")
    print("completed:", sum(r["completed"] for r in results), "/", n)
    print("trials with wall contact:", sum(r["collided"] for r in results), "/", n)
    print(f"true miss distance [m]: mean {miss.mean():.3f}  median {np.median(miss):.3f} "
          f"(95% CI {lo:.3f} to {hi:.3f})  p95 {np.percentile(miss, 95):.3f}  max {miss.max():.3f}")
    for tol in cr["tolerances_m"]:
        k = int((miss <= tol).sum())
        wl, wh = wilson_interval(k, n)
        print(f"within {tol:.2f} m of the dock point: {k}/{n} = {100 * k / n:.0f}%  (Wilson 95% CI {100 * wl:.0f}% to {100 * wh:.0f}%)")

    out_dir = cfg["_root"] / "results"
    with open(out_dir / "crossing_baseline.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seed", "completed", "duration_s", "path_length_m", "true_miss_m", "odometry_error_m",
                     "final_heading_error_deg", "wheel_scale_left", "wheel_scale_right", "track_error", "collided"])
        for r in results:
            e = r["errors"]
            wr.writerow([r["seed"], r["completed"], f"{r['duration_s']:.2f}", f"{r['path_length_m']:.3f}",
                         f"{r['miss']:.4f}", f"{r['final_position_error_m']:.4f}",
                         f"{np.degrees(r['final_heading_error_rad']):.3f}", f"{e['scale_left']:.5f}",
                         f"{e['scale_right']:.5f}", f"{e['track_error']:.5f}", r["collided"]])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot([0, 6, 6, 0, 0], [-2, -2, 2, 2, -2], color="0.6", lw=1)
    for r in results:
        ax[0].plot(r["gt"][:, 0], r["gt"][:, 1], color="tab:blue", alpha=0.3, lw=1)
    ax[0].plot(*cr["start_pose"][:2], "gs", label="start")
    ax[0].plot(*goal, "r*", ms=14, label="dock point")
    ax[0].set_aspect("equal")
    ax[0].set(xlabel="x [m]", ylabel="y [m]", title=f"True paths, odometry-only, {n} seeds")
    ax[0].legend(loc="upper left")
    ax[1].step(np.sort(miss), 100 * np.arange(1, n + 1) / n, where="post")
    for tol in cr["tolerances_m"]:
        ax[1].axvline(tol, color="k", ls="--", lw=1)
    ax[1].set(xlabel="true miss distance [m]", ylabel="% of trials within this distance", title="Docking accuracy (cumulative)")
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "crossing_baseline.png", dpi=150)
    print("\nsaved results/crossing_baseline.csv and results/figures/crossing_baseline.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["square", "study", "crossing"])
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--laps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    {"square": square, "study": study, "crossing": crossing}[args.mode](load_config(), args)


if __name__ == "__main__":
    main()
