"""Step 6: room crossing with EKF localisation (Config B, fixed forward-looking head).

    python scripts/ekf_crossing.py --sensor synthetic [--trials 30]   # fast, no rendering
    python scripts/ekf_crossing.py --sensor camera    [--trials 30]   # real rendered ArUco pipeline

Compares, on the same seeds (so the same hidden encoder errors):
  A      odometry only            (read from results/crossing_baseline.csv, run scripts/odometry_baseline.py crossing first)
  B-3    EKF, state [x, y, theta]
  B-5    EKF with online odometry calibration, state [x, y, theta, c, s]

`synthetic` draws marker measurements from the TRUE pose using the error model fitted to the
characterisation data; it is an optimistic proxy (no occlusion, no missed detections, ideal
visibility). Report final numbers from `--sensor camera`.

Outputs: results/ekf_crossing_<sensor>.csv, results/figures/ekf_crossing_<sensor>.png
"""
import argparse
import copy
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activenav.config import load_config
from activenav.estimation.ekf import EKF
from activenav.evaluation.stats import bootstrap_ci, wilson_interval
from activenav.evaluation.synthetic import SyntheticMarkerSensor
from activenav.evaluation.trials import run_ekf_trial
from activenav.sim.env import DockingEnv

VARIANTS = {"B-3 (EKF)": False, "B-5 (EKF + calibration)": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["synthetic", "camera"], default="synthetic")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    cr = cfg["crossing"]
    n = args.trials or cfg["study"]["trials"]
    goal = np.array(cr["goal"], dtype=float)
    env = DockingEnv(cfg)
    markers = env.marker_map()
    if args.sensor == "camera":
        from activenav.sensing.markers import MarkerSensor
        make_sensor = lambda seed: MarkerSensor(cfg)
    else:
        make_sensor = lambda seed: SyntheticMarkerSensor(env, np.random.default_rng(1000 + seed))

    res = {k: [] for k in VARIANTS}
    for name, bias in VARIANTS.items():
        c = copy.deepcopy(cfg)
        c["ekf"]["estimate_bias"] = bias
        ekf = EKF(c, markers)
        for i in range(n):
            seed = cfg["seed"] + i
            r = run_ekf_trial(env, seed, cr["start_pose"], cr["goal"], make_sensor(seed), ekf,
                              cr["v_max"], cr["w_max"], cr["max_time"], cfg["camera_rate_hz"])
            r["seed"] = seed
            r["miss"] = float(np.hypot(*(r["gt"][-1, :2] - goal)))
            res[name].append(r)
            print(f"{name:24s} seed {seed:3d}: true miss {r['miss']:.3f} m  heading err "
                  f"{np.degrees(r['final_heading_error_rad']):+5.1f} deg  mean NEES {r['nees'].mean():7.1f}  "
                  f"gated {r['n_gated']}/{r['n_seen']}{'' if r['completed'] else '  (TIMED OUT)'}"
                  f"{'  (WALL CONTACT)' if r['collided'] else ''}")
    env.close()

    out_dir = cfg["_root"] / "results"
    A = None
    base = out_dir / "crossing_baseline.csv"
    if base.exists():
        A = np.array([float(row["true_miss_m"]) for row in csv.DictReader(open(base))][:n])

    print(f"\n== Room crossing, {n} trials, sensor = {args.sensor}, dock point {cr['goal']} ==")
    rows = [("A (odometry only)", A)] if A is not None else []
    rows += [(k, np.array([r["miss"] for r in v])) for k, v in res.items()]
    for name, miss in rows:
        lo, hi = bootstrap_ci(miss)
        line = f"{name:24s} median miss {np.median(miss):.3f} m (CI {lo:.3f} to {hi:.3f})  p95 {np.percentile(miss, 95):.3f}  max {miss.max():.3f}"
        for tol in cr["tolerances_m"]:
            k = int((miss <= tol).sum())
            wl, wh = wilson_interval(k, len(miss))
            line += f" | <={tol:.2f} m: {k}/{len(miss)} ({100 * wl:.0f}-{100 * wh:.0f}%)"
        print(line)
    for k, v in res.items():
        nees = np.concatenate([r["nees"] for r in v])
        print(f"{k:24s} NEES median {np.median(nees):.1f}, mean {nees.mean():.1f} (3 dof, consistent filter: mean about 3); "
              f"collisions {sum(r['collided'] for r in v)}/{n}; gated {sum(r['n_gated'] for r in v)}/{sum(r['n_seen'] for r in v)}")

    with open(out_dir / f"ekf_crossing_{args.sensor}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "completed", "duration_s", "true_miss_m", "final_heading_error_deg",
                    "mean_nees", "n_seen", "n_used", "n_gated", "c_est", "s_est", "collided"])
        for k, v in res.items():
            for r in v:
                w.writerow([k, r["seed"], r["completed"], f"{r['duration_s']:.2f}", f"{r['miss']:.4f}",
                            f"{np.degrees(r['final_heading_error_rad']):.3f}", f"{r['nees'].mean():.2f}",
                            r["n_seen"], r["n_used"], r["n_gated"],
                            f"{r['bias_est'][0]:.5f}", f"{r['bias_est'][1]:.5f}", r["collided"]])

    best = res["B-5 (EKF + calibration)"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot([0, 6, 6, 0, 0], [-2, -2, 2, 2, -2], color="0.6", lw=1)
    for mk in markers.values():
        ax[0].plot(mk["x"], mk["y"], "ks", ms=4)
    for r in best:
        ax[0].plot(r["gt"][:, 0], r["gt"][:, 1], color="tab:green", alpha=0.3, lw=1)
    ax[0].plot(*cr["start_pose"][:2], "gs", label="start")
    ax[0].plot(*goal, "r*", ms=14, label="dock point")
    ax[0].set_aspect("equal")
    ax[0].set_title("True paths, EKF + calibration")
    ax[0].legend(loc="upper left")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    for name, miss in rows:
        ax[1].step(np.sort(miss), 100 * np.arange(1, len(miss) + 1) / len(miss), where="post", label=name)
    for tol in cr["tolerances_m"]:
        ax[1].axvline(tol, color="k", ls="--", lw=1)
    ax[1].set_xlabel("true miss distance [m]")
    ax[1].set_ylabel("% of trials within this distance")
    ax[1].set_title("Docking accuracy (cumulative)")
    ax[1].legend(loc="lower right")
    ax[1].grid(True)
    for k, v in res.items():
        tmin = min(r["t"][-1] for r in v)
        grid = np.linspace(0, tmin, 120)
        med = np.median([np.interp(grid, r["t"][1:], r["nees"]) for r in v], axis=0)
        ax[2].semilogy(grid, med, label=k)
    ax[2].axhline(3, color="k", ls="--", lw=1, label="ideal (3 dof)")
    ax[2].set_xlabel("time [s]")
    ax[2].set_ylabel("median NEES")
    ax[2].set_title("Filter consistency (lower is not better: aim for 3)")
    ax[2].legend()
    ax[2].grid(True)
    fig.tight_layout()
    path = out_dir / "figures" / f"ekf_crossing_{args.sensor}.png"
    fig.savefig(path, dpi=140)
    print(f"\nsaved results/ekf_crossing_{args.sensor}.csv and {path.relative_to(cfg['_root'])}")


if __name__ == "__main__":
    main()
