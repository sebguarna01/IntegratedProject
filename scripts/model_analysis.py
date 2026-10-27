"""Controllability and observability of the platform and the estimator (Part A of the brief).

    python scripts/model_analysis.py

Prints the tables and writes results/model_analysis.csv and results/figures/model_analysis.png.
The maths lives in src/activenav/analysis.py and uses the filters' own Jacobians, so it describes the code that runs.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from activenav import analysis as an
from activenav.config import load_config

MARKERS = {1: dict(x=4.0, y=1.0, normal_angle=np.pi), 2: dict(x=2.0, y=-1.5, normal_angle=np.pi / 2)}
LANDMARKS = [(4.0, 1.0, np.pi), (2.0, -1.5, np.pi / 2)]
SETS = [("none (odometry only)", ()), ("bearing", (1,)), ("range + bearing", (0, 1)), ("range + bearing + normal", (0, 1, 2))]


def main():
    cfg = load_config()
    Ts = cfg["sim"]["control_dt"]
    rows = []

    print("== Controllability ==")
    for v in (0.0, 0.05, 0.2):                               # pose model about a straight run at heading 0.7 rad
        A, B = an.unicycle_linearisation(v, 0.7)
        r, s = an.rank_and_sv(an.ctrb(A, B))
        print(f"pose model [x y th], inputs [v w], speed {v:4.2f} m/s: rank {r}/3")
        rows.append(["controllability", f"pose model, v={v}", r, 3, s.min(), "rank 3 only while moving: a robot at rest cannot step sideways"])
    for v, kappa in [(0.0, 0.0), (0.05, 0.0), (0.2, 0.0), (0.2, 0.5)]:     # path-error model, e = [e_y, e_th]
        A, B = an.error_model(v, kappa)
        Ad, Bd = an.discretise(A, B, Ts)
        r, s = an.rank_and_sv(an.ctrb(A, B))
        rd, sd = an.rank_and_sv(an.ctrb(Ad, Bd))
        print(f"path-error model, v={v:4.2f} m/s, kappa={kappa:3.1f}/m: det C = {np.linalg.det(an.ctrb(A, B)):+.3f}, "
              f"rank {r}/2 (continuous), {rd}/2 (ZOH at {Ts} s, smallest singular value {sd.min():.2e})")
        rows.append(["controllability", f"path-error, v={v}, kappa={kappa}", rd, 2, sd.min(), "ZOH discretisation"])

    print("\n== Observability of the pose (known-map EKF, markers always in view, 3 m with a turn) ==")
    print(f"{'measurement':26s} {'markers':>7s}  pose only  with bias states c, s, k  (rank / states, smallest scaled singular value)")
    for name, rws in SETS:
        for k, mk in ((1, {1: MARKERS[1]}), (2, MARKERS)):
            cells = []
            for bias in (False, True):
                r, s, n = an.pose_observability(cfg, mk, rws, bias)
                cells.append(f"{r}/{n}  ({s.min():.1e})")
                rows.append(["observability", f"{name}, {k} marker(s), {'bias' if bias else 'pose'}", r, n, s.min(), "known map"])
            print(f"{name:26s} {k:7d}  {cells[0]:18s} {cells[1]}")

    print("\n== EKF-SLAM: robot + landmarks, with and without the start-pose anchor ==")
    for k in (1, 2):
        for anchored in (False, True):
            r, d = an.slam_observability(cfg, LANDMARKS[:k], anchored, bias=True)
            print(f"{k} landmark(s), start pose {'anchored' if anchored else 'free    '}: rank {r}/{d}  (unobservable directions: {d - r})")
            rows.append(["slam gauge", f"{k} landmarks, {'anchored' if anchored else 'free'}", r, d, float(d - r),
                         "gauge = global translation and rotation"])

    sig = an.blind_growth(cfg, 5.0)
    print(f"\n== Marker out of view: 1-sigma after 5 m with no measurement: x {sig[0]:.3f} m, y {sig[1]:.3f} m, th {np.degrees(sig[2]):.1f} deg ==")
    rows.append(["blind growth", "5 m straight, no marker", 0, 3, float(np.hypot(sig[0], sig[1])), "1-sigma position [m]"])

    with open(cfg["_root"] / "results" / "model_analysis.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "case", "rank", "dim", "min_singular_value_or_gap", "note"])
        w.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    vs = np.linspace(0.0, 0.3, 61)
    ax[0].plot(vs, [abs(np.linalg.det(an.ctrb(*an.error_model(v)))) for v in vs], lw=2)
    ax[0].set(xlabel="forward speed v [m/s]", ylabel="|det C|  (= v)", title="Path-error model: controllability fades to zero at rest")
    ax[0].grid(True)
    labels, vals = [], []
    for name, rws in SETS[1:]:
        for k, mk in ((1, {1: MARKERS[1]}), (2, MARKERS)):
            _, s, _ = an.pose_observability(cfg, mk, rws, True)
            labels.append(f"{name.replace('range + bearing', 'r+b')}, {k}")
            vals.append(max(s.min(), 1e-17))
    ax[1].barh(labels, vals)
    ax[1].set_xscale("log")
    ax[1].axvline(1e-9, color="r", ls="--", lw=1, label="rank tolerance")
    ax[1].legend(loc="lower right")
    ax[1].set(xlabel="smallest scaled singular value (6 states)", title="Observability with bias states")
    ax[1].grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(cfg["_root"] / "results" / "figures" / "model_analysis.png", dpi=150)
    print("\nsaved results/model_analysis.csv and results/figures/model_analysis.png")


if __name__ == "__main__":
    main()
