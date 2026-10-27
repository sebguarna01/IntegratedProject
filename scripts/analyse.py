"""All results of the study in one place: every table (results/final_tables.md) and figure (results/figures/final/) is rebuilt
from the CSVs under results/, nothing is typed in by hand.

    python scripts/analyse.py            # everything that has data (a missing block is skipped)
    python scripts/analyse.py --cap 300  # common time cap in seconds (default 300): a longer run counts as a timeout

Study folders (written by scripts/run_studies.sh):
  results/                main block, 12 rooms, seeds 0..19
  results/heldout/        held-out seeds 5000..5019 (rooms and dock heights): never used while tuning
  results/height/         dock-height study, large_6_pillars   results/height_rooms/  other rooms   results/dock02/  fresh 0.2 m seeds
  results/tilt_free/      B with its tilt free (two preset tilts)    results/lqr/  the LQR path follower on the same seeds
  results/sens_*/         estimator noise Q, R scaled by 0.5 and 2   results/gaze/  head-behaviour columns (same seeds as the main block)
  results/mixed/          every marker at a random height            results/tall_hard/  tall docks in the hardest rooms

Statistics: Wilson intervals for proportions; paired differences (same room, layout and seed under two configs) with a bootstrap over
seeds (evaluation/paired.py); exact McNemar for paired success.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from activenav.evaluation import paired                                  # noqa: E402
from activenav.evaluation.stats import wilson_interval                   # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures" / "final"
CFGS = ["B", "C-sweep", "C-info"]
COL = {"B": "#2a78d6", "C-sweep": "#eb6a34", "C-info": "#1bb07a", "B, tilt free": "#7fa6dc"}
OUTCOMES = ["docked", "false dock", "contact", "timeout", "gave up"]
OUT_COL = {"docked": "#1b8a6b", "false dock": "#d9a400", "contact": "#c23b3b", "timeout": "#8c8c88", "gave up": "#c9c7c0"}
CAP = 300.0
ROOMS = ["alcove", "cluttered_14", "corridor", "l_room", "large_6_pillars", "large_9_pillars", "large_room",
         "people_constant", "people_random", "sparse_noisy_odom", "two_rooms", "zigzag"]


# ------------------------------------------------------------------------------------------------ loading
def num(r, k):
    try:
        return float(r.get(k, "nan") or "nan")
    except (ValueError, TypeError):
        return float("nan")


def load(folder: str) -> list[dict]:
    """Camera study rows of one results folder ('.' = results/ itself). A run longer than the cap counts as a timeout."""
    d = RES if folder == "." else RES / folder
    rows = []
    for p in sorted(d.glob("search_study_camera*.csv")):
        for r in csv.DictReader(open(p, newline="")):
            r["seed"] = int(r["seed"])
            r["z"] = num(r, "dock_height_m")
            if r["docked"] == "1" and float(r["duration_s"]) > CAP or (r["outcome"] == "docked" and float(r["duration_s"]) > CAP):
                r["docked"], r["outcome"] = "0", "timeout"
            r["ok"] = r["docked"] == "1"
            r["block"] = folder
            rows.append(r)
    return rows


def pick(rows, config=None, level=None, z=None, layout=None):
    out = rows
    if config is not None:
        out = [r for r in out if r["config"] == config]
    if level is not None:
        out = [r for r in out if r["level"] == level]
    if z is not None:
        out = [r for r in out if abs(r["z"] - z) < 1e-6]
    if layout is not None:
        out = [r for r in out if r["layout"] == layout]
    return out


def vals(rows, key, scale=1.0):
    v = np.array([num(r, key) for r in rows]) * scale
    return v[np.isfinite(v)]


def med(rows, key, scale=1.0):
    v = vals(rows, key, scale)
    return float(np.median(v)) if len(v) else float("nan")


def frac(rows) -> str:
    n = len(rows)
    return "-" if n == 0 else f"{sum(r['ok'] for r in rows)}/{n} ({100 * sum(r['ok'] for r in rows) / n:.0f}%)"


def pct(rows):
    """(percent docked, lower, upper) with a Wilson 95% interval."""
    n, k = len(rows), sum(r["ok"] for r in rows)
    lo, hi = wilson_interval(k, n)
    return 100 * k / n, 100 * lo, 100 * hi


def md_table(head, body):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in row) + " |" for row in body])


def cmp_row(rows, a, b, metric):
    """paired.compare for one metric (seed-cluster bootstrap): dict with diff, lo, hi, n, wins, losses, p_mcnemar."""
    return next(m for m in paired.compare(rows, a, b) if m["metric"] == metric)


def fmt_ci(m, scale=1.0, nd=1):
    if m["n"] == 0 or m["diff"] != m["diff"]:
        return "-"
    return f"{m['diff'] * scale:+.{nd}f} [{m['lo'] * scale:+.{nd}f}, {m['hi'] * scale:+.{nd}f}]"


def style(ax, grid="y"):
    ax.grid(alpha=0.25, axis=grid)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    global CAP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=float, default=300.0)
    CAP = ap.parse_args().cap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    FIG.mkdir(parents=True, exist_ok=True)
    out = ["# Final study tables (generated by scripts/analyse.py)\n", f"Common cap {CAP:.0f} s: a longer run counts as a timeout. "
           "Intervals: Wilson (proportions), bootstrap over seeds (paired differences).\n"]

    main_rows = load(".")
    held = load("heldout")
    rooms_rows = main_rows + [r for r in held if r["layout"] == "fixed"]                 # 12 rooms, standard marker heights, dev + held-out seeds
    hrows = load("height") + [r for r in held if r["layout"].startswith("fixed+dock")] + load("dock02")
    other = load("height_rooms")
    tilt_free, lqr = load("tilt_free"), load("lqr")
    sens = {v: load(f"sens_{v}") for v in ["Q05", "Q2", "R05", "R2"]}
    gaze, mixed, tall = load("gaze"), load("mixed"), load("tall_hard")
    heights = sorted({r["z"] for r in hrows})
    n_dev = sum(r["seed"] < 1000 for r in rooms_rows) // 3
    n_held = sum(r["seed"] >= 1000 for r in rooms_rows) // 3
    out.append(f"Rooms block: {len(rooms_rows)} trials ({n_dev} development and {n_held} held-out trials per config). "
               f"Height block: {len(hrows)} trials over dock heights {', '.join(f'{z:g}' for z in heights)} m.\n")

    # ============================================================== 1. pooled scorecard
    scopes = [("Rooms at standard marker heights (12 rooms, development + held-out seeds)", rooms_rows),
              ("All dock heights pooled (height studies, large_6_pillars)", hrows)]
    fig, axs = plt.subplots(len(scopes), 4, figsize=(12, 3.0 * len(scopes)))
    body = []
    for si, (title, rs) in enumerate(scopes):
        axs[si, 0].set_ylabel(title.split(" (")[0], fontsize=8, fontweight="bold")
        for ci, c in enumerate(CFGS):
            sub = pick(rs, c)
            if not sub:
                continue
            p, lo, hi = pct(sub)
            axs[si, 0].bar(ci, p, color=COL[c], width=0.6)
            axs[si, 0].errorbar(ci, p, yerr=[[p - lo], [hi - p]], color="k", capsize=3)
            axs[si, 0].text(ci, 4, f"{sum(r['ok'] for r in sub)}/{len(sub)}", ha="center", fontsize=7, color="w")
            for k, (key, scale, ax) in enumerate([("duration_s", 1.0, axs[si, 1]), ("pos_rmse_m", 1000.0, axs[si, 2]),
                                                   ("min_clearance_m", 1.0, axs[si, 3])]):
                v = vals([r for r in sub if r["ok"] or key != "duration_s"], key, scale)
                if len(v) == 0:
                    continue
                q = np.percentile(v, [10, 25, 50, 75, 90])
                ax.add_patch(plt.Rectangle((ci - 0.3, q[1]), 0.6, q[3] - q[1], color=COL[c], alpha=0.45))
                ax.plot([ci - 0.3, ci + 0.3], [q[2], q[2]], color=COL[c], lw=2.5)
                ax.plot([ci, ci], [q[0], q[4]], color=COL[c], lw=1)
            body.append([title.split(" (")[0], c, len(sub), frac(sub), f"{med([r for r in sub if r['ok']], 'duration_s'):.0f}",
                         f"{med(sub, 'pos_rmse_m', 1000):.1f}", f"{med(sub, 'nees_consistent'):.2f}", f"{med(sub, 'min_clearance_m'):.2f}",
                         sum(r["outcome"] == "contact" for r in sub)])
        for ax, t in zip(axs[si], ["Success [%] (Wilson 95%)", "Time to dock [s] (successful runs)", "Position RMSE [mm]", "Closest approach [m] (below 0 = contact)"]):
            ax.set_xticks(range(3)); ax.set_xticklabels(CFGS); ax.set_title(t, fontsize=9); style(ax)
            ax.set_xlim(-0.6, 2.6)
        axs[si, 0].set_ylim(0, 110)
        axs[si, 3].axhline(0, color="k", lw=0.8)
        axs[si, 1].set_ylim(bottom=0); axs[si, 2].set_ylim(bottom=0)
    save(fig, "01_scorecard.png")
    out += ["## 1. Pooled scorecard (box: 25-75th percentile, line: median, whisker: 10-90th)\n",
            md_table(["scope", "config", "n", "docked", "median time to dock [s]", "median pos RMSE [mm]", "NEES in 95% band", "median closest approach [m]", "contacts"], body), ""]

    # ============================================================== 2. dock height
    series = {c: [r for r in hrows if r["config"] == c] for c in CFGS}
    if tilt_free:
        series["B, tilt free"] = tilt_free
    body = [[f"{z:g} m"] + [frac(pick(rs, z=z)) for rs in series.values()] for z in heights]
    out += ["## 2. Docking success by dock-marker height (development, held-out and fresh seeds pooled)\n",
            md_table(["height"] + list(series), body), "",
            "Seeds: development 0..19, held-out 5000..5019 (both at every height), 6000..6039 (0.2 m only). B and C-sweep were not run at 0.9 m (cannot dock above about 0.33 m).\n"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for name, rs in series.items():
        xs, ys, lo, hi = [], [], [], []
        for z in heights:
            sub = pick(rs, z=z)
            if sub:
                p, l, h = pct(sub)
                xs.append(z); ys.append(p); lo.append(p - l); hi.append(h - p)
        base = name.split(",")[0]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker={"B": "o", "C-sweep": "s", "C-info": "^"}[base], ls=":" if "free" in name else "-",
                    color=COL[name], label=name, capsize=3, lw=1.6, ms=6)
    ax.axvspan(0.33, max(heights) + 0.03, color="#999", alpha=0.07)
    ax.text(0.34, 40, "above about 0.33 m a level head cannot see\nthe whole plate from the 0.5 m standoff", fontsize=7, color="#555")
    ax.set_xlabel("dock marker centre height [m]"); ax.set_ylabel("docked [%]"); ax.set_ylim(-5, 108); ax.set_xticks(heights)
    ax.set_title("Docking success against dock height"); ax.legend(fontsize=8, frameon=False, loc="lower left"); style(ax, "both")
    if other:
        lab = []
        for li, lv in enumerate(sorted({r["level"] for r in other})):
            for zi, z in enumerate(sorted({r["z"] for r in other})):
                x = li * 3 + zi
                lab.append((x, f"{lv.replace('_pillars', '')}\n{z:g} m"))
                for ci, c in enumerate(CFGS):
                    sub = pick(other, c, lv, z)
                    p, l, h = pct(sub)
                    ax2.bar(x + (ci - 1) * 0.26, p, width=0.24, color=COL[c])
                    ax2.errorbar(x + (ci - 1) * 0.26, p, yerr=[[p - l], [h - p]], color="k", capsize=2, lw=0.8)
        ax2.set_xticks([a for a, _ in lab]); ax2.set_xticklabels([b for _, b in lab], fontsize=7)
        ax2.set_title("Other rooms (seeds 5000..)"); ax2.set_ylim(0, 110); style(ax2)
    save(fig, "02_height_success.png")

    # ============================================================== 3. success and estimation by room (tables as heatmaps)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    hm = {"success [%]": lambda s: 100 * sum(r["ok"] for r in s) / len(s), "median position RMSE [mm]": lambda s: med(s, "pos_rmse_m", 1000),
          "median time to dock [s]": lambda s: med([r for r in s if r["ok"]], "duration_s")}
    body = []
    for ax, (title, fn) in zip(axs, hm.items()):
        M = np.array([[fn(pick(rooms_rows, c, lv)) for c in CFGS] for lv in ROOMS])
        ax.imshow(M, aspect="auto", cmap="viridis_r" if "success" not in title else "viridis")
        ax.set_xticks(range(3)); ax.set_xticklabels(CFGS); ax.set_yticks(range(len(ROOMS))); ax.set_yticklabels(ROOMS if ax is axs[0] else [])
        for i in range(len(ROOMS)):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7, color="w" if (M[i, j] - np.nanmin(M)) / (np.ptp(M) + 1e-9) < 0.6 else "k")
        ax.set_title(title, fontsize=9)
    save(fig, "03_rooms_heatmap.png")
    for lv in ROOMS + ["ALL"]:
        row = [lv]
        for c in CFGS:
            s = pick(rooms_rows, c) if lv == "ALL" else pick(rooms_rows, c, lv)
            row.append(f"{frac(s)}; {med(s, 'pos_rmse_m', 1000):.1f} mm; {med([r for r in s if r['ok']], 'duration_s'):.0f} s")
        body.append(row)
    out += ["## 3. Rooms at standard marker heights: docked; median position RMSE; median time to dock (development + held-out seeds)\n",
            md_table(["room", "B", "C-sweep", "C-info"], body), ""]

    # ============================================================== 4. paired differences by room
    comps = [("C-info", "B"), ("C-sweep", "B"), ("C-info", "C-sweep")]
    cmark = {("C-info", "B"): ("^", COL["C-info"]), ("C-sweep", "B"): ("s", COL["C-sweep"]), ("C-info", "C-sweep"): ("D", "#555555")}
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    labels = ROOMS + ["ALL ROOMS"]
    body = []
    for ax, (metric, scale, xl, ttl) in zip(axs, [("pos_rmse_m", 1000, "mm (left of 0: first config better)", "Position RMSE"),
                                                  ("duration_s", 1, "s (left of 0: first config faster)", "Time to dock (both docked)"),
                                                  ("min_clearance_m", 100, "cm (right of 0: first config keeps more room)", "Closest approach")]):
        for ci, (a, b) in enumerate(comps):
            for ri, lv in enumerate(labels):
                rs = rooms_rows if lv == "ALL ROOMS" else pick(rooms_rows, level=lv)
                m = cmp_row(rs, a, b, metric)
                if m["n"] == 0 or m["lo"] != m["lo"]:
                    continue
                y = ri + (ci - 1) * 0.23
                mk, col = cmark[(a, b)]
                ax.plot([m["lo"] * scale, m["hi"] * scale], [y, y], color=col, lw=1.3)
                ax.plot(m["diff"] * scale, y, mk, color=col, ms=5, label=f"{a} - {b}" if ri == 0 else None)
                if metric == "pos_rmse_m":
                    body.append([lv, f"{a} - {b}", m["n_seeds"], fmt_ci(m, 1000)])
        ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(xl, fontsize=8); ax.set_title(ttl); style(ax, "x")
    axs[0].set_yticks(range(len(labels))); axs[0].set_yticklabels(labels, fontsize=8); axs[0].invert_yaxis(); fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "04_paired_by_room.png", dpi=150); plt.close(fig)
    out += ["## 4. Paired position-RMSE difference by room [mm, median [95% CI over seeds]] (first minus second; negative = first better)\n",
            md_table(["room", "pair", "seeds", "RMSE difference"], body), ""]
    body = []
    for a, b in comps:
        for metric, scale, nd, nm in [("docked", 1, 1, "docking [pp]"), ("duration_s", 1, 1, "time to dock [s]"), ("pos_rmse_m", 1000, 1, "RMSE [mm]"),
                                      ("min_clearance_m", 100, 1, "closest approach [cm]")]:
            m = cmp_row(rooms_rows, a, b, metric)
            extra = f" ({m['wins']}W {m['losses']}L, McNemar p={m['p_mcnemar']:.2f})" if metric == "docked" and "wins" in m else ""
            body.append([f"{a} - {b}", nm, m["n_seeds"], fmt_ci(m, scale, nd) + extra])
    out += ["Pooled over all 12 rooms:\n", md_table(["pair", "metric", "seeds", "difference [95% CI]"], body), ""]

    # ============================================================== 5. estimation by height, and how trials ended
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for ci, c in enumerate(CFGS):
        for zi, z in enumerate(heights):
            v = vals(pick(hrows, c, z=z), "pos_rmse_m", 1000)
            if len(v):
                ax.scatter(np.full(len(v), zi + (ci - 1) * 0.25) + np.linspace(-0.06, 0.06, len(v)), v, s=5, color=COL[c], alpha=0.3)
                ax.plot(zi + (ci - 1) * 0.25, np.median(v), "_", ms=14, mew=2.5, color=COL[c], label=c if zi == 0 else None)
    ax.set_xticks(range(len(heights))); ax.set_xticklabels([f"{z:g}" for z in heights]); ax.set_yscale("log")
    ax.set_xlabel("dock marker height [m]"); ax.set_ylabel("position RMSE [mm] (dot = one trial, bar = median)")
    ax.set_title("Estimation error by dock height")
    ax.legend(frameon=False, fontsize=8); style(ax)
    groups = [("rooms", rooms_rows)] + [(f"{z:g} m", pick(hrows, z=z)) for z in heights]
    for gi, (gname, rs) in enumerate(groups):
        for ci, c in enumerate(CFGS):
            sub = pick(rs, c)
            if not sub:
                continue
            left = 0
            for o in OUTCOMES:
                w = 100 * sum(r["outcome"] == o for r in sub) / len(sub)
                if w:
                    ax2.barh(gi * 4 + ci, w, left=left, color=OUT_COL[o], label=o if (gi == 0 and ci == 0) or (o not in ax2.get_legend_handles_labels()[1] and w) else None)
                left += w
    ax2.set_yticks([gi * 4 + 1 for gi in range(len(groups))]); ax2.set_yticklabels([g for g, _ in groups]); ax2.invert_yaxis()
    ax2.set_xlabel("share of trials [%]  (rows within a group: B, C-sweep, C-info)"); ax2.set_title("How trials ended"); ax2.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    style(ax2, "x")
    save(fig, "05_estimation_height_outcomes.png")
    body = []
    for gname, rs in groups:
        for c in CFGS:
            sub = pick(rs, c)
            if sub:
                body.append([gname, c, len(sub)] + [sum(r["outcome"] == o for r in sub) for o in OUTCOMES])
    out += ["## 5. How trials ended\n", md_table(["group", "config", "n"] + OUTCOMES, body), ""]

    # ============================================================== 6. false docks
    allr = rooms_rows + hrows
    fd = [r for r in allr if r["outcome"] == "false dock"]
    if fd:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        zs = sorted({r["z"] for r in allr})
        for ci, c in enumerate(CFGS):
            ax.bar(np.arange(len(zs)) + (ci - 1) * 0.27, [100 * sum(r["outcome"] == "false dock" for r in pick(allr, c, z=z)) / max(1, len(pick(allr, c, z=z))) for z in zs],
                   width=0.25, color=COL[c], label=c)
        ax.set_xticks(range(len(zs))); ax.set_xticklabels([f"{z:g}" for z in zs]); ax.set_xlabel("dock marker height [m]"); ax.set_ylabel("false docks [% of trials]")
        ax.set_title("False docks by dock height"); ax.legend(frameon=False, fontsize=8); style(ax)
        ax2.hist(vals(fd, "yaw_err_deg"), bins=np.arange(5, 17, 1), color="#d9a400"); ax2.axvline(5, color="k", lw=0.8, ls="--")
        ax2.set_xlabel("heading error at the dock [deg] (limit 5)"); ax2.set_title(f"False docks: {np.mean(vals(fd, 'miss_m') <= 0.10) * 100:.0f}% were inside the 0.10 m position limit"); ax2.title.set_fontsize(8); style(ax2)
        save(fig, "06_false_docks.png")
        out += ["## 6. False docks (the robot stops and claims the dock but misses the 0.10 m / 5 degree tolerance)\n",
                md_table(["height"] + CFGS, [[f"{z:g} m"] + [f"{sum(r['outcome'] == 'false dock' for r in pick(allr, c, z=z))}/{len(pick(allr, c, z=z))}" for c in CFGS] for z in zs]),
                f"\n{len(fd)} false docks in all; median heading error {np.median(vals(fd, 'yaw_err_deg')):.1f} deg, median position miss {np.median(vals(fd, 'miss_m')) * 1000:.0f} mm.\n"]

    # ============================================================== 7. LQR follower
    if lqr:
        gtp = main_rows + load("height") + held
        cells = [("large_6_pillars", lambda r: r["level"] == "large_6_pillars" and r["layout"] == "fixed" and r["seed"] < 1000),
                 ("cluttered_14", lambda r: r["level"] == "cluttered_14" and r["layout"] == "fixed" and r["seed"] < 1000),
                 ("height study", lambda r: r["layout"].startswith("fixed+dock") and r["seed"] < 1000 and r["z"] in {0.2, 0.3, 0.45, 0.6})]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        body = []
        for ki, (name, sel) in enumerate(cells):
            for ci, c in enumerate(CFGS):
                A = [r for r in gtp if r["config"] == c and sel(r)]
                B = [r for r in lqr if r["config"] == c and sel(r)]
                ix = {(r["level"], r["layout"], r["seed"]): r for r in B}
                pr = [(r, ix[(r["level"], r["layout"], r["seed"])]) for r in A if (r["level"], r["layout"], r["seed"]) in ix]
                if not pr:
                    continue
                n = len(pr); g = sum(x["ok"] for x, _ in pr); q = sum(y["ok"] for _, y in pr)
                w = sum(x["ok"] and not y["ok"] for x, y in pr); l = sum(y["ok"] and not x["ok"] for x, y in pr)
                x0 = ki * 4 + ci
                ax.bar(x0 - 0.2, 100 * g / n, width=0.38, color=COL[c]); ax.bar(x0 + 0.2, 100 * q / n, width=0.38, color=COL[c], hatch="//", edgecolor="w")
                body.append([name, c, n, g, q, f"{w}/{l}", f"{paired.mcnemar_exact(w, l):.2f}"])
        ax.set_xticks([ki * 4 + 1 for ki in range(len(cells))]); ax.set_xticklabels([c[0] for c in cells]); ax.set_ylabel("docked [%]"); ax.set_ylim(0, 108)
        ax.set_title("go_to_point (solid) against LQR (hatched), same seeds"); style(ax)
        save(fig, "07_lqr_follower.png")
        out += ["## 7. LQR path follower against go_to_point, same seeds\n",
                md_table(["scope", "config", "pairs", "go_to_point docked", "LQR docked", "gtp-only / LQR-only", "McNemar p"], body), ""]

    # ============================================================== 8. estimator sensitivity
    if any(sens.values()):
        base = [r for r in main_rows if r["level"] == "large_6_pillars"]
        runs = [("Q x0.5", sens["Q05"]), ("Q x1", base), ("Q x2", sens["Q2"]), ("R x0.5", sens["R05"]), ("R x1", base), ("R x2", sens["R2"])]
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
        body = []
        for ax, (key, scale, ttl) in zip(axs, [("pos_rmse_m", 1000, "Position RMSE [mm]"), ("nees_consistent", 1, "NEES in the 95% band (nominal 0.95)"),
                                               ("sigma_xy_mean_m", 1000, "Filter's own sigma [mm]")]):
            for c in ["B", "C-info"]:
                for off, grp in [(0, runs[:3]), (4, runs[3:])]:
                    ys = [med(pick(r, c), key, scale) for _, r in grp if r]
                    ax.plot(np.arange(len(ys)) + off, ys, "o-", color=COL[c], label=c if off == 0 else None)
            ax.set_xticks(list(range(3)) + list(range(4, 7))); ax.set_xticklabels(["x0.5", "x1", "x2"] * 2); ax.set_title(ttl, fontsize=9)
            ax.text(1, ax.get_ylim()[0], "Q (odometry noise)", ha="center", va="bottom", fontsize=7); ax.text(5, ax.get_ylim()[0], "R (marker noise)", ha="center", va="bottom", fontsize=7); style(ax)
        axs[0].legend(frameon=False, fontsize=8)
        save(fig, "08_sensitivity.png")
        for name, r in runs:
            if r:
                body.append([name] + [f"{frac(pick(r, c))}; {med(pick(r, c), 'pos_rmse_m', 1000):.1f} mm; NEES {med(pick(r, c), 'nees_consistent'):.2f}; sigma {med(pick(r, c), 'sigma_xy_mean_m', 1000):.1f} mm" for c in ["B", "C-info"]])
        out += ["## 8. Estimator sensitivity (Q: odometry noise, R: marker noise, scaled by 0.5 and 2), large_6_pillars\n", md_table(["run", "B", "C-info"], body), ""]

    # ============================================================== 9. gaze behaviour
    if gaze:
        keys = [("gaze_switch_per_min", "Pan reversals per minute"), ("gaze_toward_travel", "Share of moving time looking where it is going"),
                ("gaze_dwell_med_s", "Median time the gaze stays still [s]"), ("pan_std_rad", "Pan spread [rad]"), ("head_tilt_travel_rad", "Tilt travel over a run [rad]")]
        fig, axs = plt.subplots(1, len(keys), figsize=(14, 3.4))
        for ax, (k, t) in zip(axs, keys):
            for ci, c in enumerate(CFGS):
                v = vals(pick(gaze, c), k)
                ax.boxplot(v, positions=[ci], widths=0.5, patch_artist=True, showfliers=False, boxprops=dict(facecolor=COL[c], alpha=0.5), medianprops=dict(color="k"))
                ax.scatter(np.full(len(ROOMS), ci) + np.linspace(-0.12, 0.12, len(ROOMS)), [med(pick(gaze, c, lv), k) for lv in ROOMS], s=8, color=COL[c], zorder=3)
            ax.set_xticks(range(3)); ax.set_xticklabels(CFGS); ax.set_title(t, fontsize=8); style(ax)
        save(fig, "09_gaze_behaviour.png")
        body = []
        for lv in ROOMS + ["ALL"]:
            for k, t in keys:
                row = [lv, t]
                for c in CFGS:
                    v = vals(pick(gaze, c) if lv == "ALL" else pick(gaze, c, lv), k)
                    q = np.percentile(v, [25, 50, 75]) if len(v) else [np.nan] * 3
                    row.append(f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]")
                body.append(row)
        out += ["## 9. Gaze behaviour: median [IQR] per config (descriptive cues, not a perception result; B faces forward by construction)\n",
                md_table(["room", "metric", "B", "C-sweep", "C-info"], body), ""]
        rep = [r for r in gaze if r["level"] in ROOMS]
        same = 0; tot = 0
        ref = {(r["level"], r["config"], r["seed"]): r for r in main_rows}
        for r in rep:
            m = ref.get((r["level"], r["config"], r["seed"]))
            if m:
                tot += 1; same += (m["outcome"] == r["outcome"] and abs(num(m, "duration_s") - num(r, "duration_s")) < 0.05)
        out += [f"Reproducibility: {same} of {tot} gaze-block trials repeated the main-block outcome and duration exactly.\n"]

    # ============================================================== 10. mixed marker heights
    if mixed:
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
        body = []
        for ci, c in enumerate(CFGS):
            sub = pick(mixed, c)
            p, lo, hi = pct(sub)
            axs[0].bar(ci, p, color=COL[c]); axs[0].errorbar(ci, p, yerr=[[p - lo], [hi - p]], color="k", capsize=3)
            for ax, (k, sc) in zip(axs[1:], [("pos_rmse_m", 1000), ("n_landmarks", 1), ("coverage", 100)]):
                v = vals(sub, k, sc)
                if len(v):
                    ax.boxplot(v, positions=[ci], widths=0.5, patch_artist=True, showfliers=False, boxprops=dict(facecolor=COL[c], alpha=0.5), medianprops=dict(color="k"))
            body.append([c, len(sub), frac(sub), f"{med(sub, 'pos_rmse_m', 1000):.1f}", f"{med(sub, 'n_landmarks'):.0f}", f"{med(sub, 'coverage') * 100:.0f}"])
        for ax, t in zip(axs, ["Docked [%]", "Position RMSE [mm]", "Landmarks mapped", "Floor coverage at the end [%]"]):
            ax.set_xticks(range(3)); ax.set_xticklabels(CFGS); ax.set_title(t, fontsize=9); style(ax)
        axs[0].set_ylim(0, 108)
        save(fig, "10_mixed_heights.png")
        out += ["## 10. Mixed marker heights (every marker at a random height 0.2-0.6 m; seeds 8000..)\n",
                md_table(["config", "n", "docked", "median RMSE [mm]", "median landmarks", "median coverage [%]"], body), ""]
        pr = [("C-info", "B"), ("C-sweep", "B"), ("C-info", "C-sweep")]
        out += [md_table(["pair", "metric", "seeds", "difference [95% CI]"], [[f"{a} - {b}", nm, cmp_row(mixed, a, b, k)["n_seeds"], fmt_ci(cmp_row(mixed, a, b, k), sc)]
                                                                       for a, b in pr for k, sc, nm in [("docked", 1, "docking [pp]"), ("pos_rmse_m", 1000, "RMSE [mm]")]]), ""]

    # ============================================================== 11. tall docks in the hardest rooms
    if tall:
        lv_t = sorted({r["level"] for r in tall}); zs = sorted({r["z"] for r in tall})
        fig, ax = plt.subplots(figsize=(7, 3.6))
        body = []
        for li, lv in enumerate(lv_t):
            for zi, z in enumerate(zs):
                sub = pick(tall, "C-info", lv, z)
                if not sub:
                    continue
                p, lo, hi = pct(sub); x = li * 3 + zi
                ax.bar(x, p, color=COL["C-info"], width=0.8); ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color="k", capsize=3)
                body.append([lv, f"{z:g} m", frac(sub), f"{med([r for r in sub if r['ok']], 'duration_s'):.0f}", f"{med(sub, 'pos_rmse_m', 1000):.1f}",
                             sum(r["outcome"] == "contact" for r in sub), sum(r["outcome"] == "timeout" for r in sub)])
        ax.set_xticks([li * 3 + 0.5 for li in range(len(lv_t))]); ax.set_xticklabels(lv_t); ax.set_ylim(0, 108); ax.set_ylabel("docked [%]")
        ax.set_title("C-info with a tall dock marker in the hardest rooms (0.45 left, 0.6 m right of each pair)"); ax.title.set_fontsize(8); style(ax)
        save(fig, "11_tall_hard_rooms.png")
        out += ["## 11. Tall docks (0.45 and 0.6 m) in the hardest rooms, C-info (seeds 9000..)\n",
                md_table(["room", "height", "docked", "median time [s]", "median RMSE [mm]", "contacts", "timeouts"], body), ""]

    (RES / "final_tables.md").write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwrote {RES / 'final_tables.md'} and {len(list(FIG.glob('*.png')))} figures in {FIG}")


if __name__ == "__main__":
    main()
