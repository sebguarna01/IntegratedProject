"""Search-and-dock study: drop the robot into an unknown room and let it find marker 10 and dock.

    python scripts/search_study.py --sensor synthetic --trials 30 --level large_6_pillars --random-markers --random-dock
    python scripts/search_study.py --sensor camera --trials 20            # real rendered camera + floor ranging
    python scripts/search_study.py --report --sensor synthetic            # tables from the CSV

Configs: B (fixed head), C-sweep (sweeping head), C-info (active gaze). A is not run: odometry alone cannot see
marker 10 or the walls, so it cannot search. Each seed fixes the start pose (random, clear of obstacles), the
hidden encoder errors and, with --random-markers / --random-dock, the marker layout, so every config faces the
same trial. The robot knows no map; ground truth is used only to score (docked = true position within 0.10 m of
the dock point and heading within 5 deg, no contact). Results append to results/search_study_<sensor>.csv and
finished rows are skipped, so an interrupted run resumes.
"""
import argparse
import csv

import numpy as np

from activenav.config import apply_overrides, load_config
from activenav.evaluation.worlds import ScenarioError
from activenav.evaluation.search import SEARCH_CONFIGS, make_search_session
from activenav.evaluation.stats import bootstrap_ci, wilson_interval

FIELDS = ["level", "layout", "config", "seed", "outcome", "docked", "duration_s", "time_to_first_sight_s", "path_length_m",
          "miss_m", "yaw_err_deg", "min_clearance_m", "pos_rmse_m", "n_landmarks", "landmark_err_m", "coverage", "map_precision",
          # added with the height factor and the extra metrics (older files lack these columns; the report copes)
          "dock_height_m", "approach_s", "nees_mean", "nees_consistent", "sigma_xy_mean_m", "sigma_xy_final_m", "final_pos_err_m",
          "coverage_30s", "coverage_60s", "coverage_90s", "head_pan_travel_rad", "head_tilt_travel_rad", "tilt_mean_rad",
          "tilt_std_rad", "near_time_s", "blocked_s", "n_recover", "n_replans", "dock_seen_frac", "n_spurious_landmarks",
          # walking-people levels (nan elsewhere)
          "min_person_gap_m", "person_near_s", "person_contact",
          # gaze behaviour (runs from the final study on; nan in older files)
          "gaze_switch_per_min", "gaze_dwell_med_s", "gaze_toward_travel", "gaze_lead_s", "pan_std_rad"]
NEW = FIELDS[17:]


def rdir(args, cfg):
    """Results folder: results/ by default, or --dir (a different study keeps its own files, so seeds never overwrite each other)."""
    return cfg["_root"] / args.dir


def layout_name(args, dock_height=None):
    return ("randmk" if args.random_markers else "fixed") + ("+randdock" if args.random_dock else "") + \
        ("+heights" if args.heights else "") + (f"+dock{dock_height:g}" if dock_height is not None else "")


def run(args, cfg):
    path = rdir(args, cfg) / f"search_study_{args.sensor}{args.tag}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        from activenav.evaluation.stats import read_rows, repair
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), [])
        rows0, dropped = read_rows(path, FIELDS)
        if dropped or header != FIELDS:                    # older header or an interrupted write: rewrite it first (original kept as .bak)
            kept, dropped = repair(path, FIELDS)
            print(f"{path.name}: header brought up to date, {kept} rows kept, {dropped} damaged row(s) dropped (original: {path.name}.bak)")
            rows0, _ = read_rows(path, FIELDS)
        if args.redo:                                      # forget the rows this command would produce, so they run again
            want = {(args.level, layout_name(args, h), c, cfg["seed"] + i)
                    for h in (args.dock_height or [None]) for c in args.configs for i in range(args.trials)}
            keep = [r for r in rows0 if (r["level"], r["layout"], r["config"], int(r["seed"])) not in want]
            if len(keep) != len(rows0):
                import shutil
                shutil.copy(path, str(path) + ".bak")
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore", restval="")
                    w.writeheader()
                    w.writerows(keep)
                print(f"--redo: removed {len(rows0) - len(keep)} existing row(s) from {path.name} (original: {path.name}.bak); running them again")
            rows0 = keep
        done = {(r["level"], r["layout"], r["config"], int(r["seed"])) for r in rows0}
    new = not path.exists()
    f = open(path, "a", newline="")
    wr = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        wr.writeheader()
    camera = None
    if args.sensor == "camera":
        from activenav.sensing.markers import MarkerSensor
        camera = MarkerSensor(cfg)
    cache = {}
    heights = args.dock_height or [None]                   # --dock-height 0.2 0.45 0.9: the same seeds at every height (paired)
    for dh, conf in [(h, c) for h in heights for c in args.configs]:
        for i in range(args.trials):
            seed = cfg["seed"] + i
            key = (args.level, layout_name(args, dh), conf, seed)
            if key in done:
                continue
            try:
                s, _ = make_search_session(cfg, cache, conf, seed, level=args.level, start=args.start, sensor=args.sensor,
                                           camera_sensor=camera, random_markers=args.random_markers,
                                           random_dock=args.random_dock, max_time=args.max_time, heights=args.heights,
                                           dock_height=dh)
            except ScenarioError as e:
                raise SystemExit(f"error: {e}")
            while s.step():
                pass
            r = s.result()
            ts = r["time_to_first_sight_s"]
            row = dict(level=args.level, layout=layout_name(args, dh), config=conf, seed=seed, outcome=r["outcome"],
                       docked=int(r["docked"]), duration_s=round(r["duration_s"], 1),
                       time_to_first_sight_s="nan" if ts is None else round(ts, 1), path_length_m=round(r["path_length_m"], 2),
                       miss_m=round(r["miss_m"], 4), yaw_err_deg=round(r["yaw_err_deg"], 2),
                       min_clearance_m=round(r["min_clearance_m"], 3), pos_rmse_m=round(r["pos_rmse_m"], 4),
                       n_landmarks=r["n_landmarks"], landmark_err_m=round(r["landmark_err_m"], 4),
                       coverage=round(r["coverage"], 3), map_precision=round(r["map_precision"], 3),
                       **{k: ("nan" if r[k] != r[k] else round(float(r[k]), 4)) for k in NEW})
            wr.writerow(row)
            f.flush()
            print(f"{args.level} {conf:8s} seed {seed:3d}: {r['outcome']:10s} {r['duration_s']:5.0f} s  "
                  f"miss {r['miss_m']:.3f} m  yaw {r['yaw_err_deg']:4.1f} deg  clearance {r['min_clearance_m']:+.2f} m", flush=True)
    f.close()
    for ctx in cache.values():
        ctx.env.close()


def _col(rs, k):
    """Numeric column as floats; rows from older files (no such column) or 'nan' give nan."""
    out = []
    for r in rs:
        try:
            out.append(float(r.get(k, "nan") or "nan"))
        except ValueError:
            out.append(float("nan"))
    return np.array(out, float)


def extra_report(sel, per):
    """Secondary metrics (medians over every run of a config), success over time, and success by dock-marker height."""
    cfgs = [c for c in SEARCH_CONFIGS if c in per]
    rows = {c: list(per[c].values()) for c in cfgs}
    if not any(np.isfinite(_col(rows[c], "nees_mean")).any() for c in cfgs):
        return                                                    # files from before these metrics existed
    cols = [("nees_mean", "NEES (ideal 3)"), ("nees_consistent", "NEES in band"), ("sigma_xy_mean_m", "sigma xy [m]"),
            ("final_pos_err_m", "final pos err"), ("coverage_60s", "cov@60s"), ("approach_s", "approach s"),
            ("dock_seen_frac", "dock seen"), ("head_pan_travel_rad", "pan travel"), ("head_tilt_travel_rad", "tilt travel"),
            ("near_time_s", "near obst s"), ("blocked_s", "blocked s"), ("n_recover", "recoveries"), ("n_replans", "replans")]
    print("  secondary metrics (median over all runs of the config; nan runs ignored):")
    print("  " + f"{'config':8s}" + "".join(f"{h:>15s}" for _, h in cols))
    for c in cfgs:
        print("  " + f"{c:8s}" + "".join(f"{np.nanmedian(_col(rows[c], k)) if np.isfinite(_col(rows[c], k)).any() else float('nan'):15.3f}" for k, _ in cols))
    # success over time: a failure is never counted as 'slow', it simply has not succeeded by that time
    print("  docked within:   " + "   ".join(f"{c}: " + " ".join(
        f"{int(sum(r['docked'] == '1' and float(r['duration_s']) <= T for r in rows[c]))}/{len(rows[c])}@{T}s" for T in (60, 100, 150))
        for c in cfgs))
    hs = sorted({h for c in cfgs for h in _col(rows[c], "dock_height_m") if np.isfinite(h)})
    if len(hs) > 1:
        print("  docking success by dock-marker height:")
        print("  " + f"{'height m':>9s}" + "".join(f"{c:>16s}" for c in cfgs))
        for h in hs:
            cells = []
            for c in cfgs:
                sub = [r for r in rows[c] if abs(float(r.get("dock_height_m") or "nan") - h) < 1e-6]
                k = sum(r["docked"] == "1" for r in sub)
                cells.append(f"{k}/{len(sub)} ({100 * k / max(1, len(sub)):.0f}%)")
            print("  " + f"{h:9.2f}" + "".join(f"{x:>16s}" for x in cells))


def load_rows(args, cfg):
    """Every results file of this sensor (tagged runs included). A seed that appears in several files counts once: newest wins."""
    latest = {}
    files = sorted(rdir(args, cfg).glob(f"search_study_{args.sensor}*.csv"), key=lambda p: p.stat().st_mtime)
    for path in files:
        if "pre_sweepfix" not in path.name:
            from activenav.evaluation.stats import read_rows
            for r in read_rows(path, FIELDS)[0]:
                latest[(r["level"], r["layout"], r["config"], int(r["seed"]))] = r
    from activenav.evaluation.stats import apply_cap
    rows = apply_cap(list(latest.values()), getattr(args, "cap", None))
    if getattr(args, "cap", None):
        print(f"(analysis cap {args.cap:g} s: runs longer than that count as timeouts)")
    return rows


def plots(args, cfg):
    from activenav.evaluation.study_plots import make_all
    rows = load_rows(args, cfg)
    if not rows:
        raise SystemExit("no results to plot")
    out = rdir(args, cfg) / "figures" / f"study_{args.sensor}"
    for p in make_all(rows, out, getattr(args, "view", "both")):
        print("saved", p)


def summary_report(rows):
    """The high-level text view: one table per room (every layout and dock height pooled), then success for low and high dock markers."""
    print("\n##### SUMMARY: one block per room, every layout and dock height pooled #####")
    for lvl in dict.fromkeys(r["level"] for r in rows):
        sel = [r for r in rows if r["level"] == lvl]
        print(f"\n=== {lvl}  ({len(sel)} trials) ===")
        print(f"{'config':8s} {'n':>4s} {'docked (Wilson 95%)':>22s} {'false':>6s} {'contact':>8s} {'timeout/gave up':>16s} {'t_dock med':>11s} "
              f"{'NEES in band':>13s} {'final err':>10s} {'min clr':>8s} {'near s':>7s}")
        for c in SEARCH_CONFIGS:
            rs = [r for r in sel if r["config"] == c]
            if not rs:
                continue
            ok = [r for r in rs if r["docked"] == "1"]
            lo, hi = wilson_interval(len(ok), len(rs))
            td = np.array([float(r["duration_s"]) for r in ok]) if ok else np.array([np.nan])
            med = lambda k: np.nanmedian(_col(rs, k)) if np.isfinite(_col(rs, k)).any() else float("nan")
            print(f"{c:8s} {len(rs):4d} {len(ok):3d}/{len(rs):<4d} ({lo:.2f}-{hi:.2f}) {sum(r['outcome'] == 'false dock' for r in rs):6d} "
                  f"{sum(r['outcome'] == 'contact' for r in rs):8d} {sum(r['outcome'] in ('timeout', 'gave up') for r in rs):16d} "
                  f"{np.median(td):11.0f} {med('nees_consistent'):13.2f} {med('final_pos_err_m'):10.3f} {med('min_clearance_m'):+8.2f} {med('near_time_s'):7.1f}")
    hs = _col(rows, "dock_height_m")
    from activenav.evaluation import paired
    bands = [(label, f) for _, label, f in paired.BANDS if any(np.isfinite(h) and f(h) for h in hs)]
    if len(bands) >= 2:
        print("\n##### docking success by room and dock-marker height band #####")
        print(f"{'room':18s} {'band':>32s}" + "".join(f"{c:>16s}" for c in SEARCH_CONFIGS))
        for lvl in dict.fromkeys(r["level"] for r in rows):
            for name, f in bands:
                cells = []
                for c in SEARCH_CONFIGS:
                    sub = [r for r in rows if r["level"] == lvl and r["config"] == c and np.isfinite(float(r.get("dock_height_m") or "nan"))
                           and f(float(r["dock_height_m"]))]
                    k = sum(r["docked"] == "1" for r in sub)
                    cells.append(f"{k}/{len(sub)} ({100 * k / len(sub):.0f}%)" if sub else "-")
                if any(x != "-" for x in cells):
                    print(f"{lvl:18s} {name:>32s}" + "".join(f"{x:>16s}" for x in cells))


def report(args, cfg):
    rows = load_rows(args, cfg)
    view = getattr(args, "view", "both")
    if view in ("summary", "both"):
        summary_report(rows)
        if view == "summary":
            height_report(rows)
            paired_report(rows, "room")
            return
    height_report_rows = rows
    num = lambda r, k: float(r[k])
    groups = list(dict.fromkeys((r["level"], r["layout"]) for r in rows))
    for lvl, lay in groups:
        sel = [r for r in rows if r["level"] == lvl and r["layout"] == lay]
        print(f"\n=== {lvl}, markers {lay}, sensor {args.sensor} ===")
        print(f"{'config':8s} {'n':>3s} {'docked (Wilson 95%)':>22s} {'contact':>8s} {'timeout/gave up':>16s} {'t_dock med [CI]':>20s} "
              f"{'p90':>6s} {'t_first med':>11s} {'path':>6s} {'min clr':>8s} {'coverage':>9s} {'lm err':>7s}")
        per = {}
        for c in SEARCH_CONFIGS:
            rs = sorted([r for r in sel if r["config"] == c], key=lambda r: int(r["seed"]))
            if not rs:
                continue
            per[c] = {int(r["seed"]): r for r in rs}
            ok = [r for r in rs if r["docked"] == "1"]
            lo, hi = wilson_interval(len(ok), len(rs))
            td = np.array([num(r, "duration_s") for r in ok]) if ok else np.array([np.nan])
            ci = bootstrap_ci(td) if len(ok) > 2 else (float("nan"),) * 2
            tf = np.array([num(r, "time_to_first_sight_s") for r in rs])
            print(f"{c:8s} {len(rs):3d} {len(ok):3d}/{len(rs):<3d} ({lo:.2f}-{hi:.2f}) {sum(r['outcome'] == 'contact' for r in rs):8d} "
                  f"{sum(r['outcome'] in ('timeout', 'gave up') for r in rs):16d} {np.median(td):8.0f} [{ci[0]:.0f}-{ci[1]:.0f}]"
                  f" {np.nanpercentile(td, 90):6.0f} {np.nanmedian(tf):11.1f} {np.median([num(r, 'path_length_m') for r in rs]):6.1f} "
                  f"{np.median([num(r, 'min_clearance_m') for r in rs]):+8.2f} {np.median([num(r, 'coverage') for r in rs]):9.2f} "
                  f"{np.nanmedian([num(r, 'landmark_err_m') for r in rs]):7.3f}")
        extra_report(sel, per)
        try:
            from scipy.stats import fisher_exact, wilcoxon
        except ImportError:
            continue
        for a, b in (("B", "C-info"), ("B", "C-sweep"), ("C-sweep", "C-info")):
            if a in per and b in per:
                seeds = sorted(set(per[a]) & set(per[b]))
                ta = np.array([num(per[a][s], "duration_s") for s in seeds])
                tb = np.array([num(per[b][s], "duration_s") for s in seeds])
                fa = sum(per[a][s]["docked"] != "1" for s in seeds)
                fb = sum(per[b][s]["docked"] != "1" for s in seeds)
                try:
                    pw = wilcoxon(ta, tb).pvalue if np.any(ta != tb) else 1.0
                except ValueError:
                    pw = float("nan")
                pf = fisher_exact([[fa, len(seeds) - fa], [fb, len(seeds) - fb]])[1]
                print(f"  {a} vs {b}: paired time to finish (failures count as the run length) {np.median(ta):.0f} vs {np.median(tb):.0f} s (Wilcoxon p={pw:.3f}); "
                      f"failures {fa} vs {fb} (Fisher p={pf:.3f})   [n={len(seeds)} seeds]")

    height_report(height_report_rows)
    paired_report(height_report_rows, "room")
    paired_report(height_report_rows, "band")


def paired_report(rows, by="room"):
    """Paired comparison of the configs, seed by seed (evaluation/paired.py): wins/losses, success difference with a CI over seeds."""
    from activenav.evaluation import paired
    if not any(paired.pairs(rows, a, b) for a, b in paired.COMPARISONS):
        return
    what = "room" if by == "room" else "room and dock-height band"
    print(f"\n##### PAIRED COMPARISON by {what}: the same seed and layout under two configs (A - B, 95% CI over seeds) #####")
    print(paired.format_report(rows, by))


def height_report(rows):
    """Success by dock-marker height for each level, pooling every layout (a --dock-height sweep has one layout per height)."""
    for lvl in dict.fromkeys(r["level"] for r in rows):
        sel = [r for r in rows if r["level"] == lvl]
        hs = sorted({h for h in _col(sel, "dock_height_m") if np.isfinite(h)})
        if len(hs) < 2:
            continue
        print(f"\n=== {lvl}: docking success by dock-marker height (all layouts pooled) ===")
        print(f"{'height m':>9s}" + "".join(f"{c:>18s}" for c in SEARCH_CONFIGS))
        for h in hs:
            cells = []
            for c in SEARCH_CONFIGS:
                sub = [r for r in sel if r["config"] == c and abs(float(r.get("dock_height_m") or "nan") - h) < 1e-6]
                k = sum(r["docked"] == "1" for r in sub)
                cells.append(f"{k}/{len(sub)} ({100 * k / max(1, len(sub)):.0f}%)")
            print(f"{h:9.2f}" + "".join(f"{x:>18s}" for x in cells))


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["synthetic", "camera"], default="synthetic")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for the results file, so several runs can go in parallel (e.g. _a)")
    ap.add_argument("--level", default="large_6_pillars")
    ap.add_argument("--configs", nargs="+", default=SEARCH_CONFIGS, choices=SEARCH_CONFIGS)
    ap.add_argument("--random-markers", action="store_true", help="re-place the markers per seed")
    ap.add_argument("--random-dock", action="store_true", help="also re-place marker 10 per seed (implies a new layout)")
    ap.add_argument("--heights", action="store_true", help="every marker (dock included) hangs at a height from `marker_heights`, per seed")
    ap.add_argument("--dock-height", type=float, nargs="+", default=None, metavar="Z",
                    help="centre height(s) of the dock marker only [m] (0.2..wall_height-0.2); several values run --trials seeds at "
                         "each height (the same seeds, so the heights are compared on identical starts); other markers keep their heights")
    ap.add_argument("--start", nargs=3, type=float, default=None, metavar=("X", "Y", "YAW"), help="fixed start pose (default: random per seed)")
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--redo", action="store_true", help="delete the existing rows of this exact command (level, configs, dock heights, seeds) first, then run them again")
    ap.add_argument("--cap", type=float, default=None, metavar="SECONDS",
                    help="with --report/--plots: analyse every run as if capped at this many seconds (a longer run counts as a timeout), so rooms run with "
                         "different --max-time stay comparable; it can only shorten runs")
    ap.add_argument("--repair", action="store_true", help="bring every results file of this sensor up to date (current header, damaged rows dropped; originals kept as .bak) and exit")
    ap.add_argument("--report", action="store_true", help="print the tables and also write the charts (results/figures/study_<sensor>/)")
    ap.add_argument("--plots", action="store_true", help="only write the charts")
    ap.add_argument("--no-plots", action="store_true", help="with --report: tables only")
    ap.add_argument("--view", choices=["summary", "detail", "both"], default="both",
                    help="with --report/--plots: 'summary' = one block/figure set per room (pooled), 'detail' = per layout and per dock height, "
                         "'both' (default) = summary first, then the detail")
    ap.add_argument("--dir", default="results", help="results folder, relative to the project root (e.g. results/heldout, results/sensitivity_q2)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    apply_overrides(cfg, args.set)
    if args.repair:
        from activenav.evaluation.stats import repair
        for path in sorted(rdir(args, cfg).glob(f"search_study_{args.sensor}*.csv")):
            kept, dropped = repair(path, FIELDS)
            print(f"{path.name}: {kept} rows kept, {dropped} damaged row(s) dropped (original: {path.name}.bak)")
        return
    if args.report or args.plots:
        if args.report:
            report(args, cfg)
        if args.plots or not args.no_plots:
            plots(args, cfg)
    else:
        run(args, cfg)


if __name__ == "__main__":
    main()
