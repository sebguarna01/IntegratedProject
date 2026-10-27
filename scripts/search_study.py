"""Search-and-dock study: drop the robot into an unknown room and let it find marker 10 and dock.

    python scripts/search_study.py --sensor synthetic --trials 30 --level large_6_pillars --random-markers --random-dock
    python scripts/search_study.py --sensor camera --trials 20            # real rendered camera + floor ranging
    python scripts/analyse.py                                             # every table and figure, from the CSVs

Configs: B (fixed head), C-sweep (sweeping head), C-info (active gaze). A is not run: odometry alone cannot see
marker 10 or the walls, so it cannot search. Each seed fixes the start pose (random, clear of obstacles), the
hidden encoder errors and, with --random-markers / --random-dock, the marker layout, so every config faces the
same trial. The robot knows no map; ground truth is used only to score (docked = true position within 0.10 m of
the dock point and heading within 5 deg, no contact). Results append to results/search_study_<sensor>.csv (--dir and --tag choose the file) and
finished rows are skipped, so an interrupted run resumes.
"""
import argparse
import csv

from activenav.config import apply_overrides, load_config
from activenav.evaluation.worlds import ScenarioError
from activenav.evaluation.search import SEARCH_CONFIGS, make_search_session

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
    # a camera profile other than the ideal one is part of the label, so its rows never mix with the ideal camera's
    cam = getattr(args, "camera", "ideal")
    return ("randmk" if args.random_markers else "fixed") + ("+randdock" if args.random_dock else "") + \
        ("+heights" if args.heights else "") + (f"+dock{dock_height:g}" if dock_height is not None else "") + \
        ("" if cam == "ideal" else f"+{cam}")


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
    ap.add_argument("--repair", action="store_true", help="bring every results file of this sensor up to date (current header, damaged rows dropped; originals kept as .bak) and exit")
    ap.add_argument("--dir", default="results", help="results folder, relative to the project root (e.g. results/heldout, results/sensitivity_q2)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    apply_overrides(cfg, args.set)
    from activenav.sensing.camera import camera_profile
    args.camera = camera_profile(cfg)                      # head camera profile (--set camera=d435i)
    if args.repair:
        from activenav.evaluation.stats import repair
        for path in sorted(rdir(args, cfg).glob(f"search_study_{args.sensor}*.csv")):
            kept, dropped = repair(path, FIELDS)
            print(f"{path.name}: {kept} rows kept, {dropped} damaged row(s) dropped (original: {path.name}.bak)")
        return
    run(args, cfg)


if __name__ == "__main__":
    main()
