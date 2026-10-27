"""Does camera-only obstacle ranging work well enough? The gate before anything is built on it.

    python scripts/obstacle_diagnostic.py                      # real rendered camera, 40 random poses x 2 tilts
    python scripts/obstacle_diagnostic.py --world room_10x6_p4 --poses 60
    python scripts/obstacle_diagnostic.py --dry-run            # no rendering: exercises the statistics only

For random free poses it renders the head camera, runs the floor-ranging front-end (sensing/floor.py)
and compares every ranged direction with the TRUE distance to the wall or pillar along the same direction
(ground truth is used only to score; this is an evaluation script). It prints
  - range error by true distance (median / 90th percentile of |error|),
  - how often a real obstacle within range is reported (recall) and how often free floor is called an obstacle,
and writes a contact sheet (results/obstacle_diag/sheet.png: image, floor mask, ranged points) and a CSV.

Decision rule: median |error| <= 0.10 m for obstacles within 3 m, recall >= 0.85 and false hits <= 0.05 -> keep
the OpenCV floor-ranging front-end. Otherwise the fallback is a learned floor segmenter (same geometry).
Please send the printed summary and sheet.png.
"""
import argparse
import csv

import cv2
import numpy as np

from activenav.autonomy.obstacle_sensor import GroundGeometry
from activenav.config import apply_overrides, load_config
from activenav.evaluation.worlds import get_world
from activenav.evaluation.synthetic import SyntheticObstacleSensor
from activenav.sensing.floor import FloorRanger
from activenav.types import Command


def settle(env, tilt, steps=60):
    sb = None
    for _ in range(steps):
        sb = env.step(Command(0.0, 0.0, 0.0, tilt), render=False)
    return sb


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="room_10x6")
    ap.add_argument("--poses", type=int, default=40)
    ap.add_argument("--tilts", type=float, nargs="+", default=[0.2, 0.45])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    apply_overrides(cfg, args.set)

    ctx = get_world(cfg, {}, args.world)
    env = ctx.env
    truth = SyntheticObstacleSensor(env, np.random.default_rng(0), cfg)
    geo = GroundGeometry(cfg)
    rng = np.random.default_rng(args.seed)
    x0, x1, y0, y1 = env.room_bounds()
    from activenav.sensing.camera import camera_profile
    cam_name = camera_profile(cfg)                    # another camera profile writes to its own folder
    out = cfg["_root"] / "results" / ("obstacle_diag" if cam_name == "ideal" else f"obstacle_diag_{cam_name}")
    out.mkdir(parents=True, exist_ok=True)
    rows, sheet = [], []
    maxr = float(cfg["obstacles"]["max_range"])

    n = 0
    while n < args.poses:
        p = (rng.uniform(x0 + 0.5, x1 - 0.5), rng.uniform(y0 + 0.5, y1 - 0.5), rng.uniform(-np.pi, np.pi))
        if not ctx.grid.is_free(p[0], p[1]):
            continue
        n += 1
        for tilt in args.tilts:
            env.reset(int(rng.integers(1 << 30)), start_pose=p)
            sb = settle(env, tilt)
            if args.dry_run:
                scan = SyntheticObstacleSensor(env, rng, cfg).scan(sb)
                img = np.full((geo.H, geo.W, 3), 128, np.uint8)
                fr = None
            else:
                sb = env.step(Command(0.0, 0.0, 0.0, tilt), render=True)
                img = sb.image
                fr = FloorRanger(cfg)
                scan = fr.scan(img, sb.view_pan, sb.view_tilt)       # head angles when the image was taken
            x, y, th = env.ground_truth()
            c, s = np.cos(th), np.sin(th)
            o_w = np.array([x, y]) + np.array([[c, -s], [s, c]]) @ scan.origin
            for q, hit in zip(scan.points, scan.hit):
                e_w = np.array([x, y]) + np.array([[c, -s], [s, c]]) @ q
                ang = np.arctan2(e_w[1] - o_w[1], e_w[0] - o_w[0])
                rng_meas = float(np.hypot(*(e_w - o_w)))
                t_true = truth.cast(o_w, ang)
                rows.append(dict(pose=n, tilt=tilt, true_range=t_true, meas_range=rng_meas, hit=int(hit),
                                 true_hit=int(t_true <= maxr), err=rng_meas - t_true))
            if fr is not None and len(sheet) < 12:
                vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
                m = np.zeros((geo.H, geo.W), np.uint8)
                for j, col in enumerate(fr.last_cols):
                    m[:, max(0, col - 2):col + 3] = np.where(fr.last_mask[:, j], 255, 0)[:, None]
                vis[m == 255] = (0.6 * vis[m == 255] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
                sheet.append(cv2.resize(vis, (320, int(round(320 * geo.H / geo.W)))))

    with open(out / "obstacle_diag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        grid = np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)])
        cv2.imwrite(str(out / "sheet.png"), grid)

    r = {k: np.array([row[k] for row in rows], float) for k in rows[0]}
    print(f"{len(rows)} rays from {n} poses x {len(args.tilts)} tilts, world {args.world}, {'DRY RUN (synthetic)' if args.dry_run else 'rendered camera'}")
    print("\nrange error of reported obstacles (measured - true), obstacles within range only")
    both = (r["hit"] == 1) & (r["true_hit"] == 1)
    print(f"{'true range [m]':>16s} {'n':>6s} {'median |err|':>13s} {'p90 |err|':>10s} {'bias':>8s}")
    for lo, hi in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]:
        m = both & (r["true_range"] >= lo) & (r["true_range"] < hi)
        if m.sum():
            e = r["err"][m]
            print(f"{lo:5.1f} - {hi:4.1f}     {m.sum():6d} {np.median(np.abs(e)):13.3f} {np.percentile(np.abs(e), 90):10.3f} {np.mean(e):+8.3f}")
    th = r["true_hit"] == 1
    recall = float(((r["hit"] == 1) & th).sum() / max(1, th.sum()))
    false_hit = float(((r["hit"] == 1) & ~th).sum() / max(1, (~th).sum()))
    med = float(np.median(np.abs(r["err"][both]))) if both.any() else float("nan")
    print(f"\nrecall (real obstacle within {maxr:.0f} m reported): {recall:.2f}    false hits (free floor called obstacle): {false_hit:.3f}")
    print("GATE:", "PASS (keep floor ranging)" if med <= 0.10 and recall >= 0.85 and false_hit <= 0.05
          else f"FAIL (median |err| {med:.3f}, recall {recall:.2f}, false hits {false_hit:.3f}): move to the learned segmenter")
    print("saved", out)


if __name__ == "__main__":
    main()
