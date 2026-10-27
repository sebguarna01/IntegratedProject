"""Watch the robot search an unknown room for marker 10 and dock (no map, no waypoints).

    python scripts/watch_search.py                                   # C-info, large room with 6 pillars, random start
    python scripts/watch_search.py --config B --level large_4_pillars --seed 3
    python scripts/watch_search.py --random-markers --random-dock --seeds 0-9
    python scripts/watch_search.py --start 1.0 -2.0 0.5              # fixed start pose x y yaw
    python scripts/watch_search.py --list
    python scripts/watch_search.py --video results/videos/search.mp4 --seeds 0-2      # offscreen

Left: the head camera. Right: the map the robot is building in its OWN start frame (grey unknown, white free,
dark obstacle), landmarks it found (marker 10 red), its path and frontier target (star), belief (orange with 2 sigma
ellipse) and the head's field of view. `t` overlays the true room (green) so you can judge the map. Configs: 1 B
(fixed head), 2 C-sweep, 3 C-info. Quitting (or a video finishing) writes results/watch_search_*.csv.
Keys: space pause | + / - speed | r restart | n / p seed | 1-3 config | l level | m random markers | d random dock | h marker heights |
      t truth | v record | s screenshot | q quit
"""
import argparse
import time

import cv2

from activenav.config import apply_overrides, load_config
from activenav.evaluation.worlds import ScenarioError
from activenav.evaluation.search import SEARCH_CONFIGS
from activenav.evaluation.viz import CAM_H, CAM_W, HUD_H, MAP_W
from activenav.evaluation.viz import SearchWatcher


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C-info", choices=SEARCH_CONFIGS)
    ap.add_argument("--level", default="large_6_pillars")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", default=None, metavar="N|A-B")
    ap.add_argument("--start", nargs=3, type=float, default=None, metavar=("X", "Y", "YAW"), help="default: random per seed")
    ap.add_argument("--random-markers", action="store_true")
    ap.add_argument("--random-dock", action="store_true")
    ap.add_argument("--heights", action="store_true", help="markers (dock included) hang at different heights per seed")
    ap.add_argument("--sensor", choices=["camera", "synthetic"], default="camera")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--video", metavar="FILE.mp4")
    ap.add_argument("--png", metavar="FILE.png")
    ap.add_argument("--at", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    apply_overrides(cfg, args.set)
    if args.list:
        print("levels:")
        for k, v in cfg["levels"].items():
            print(f"  {k:26s} {v['description']}{'' if v.get('available', True) else '   [not available yet]'}")
        print("configs: B (fixed head) | C-sweep (sweeping head) | C-info (active gaze). A cannot search: it has no way to perceive.")
        return
    seeds = None
    if args.seeds:
        seeds = []
        for tok in args.seeds:
            a, _, b = tok.partition("-")
            seeds += list(range(int(a), int(b) + 1)) if b else [int(a)]
    try:
        w = SearchWatcher(cfg, args.config, args.seed, args.level, args.start, args.sensor, args.speed,
                          args.random_markers, args.random_dock, seeds, args.heights)
    except ScenarioError as e:
        raise SystemExit(f"error: {e}")

    if args.video or args.png:
        import os
        writer = None
        if args.video:
            os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
            writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (CAM_W + MAP_W, CAM_H + HUD_H))
        for si in range(len(seeds) if seeds else 1):
            if si:
                w.seed = seeds[si]
                w.restart()
            while w.result is None:
                w.advance_steps(2)
                if writer:
                    writer.write(w.compose())
                if args.png and args.at is not None and w.session.t >= args.at:
                    break
            if args.png:
                path = args.png if not seeds else args.png.replace(".png", f"_s{w.seed}.png")
                cv2.imwrite(path, w.compose())
                print("saved", path)
            if writer:
                for _ in range(50):
                    writer.write(w.compose())
        if writer:
            writer.release()
            print("saved", args.video)
        w.save_log()
        return

    win = "Active visual navigation: search and dock"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    last = time.perf_counter()
    while True:
        now = time.perf_counter()
        w.advance(now - last)
        last = now
        frame = w.compose()
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(win, frame)
        if w.writer is not None:
            w.writer.write(w.compose())
        k = cv2.waitKey(1) & 0xFF
        if k != 255 and not w.handle_key(k):
            break
        time.sleep(max(0.0, 0.05 - (time.perf_counter() - now)))
    if w.writer is not None:
        w.writer.release()
    cv2.destroyAllWindows()
    w.save_log()


if __name__ == "__main__":
    main()
