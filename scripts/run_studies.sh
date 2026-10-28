#!/usr/bin/env bash
# Every trial of the final study, in one place. One time cap (CAP seconds) for all of it, so no run is censored differently.
#   SENSOR=camera JOBS=4 bash scripts/run_studies.sh          # everything (slow: hours; JOBS processes in parallel)
#   SENSOR=synthetic TRIALS=2 JOBS=2 bash scripts/run_studies.sh   # quick smoke test of the plumbing
# Re-running continues where it left off (finished seeds are skipped). One --tag per command, so no two processes share a file.
# Blocks: main (levels, seeds 0..), height (dock height), heldout (seeds 5000.., never tuned on), tilt (B with its tilt free: separates
# the value of tilt from the value of panning), d455pilot / d455full (the RealSense D455 camera profile; not in the default BLOCKS), lqr (the LQR follower on the same seeds), sens (estimator noise Q and R scaled by 0.5 and 2).
set -u
SENSOR=${SENSOR:-camera}; TRIALS=${TRIALS:-20}; CAP=${CAP:-300}; JOBS=${JOBS:-2}
BLOCKS=${BLOCKS:-"main height heldout tilt lqr sens gaze mixed tallhard height_more false02 heldout_all"}
PY="python scripts/search_study.py --sensor $SENSOR --trials $TRIALS --max-time $CAP"
HEIGHTS="0.2 0.3 0.45 0.6"
LEVELS="large_room large_6_pillars large_9_pillars l_room two_rooms zigzag corridor alcove cluttered_14 sparse_noisy_odom people_constant people_random"
cmds=()
for b in $BLOCKS; do case $b in
  main)    for l in $LEVELS; do cmds+=("$PY --level $l --tag _$l"); done ;;
  height)  cmds+=("$PY --level large_6_pillars --dock-height $HEIGHTS --dir results/height --tag _large_6_pillars") ;;
  heldout) for l in large_6_pillars cluttered_14; do cmds+=("$PY --level $l --dir results/heldout --tag _$l --set seed=5000"); done
           cmds+=("$PY --level large_6_pillars --dock-height $HEIGHTS --dir results/heldout --tag _height --set seed=5000") ;;
  lqr)     # the LQR path follower on the same seeds as the go_to_point runs above (paired comparison)
           for l in large_6_pillars cluttered_14; do cmds+=("$PY --level $l --dir results/lqr --tag _$l --set search.follow=lqr"); done
           cmds+=("$PY --level large_6_pillars --dock-height $HEIGHTS --dir results/lqr --tag _height --set search.follow=lqr") ;;
  false02) # replicate the 0.2 m dock height on fresh seeds (the one development-seed result that did not hold on held-out seeds)
           cmds+=("$PY --level large_6_pillars --trials 40 --dock-height 0.2 --dir results/dock02 --tag _large_6_pillars --set seed=6000") ;;
  height2) # does the dock-height result hold in other rooms? Seeds 5000.. (never tuned on), the two heights that matter most
           for l in large_9_pillars two_rooms; do
             cmds+=("$PY --level $l --dock-height 0.2 0.45 --dir results/height_rooms --tag _$l --set seed=5000"); done ;;
  gaze)    # head behaviour (pan reversals, dwell, looking where it is going) in every room, same seeds as the main block: the trials are
           # deterministic, so outcomes must reproduce the main block exactly (a check), and the gaze columns come for free.
           # Longest explorations first, so stopping early still leaves the informative rooms.
           for l in cluttered_14 large_9_pillars two_rooms zigzag l_room corridor sparse_noisy_odom people_random people_constant large_6_pillars large_room alcove; do
             cmds+=("$PY --level $l --dir results/gaze --tag _$l"); done ;;
  mixed)   # realistic rooms: EVERY marker (the dock included) hangs at a random height from marker_heights (0.2..0.6 m) per seed. A level head
           # cannot see the tall ones, so this tests state estimation (landmarks seen), not only docking.
           for l in large_6_pillars large_9_pillars cluttered_14; do cmds+=("$PY --level $l --heights --dir results/mixed --tag _$l --set seed=8000"); done ;;
  tallhard) # tall dock markers in the hardest rooms (C-info only: B and C-sweep cannot dock a tall marker, and each would burn the full cap)
           for l in cluttered_14 sparse_noisy_odom people_random; do
             cmds+=("$PY --level $l --configs C-info --dock-height 0.45 0.6 --dir results/tall_hard --tag _$l --set seed=9000"); done ;;
  height_more) # more dock heights in large_6_pillars: 0.25 and 0.35 for all three configs, 0.9 for C-info only (B and C-sweep cannot dock
           # above 0.33 m and each failure burns the full cap). Development seeds (0..) and held-out seeds (5000..).
           cmds+=("$PY --level large_6_pillars --dock-height 0.25 0.35 --dir results/height --tag _more")
           cmds+=("$PY --level large_6_pillars --configs C-info --dock-height 0.9 --dir results/height --tag _more09")
           cmds+=("$PY --level large_6_pillars --dock-height 0.25 0.35 --dir results/heldout --tag _height_more --set seed=5000")
           cmds+=("$PY --level large_6_pillars --configs C-info --dock-height 0.9 --dir results/heldout --tag _height_more09 --set seed=5000") ;;
  heldout_all) # every room on held-out seeds (the rooms already done are skipped): per-room comparisons on 40 seeds, never tuned on
           for l in $LEVELS; do cmds+=("$PY --level $l --dir results/heldout --tag _$l --set seed=5000"); done ;;
  d455pilot) # a small first look at the RealSense D455 camera profile (global shutter, 90 x 59 deg, real lens, latency 0.06 s; the
           # robot's algorithms are unchanged). 10 seeds, same seeds as the ideal-camera runs, so every trial can be paired with its ideal twin:
           # three rooms at the standard marker heights, then the dock-height effect (0.3 / 0.45 / 0.6 m) in large_6_pillars.
           for l in large_6_pillars cluttered_14 people_random; do
             cmds+=("$PY --trials 10 --level $l --dir results/d455_pilot --tag _$l --set camera=d455"); done
           cmds+=("$PY --trials 10 --level large_6_pillars --dock-height 0.3 0.45 0.6 --dir results/d455_pilot --tag _height --set camera=d455") ;;
  d455full) # the D455 as a robustness check on the ideal-camera study. Same folder as d455pilot (finished seeds are skipped, so the two
           # never double count). TRIALS=10 gives seeds 0..9; re-run with TRIALS=20 to add 10..19. Order = priority: rooms first, then the
           # dock heights the pilot did not cover (0.2 / 0.25 / 0.35 m, all configs; 0.9 m, C-info only), then mixed marker heights.
           for l in $LEVELS; do cmds+=("$PY --level $l --dir results/d455_pilot --tag _$l --set camera=d455"); done
           cmds+=("$PY --level large_6_pillars --dock-height 0.2 0.25 0.35 0.3 0.45 0.6 --dir results/d455_pilot --tag _height --set camera=d455")
           cmds+=("$PY --level large_6_pillars --configs C-info --dock-height 0.9 --dir results/d455_pilot --tag _height09 --set camera=d455")
           for l in large_6_pillars large_9_pillars cluttered_14; do cmds+=("$PY --level $l --heights --dir results/d455_pilot --tag _mixed_$l --set camera=d455 seed=8000"); done ;;
  tilt)    cmds+=("$PY --level large_6_pillars --configs B --dock-height $HEIGHTS --dir results/tilt_free --tag _B_tilt_free --set search.fixed_tilt=null") ;;
  sens)    # Q: odometry process noise, R: marker measurement noise
           Q05="ekf.k_ds=0.01 ekf.k_dth=0.015 ekf.k_dth_ds=0.005";  Q2="ekf.k_ds=0.04 ekf.k_dth=0.06 ekf.k_dth_ds=0.02"
           R05="ekf.range_sigma=[0.002,0.001] ekf.bearing_sigma_rad=0.0025 ekf.normal_sigma_deg=[0.15,0.03]"
           R2="ekf.range_sigma=[0.008,0.004] ekf.bearing_sigma_rad=0.01 ekf.normal_sigma_deg=[0.6,0.12]"
           for v in Q05 Q2 R05 R2; do
             cmds+=("$PY --level large_6_pillars --configs B C-info --dir results/sens_$v --tag _$v --set ${!v}")
           done ;;
esac; done
# NUL-separated and passed as $0 (macOS xargs refuses to substitute -I{} into lines longer than 255 bytes)
printf '%s\0' "${cmds[@]}" | xargs -0 -n1 -P "$JOBS" bash -c 'eval "$0"'