# Search and dock: the robot knows nothing

**Task.** Drop the robot at any pose in an unknown room. It has to find ArUco marker 10 and dock in front of it
(true position within 0.10 m of the dock point, 0.5 m out along the marker's normal, and heading within 5 deg of
facing the marker), without touching anything.

**What it is given.** Wheel odometry, the head camera, the marker size (0.30 m), its own geometry, the id of the
dock marker and the dock stand-off. **What it is not given.** The room size, obstacles, any marker position
(marker 10 included), its own pose in the room. Its frame is its start pose.

**Pipeline (src/activenav/autonomy, estimation/slam.py).**
1. `estimation/slam.py`: EKF-SLAM. Robot pose and odometry calibration (c, s, k) plus every marker seen as a landmark
   (position and normal direction). Same noise model, gating and correlated-frame throttle as the known-map EKF.
2. `sensing/floor.py`: ground-plane ranging. Floor *chroma* (a, b in Lab) is learned from the strip at the
   bottom of the image, and a lightness step against the pixels just below also ends the floor. Lightness is ignored on
   purpose: the room lamp is a point source, so the floor goes from L~220 near the lamp to L~125 elsewhere, and a wall in
   shade is about as light as the floor. A first version matched the full Lab colour and failed the real-camera gate
   (false hits 50%, recall 74%); the chroma model passes (median error 0.020 m, false hits 1.9%, recall 94%). Limit: a
   grey obstacle with no lightness step against the floor is invisible. In each column the floor boundary row is converted to a distance with the camera height (0.19 m) and tilt.
3. `autonomy/occupancy.py`: log-odds grid in the start frame. Free space stops 2 sigma short of each hit.
4. `autonomy/explore.py`: frontier exploration through known-free cells (unknown is not traversable), margin = 0.15 m
   + 2 sigma of position uncertainty (max 0.30 m).
5. `autonomy/agent.py`: SCAN, EXPLORE, APPROACH, DOCK. Docking servoes on the live marker-10 measurements and judges
   the finish on the average of the last five frames.
6. `autonomy/search_gaze.py`: B (fixed head, turns its body for the initial look-around), C-sweep, C-info (unknown map in
   view + landmarks when position uncertainty is high). When the next metre of path is near unknown space or obstacles,
   the head must keep the direction of travel in view.

**Scoring.** `evaluation/search.py` reads ground truth only to score; pose and map errors are measured in the start
frame. Config A cannot perceive, so it cannot search and is not run.

**Limits to state in the write-up.** Floor ranging needs a floor that looks different from walls and obstacles (true in
the simulator, not in general); the synthetic obstacle sensor is a proxy for it (run `scripts/obstacle_diagnostic.py`
on the real camera first); static obstacles only so far.

**Scripts.** `watch_search.py` (live viewer), `search_study.py` (batch study + report),
`obstacle_diagnostic.py` (camera ranging accuracy gate).

## Landmark recovery
A landmark first seen from far away and off-axis can be born with a wrong normal; the gate then rejects every better
close-up. After 8 consecutive gated updates at range <= 2 m the landmark is re-initialised from the current view
(`SlamEKF._reinitialise`). Found on the camera study (seed 12: the robot stood 0.9 m from marker 10 and never docked).

## Marker heights (`--heights`, key `h` in the viewer)
Every marker, the dock marker included, hangs at a centre height drawn per seed from `marker_heights` (0.2, 0.3, 0.45,
0.6 m; the plate is 0.40 m, so 0.2 m is the lowest it can go). The camera is 0.19 m up and sees +-30 degrees vertically,
so at the default search tilt (0.2 rad down) a 0.6 m marker is cut off when the robot is closer than about 1.7 m.
- The marker measurement now carries `height` (from the 3D pose). The agent keeps the median per marker.
- Docking sets the tilt from the measured height and the range so the whole marker stays in view and is centred when
  close (`autonomy/marker_view.py`). Head tilt range is -0.6 .. 1.0 rad.
- C-sweep sweeps the PAN only; its tilt stays at `tilt_search`. B is fixed.
- C-info chooses pan and tilt (`gaze.tilt_cands`). Terms: unknown floor in view (looking up hides the near floor), not-yet-
  searched wall/unknown space out to marker range, weighted by the share of plausible heights that fit the vertical view,
  and known landmarks (with their measured height) when position uncertainty is high. It keeps a coarse memory of what
  it has searched, because the floor is only mapped to 3 m while markers are detected to ~5 m. Near known obstacles it may
  only use tilts that see the floor from 0.45 m (`near_limit`); an earlier version tilted up beside a pillar and clipped it.

## Layouts
Interior wall blocks (`blocks: [[x0, y0, x1, y1], ...]` in a world) are full-height boxes: partitions, filled-in corners.
Planner grid, line-of-sight tests, ray casting, clearance and the viewer all treat obstacles as circles (pillars) or
rectangles (blocks). Markers hang on the outer walls only, never where a block covers the wall.
| level | world | what it adds |
|---|---|---|
| large_9_pillars | 10 x 10 m, 3 x 3 pillars, 10 markers | clutter, long occlusions |
| l_room | L-shape (10 x 10 with a quarter filled in), 8 markers | a corner nothing can see round |
| two_rooms | two 6 x 6 rooms, 1.4 m doorway, dock in the far room | doorway as the only way through |
| zigzag | 12 x 6 with two staggered partitions, 9 markers | S-shaped route |
A narrow doorway gives a short ragged frontier; frontier cells up to 2 coarse cells apart are merged before counting.

## Mapping and navigation robustness (added with the harder layouts)

- **Ray-wedge fill.** Between two neighbouring free rays whose endpoints are within 0.4 m, the triangle is also marked
  free. Without it, a doorway interior stays as unknown stripes between rays and no frontier is ever large enough.
- **Doorway frontiers.** Frontier clusters are merged with a 5x5 dilation; if no frontier is found on the first look,
  the minimum cluster size drops to one cell on the retry.
- **Planner clearance.** The required clearance is `radius + margin`, and the floor on the allowance is
  `radius + 0.03` (robot circumscribed radius 0.25 m). A lower floor let the planner ratchet in next to pillars.
- **Safety hysteresis.** The reflex that stops the robot when the map blocks the way ahead uses `radius`, 3 cm below the
  planner floor, so the planner never hands over a path the reflex immediately refuses.
- **Stuck detector.** If the robot has not moved for `search.stuck_s` (8 s) while a path exists, the target is
  blacklisted, and it backs off (if clear behind) or turns for 3 s, then replans.
- **C-sweep** is a pure pan triangle wave (`sweep_amp`, `sweep_period`) at fixed tilt; only C-info chooses tilt.

## Study metrics and the dock-height factor

`scripts/search_study.py` now writes, per trial, the dock marker's height (`dock_height_m`) and these extra columns
(older CSVs without them still load; the report ignores the missing values):

| group | columns |
|---|---|
| estimation | `nees_mean` (3 dof, about 3 when the filter is honest), `nees_consistent` (share of time inside the 95% chi-square band), `sigma_xy_mean_m`, `sigma_xy_final_m`, `final_pos_err_m` |
| search | `coverage_30s/60s/90s` (last value held after the run ends), `approach_s` (first sight to end), `dock_seen_frac` (share of camera frames, after first sight, that show marker 10) |
| effort | `head_pan_travel_rad`, `head_tilt_travel_rad` (commanded angles, not encoder readings, which add noise), `tilt_mean_rad`, `tilt_std_rad`, `n_replans` |
| safety | `near_time_s` (time with the robot's edge under 0.10 m from an obstacle), `blocked_s` (safety reflex holding the robot back), `n_recover` (stuck recoveries) |

`--dock-height Z` fixes the height of marker 10 only (0.2 m to wall height minus 0.2 m); the other markers keep their
heights (or use `--heights`). Rows carry the layout label `+dockZ`, so they never mix with other runs in the report.
The report prints secondary metrics, "docked within 60/100/150 s" (a failure is never counted as slow), and docking success by
dock height. A seed that appears in several result files counts once (the newest file wins).

## Locked head tilt for B and C-sweep

`search.fixed_tilt` (default 0.0 rad) is the head tilt that B and C-sweep hold for the entire run: exploring, approaching and
docking. Only C-info chooses its own tilt. (Set it to `null` to get the older behaviour: `tilt_search` while exploring and
`tilt_dock` close to the marker.) The camera floor-ranging gate passes at a locked 0 deg tilt on `room_two_rooms`, `room_l`
and `room_10x10_p9`. With the plate seen at the 0.5 m standoff, a locked 0 deg tilt shows a whole marker only up to about
0.33 m centre height.

## More environments (levels)

| level | world | what it tests |
|---|---|---|
| `corridor` | 16 x 2.4 m, 8 markers, dock on the far end wall | long travel with side markers always seen obliquely; little room to manoeuvre |
| `alcove` | 10 x 6 m, dock at the back of a 1.6 m wide, 2.5 m deep niche | the final approach has to drive down a short dead-end and stand 0.5 m from the back wall |
| `cluttered_14` | 14 x 14 m, 16 pillars (3.2 m pitch) + 6 crates, 11 markers; every gap >= 1.5 m | large, cluttered, many occlusions; slow exploration |
| `people_constant` | 10 x 6 m, 4 pillars, 7 markers, **three people** walking straight at 0.4 m/s | moving obstacles: the map goes stale, people hide markers, and the robot must not touch them |
| `people_random` | same room; the people **wander** (random turns, speed drifting around 0.4 m/s) | as above, but a person's path cannot be predicted from two sightings |
| `sparse_noisy_odom` | the 10 x 6 m, 6-pillar room with only 3 markers, wheel scale error 4x and encoder noise 3x worse (track width unchanged) | state estimation under long stretches without a marker; the EKF noise settings are left unchanged on purpose |

**Walking people** (`people_*` levels; `worlds.<name>.people`): each person is a 0.2 m radius, 1.7 m tall cylinder moved by the
simulator (`sim/people.py`). They bounce off walls, pillars and each other and never steer around the robot, with one courtesy:
a person stops while the robot is within `yield_gap` (0.35 m, surface to surface); if the robot is still that close after
`yield_wait` (3 s) the person walks directly away from it until 0.3 m clear of the gap and then carries on. So nobody walks into
a standing robot and nobody blocks the robot for ever (`yield_gap: 0` removes the courtesy). People start at random free spots at least 1.5 m from the robot's start; the
seed fixes their start and their walk, so every config meets the same people on the same seed. The agent only sees them through
the camera (floor ranging, like any obstacle); they also hide markers. Touching a person ends the trial as `contact`, with
`person_contact = 1` telling it apart from a wall. Extra metrics: `min_person_gap_m` (closest surface-to-surface distance to a
person over the trial) and `person_near_s` (time with one closer than 0.30 m); both are nan in rooms without people. The gap is measured from the robot's 0.25 m planning radius (like `min_clearance_m`), so it can read a few centimetres negative without a physical touch; `person_contact` is the real collision flag.

A level can carry `set: ["a.b=value", ...]` (applied like `--set`); `sparse_noisy_odom` uses it for
`noise.wheel_scale_std` and `noise.encoder_mult_std`. All four pass the camera floor-ranging gate
(`scripts/obstacle_diagnostic.py --world <name>`): recall 0.95-0.98, false hits <= 0.015.
Run them like any other level, e.g. `python scripts/search_study.py --sensor camera --level corridor --dock-height 0.25 0.3 0.45 0.6 --trials 20 --tag _corridor`.


---

## Study metrics: what each column means, and which ones to lead with

Every trial of `scripts/search_study.py` writes one CSV row. `--report` prints the tables and writes charts to
`results/figures/study_<sensor>/` (`--plots` for charts only, `--no-plots` for tables only).

### Two views: `--view summary | detail | both` (default both)

| | text report | charts |
|---|---|---|
| **summary** | one block per room, every layout and dock height pooled; success by dock-marker height band (level camera sees it <= 0.30 m / borderline 0.35 m / tilt needed >= 0.40 m); **paired comparison by room** | `summary/01_scorecard_all_rooms.png`, `02_headline_by_room.png` (rooms x the four headline metrics), `03_success_by_dock_height_band.png`, **`04_paired_difference_vs_B.png`** (forest plot: all rooms per height band, then each room in the level band), **`05_how_trials_ended.png`** (failure taxonomy) |
| **detail** | one block per layout (per dock height), paired tests, success by height, **paired comparison by room and dock-height band** | `detail/*.png` (per-level panels, `paired_difference_by_height_band.png`, `effects_of_dock_height.png`: every metric against the height, `docked_over_time`, estimation, safety and effort) and `detail/rooms/<level>_*.png` |

Read the summary first; go to the detail to explain a number. The detail's `effects_of_dock_height` is also drawn per room.

### The four headline (umbrella) metrics: summary/01_scorecard_all_rooms.png and summary/02_headline_by_room.png

| # | Metric | Column(s) | Why it leads |
|---|---|---|---|
| 1 | **Docking success** | `docked`, `outcome` | The task itself. Binary per trial, reported with a 95% Wilson interval. |
| 2 | **Time to dock** | `duration_s` (successful runs) and "docked within 60/100/150 s" | Speed. A failure is never counted as slow: it is simply not docked by then. |
| 3 | **State-estimation quality** | `nees_consistent`, `nees_mean`, `final_pos_err_m`, `landmark_err_m` | The research question's second half. Consistency says whether the filter's uncertainty is honest. |
| 4 | **Safety** | `min_clearance_m`, `outcome == contact`, `near_time_s` | A config that docks by brushing obstacles has not really succeeded. |

Pre-register these four as the primary comparison; everything else is there to explain *why* the primary numbers differ.

### Outcome and speed
| Column | Meaning |
|---|---|
| `outcome` | docked / contact (touched a wall or obstacle) / false dock (claimed done but outside 0.10 m or 5 deg) / gave up / timeout |
| `docked` | 1 if outcome is docked |
| `duration_s` | Run length; for a success, the time to dock |
| `time_to_first_sight_s` | Time until marker 10 was first measured (blank/nan if never) |
| `approach_s` | First sight to the end of the run (how long the approach and docking took) |
| `path_length_m` | Distance driven |
| `miss_m`, `yaw_err_deg` | Final position error to the dock point and heading error (the dock needs 0.10 m and 5 deg) |
| `dock_height_m` | Centre height of marker 10. Always written, so any study can be filtered or split by height |

### State estimation (ground truth is used only to score)
| Column | Meaning |
|---|---|
| `nees_mean` | Normalised estimation error squared, 3 degrees of freedom: about 3 when the filter's covariance is honest. Well above 3 = overconfident; well below = too cautious |
| `nees_consistent` | Share of the run with NEES inside the 95% chi-square band (0.22 to 9.35). Higher is better |
| `pos_rmse_m` | Root-mean-square position error over the run |
| `final_pos_err_m` | Position error at the end (what matters for docking) |
| `sigma_xy_mean_m`, `sigma_xy_final_m` | The filter's own position uncertainty (what it believes) |
| `n_landmarks`, `landmark_err_m` | Markers mapped and their mean position error |

### Search and mapping
| Column | Meaning |
|---|---|
| `coverage`, `coverage_30s/60s/90s` | Share of the true free floor marked free in the map at the end / at that time (last value held if the run ended earlier) |
| `map_precision` | Share of cells marked occupied that really are obstacles |
| `dock_seen_frac` | Share of camera frames, after first sight, that contain marker 10 (how well the head keeps the target in view) |

### Effort and safety detail
| Column | Meaning |
|---|---|
| `head_pan_travel_rad`, `head_tilt_travel_rad` | Total commanded head motion (commanded, because encoder readings add noise) |
| `tilt_mean_rad`, `tilt_std_rad` | Where the head pointed (B and C-sweep: constant) |
| `near_time_s` | Seconds with the robot's edge under 0.10 m from an obstacle |
| `min_person_gap_m` | People levels only: closest surface-to-surface distance to a walking person over the trial (negative = touching) |
| `person_near_s` | People levels only: seconds with a person closer than 0.30 m |
| `person_contact` | People levels only: 1 if the trial ended by touching a person (the outcome is still `contact`) |
| `blocked_s` | Seconds the safety reflex held the robot back |
| `n_recover` | Stuck-detector recoveries (deadlocks broken) |
| `n_replans` | Exploration replans |
| `min_clearance_m` | Closest approach of the robot's edge to an obstacle (below 0 = contact) |

### Paired comparison (`evaluation/paired.py`; the last tables of `--report`, figures 04 and 05)
Every config meets the same start, layout, dock height and people on the same seed, so the question to ask of two configs is
"on the same trial, who did better?". A **pair** is one (room, layout, seed) that both configs finished. For `C-info vs B`,
`C-sweep vs B` and `C-info vs C-sweep` the report gives, per room (all rooms pooled first) and per room x dock-height band:

| column | meaning |
|---|---|
| `W / L / both / neither` | pairs where A docked and B did not / the reverse / both docked / neither |
| success diff, pp | A minus B in percentage points, with a 95% interval; exact McNemar p on the W/L pairs |
| time to dock diff | median of A minus B in seconds, **only pairs where both docked** (a failed run has no docking time); negative = A faster |
| position RMSE diff, closest approach diff | median paired differences over all pairs; RMSE negative = A estimates better, closest approach positive = A keeps more room |

**Height bands.** Three, not two. With the head locked level (camera 0.19 m up, 0.30 m black square) the whole square is in view
from the 0.5 m standoff only up to about 0.33 m: at <= 0.30 m a level head can dock (the band where B, C-sweep and C-info are
comparable), at 0.35 m the top is cut off so a level head mostly fails, and from 0.40 m up it cannot see the marker at all (B and
C-sweep lock the tilt, so only C-info docks there: that band tests tilt control, not pan control).

The intervals resample **whole seeds**, not single trials: the several dock heights of one seed share a start and a room, so
they are not independent. Under 4 seeds the interval is withheld (`few seeds`). The forest plot (fig 04) flips signs so that
right of zero always means "better than B"; fig 05 shows how every trial ended (docked, false dock, contact, timeout, gave up).
Read a difference as an effect size with an interval, not as a verdict: a statistically clear 5 mm RMSE gain is still 5 mm. With many
rooms, metrics and bands there are many intervals, so lead with docking success and the headline metrics and treat the rest as exploratory.

### Comparing configs
All configs run the same seeds (same start pose and layout), so compare paired. For success, the 95% interval on each rate
and a Fisher test on the pair; for time, the "docked within T" curves (fig3) rather than a mean that failures distort.
With this many metrics, treat the four headline metrics as the confirmatory ones and the rest as exploratory.


### Damaged result files
A run killed in the middle of a write leaves a truncated row, and files written over several versions have an old header.
The study now reads them tolerantly (`evaluation/stats.py`: extra columns keep their names, a truncated row is dropped) and brings
the header up to date before appending. `python scripts/search_study.py --sensor camera --repair` does this for every results
file at once (originals are kept as `.bak`).

### A common time cap, and re-running rows
`--max-time` only truncates a run; the run itself is deterministic. So any run can be analysed as if it had a shorter cap:
`--report --cap 200` counts every run longer than 200 s as a timeout (columns that summarise the whole run are blanked for those rows).
It cannot lengthen a run, so 200 s is the longest cap you can impose on files that were run with 200 s. In the camera dataset the cap
changes the size of the gaps, not the ranking: median docking time is 74 s and 99% of docked runs finish within 235 s.
`--redo` deletes the existing rows of exactly the command you give (level, configs, dock heights, seeds; the original file is kept as
`.bak`) and runs them again, e.g. after a bug fix. Never run two processes on the same `--tag`: they append to one CSV and can interleave
a row; use one tag per process.

### Reading the numbers
Trials of one seed at different dock heights are not independent: until the dock marker is first seen the robot behaves exactly
the same, so one bad event (a contact, say) can appear in every height of that seed. Wilson intervals on pooled trials are therefore
a guide only; the paired comparison resamples seeds. Count incidents per seed when you quote them. Keep the time cap (`--max-time`)
the same in every room you compare: a timeout at 200 s and one at 900 s are different things.


---

# Model, controllability and observability

Code: `src/activenav/analysis.py`, run `python scripts/model_analysis.py` (writes `results/model_analysis.csv`, `results/figures/model_analysis.png`). Tests: `tests/test_analysis.py`. The numbers below are from that run; the observability analysis uses the filters' own Jacobians (`EKF._predict_parts`, `predict_measurement`, `SlamEKF._expected`).

## Platform model

State x = [x, y, theta] (world frame, x forward, y left, theta counter-clockwise), input u = [v, w]:

    x' = [v cos(theta), v sin(theta), w],    wheel speeds from  v = r (wl + wr) / 2,  w = r (wr - wl) / W

Assumptions: planar motion, no wheel slip, rigid base, wheel radius r and track width W constant within a run but wrong by a per-trial error (`noise:` in the config). Odometry integrates the encoder increments (ds, dtheta) with the midpoint rule; the EKF corrects them with three slow states, c (heading drift per metre), s (distance scale), k (turn scale): ds' = ds (1 + s), dtheta' = dtheta (1 + k) - c ds.

## Linearisation and controllability

About a run at speed v0 and heading theta0 the pose model has A = [[0,0,-v0 sin th0],[0,0,v0 cos th0],[0,0,0]], B = [[cos th0, 0],[sin th0, 0],[0, 1]]. The controllability matrix has rank 3 for v0 > 0 and rank 2 at v0 = 0: a robot at rest cannot step sideways (the nonholonomic constraint). The nonlinear system is still controllable by manoeuvring, but a linear controller about a standing pose is not defined, which is why the tracking controller is designed about a moving reference.

For path following use the error coordinates e = [e_y, e_theta] (lateral offset, heading error to the path tangent), path curvature kappa and input dw = w - v kappa:

    e_y' = v sin(e_theta),    e_theta' = dw - v kappa (cos(e_theta) / (1 - kappa e_y) - 1)
    A = [[0, v], [-v kappa^2, 0]],    B = [[0], [1]]        (checked against finite differences in the tests)

C = [B, AB] = [[0, v], [1, 0]], so det C = -v: rank 2 for any v > 0 (also on a turn, kappa does not change this), rank 1 at v = 0. At 0.2 m/s with zero-order-hold sampling at 0.02 s: Ad = [[1, 0.004], [0, 1]], Bd = [[4e-5], [0.02]], rank 2, and the open-loop eigenvalues sit at z = 1 (a double integrator). Slow driving is weakly controllable (smallest singular value 1.4e-5 at 0.05 m/s against 5.7e-5 at 0.2 m/s), so a gain that works at 0.2 m/s needs more effort at crawl speed.

## Observability of the pose (known-map EKF)

Observability matrix along 3 m of odometry (straight, left turn, straight) with the markers always in view; columns scaled by the filter's initial standard deviations. Rank out of states:

| Measurements | 1 marker, pose | 1 marker, pose + c, s, k | 2 markers, pose | 2 markers, pose + c, s, k |
|---|---|---|---|---|
| none (odometry only) | 0/3 | 0/6 | 0/3 | 0/6 |
| bearing | 2/3 | 4/6 | 3/3 | 6/6 |
| range + bearing | 2/3 | 5/6 | 3/3 | 6/6 |
| range + bearing + marker normal | 3/3 | 6/6 | 3/3 | 6/6 |

Reading: odometry alone observes nothing about the pose, so error only grows. One marker with range and bearing leaves one direction free (rotation about the marker); the marker normal closes it, which is why `ekf.use_normal` is on. Two markers need no normal. The odometry-calibration states are observable once the pose is, provided the robot turns and drives.

When the marker leaves view the measurement rows vanish. With no marker in view for 5 m the filter's own 1-sigma grows to x 0.055 m, y 0.39 m, theta 8.7 deg (driven mostly by the prior on c), and the next sighting pulls it back. Idealisation: the analysis assumes markers stay in view; the real condition is that enough are in view often enough, which is what the gaze policy is for.

## EKF-SLAM and its gauge

With landmarks in the state, every measurement is relative, so moving the robot start pose and all landmarks together by any rigid motion changes no measurement. The observability matrix has three unobservable directions (global x, y and rotation): rank 6/9 for one landmark and 9/12 for two, with the start pose free. `SlamEKF.reset` fixes the start pose with a tiny covariance, which anchors the frame: rank 9/9 and 12/12. The map is therefore expressed in the start frame and says nothing about the world frame, which is all docking needs.


---

# Camera calibration and frame chain

Code: `scripts/calibrate_camera.py`, helpers in `src/activenav/sensing/camera.py` (`board_points`, `calibrate`, `pose_difference`), tests `tests/test_calibration.py`. Needs a GL context (`MUJOCO_GL=glfw` or `egl`). Outputs: `results/camera_calibration.csv`, `results/figures/camera_calibration.png`.

## Method

The script builds a small scene (floor plus one 9 x 6 inner-corner checkerboard, 0.05 m squares, one white square of border) around the real robot, then renders the real head camera (640 x 480, with the configured pixel noise) from 30 random robot poses, pan and tilt angles in which all 54 corners are found and inside the image. It runs `cv2.findChessboardCorners`, `cornerSubPix` and `cv2.calibrateCamera`, once with the full model (k1, k2, p1, p2, k3) and once pinhole-only (distortion fixed at zero).

## Result (seed 0, 30 views)

| | analytic (`fovy_deg` = 60) | calibrated, full model |
|---|---|---|
| fx, fy [px] | 415.69, 415.69 | 415.53, 415.50 |
| cx, cy [px] | 319.50, 239.50 | 319.94, 238.65 |
| distortion k1 k2 p1 p2 k3 | 0 | -0.004, 0.010, -0.001, 0.000, -0.009 |
| reprojection RMS [px] | | 0.090 (pinhole only: 0.091) |

The calibrated matrix agrees with the analytic one to within 1 px, and the pinhole-only model fits as well as the full model, so the distortion terms are noise: the renderer is an ideal pinhole and the pipeline's zero-distortion assumption (`perception` / `sensing/markers.py`) is justified for the simulated camera. Calibration therefore confirms the analytic matrix; the config keeps the analytic values and records the calibrated ones. This says nothing about a real lens: a real camera needs its own calibration, and distortion would then have to be passed to the ArUco and floor-ranging code.

## Frame chain

For every view the camera pose in the world is compared with MuJoCo's own camera pose (ground truth, used only to score):

| Camera pose from | translation, median (max) | rotation, median (max) |
|---|---|---|
| base pose and noisy head encoders through `camera_to_base` | 0.2 mm (0.3) | 0.16 deg (0.29) |
| checkerboard `solvePnP` with the calibrated camera | 4.1 mm (12.9) | 0.19 deg (0.50) |

The pan-tilt kinematics in `sensing/camera.py` match the simulator to 0.3 mm; the rotation error is the head encoder noise (0.002 rad per joint). The PnP numbers are larger because they inherit corner noise and board distance, and they show the image-to-world chain (calibrated camera, board frame, world) is consistent. The board frame uses the same plate convention as the marker plates, so this also checks that convention.

Limits: one rendered scene, no lens model, no motion blur or exposure change; the check says the simulated chain is correct, not that a real camera will behave.


## Path tracking: LQR vs go-to-point

`control/lqr.py` linearises the path-error model (see "Model, controllability and observability") and designs a discrete LQR with `solve_discrete_are` at 0.02 s, using Bryson's rule (max errors 0.05 m, 0.20 rad, integral 0.05 m, max effort 0.5 rad/s). The state is [e_y, e_theta, integral of e_y]; the command is w = v_ref*kappa + dw with dw = -K x. Integration is conditional (frozen when w saturates or |e_y| > 0.10 m). Gain at v = 0.2 m/s: K = [21.45, 3.78, 9.62], closed-loop poles |z| 0.95 to 0.985.

`scripts/lqr_tracking.py` drives a stadium loop (10.28 m) in the default 6x4 room for 3 laps with the known-map EKF (with bias states) and the synthetic marker sensor, with a sinusoidal head pan. Both controllers see only the EKF estimate and share seeds. Errors are scored on the last lap against the true pose. 12 seeds, median [min, max]:

| Metric | go_to_point | LQR | LQR no integral |
|---|---|---|---|
| RMS cross-track, nominal | 15.4 mm | 1.8 mm | 2.0 mm |
| RMS cross-track, 0.10 rad/s yaw bias | 27.5 mm | 1.8 mm | 8.7 mm |
| Worst cross-track, yaw bias | 38.8 mm | 3.2 mm | 10.2 mm |
| RMS effort w - v*kappa, nominal | 0.130 rad/s | 0.018 | 0.015 |
| Estimate RMSE | 2.0 mm | 2.2 mm | 2.1 mm |

Paired difference in RMS error (go_to_point minus LQR): +13.7 mm nominal and +25.8 mm with bias, 95% CIs excluding zero. Integral action matters only under the unmodelled disturbance. NEES consistency and estimate error are unchanged, so the controller does not degrade estimation.

Caveats. The setting is benign (markers always visible, 2 mm estimate error), so both controllers meet the tracking limits. Much of the baseline error is chord-cutting between waypoints placed every 0.4 m plus proportional heading control, which a path with curvature feed-forward avoids by construction. The agent's A* paths are polylines with corners, so using LQR inside `agent._follow` needs smoothing first. This comparison is on a smooth reference only.

### LQR inside the agent (tested, not adopted)

`search.follow: lqr` makes `SearchAgent._follow` track the planner's polyline with the LQR on a filleted copy of it (`fillet_path`, corner radius `search.fillet_radius`). It was compared with `go_to_point` on identical seeds (synthetic sensor, 300 s cap). In `large_room` and `zigzag` both followers docked 16/16 (8 seeds, B and C-info) with similar time and miss distance. In `cluttered_14` C-info over 24 seeds, go_to_point docked 21/24 (median 146 s) and LQR 18/24 (median 116 s); five seeds were lost only by LQR and two only by go_to_point (McNemar p about 0.45), and rounded corners produced contacts in some runs. The tracking gain shown above does not carry into whole-system performance, where the limiting factors are perception and planning, so the default stays `go_to_point` and the LQR is reported as a separately validated tracker.


## The final study (`scripts/run_studies.sh`)

One script holds every trial, with one time cap (`CAP`, 300 s) for all of it. `--dir` gives each block its own results folder, so no block overwrites another's seeds.

| block | what | folder |
|---|---|---|
| main | 12 levels, B / C-sweep / C-info, seeds 0..19 | `results/` |
| height | `large_6_pillars`, dock height 0.2 / 0.45 / 0.9 m, same seeds | `results/height/` |
| heldout | seeds 5000..5019 (`--set seed=5000`), never used while tuning: `large_6_pillars`, `cluttered_14`, and the height study | `results/heldout/` |
| tilt | config B with the tilt free (`search.fixed_tilt=null`: search tilt while exploring, dock tilt close in) at the same heights, to separate the effect of tilt from that of panning. It still has only two preset tilts | `results/tilt_free/` |
| sens | estimator noise scaled by 0.5 and 2: Q (`k_ds`, `k_dth`, `k_dth_ds`) and R (`range_sigma`, `bearing_sigma_rad`, `normal_sigma_deg`), B and C-info, `large_6_pillars` | `results/sens_Q05`, `sens_Q2`, `sens_R05`, `sens_R2` |

Report any block with `python scripts/search_study.py --sensor camera --dir results/heldout --report --cap 300`. A smoke test of the plumbing: `SENSOR=synthetic TRIALS=2 CAP=120 bash scripts/run_studies.sh`.

## Final study results (camera sensor, 300 s cap, 20 seeds per cell)

Data: `results/` (main), `results/height/`, `results/heldout/` (seeds 5000..5019), `results/tilt_free/`, `results/lqr/`, `results/sens_*/`. Reports: `python scripts/search_study.py --sensor camera --dir <folder> --report --cap 300`. Intervals are Wilson (success) or cluster bootstrap over seeds (paired).

**Main block, 12 rooms (240 trials per config).** Docking: B 234/240, C-sweep 236/240, C-info 234/240. Paired success differences are not significant (C-info vs B +0 pp [-2, +2], McNemar p = 1.0). All these rooms hang the markers where the level camera can see them, so the head has no docking advantage to give. What the head does change is estimation: median position RMSE B 20.1 mm, C-sweep 12.1 mm, C-info 12.2 mm; paired RMSE difference vs B -4 mm [-6, -2] (C-info) and -5 mm [-7, -3] (C-sweep). The gap is larger in the hard rooms (`cluttered_14`: 75.6 / 61.9 / 45.3 mm; `sparse_noisy_odom`: 64.0 / 43.5 / 44.7 mm). NEES inside the 95% band: B 0.90, active configs 0.79 (the active heads are less consistent than B, which sits nearer the nominal 0.95; this study does not tell over- from under-confidence).

**Dock height (`large_6_pillars`, same seeds at each height, development seeds 0..19 and held-out seeds 5000..5019).**

| dock height | B | C-sweep | C-info |
|---|---|---|---|
| 0.20 m | 29/40 | 30/40 | 31/40 |
| 0.30 m | 38/40 | 40/40 | 40/40 |
| 0.45 m | 0/40 | 0/40 | 40/40 |
| 0.60 m | 0/40 | 0/40 | 39/40 |

Above about 0.33 m a locked tilt cannot see the whole plate at the standoff, so B and C-sweep never dock. C-info docks 79/80 at those heights, and the held-out seeds reproduce it (39/40). Paired over all heights, C-info vs B is +55 pp [+50, +60] (45 wins, 1 loss; development seeds) and +40 pp [+33, +47] on the held-out large room. C-sweep vs B is +0 pp [-6, +6]: panning alone gives no docking benefit when the tilt is wrong. B with its tilt freed (`tilt_free`, two preset tilts only) also fails at 0.45 and 0.60 m (0/40), so the benefit comes from choosing the tilt from what the map and markers need, not from having a tilt at all.

**A result that did not replicate: 0.20 m.** On development seeds C-info had 19/20 at 0.20 m against 15/20 for B and C-sweep; on held-out seeds it had 12/20 against 14 and 15. Pooled, the three configs are the same (29, 30, 31 of 40). The failures at 0.20 m are mostly false docks (the robot reports a dock within tolerance but the yaw is off by 6 to 10 degrees, because a face-on plate gives an ambiguous planar pose): B 11, C-sweep 10, C-info 9 of 40. The earlier impression that C-info reduces them came from the development seeds only.

**LQR vs go_to_point inside the agent (same seeds, camera).** No significant difference in any cell. Height study: C-info 79 vs 74 of 80 (LQR-only wins 1, go_to_point-only 6, McNemar p = 0.12), B 35 vs 34, C-sweep 35 vs 37. `cluttered_14`: B 18 vs 18, C-sweep 19 vs 20, C-info 18 vs 18. `large_6_pillars`: 20/20 for both in every config. The default stays go_to_point.

**Sensitivity of the estimator tuning (`large_6_pillars`, B and C-info, 20 seeds each).** Docking is unchanged by any scaling (19 or 20 of 20 in every run).

| run | B pos RMSE | B NEES in band | C-info pos RMSE | C-info NEES in band | C-info mean sigma_xy |
|---|---|---|---|---|---|
| baseline | 15.4 mm | 0.95 | 9.7 mm | 0.79 | 12.1 mm |
| Q x0.5 | 15.5 | 0.94 | 9.2 | 0.80 | 11.1 |
| Q x2 | 14.2 | 0.93 | 9.9 | 0.79 | 11.5 |
| R x0.5 | 16.1 | 0.93 | 12.4 | 0.59 | 8.1 |
| R x2 | 14.1 | 0.87 | 10.3 | 0.76 | 17.9 |

The process-noise scale (Q) and the doubled measurement noise change almost nothing. Halving R is the one setting that matters: C-info becomes clearly over-confident (reported sigma 8.1 mm against an actual RMSE of 12.4 mm, NEES in band 0.59), because it looks at markers far more often and each look is trusted too much. So the measurement-noise model is the part of the tuning that needs to stay conservative, and the active head is more sensitive to it than the fixed one.

## Rebuilding the numbers, and the gaze-behaviour metrics

`python scripts/analyse_final.py` rebuilds every table and figure above from the CSVs (`results/final_tables.md`, `results/figures/final/`), skipping any block that is missing. Blocks `adapt` (B and C-sweep with the dock-time tilt rule of C-info, tilt otherwise free: separates the dock rule from information-driven search) and `gaze` (head behaviour) are in `scripts/run_studies.sh`.

`evaluation/search.gaze_metrics` adds five descriptors of the head to each new trial: `gaze_switch_per_min` (pan reversals), `gaze_dwell_med_s` (median time the world-frame gaze stays still), `gaze_toward_travel` (share of moving time looking within 0.6 rad of the direction of travel), `gaze_lead_s` (lag at which the pan best predicts the body's later heading change; exploratory, can sit at the 4 s edge of its search range) and `pan_std_rad`. They are descriptive cues in the sense used in the human-robot interaction literature on legible, goal-directed motion; they say nothing about how people perceive the robot, which would need participants.

## Head camera profiles: the ideal camera and an Intel RealSense D435i

`camera:` in the config picks the head camera. **`ideal`** (the default) is the camera every result above was collected
with: a pinhole, 60 deg vertical field of view, 640 x 480, Gaussian pixel noise, no lens distortion, no blur, no delay.
Selecting it reproduces those results bit for bit. **`d435i`** is the colour camera of an Intel RealSense D435i,
simulated as a real camera. Any script with `--set` takes it: `watch_search.py`, `search_study.py`,
`calibrate_camera.py`, `characterise_markers.py`, `check_markers.py`, `obstacle_diagnostic.py`, e.g.

    python scripts/watch_search.py --set camera=d435i
    python scripts/search_study.py --sensor camera --level large_6_pillars --set camera=d435i --dir results/d435i

It changes the image the robot gets and what the robot knows about its camera, not the robot's algorithms.
Only the colour stream is modelled: the D435i's stereo depth is not simulated and the robot stays monocular.

**Image formation** (`sim/realsense.py`). MuJoCo renders an ideal pinhole image a little wider than the stream. It
then goes through:
1. Optics: resampled through the stream's intrinsics (848 x 480, fx = 609.3, fy = 609.0 px, principal point a
   few pixels off centre) and Brown-Conrady lens distortion (mild barrel, about 6 px at the corners), then vignetting.
2. Motion: rolling shutter (the rows are read over 20 ms) and motion blur (10 ms exposure), from the camera's angular
   velocity. While the head pans at the gaze slew limit (1.5 rad/s) the image is blurred by about 9 px and skewed by
   about 18 px from top to bottom.
3. Sensor: Bayer RGGB colour filter, photon shot noise and read noise counted in electrons, 10-bit ADC.
4. ISP: demosaicing (OpenCV edge-aware) and sRGB gamma.

Light is handled as linear intensity (blur, vignetting, noise) and gamma-encoded at the end, as in a real camera.
`results/figures/camera_profiles.png` (from `scripts/camera_profile_figure.py`) shows the camera on the head and the
same scene through both profiles.

**Timing.** Frames come out on a 30 Hz grid and are delivered 60 ms after capture. `DockingEnv` keeps the last
states (one per control step) and renders a delivered frame from the state at its capture time. It uses a separate
`MjData`, so the physics is untouched. A frame carries its capture time and the head encoder readings of that moment
(`SensorBundle.image_t`, `image_pan`, `image_tilt`), as a driver's time stamps would. When the robot fuses a late
frame, it carries the measurements into its current base frame with its own wheel odometry (`sensing/delay.py`).
Without this, a 60 to 90 ms old frame taken while the robot turns in place would be off by several degrees.

**What the robot knows** (`sensing/camera.py: camera_model`): image size, intrinsics, distortion, fields of view and
the lens position on the head. ArUco PnP uses the distortion, floor ranging removes it before intersecting rays with
the floor, and the gaze policies and synthetic sensors use the profile's horizontal field of view (69.7 deg instead of
75.2 deg). The narrower vertical field of view (43.0 deg instead of 60 deg) matters most. At a level head the floor is
visible only from 0.49 m (0.33 m with the ideal camera), and tall markers leave the image sooner.

**Body.** The head's generic camera block is replaced (through `MjSpec`, in any scene) by the D435i body: 90 x 25 x
25 mm, front glass, colour lens and the two IR windows 50 mm apart. The colour lens is on the front face, 12.5 mm
ahead of the tilt axis and centred sideways. The centring is a simplification: on the module, the colour lens sits a
few centimetres to one side. The head's mass and inertia are unchanged: 0.1 kg covers the 72 g camera and a bracket,
and the servo's reflected inertia dominates the payload's.

**Calibration** (`calibrate_camera.py --set camera=d435i`, 30 views). The calibrated focal lengths are 0.7 px from
the profile, the principal point is within 0.3 px, and the distortion coefficients are recovered. The assumed and
calibrated cameras differ by 0.5 px median (1.3 px max) over the image. Reprojection RMS is 0.087 px with
distortion and 0.127 px as a pure pinhole: for this camera, modelling distortion matters, unlike for the ideal
camera. The frame chain is unchanged (0.2 mm, 0.12 deg).

**Marker characterisation** (`characterise_markers.py --set camera=d435i`, 300 poses, refined detections up to 5 m):

| | range bias | range noise at 4-5 m | bearing noise | normal noise at 4-5 m |
|---|---|---|---|---|
| ideal camera | 0.0020 r^2 | 27 mm | 0.0020 rad | 4.3 deg |
| D435i | 0.0033 r^2 | 17 mm | 0.0020 rad | 3.4 deg |

The D435i sees the markers with more pixels per degree (narrower field of view), so its noise is at or below the
ideal camera's. Its range bias is larger: blurring happens in linear light, and after the gamma the black square
looks slightly smaller, so it looks farther. The profile's `set:` therefore refits only the EKF's range-bias
correction (`ekf.range_bias_coeff=0.0033`). The noise model R, fitted to the ideal camera and then inflated, still
bounds this camera's measured noise and is kept.

**Datasheet values vs assumptions.** From the D435i datasheet: colour field of view 69 x 42 deg (+-3), 30 fps at this
resolution, body size and mass, 50 mm stereo baseline, OV2740 colour sensor with 1.4 um pixels and 10-bit output.
Assumed (typical values, marked in `configs/default.yaml`): the detailed intrinsics and distortion (every unit has its
own factory calibration), exposure, readout time, latency, full well, read noise and vignetting. They are parameters
and can be changed in the config.

**First end-to-end check with the D435i (robot software unchanged).** `large_6_pillars`, the same seeds as the
ideal-camera study, camera sensor, 300 s cap (`results/d435i/`):

| config | ideal camera, seeds 0-4 | D435i |
|---|---|---|
| B | 5 / 5 docked | 1 / 5: 4 timeouts |
| C-sweep | 5 / 5 | 0 / 5: 3 timeouts, 2 false docks (yaw 33-39 deg) |
| C-info | 5 / 5 | 0 / 5: 5 false docks (yaw 31-34 deg) |

This is not a fault of the camera model. With its motion effects switched off (latency, exposure and readout set to
0), the same C-info trial docks in 23 s, as with the ideal camera. Two real effects of the camera break docking logic
that was tuned on the ideal camera:

1. **Vertical field of view.** The dock marker's centre is at 0.25 m, so its black square reaches 0.40 m. At the
   0.5 m stand-off, a level head sees up to 0.48 m with the ideal camera but only 0.39 m with the D435i (43 deg). B and
   C-sweep lock the tilt at 0, so they cannot see the whole marker at the dock point. They time out, or dock only
   when they stop a little farther back (B seed 3: 6.3 cm).
2. **The camera turns while it looks.** While docking, the body turns in place at 0.4 rad/s and the head pans at up
   to about 1.9 rad/s to keep the marker centred.
   - The rolling shutter shears the marker, so its measured normal is off by about 5-10 deg, with a sign that
     follows the turn. The marker is still detected in 90 % of frames up to 0.6 rad/s, 60-70 % at 1-1.5 rad/s and
     30 % above.
   - The EKF expects about 0.3 deg on the normal at 0.5 m (R was fitted on still views), so its gate rejects these
     frames. The dock servo then gets an accepted view only about once a second. It acts on the last view for up
     to 0.8 s, overshoots and swings by about +-40 deg.
   - The finish test averages the last five views. Over a symmetric swing (-36, -18, +2, +20, +39 deg) the mean is
     about 0, so it declares "docked" while 30-40 deg off.

Possible changes to the robot, not made (to decide as a team):
- measurement noise that grows with the camera's rotation rate, which the robot knows from the head encoders and
  odometry;
- a dock servo that stops and looks instead of acting on a stale view;
- a finish test on every one of the last views, not their mean;
- a dock tilt, stand-off or marker height compatible with a 43 deg vertical field of view for B and C-sweep.
