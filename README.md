# Active Visual Navigation and Autonomous Docking

Camera-only navigation and docking for a small mobile robot in MuJoCo, using a TurtleBot3 base with an actively controlled pan-tilt camera head.

**Course:** 41014 / 42043 Sensors and Control in Mechatronics Systems, Assessment 4 (Integrated Project)
**Track:** Own project (Track A extension)
**Team:** Team 10
**Demo video:** TBC

> Status: work in progress. Sections marked **TBC** are not yet finished.

---

## 1. Overview

This project investigates how far a small mobile robot can navigate and dock using minimal sensing: wheel-encoder odometry and an actively controlled pan-tilt camera.

The robot has **no LiDAR**. It uses only wheel-encoder odometry and a pan-tilt RGB camera. It must cross a rectangular room, avoid static and moving obstacles, build a map as it goes, and dock at an ArUco-marked charging station at the far end.

**Research question:** Can active control of a pan-tilt monocular camera improve state estimation and docking performance in a minimally instrumented mobile robot?

The camera head is not passive. A small behaviour layer (Curious, Attentive, Cautious, Docking) decides where the camera looks based on the robot's state and how uncertain its pose estimate is.

## 2. System summary

| Part | Approach |
|---|---|
| Simulator | MuJoCo, TurtleBot3 from `robotis_mujoco_menagerie` with a pan-tilt head added |
| Sensors | Wheel encoders, pan/tilt joint encoders, monocular RGB camera (ideal pinhole by default; `--set camera=d435i` or `camera=d455` simulates an Intel RealSense colour camera: real optics, shutter, motion blur, sensor noise, latency) |
| Landmarks and dock | ArUco markers at known positions |
| State estimation | Extended Kalman Filter, state `[x, y, theta]` |
| Obstacle detection | Colour segmentation plus monocular ground-plane ranging |
| Mapping | Occupancy grid built at the estimated pose |
| Obstacle tracking | Constant-velocity Kalman filter per obstacle |
| Planning | A* on the inflated map, with replanning |
| Control | Gain-scheduled LQR trajectory tracker |
| Behaviour | State machine plus gaze selection for the pan-tilt head |

**Key constraint:** the controller, planner and behaviour layer only use the estimated state and the map built from it. Simulator ground truth is used only in the `evaluation/` package.

## 3. Model and frames

- State-space model: differential-drive (unicycle) model. Derivation: TBC (`docs/`)
- Frame chain: World, Base, Pan, Tilt, Camera, Marker. Frame diagram: TBC (`docs/`)
- Controllability and observability analysis: TBC (`docs/`)

## 4. Estimator and controller design

Parameter values and justification will be added here once tuned.

| Item | Value | Where it comes from |
|---|---|---|
| EKF process noise Q | TBC | TBC |
| EKF measurement noise R | TBC | TBC |
| Gating threshold | TBC | TBC |
| LQR weights | TBC | TBC |
| Velocity limits | TBC | TBC |
| Camera intrinsics | fx = fy = 415.5 px, cx = 319.9, cy = 238.7 (640 x 480), distortion about zero | Checkerboard calibration in simulation, `scripts/calibrate_camera.py`; reprojection RMS 0.09 px |

## 5. Experiments

Three sensing configurations are compared over randomised trials:

| Config | Localisation | Camera head |
|---|---|---|
| A | Odometry only (dead reckoning) | Fixed |
| B | EKF with marker updates | Fixed |
| C | EKF with marker updates | Active (behaviour-driven) |

Scenarios increase in difficulty: static obstacles, unknown obstacles with online mapping, constant-velocity obstacles, then random or alternating-velocity obstacles. Number of trials per configuration: TBC.

## 6. Results

All results are TBC.

| Metric | A | B | C |
|---|---|---|---|
| Docking success rate | TBC | TBC | TBC |
| Position RMSE (m) | TBC | TBC | TBC |
| Heading RMSE (rad) | TBC | TBC | TBC |
| Map IoU | TBC | TBC | TBC |
| Collisions per trial | TBC | TBC | TBC |
| Docking time (s) | TBC | TBC | TBC |

Failure analysis: TBC.

## 7. Installation

Developed on macOS (Apple silicon). Python version: TBC.

```bash
git clone <this repository> active-visual-navigation   # URL: TBC
cd active-visual-navigation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                 # makes `import activenav` work
```

Copy the Waffle Pi mesh files (`waffle_pi_base.stl`, `left_tire.stl`, `right_tire.stl`) from the
`assets/` folder of the TurtleBot3 model in
[robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
into `models/assets/`. The robot, head and room are defined in `models/docking_scene.xml`.

On macOS, the interactive MuJoCo viewer must be launched with `mjpython` instead of `python`.

Editor: Visual Studio Code. After creating `.venv`, select it as the Python interpreter (Command Palette, "Python: Select Interpreter"). Further VS Code setup (debug configurations, recommended extensions): TBC.

## 8. Running the code

Current scripts (details in `docs/methods.md`):

```bash
python scripts/odometry_baseline.py square|study|crossing   # odometry-only baselines
python scripts/characterise_markers.py                      # marker measurement error vs range and view angle
python scripts/ekf_crossing.py                              # known-map EKF check (consistency, bias states)
python scripts/search_study.py --sensor camera --level large_room --trials 20   # one study
SENSOR=camera JOBS=4 bash scripts/run_studies.sh   # the whole final study: all levels, dock heights, held-out seeds, tilt-free B, Q/R sensitivity (one 300 s cap)
python scripts/analyse.py                                   # every table (results/final_tables.md) and figure (results/figures/final/) from the CSVs
python scripts/watch_search.py                              # live viewer (3D view: mouse to rotate / move / zoom)
python scripts/watch_search.py --set camera=d455            # same, with the simulated RealSense D455 colour camera (global shutter, 90 deg)
pytest                                                      # unit tests
```

```bash
python scripts/model_analysis.py                           # controllability and observability
python scripts/calibrate_camera.py                         # checkerboard calibration and frame-chain check (needs GL)
```

Camera profiles (`docs/methods.md`, "Head camera profiles"): `--set camera=d435i|d455` works with every script that takes `--set`; calibration, marker characterisation and figures are in `results/` (`scripts/camera_profile_figure.py`).

LQR vs go-to-point tracking comparison: `scripts/lqr_tracking.py` (results in docs/methods.md).

## 9. Repository structure

```
configs/        parameters, worlds and levels
models/         MuJoCo robot template, base scene, assets (room XML is generated)
src/activenav/
  sim/          MuJoCo wrapper, world and scene builder, walking people
  sensing/      odometry, camera and head kinematics, ArUco markers, floor ranging
  estimation/   EKF and EKF-SLAM
  control/      go-to-point, lqr (path tracker)
  autonomy/     search-and-dock agent, exploration, gaze, occupancy map, planner
  evaluation/   trials, search study, paired statistics, plots, viewer (only place ground truth is used)
scripts/        run and experiment scripts
tests/
docs/           methods and metrics
results/        final tables and figures
```

## 10. Contributions

| Member | Contribution |
|---|---|
| TBC | TBC |
| TBC | TBC |

## 11. Generative AI declaration

Tools used: TBC
Purpose: TBC
What was checked or changed by the team: TBC

## 12. Acknowledgements and references

- Robot models: [ROBOTIS MuJoCo Menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
- MuJoCo physics engine
- Code adapted from weekly tutorials, MuJoCo examples and the Robotics Toolbox: TBC (list specific files)

## 13. Licence

TBC
