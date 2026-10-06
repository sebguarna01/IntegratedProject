# Active Visual Navigation and Autonomous Docking
### MuJoCo TurtleBot3 with a Pan-Tilt RGB Camera

Camera-only navigation and docking for a small mobile robot in MuJoCo, using a TurtleBot3 base with an actively controlled pan-tilt camera head.

**Course:** 41014 / 42043 Sensors and Control in Mechatronics Systems, Integrated Project
**Track:** Own project (Track A with extension)
**Team:** Team 10
**Demo video:** TBC

> Status: work in progress. Sections marked **TBC** are not yet finished.

---

## 1. Overview

This project is a simulation-first prototype of the navigation and perception system for MOXI, a proposed small non-humanoid companion robot that communicates through movement, gaze and sound rather than speech or screens.

The robot has **no LiDAR**. It uses only wheel-encoder odometry and a pan-tilt RGB camera. It must cross a rectangular room, avoid static and moving obstacles, build a map as it goes, and dock at an ArUco-marked charging station at the far end.

**Research question:** Can active control of a pan-tilt monocular camera improve state estimation and docking performance in a minimally instrumented mobile robot?

The camera head is not passive. A small behaviour layer (Curious, Attentive, Cautious, Docking) decides where the camera looks based on the robot's state and how uncertain its pose estimate is.

## 2. System summary

| Part | Approach |
|---|---|
| Simulator | MuJoCo, TurtleBot3 from `robotis_mujoco_menagerie` with a pan-tilt head added |
| Sensors | Wheel encoders, pan/tilt joint encoders, monocular RGB camera |
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

- State-space model: differential-drive model. Derivation: TBC (`docs/`)
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
| Camera intrinsics | TBC | Calibration, TBC |

## 5. Experiments

TBC... Proposed:
Three sensing configurations are compared over randomised trials:

| Config | Localisation | Camera head |
|---|---|---|
| A | Odometry only | Fixed |
| B | EKF with marker updates | Fixed |
| C | EKF with marker updates | Active (behaviour-driven) |

Scenarios increase in difficulty: static obstacles, unknown obstacles with online mapping, constant-velocity obstacles, then random or alternating-velocity obstacles. Number of trials per configuration: TBC.

## 6. Results

All results are TBC. Sample results breakdown/analysis below

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

 TBC.

## 8. Running the code

Commands are TBC.

## 9. Repository structure

Planned layout (TBC):

```
configs/        parameters and scenario definitions
models/         MuJoCo additions: head, room, markers
src/
  sim/          MuJoCo wrapper
  sensing/      odometry, calibration, head kinematics
  perception/   ArUco detection, obstacle detection
  mapping/      occupancy grid
  planning/     A* and path smoothing
  control/      LQR, pan-tilt control
  behaviour/    state machine, gaze selection
  analysis/     controllability and observability checks
  evaluation/   metrics and plots (only place ground truth is used)
scripts/        run and experiment scripts
tests/
docs/           derivations, frame diagram, figures
results/        summary tables and figures
```

## 10. Contributions

| Member | Contribution |
|---|---|
| Sebastiano Guarna | TBC |
| Dorian Veloso | TBC |

## 11. Generative AI declaration

 TBC

## 12. Acknowledgements and references

- Robot models: [ROBOTIS MuJoCo Menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
- MuJoCo physics engine
- Code adapted from weekly tutorials, MuJoCo examples and the Robotics Toolbox: TBC (list specific files)

## 13. Licence

TBC
