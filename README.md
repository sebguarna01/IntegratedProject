# Vision-Based Navigation, Mapping and Autonomous Docking with an RGB-D TurtleBot3

**42043 Sensors and Control in Mechatronics Systems — Assessment 4, Integrated Project**

Track A (personalised) · MuJoCo simulation · TurtleBot3 Burger with a single RGB-D camera

---

## 1. Summary

This project implements a complete sensing, estimation, planning and control pipeline for a differential-drive mobile robot that navigates an unknown cluttered room and docks at an ArUco-marked charging station, using **no LiDAR and no ground truth**.

The robot carries one RGB-D camera. The colour stream is used for landmark perception — detecting ArUco markers at surveyed positions and the docking station itself. The depth stream is reduced to a planar range scan and accumulated into an occupancy grid, which a global planner searches for a collision-free path. An Extended Kalman Filter fuses wheel-encoder odometry with landmark range and bearing observations, and an LQR tracking controller follows the planned path using the filtered estimate alone.

The system therefore closes a full loop: perception feeds both the estimator and the map, the map feeds the planner, the planner feeds the controller, and the controller changes what the camera sees next.

Simulator ground truth is read by a separate evaluation module and never enters the control path.

---

## 2. Task

1. The robot starts at a randomised pose in a room whose obstacle layout it does not know.
2. It builds an occupancy grid of the room from depth measurements as it moves.
3. It localises against ArUco landmarks at known positions, correcting wheel-odometry drift.
4. It plans a collision-free path to the charging station and tracks it.
5. It visually aligns with the station's ArUco marker and executes the docking approach.

Docking succeeds when the final pose is within a stated position and heading tolerance of the station, measured against ground truth. Thresholds are fixed before the experiments and stated in Section 8.

---

## 3. System architecture

| Layer | Input | Output | Rate |
|---|---|---|---|
| Perception — landmarks | RGB image | marker ID, range, bearing | 10 Hz |
| Perception — obstacles | depth image | planar range scan | 10 Hz |
| Estimation | encoders, landmark observations | pose estimate and covariance | 50 Hz predict / 10 Hz update |
| Mapping | range scan, pose estimate | occupancy grid (log-odds) | 10 Hz |
| Global planning | occupancy grid, goal | waypoint path | on map change |
| Local avoidance | range scan, path | velocity reference | 10 Hz |
| Control | pose estimate, velocity reference | wheel velocity commands | 50 Hz |

**Architectural constraint.** The estimator, mapper, planner and controller are given the EKF estimate. None of them has a reference to the simulator state. Ground truth is read only inside `eval/`, and every such call is marked.

---

## 4. Components

### 4.1 Platform model (Criterion A)

Unicycle kinematic model of the TurtleBot3 Burger, with its published parameters: wheel radius 33 mm, wheel separation 160 mm, maximum linear velocity 0.22 m/s, maximum angular velocity 2.84 rad/s. Wheel speeds map to body velocities `(v, ω)` through the differential-drive Jacobian.

The model is discretised with a midpoint rule rather than forward Euler, which is second-order accurate in heading and has no singularity as `ω → 0`. Stated assumptions: no wheel slip, rigid wheels, planar motion, velocity-level actuation.

Two linearisations are performed and kept distinct:

- About the current estimate, giving the Jacobians `Fx` and `Fu` used by the EKF.
- About the reference trajectory, giving the error dynamics `(A, B)` used by the controller.

**Controllability.** The controllability matrix of the linearised error dynamics has rank 3 whenever the reference is moving, that is `(v_r, ω_r) ≠ (0, 0)`, and rank 2 at a complete standstill, where the lost direction is lateral error. This is the nonholonomic constraint appearing in the rank test. Because Brockett's condition rules out smooth time-invariant stabilisation to a fixed pose, docking is implemented in two phases: tracking to a pre-dock pose, then a separate alignment and approach law.

**Observability.** Analysed per sensing condition. With encoders alone the pose is unobservable and error grows without bound. With range and bearing to a *single* known landmark the measurement Jacobian has rank 2, so the pose is **not** instantaneously observable; the unobservable mode is an orbit about the landmark, in which the robot circles it while rotating by the same amount, leaving both range and bearing unchanged. Observability is recovered either through motion, since odometry excludes that orbit, or instantaneously when two or more landmarks are in view. Ranks and the conditioning of the observability matrix are reported numerically along representative trajectories.

### 4.2 Sensing and calibration (Criterion B)

**Intrinsic calibration.** MuJoCo specifies the camera by its vertical field of view, so the intrinsic matrix is known analytically. We nonetheless run a full chessboard calibration in simulation and report the RMS reprojection error together with the relative error against the analytic `K`. The simulated renderer is a perfect pinhole, so distortion coefficients are zero; this is a documented difference from a real camera.

**Frame chain.** `map → odom → base_footprint → camera_link → camera_optical → marker`, with the OpenGL-to-OpenCV optical convention change made explicit. A frame diagram is included below. Every measurement is transformed through this chain before it reaches the filter.

**Noise characterisation.** Measurement noise is measured, not assumed. With the robot held stationary on a grid of ranges and viewing angles, several hundred detections per cell give the empirical standard deviation of range and bearing as a function of distance, and the detection rate as a function of incidence angle. The resulting `R` is range-dependent.

**Marker pose and the planar ambiguity.** `solvePnP` on a planar square target admits two poses with near-identical reprojection error, and noise flips the solver's choice at distance. Rather than feed this unstable yaw to the filter, the measurement model uses range and bearing, which depend only on the marker centroid and do not flip. The cost of this choice — a rank-2 instead of rank-3 measurement Jacobian — is analysed in Section 4.1 and quantified in Section 8.

**Depth to planar scan.** A horizontal band of rows from the depth image is back-projected through `K`, filtered to the robot's horizontal plane, and reduced to a minimum range per angular bin. The result is a range scan spanning the camera's field of view, not the 360° of a LiDAR. Limited angular coverage is a deliberate consequence of the single-sensor design and is reported as such.

### 4.3 State estimation (Criterion C)

An EKF over the pose `[x, y, θ]`.

Prediction is driven by wheel-encoder increments. The process noise `Q` is not tuned by hand: per-wheel noise is modelled as proportional to wheel displacement, mapped to `(v, ω)` covariance through the differential-drive Jacobian, then through `Fu`. The slip coefficients are fitted from simulation runs with known commands.

Updates are applied per detected landmark, with the measurement noise `R` taken from Section 4.2 and an innovation gate on the Mahalanobis distance against a chi-square threshold. Bearing innovations are wrapped to `(−π, π]`.

Outputs are the estimate, its covariance, and a scalar confidence derived from the covariance, which the behaviour layer uses to decide when the estimate is good enough to commit to the docking approach.

Consistency is evaluated with NIS over single runs and NEES over the Monte Carlo set, against chi-square bounds. The trade-off between filter consistency and responsiveness is quantified by scaling `Q` and `R`.

### 4.4 Mapping and planning

**Mapping with estimated poses, not SLAM.** Localisation and mapping are deliberately separated. The EKF localises against landmarks at surveyed positions; the occupancy grid maps the obstacles, which are unknown. The grid is updated in log-odds from the planar scan, ray-traced from the estimated pose.

This separation is an explicit design decision with a consequence we measure: **the map inherits the estimator's error.** When no landmark has been visible for some time, the pose covariance grows and the grid smears. Quantifying that relationship is the project's main independent contribution (Section 6).

**Global planning.** A* over the inflated occupancy grid, re-planned when the map changes materially near the current path. Inflation radius is set from the robot's footprint plus a margin derived from the pose covariance.

**Local avoidance.** A reactive layer over the latest scan handles obstacles that appear inside the planned path, including the dynamic obstacles of the stretch scope. The global plan is not re-run at the reactive rate.

### 4.5 Control (Criterion D)

LQR on the linearised error dynamics, with the gain re-solved as the reference velocity and curvature change. Weights are chosen by Bryson's rule from stated acceptable errors and control effort, then defended; actuator saturation is enforced at the Burger's published limits.

Docking is two-phase, for the reason given in Section 4.1: trajectory tracking to a pre-dock pose on the station's axis, then a dedicated alignment and approach law.

The controller reads only the EKF estimate. Reported tracking error is given both against the estimate and against ground truth, so the estimation error and the control error are separated rather than conflated.

---

## 5. Scope

**Core system.** MuJoCo scene and TurtleBot3 model, RGB-D camera, intrinsic calibration, ArUco perception, depth-to-scan, EKF, occupancy mapping, A* planning, local avoidance, LQR tracking, visual docking, Monte Carlo evaluation.

**Stretch, in priority order.**

1. Dynamic obstacles — additional moving TurtleBots crossing the planned path.
2. Lighting variation — darker and brighter zones in the scene, degrading detection rate and hence landmark availability.
3. An actuated camera mount, letting the robot point the camera to extend mapping coverage and re-acquire landmarks.

The project is complete and assessable without any stretch item. Priority goes to a robust core pipeline over additional features.

---

## 6. Independent contribution

Hardware is not pursued. The independent contribution is a study the brief does not require: **how localisation uncertainty propagates into map quality, and what that costs the planner.**

Three sensing conditions are compared under identical seeds — odometry only, odometry with landmark updates, and odometry with landmark updates plus a covariance-aware behaviour that pauses to re-acquire a landmark before committing to the dock. For each, we report pose RMSE, filter consistency, occupancy-grid agreement against the true obstacle layout, path length against the optimal path on the true map, and docking success rate.

The hypothesis under test is that grid error is driven by pose covariance at the time each scan is integrated, and that a behaviour which spends time reducing covariance pays for that time in shorter and safer paths.

---

## 7. Repository layout

```
sim/            MuJoCo scene, robot model, marker assets
perception/     ArUco detection, calibration, depth-to-scan
estimation/     EKF, Jacobians, noise models
mapping/        occupancy grid, log-odds update
planning/       A*, inflation, local avoidance
control/        LQR, docking state machine
eval/           Monte Carlo runner, metrics, consistency tests  [GROUND TRUTH USED HERE ONLY]
figures/        generated plots
```

## 8. Results

*To be completed. Thresholds, metric definitions and the Monte Carlo protocol are fixed before the experiments are run and recorded here.*

## 9. Installation and running

*To be completed.*

## 10. Contribution statement

*To be completed — per-member contributions.*

## 11. Generative AI declaration

*To be completed. State each tool, what it was used for, and which parts of the code, text or analysis it touched.*

## 12. Acknowledgements

Code adapted from the weekly tutorials, MuJoCo examples and the Robotics Toolbox is acknowledged inline at the point of use.
