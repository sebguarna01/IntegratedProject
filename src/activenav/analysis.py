"""Model analysis: linearisation, controllability and observability of the platform and of the estimator.

Controllability (what the controller can steer) uses the unicycle and its path-error model (state-space form
x' = A x + B u, then zero-order-hold discretisation). Observability (what the estimator can know) uses the SAME
Jacobians the filters use: F from `EKF._predict_parts` and H from `predict_measurement` / `SlamEKF._expected`, evaluated
along a prescribed odometry trajectory, so the analysis cannot drift away from the code it describes.

Everything here is analysis, not run-time: nothing in the robot's loop imports it, and it reads no simulator state.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.signal import cont2discrete

from activenav.estimation.ekf import EKF
from activenav.estimation.slam import SlamEKF
from activenav.types import MarkerMeasurement

# ----------------------------------------------------------------------------------------- controllability
def unicycle_linearisation(v0: float, theta0: float):
    """A, B of  x' = [v cos th, v sin th, w]  about (theta0, v0, w0 = 0). State [x, y, th], input [v, w]."""
    A = np.array([[0.0, 0.0, -v0 * np.sin(theta0)],
                  [0.0, 0.0, v0 * np.cos(theta0)],
                  [0.0, 0.0, 0.0]])
    B = np.array([[np.cos(theta0), 0.0], [np.sin(theta0), 0.0], [0.0, 1.0]])
    return A, B


def error_dynamics(e, v: float, kappa: float, dw: float):
    """Nonlinear path-error model. e = [e_y, e_th] (lateral offset, heading error relative to the path tangent), path
    curvature kappa, input dw = w - v*kappa (the steering effort beyond what the path itself needs)."""
    ey, eth = e
    return np.array([v * np.sin(eth), dw - v * kappa * (np.cos(eth) / (1.0 - kappa * ey) - 1.0)])


def error_model(v: float, kappa: float = 0.0):
    """Continuous linearisation of `error_dynamics` at e = 0: A = [[0, v], [-v kappa^2, 0]], B = [0, 1]^T."""
    return np.array([[0.0, v], [-v * kappa ** 2, 0.0]]), np.array([[0.0], [1.0]])


def jacobians(f, x0, u0, eps=1e-6):
    """Central-difference A = df/dx, B = df/du of f(x, u) (to check the closed forms above)."""
    x0, u0 = np.asarray(x0, float), np.asarray(u0, float)
    A = np.column_stack([(f(x0 + eps * d, u0) - f(x0 - eps * d, u0)) / (2 * eps) for d in np.eye(len(x0))])
    B = np.column_stack([(f(x0, u0 + eps * d) - f(x0, u0 - eps * d)) / (2 * eps) for d in np.eye(len(u0))])
    return A, B


def discretise(A, B, Ts: float):
    """Zero-order-hold discretisation: returns (Ad, Bd)."""
    n, m = B.shape
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(n), np.zeros((n, m))), Ts, method="zoh")
    return Ad, Bd


def ctrb(A, B):
    n = A.shape[0]
    blocks, M = [B], B
    for _ in range(n - 1):
        M = A @ M
        blocks.append(M)
    return np.hstack(blocks)


def obsv(A, C):
    return ctrb(A.T, C.T).T


def rank_and_sv(M, rel_tol: float = 1e-9):
    """(rank, singular values). Rank counts singular values above rel_tol * the largest."""
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rel_tol * max(s.max(), 1e-300)).sum()) if s.size and s.max() > 0 else 0, s


# ------------------------------------------------------------------------------------------ observability
def _config(cfg: dict, bias: bool):
    c = copy.deepcopy(cfg)
    c["ekf"]["estimate_bias"] = bias
    return c


def default_trajectory(n_frames: int = 150, ds: float = 0.02, turn: float = 0.04):
    """Odometry increments per camera frame (ds [m], dtheta [rad]): straight, a left turn while driving, straight.
    A straight line alone cannot separate heading from a sideways offset; the turn is what makes the pose observable."""
    third = n_frames // 3
    return [(ds, 0.0)] * third + [(ds, turn)] * third + [(ds, 0.0)] * (n_frames - 2 * third)


def pose_observability(cfg: dict, markers: dict, rows=(0, 1, 2), bias: bool = False, trajectory=None,
                       start=(0.0, 0.0, 0.0)):
    """Observability matrix of the known-map EKF along `trajectory` with `markers` always in view and measuring `rows`
    of [range, bearing, normal] (rows=() is the odometry-only case). Columns are scaled by the filter's initial standard
    deviations, so singular values read as 'how much a prior-sized error shows up in the measurements'.
    Returns (rank, singular values, n states)."""
    ekf = EKF(_config(cfg, bias), markers)
    ekf.reset(np.asarray(start, float))
    n = ekf.n
    scale = np.sqrt(np.diag(ekf.P))
    Phi, blocks = np.eye(n), []
    for ds, dth in (trajectory or default_trajectory()):
        for mid in markers:
            if rows:
                _, H3 = ekf.predict_measurement(mid)
                H = np.zeros((3, n))
                H[:, :3] = H3
                blocks.append(H[list(rows)] @ Phi)
        x_new, F, _ = ekf._predict_parts(ds, dth)
        ekf.x = x_new
        Phi = F @ Phi
    if not blocks:
        return 0, np.zeros(n), n
    r, s = rank_and_sv(np.vstack(blocks) * scale)
    return r, s, n


def slam_observability(cfg: dict, landmarks: list, anchored: bool, bias: bool = False, trajectory=None):
    """Observability of robot + landmarks for EKF-SLAM, from the real SlamEKF Jacobians. `landmarks` = [(x, y, normal_angle)]
    seen at every frame. `anchored` adds the filter's prior on the starting pose (what `SlamEKF.reset` does).
    Returns (rank, state dimension): the gap is the number of directions no measurement can ever fix (the gauge)."""
    f = SlamEKF(_config(cfg, bias))
    f.reset(np.zeros(3))
    for i, (mx, my, mth) in enumerate(landmarks):                       # introduce the landmarks with exact measurements
        dx, dy = mx - f.x[0], my - f.x[1]
        f.update_marker(MarkerMeasurement(i, float(np.hypot(dx, dy)), float(np.arctan2(dy, dx) - f.x[2]),
                                          float(mth - f.x[2])))
    d, n = f.dim, f.n
    Phi, blocks = np.eye(d), []
    for ds, dth in (trajectory or default_trajectory()):
        for i in f.lm_index:
            blocks.append(f._expected(i)[1] @ Phi)
        full = f.x
        f.x = full[:n].copy()                                          # as SlamEKF.predict: the robot block only moves
        x_r, F, _ = f._predict_parts(ds, dth)
        f.x = full.copy()
        f.x[:n] = x_r
        Fd = np.eye(d)
        Fd[:n, :n] = F
        Phi = Fd @ Phi
    O = np.vstack(blocks)
    if anchored:
        A0 = np.zeros((3, d))
        A0[:, :3] = np.eye(3)
        O = np.vstack([A0, O])
    r, _ = rank_and_sv(O)
    return r, d


def blind_growth(cfg: dict, metres: float = 5.0, bias: bool = True, ds: float = 0.02):
    """1-sigma of [x, y, theta] after driving `metres` straight with no marker in view (the filter's own covariance)."""
    ekf = EKF(_config(cfg, bias), {})
    ekf.reset(np.zeros(3))
    for _ in range(int(round(metres / ds))):
        ekf.predict(ds, 0.0)
    return ekf.sigma()
