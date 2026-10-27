"""Discrete LQR path tracker with integral action, on the path-error model of the unicycle.

Error state e = [e_y, e_theta]: lateral offset (+ = left of the path) and heading error to the path tangent; path
curvature kappa. With the input dw = w - v kappa the linearised model is  e' = [[0, v], [-v kappa^2, 0]] e + [0, 1]^T dw
(analysis.error_model; kappa^2 is dropped here: it is small for the paths driven and the gain is designed on the straight
model). Integral action adds z = integral of e_y. Gains come from the discrete algebraic Riccati equation on the
zero-order-hold model at the control period, with Bryson's rule for the weights (1 / largest acceptable value squared).

The tracker only sees the pose it is given (the EKF estimate) and the path. It never reads simulator state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from activenav.analysis import discretise, error_model
from activenav.kinematics import wrap_angle


@dataclass(frozen=True)
class LqrWeights:
    """Largest acceptable values (Bryson's rule). Defaults are for a 0.2 m/s robot in a corridor-sized room."""
    ey_max: float = 0.05        # m       lateral offset
    eth_max: float = 0.20       # rad     heading error (about 11 degrees)
    int_max: float = 0.05       # m s     accumulated lateral offset
    dw_max: float = 0.50        # rad/s   steering effort beyond what the path needs


def design_lqr(v: float, Ts: float, weights: LqrWeights = LqrWeights(), integral: bool = True):
    """Gain K (1 x 3 with integral action, 1 x 2 without), closed-loop discrete poles. dw = -K [e_y, e_theta, int e_y]."""
    A, B = error_model(v, 0.0)
    Ad, Bd = discretise(A, B, Ts)
    if integral:
        Ad = np.block([[Ad, np.zeros((2, 1))], [Ts * np.array([[1.0, 0.0]]), np.ones((1, 1))]])
        Bd = np.vstack([Bd, [[0.0]]])
        Q = np.diag([1 / weights.ey_max ** 2, 1 / weights.eth_max ** 2, 1 / weights.int_max ** 2])
    else:
        Q = np.diag([1 / weights.ey_max ** 2, 1 / weights.eth_max ** 2])
    R = np.array([[1 / weights.dw_max ** 2]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, np.linalg.eigvals(Ad - Bd @ K)


class Path:
    """A dense planar path: points (N, 2), tangent headings psi (N,) and curvature kappa (N,). `closed` loops the index."""

    def __init__(self, points, psi, kappa, closed: bool = False):
        self.p, self.psi, self.kappa, self.closed = np.asarray(points, float), np.asarray(psi, float), np.asarray(kappa, float), closed
        seg = np.hypot(*np.diff(self.p, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1] + (np.hypot(*(self.p[0] - self.p[-1])) if closed else 0.0))

    def __len__(self):
        return len(self.p)

    def errors(self, pose, i: int):
        """(e_y, e_theta) of `pose` against path point i."""
        d = np.asarray(pose[:2]) - self.p[i]
        psi = self.psi[i]
        return float(-np.sin(psi) * d[0] + np.cos(psi) * d[1]), float(wrap_angle(pose[2] - psi))

    def nearest(self, pose, hint: int | None = None, window: int = 40):
        """Index of the closest point, searched in a window around `hint` (so a loop cannot jump to the other side)."""
        n = len(self.p)
        if hint is None:
            idx = np.arange(n)
        elif self.closed:
            idx = (hint + np.arange(-window // 4, window)) % n
        else:
            idx = np.arange(max(hint - window // 4, 0), min(hint + window, n))
        d = np.hypot(*(self.p[idx] - np.asarray(pose[:2])).T)
        return int(idx[int(np.argmin(d))])


def fillet_path(points, radius: float = 0.3, step: float = 0.02):
    """Dense Path through a polyline with each corner replaced by a circular arc (tangent length capped at 45 % of the
    shorter neighbouring segment, so tight corners get a smaller radius). Returns (Path, vertex_s): vertex_s[k] is the
    arc length at which the path passes input point k (the corner's arc midpoint for interior points)."""
    P = np.asarray(points, float)
    n = len(P)
    pts, psi, kap, vs = [], [], [], [0.0]

    def line(a, b, k0=0.0):
        L = float(np.hypot(*(b - a)))
        m = max(int(np.ceil(L / step)), 1)
        h = float(np.arctan2(*(b - a)[::-1]))
        for j in range(m):
            pts.append(a + (b - a) * j / m); psi.append(h); kap.append(0.0)

    cur = P[0]
    for k in range(1, n - 1):
        a, b, c = P[k - 1], P[k], P[k + 1]
        l1, l2 = float(np.hypot(*(b - a))), float(np.hypot(*(c - b)))
        if l1 < 1e-9 or l2 < 1e-9:
            vs.append(len(pts) * step)
            continue
        u1, u2 = (b - a) / l1, (c - b) / l2
        phi = float(wrap_angle(np.arctan2(u2[1], u2[0]) - np.arctan2(u1[1], u1[0])))
        if abs(phi) < 1e-3 or abs(phi) > np.pi - 1e-3:
            line(cur, b); cur = b; vs.append(len(pts) * step)
            continue
        d = min(radius * np.tan(abs(phi) / 2), 0.45 * min(l1, l2))
        r = d / np.tan(abs(phi) / 2)
        t_in, t_out = b - u1 * d, b + u2 * d
        line(cur, t_in)
        sgn = np.sign(phi)
        ctr = t_in + sgn * r * np.array([-u1[1], u1[0]])
        a0 = float(np.arctan2(*(t_in - ctr)[::-1]))
        m = max(int(np.ceil(r * abs(phi) / step)), 1)
        for j in range(m):
            a_ = a0 + sgn * abs(phi) * j / m
            pts.append(ctr + r * np.array([np.cos(a_), np.sin(a_)]))
            psi.append(a_ + sgn * np.pi / 2); kap.append(sgn / r)
        vs.append((len(pts) - m / 2) * step)
        cur = t_out
    line(cur, P[-1])
    pts.append(P[-1]); psi.append(psi[-1]); kap.append(0.0)
    path = Path(np.array(pts), np.array(psi), np.array(kap))
    vs.append(path.length)
    return path, np.array(vs)


def stadium(cx=3.0, cy=0.0, half_straight=1.0, radius=1.0, step=0.02) -> Path:
    """Counter-clockwise stadium loop, starting at the start of the lower straight, heading east."""
    pts, psi, kap = [], [], []
    x0, x1 = cx - half_straight, cx + half_straight
    for x in np.arange(x0, x1, step):                                           # lower straight, east
        pts.append((x, cy - radius)); psi.append(0.0); kap.append(0.0)
    for a in np.arange(-np.pi / 2, np.pi / 2, step / radius):                   # right semicircle, counter-clockwise
        pts.append((x1 + radius * np.cos(a), cy + radius * np.sin(a))); psi.append(a + np.pi / 2); kap.append(1 / radius)
    for x in np.arange(x1, x0, -step):                                          # upper straight, west
        pts.append((x, cy + radius)); psi.append(np.pi); kap.append(0.0)
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, step / radius):                # left semicircle
        pts.append((x0 + radius * np.cos(a), cy + radius * np.sin(a))); psi.append(a + np.pi / 2); kap.append(1 / radius)
    return Path(pts, wrap_angle(np.array(psi)), kap, closed=True)


class LqrTracker:
    """dw = -K [e_y, e_theta, int e_y] on the estimated pose, w = v kappa + dw, v constant (slowed if far off the path)."""

    def __init__(self, path: Path, v_ref: float = 0.2, w_max: float = 1.0, Ts: float = 0.02,
                 weights: LqrWeights = LqrWeights(), integral: bool = True, int_band: float = 0.10,
                 rotate_above: float | None = None):
        self.int_band, self.rotate_above = int_band, rotate_above
        self.path, self.v_ref, self.w_max, self.Ts, self.integral = path, v_ref, w_max, Ts, integral
        self.K, self.poles = design_lqr(v_ref, Ts, weights, integral)
        self.i, self.z = None, 0.0
        self.dw = 0.0

    def reset(self):
        self.i, self.z, self.dw = None, 0.0, 0.0

    def command(self, pose):
        """(v, w) for the estimated pose."""
        self.i = self.path.nearest(pose, self.i)
        ey, eth = self.path.errors(pose, self.i)
        x = np.array([ey, eth, self.z] if self.integral else [ey, eth])
        self.dw = float(-(self.K @ x)[0])
        w = float(np.clip(self.path.kappa[self.i] * self.v_ref + self.dw, -self.w_max, self.w_max))
        if self.integral and abs(w) < self.w_max and abs(ey) < self.int_band:   # anti-windup: not while saturated or far off the path
            self.z = float(np.clip(self.z + ey * self.Ts, -0.5, 0.5))
        if self.rotate_above is not None and abs(eth) > self.rotate_above:      # far from the path heading: turn on the spot
            return 0.0, float(np.clip(-2.0 * eth, -self.w_max, self.w_max))
        v = self.v_ref * (1.0 if abs(eth) < 1.0 else 0.3)
        return v, w
