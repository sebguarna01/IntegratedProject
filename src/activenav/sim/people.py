"""Walking people: kinematic cylinders that move through a world and bounce off walls, obstacles and each other.

A person is a vertical cylinder (radius `radius`, height `height`) moved by the simulator, not by physics: it is
a MuJoCo mocap body, so it pushes the robot if they touch but nothing pushes it. Two walking styles:

    constant   a fixed speed and heading; the person walks straight and reflects off whatever it meets
    random     the heading wanders (a random walk on the turn rate) and the speed drifts around its mean,
               so the path cannot be predicted from two observations

People do not plan around the robot, with one courtesy so nobody walks into a robot that is standing still or blocks
it for ever: a person stops while the robot is closer than `yield_gap` (surface to surface); if the robot is still
that close after `yield_wait` seconds the person walks directly away from it until it is `yield_gap` + 0.3 m clear,
then carries on as before. Set `yield_gap: 0` to take the courtesy away.

Everything is deterministic in the trial seed (`reset(seed)`), and independent of the robot's random numbers, so
every configuration meets the same people on the same seed and comparisons stay paired.
"""
from __future__ import annotations

import numpy as np

from activenav.geometry import obstacle_sdf

DEFAULTS = dict(count=3, mode="constant", speed=0.4, radius=0.2, height=1.7, yield_gap=0.35, yield_wait=3.0,
                turn_rate_std=0.8, speed_std=0.15, speed_min=0.1, speed_max=0.8, spawn_clear=1.5)
MODES = ("constant", "random")


def people_settings(spec: dict | None) -> dict | None:
    """Fill in defaults and check a world's `people:` block (None or empty: no people)."""
    if not spec:
        return None
    p = {**DEFAULTS, **spec}
    if p["mode"] not in MODES:
        raise ValueError(f"people.mode must be one of {MODES}, got '{p['mode']}'")
    p["count"] = int(p["count"])
    if p["count"] < 1:
        return None
    for k in ("speed", "radius", "height"):
        if float(p[k]) <= 0:
            raise ValueError(f"people.{k} must be positive")
    return p


class PeopleModel:
    def __init__(self, settings: dict, bounds, obstacles, robot_radius: float, dt: float):
        self.p = settings
        self.n = settings["count"]
        self.r = float(settings["radius"])
        self.bounds = tuple(float(v) for v in bounds)           # inner faces of the walls: x0, x1, y0, y1
        self.obstacles = list(obstacles)
        self.robot_radius = float(robot_radius)
        self.dt = float(dt)
        self.pos = np.zeros((self.n, 2))
        self.heading = np.zeros(self.n)
        self.speed = np.zeros(self.n)
        self.vel = np.zeros((self.n, 2))
        self.wait = np.zeros(self.n)                    # seconds spent standing still because the robot was close
        self.leaving = np.zeros(self.n, bool)           # walking away from a robot that did not leave
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ start
    def _free(self, x, y, others, extra=()):
        x0, x1, y0, y1 = self.bounds
        m = self.r + 0.3
        if not (x0 + m < x < x1 - m and y0 + m < y < y1 - m):
            return False
        if any(float(obstacle_sdf(o, x, y)) < self.r + 0.3 for o in self.obstacles):
            return False
        if any(np.hypot(x - a, y - b) < 2 * self.r + 0.4 for a, b in others):
            return False
        return all(np.hypot(x - a, y - b) >= d for a, b, d in extra)

    def reset(self, seed: int | None, robot_xy) -> None:
        """Place the people (random free spots away from the robot's start) and give them a heading."""
        self.rng = np.random.default_rng([0 if seed is None else int(seed), 5151])
        keep = [(float(robot_xy[0]), float(robot_xy[1]), float(self.p["spawn_clear"]))]
        x0, x1, y0, y1 = self.bounds
        placed = []
        for _ in range(self.n):
            for _ in range(5000):
                x, y = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
                if self._free(x, y, placed, keep):
                    placed.append((x, y))
                    break
            else:
                raise RuntimeError("could not place the people in free space")
        self.pos = np.array(placed, float)
        self.heading = self.rng.uniform(-np.pi, np.pi, self.n)
        self.speed = np.full(self.n, float(self.p["speed"]))
        if self.p["mode"] == "random":
            self.speed = np.clip(self.rng.normal(self.p["speed"], self.p["speed_std"], self.n),
                                 self.p["speed_min"], self.p["speed_max"])
        self.wait[:] = 0.0
        self.leaving[:] = False
        self._set_vel()

    def _set_vel(self):
        self.vel = self.speed[:, None] * np.stack([np.cos(self.heading), np.sin(self.heading)], axis=1)

    # ------------------------------------------------------------------- step
    def _normal_at(self, x, y):
        """Unit vector pointing away from the nearest wall/obstacle surface at (x, y), or None if there is none within reach."""
        x0, x1, y0, y1 = self.bounds
        best, nrm = np.inf, None
        for d, n in ((x - x0, (1.0, 0.0)), (x1 - x, (-1.0, 0.0)), (y - y0, (0.0, 1.0)), (y1 - y, (0.0, -1.0))):
            if d < best:
                best, nrm = d, n
        for o in self.obstacles:
            d = float(obstacle_sdf(o, x, y))
            if d < best:
                e = 1e-3
                g = np.array([float(obstacle_sdf(o, x + e, y)) - float(obstacle_sdf(o, x - e, y)),
                              float(obstacle_sdf(o, x, y + e)) - float(obstacle_sdf(o, x, y - e))])
                if np.hypot(*g) > 1e-9:
                    best, nrm = d, tuple(g / np.hypot(*g))
        return best, np.array(nrm)

    def _blocked(self, i, x, y):
        """Hits a wall or obstacle (person radius)?"""
        d, _ = self._normal_at(x, y)
        return d < self.r

    def step(self, robot_xy) -> np.ndarray:
        """Advance every person by one control step; returns the positions (n, 2)."""
        p, dt = self.p, self.dt
        if p["mode"] == "random":
            self.heading += self.rng.normal(0.0, p["turn_rate_std"] * np.sqrt(dt), self.n)
            self.speed = np.clip(self.speed + 0.5 * (p["speed"] - self.speed) * dt
                                 + self.rng.normal(0.0, p["speed_std"] * np.sqrt(dt), self.n),
                                 p["speed_min"], p["speed_max"])
            self._set_vel()
        gap0 = float(p["yield_gap"])
        for i in range(self.n):
            if gap0 > 0:
                away = self.pos[i] - np.asarray(robot_xy[:2])
                d = float(np.hypot(*away))
                near = d < self.r + self.robot_radius + gap0
                if self.leaving[i] and d > self.r + self.robot_radius + gap0 + 0.3:
                    self.leaving[i], self.wait[i] = False, 0.0       # clear of the robot: carry on along this heading
                if near and not self.leaving[i]:
                    self.wait[i] += dt
                    if self.wait[i] >= float(p["yield_wait"]):
                        self.leaving[i] = True
                    else:
                        continue                                      # courtesy: stand still for now
                if not near and not self.leaving[i]:
                    self.wait[i] = 0.0
                if self.leaving[i]:
                    self.heading[i] = float(np.arctan2(away[1], away[0]))
                    self.vel[i] = max(self.speed[i], 0.3) * np.array([np.cos(self.heading[i]), np.sin(self.heading[i])])
            new = self.pos[i] + self.vel[i] * dt
            if self.leaving[i] and self._blocked(i, new[0], new[1]):
                # cornered against a wall or obstacle while walking away from the robot: slide along the surface, away from it
                _, n = self._normal_at(new[0], new[1])
                t = self.vel[i] - float(self.vel[i] @ n) * n
                if np.hypot(*t) < 1e-3:
                    t = np.array([-n[1], n[0]])
                    if float(t @ (self.pos[i] - np.asarray(robot_xy[:2]))) < 0:
                        t = -t
                alt = self.pos[i] + dt * self.speed[i] * t / np.hypot(*t)
                if not self._blocked(i, alt[0], alt[1]) and not any(
                        np.hypot(*(alt - self.pos[j])) < 2 * self.r for j in range(self.n) if j != i):
                    self.pos[i] = alt
                continue
            hit_other = any(np.hypot(*(new - self.pos[j])) < 2 * self.r for j in range(self.n) if j != i)
            if self._blocked(i, new[0], new[1]) or hit_other:
                # reflect the velocity about the surface normal (or the line between two people), then wait one step
                if hit_other:
                    j = min((j for j in range(self.n) if j != i), key=lambda j: np.hypot(*(new - self.pos[j])))
                    n = self.pos[i] - self.pos[j]
                    n = n / max(np.hypot(*n), 1e-9)
                else:
                    _, n = self._normal_at(new[0], new[1])
                vn = float(self.vel[i] @ n)
                if vn < 0:
                    self.vel[i] = self.vel[i] - 2.0 * vn * n
                else:                                    # already moving away but still too close: nudge along the normal
                    self.vel[i] = self.vel[i] + 0.2 * n
                sp = max(np.hypot(*self.vel[i]), 1e-9)
                self.vel[i] *= self.speed[i] / sp
                self.heading[i] = float(np.arctan2(self.vel[i][1], self.vel[i][0]))
                continue
            self.pos[i] = new
        return self.pos

    # ------------------------------------------------------------------ views
    def circles(self) -> list:
        return [dict(x=float(x), y=float(y), r=self.r) for x, y in self.pos]
