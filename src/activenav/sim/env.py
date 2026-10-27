"""MuJoCo wrapper. Exposes sensor outputs only.

The simulator's true state is available through ground_truth(), which is for the
evaluation package ONLY (enforced by tests/test_ground_truth_guard.py).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from activenav.geometry import obstacle_sdf

from activenav.kinematics import wheel_speeds_from_cmd, wrap_angle
from activenav.sensing.odometry import EncoderErrorModel
from activenav.types import Command, SensorBundle


def _id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise KeyError(f"'{name}' not found in the model")
    return i


class DockingEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        if cfg.get("world"):
            from activenav.sim.scene_builder import build_scene
            model_path = build_scene(cfg, cfg["world"], cfg.get("world_markers"), cfg.get("world_tag", ""))      # generated from configs/default.yaml `worlds`
        else:
            model_path = Path(cfg["_root"]) / cfg["sim"]["model"]
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        # MuJoCo's near clip is znear * model extent, so it grew with the room (0.17 m in 10x6, 0.21 m in 10x10) and
        # clipped the nearest floor at steep tilt. Pin it to the 10x6 value: it also hides the robot's own body.
        self.model.vis.map.znear = 0.17 / float(self.model.stat.extent)
        self.data = mujoco.MjData(self.model)
        m = self.model

        J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        self._q = {n: m.jnt_qposadr[_id(m, J, n)]
                   for n in ("base_joint", "wheel_left", "wheel_right", "head_pan", "head_tilt")}
        self._a = {n: _id(m, A, n) for n in ("wheel_left", "wheel_right", "head_pan", "head_tilt")}

        # Robot geometry read from the model (single source of truth)
        self.wheel_radius = float(m.body("wheel_left").pos[2])
        self.track_width = float(m.body("wheel_left").pos[1] - m.body("wheel_right").pos[1])
        self.max_wheel_speed = float(m.actuator_ctrlrange[self._a["wheel_left"], 1])
        self.pan_range = tuple(m.actuator_ctrlrange[self._a["head_pan"]])
        self.tilt_range = tuple(m.actuator_ctrlrange[self._a["head_tilt"]])

        self._robot_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                              for n in ("base", "wheel_left", "wheel_right", "head_pan", "head_tilt")}

        self.control_dt = float(cfg["sim"]["control_dt"])
        self._n_sub = int(round(self.control_dt / m.opt.timestep))
        if abs(self._n_sub * m.opt.timestep - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be a multiple of the model timestep")

        # walking people (mocap bodies person_i), moved once per control step; none in most worlds
        self.people = None
        self._person_mocap = []
        pset = None
        if cfg.get("world"):
            from activenav.sim.people import PeopleModel, people_settings
            pset = people_settings(cfg["worlds"][cfg["world"]].get("people"))
        if pset:
            self._person_mocap = [int(m.body_mocapid[_id(m, mujoco.mjtObj.mjOBJ_BODY, f"person_{i}")]) for i in range(pset["count"])]
            self.people = PeopleModel(pset, self.room_bounds(), self.obstacle_map(), float(cfg["robot"]["radius"]), self.control_dt)
        self._noise = cfg["noise"]
        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self._enc = EncoderErrorModel(self._noise, self._rng)
        self._renderer = None

    # ------------------------------------------------------------------ control
    def reset(self, seed: int | None = None, start_pose=(1.0, -0.5, 0.0)) -> SensorBundle:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        x, y, yaw = start_pose
        if self.people is not None:                    # the same seed gives the same people for every configuration
            self.people.reset(seed, (x, y))
            self._place_people()
        a = self._q["base_joint"]
        d.qpos[a:a + 3] = (x, y, 0.002)
        d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(m, d)
        for _ in range(int(0.3 / m.opt.timestep)):    # let the robot settle on the floor
            mujoco.mj_step(m, d)
        d.qpos[a:a + 2] = (x, y)                       # undo any settling drift
        mujoco.mj_forward(m, d)

        self._enc.reset(self._true_encoders(), rng=self._rng)   # draws this trial's systematic errors
        return self._sensors(render=False)

    def step(self, cmd: Command, render: bool = False) -> SensorBundle:
        d = self.data
        wl, wr = wheel_speeds_from_cmd(cmd.v, cmd.w, self.wheel_radius, self.track_width)
        lim = self.max_wheel_speed
        d.ctrl[self._a["wheel_left"]] = np.clip(wl, -lim, lim)
        d.ctrl[self._a["wheel_right"]] = np.clip(wr, -lim, lim)
        d.ctrl[self._a["head_pan"]] = np.clip(cmd.pan_target, *self.pan_range)
        d.ctrl[self._a["head_tilt"]] = np.clip(cmd.tilt_target, *self.tilt_range)
        if self.people is not None:
            self.people.step(self._base_xy())
            self._place_people()
        for _ in range(self._n_sub):
            mujoco.mj_step(self.model, d)
        return self._sensors(render)

    def _base_xy(self):
        a = self._q["base_joint"]
        return float(self.data.qpos[a]), float(self.data.qpos[a + 1])

    def _place_people(self) -> None:
        for k, (x, y) in zip(self._person_mocap, self.people.pos):
            self.data.mocap_pos[k] = (x, y, 0.0)

    # ------------------------------------------------------------------ sensors
    def _true_encoders(self) -> np.ndarray:
        d = self.data
        return np.array([d.qpos[self._q["wheel_left"]], d.qpos[self._q["wheel_right"]]])

    def _sensors(self, render: bool) -> SensorBundle:
        d = self.data
        enc = self._enc.measure(self._true_encoders())

        head_std = self._noise["head_encoder_std"]
        pan = d.qpos[self._q["head_pan"]] + self._rng.normal(0.0, head_std)
        tilt = d.qpos[self._q["head_tilt"]] + self._rng.normal(0.0, head_std)
        image = self.render() if render else None
        return SensorBundle(t=float(d.time), enc_left=float(enc[0]), enc_right=float(enc[1]),
                            pan=float(pan), tilt=float(tilt), image=image)

    def render(self, noise: bool = True) -> np.ndarray:
        """RGB image from the head camera, uint8 (H, W, 3).

        noise=False skips the sensor noise AND does not touch the random generator, so rendering extra frames
        purely for display (the live viewer) cannot change the trial's random sequence."""
        cam = self.cfg["sim"]["camera"]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=cam["height"], width=cam["width"])
        self._renderer.update_scene(self.data, camera=cam["name"])
        img = self._renderer.render().copy()
        std = float(self.cfg.get("camera_model", {}).get("image_noise_std", 0.0))
        if noise and std > 0.0:                         # sensor noise: Gaussian, in gray levels
            img = np.clip(img.astype(np.float32) + self._rng.normal(0.0, std, img.shape),
                          0, 255).astype(np.uint8)
        return img

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------ evaluation only
    def ground_truth(self) -> np.ndarray:
        """True base pose [x, y, yaw]. EVALUATION ONLY: never call from perception,
        estimation, mapping, planning, control or behaviour code."""
        a = self._q["base_joint"]
        x, y = self.data.qpos[a], self.data.qpos[a + 1]
        qw, qx, qy, qz = self.data.qpos[a + 3:a + 7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return np.array([x, y, float(wrap_angle(yaw))])

    def hidden_encoder_errors(self) -> dict:
        """This trial's hidden systematic encoder errors. EVALUATION / REPORTING ONLY."""
        return self._enc.errors()

    # ------------------------------------------------------------- known landmark map
    def marker_map(self) -> dict:
        """The surveyed marker layout: {id: dict(x, y, z, normal_angle, size)}.

        Markers are static landmarks, known in advance (design decision D4). This is NOT
        simulator state: it is the map the estimator is allowed to use.
        """
        m = self.model
        out = {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not name or not name.startswith("marker_"):
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            normal = R.reshape(3, 3)[:, 2]
            out[int(name.split("_")[1])] = {
                "x": float(m.geom_pos[g][0]), "y": float(m.geom_pos[g][1]), "z": float(m.geom_pos[g][2]),
                "normal_angle": float(np.arctan2(normal[1], normal[0])),
                "size": float(2.0 * m.geom_size[g][0] * 0.75),     # black square = 75% of the plate
            }
        return out

    def obstacle_map(self) -> list:
        """The surveyed obstacle layout: pillars [dict(x, y, r)] and wall blocks [dict(x0, x1, y0, y1)]. Like
        marker_map(), this is the map the planner and the gaze policy are allowed to use, read from the model's
        pillar_* and wall_block_* geoms. Empty in the 6x4 room."""
        m = self.model
        out = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name and name.startswith("pillar_"):
                out.append({"x": float(m.geom_pos[g][0]), "y": float(m.geom_pos[g][1]), "r": float(m.geom_size[g][0])})
            elif name and name.startswith("wall_block_"):
                cx, cy = (float(v) for v in m.geom_pos[g][:2])
                hx, hy = (float(v) for v in m.geom_size[g][:2])
                out.append({"x0": cx - hx, "x1": cx + hx, "y0": cy - hy, "y1": cy + hy})
        return out

    def room_bounds(self) -> tuple:
        """Inner faces of the four walls: (x_min, x_max, y_min, y_max). Part of the surveyed map."""
        m = self.model
        b = {}
        for side in ("north", "south", "east", "west"):
            g = _id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{side}")
            px, py = m.geom_pos[g][:2]
            sx, sy = m.geom_size[g][:2]
            b[side] = {"north": py - sy, "south": py + sy, "east": px - sx, "west": px + sx}[side]
        return float(b["west"]), float(b["east"]), float(b["south"]), float(b["north"])

    def min_clearance(self) -> float:
        """Distance from the TRUE base origin to the nearest wall or pillar surface. EVALUATION ONLY."""
        x, y, _ = self.ground_truth()
        x0, x1, y0, y1 = self.room_bounds()
        c = min(x - x0, x1 - x, y - y0, y1 - y)
        for o in self.obstacle_map():
            c = min(c, float(obstacle_sdf(o, x, y)))
        return c

    def people_state(self) -> list:
        """Where the walking people are right now: [dict(x, y, r)] (empty when the world has none). SIMULATOR TRUTH:
        for the stand-in sensors and scoring in evaluation/ only. The agent has to see people through the camera."""
        return self.people.circles() if self.people is not None else []

    def person_gap(self) -> float:
        """Surface-to-surface distance from the robot body to the nearest person (inf with nobody). EVALUATION ONLY."""
        if self.people is None:
            return float("inf")
        bx, by = self._base_xy()
        return float(min(np.hypot(c["x"] - bx, c["y"] - by) - c["r"] for c in self.people_state()) - self.cfg["robot"]["radius"])

    def person_contact(self) -> bool:
        """True if the robot is touching a person. EVALUATION ONLY (collision metric)."""
        if self.people is None:
            return False
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a)
                if name and name.startswith("person_") and m.geom_bodyid[b] in self._robot_bodies:
                    return True
        return False

    def wall_contact(self) -> bool:
        """True if any part of the robot is touching a wall or pillar. EVALUATION ONLY (collision metric)."""
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a)
                if name and name.startswith(("wall_", "pillar_")) and m.geom_bodyid[b] in self._robot_bodies:
                    return True
        return False
