"""Intel RealSense D435i colour camera, simulated as a real camera (the `d435i` camera profile; used by sim/env.py).

MuJoCo renders an ideal pinhole image. ColorCamera turns it into what the D435i colour stream would deliver:
  optics   the stream's intrinsics and Brown-Conrady lens distortion (the image is resampled through them), vignetting
  motion   motion blur (the pixels integrate the scene over the exposure while the camera turns) and rolling shutter (the
           rows are read one after another over `readout_s`, so a turning camera skews the image). Both come from the
           camera's angular velocity; its translation over 10-30 ms is a few millimetres and is ignored
  sensor   Bayer colour filter (one colour per pixel), photon shot noise and read noise in electrons, 10-bit ADC
  ISP      demosaicing (OpenCV edge-aware), gain, sRGB gamma
Timing (30 Hz frame grid, capture-to-delivery latency) is handled by DockingEnv, which renders the scene as it was at
capture time. mount_on_head puts the camera body (90 x 25 x 25 mm) on the pan-tilt head and places the colour lens.
Profile values: configs/default.yaml `cameras.d435i` (datasheet values and assumptions are marked there).
"""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

from activenav.sensing.camera import intrinsics_from_fovy

# cv2.COLOR_BayerBG2RGB decodes an R G / G B mosaic (OpenCV names the pattern from the second row); checked in the tests
_DEMOSAIC = {"RGGB": cv2.COLOR_BayerBG2RGB_EA, "BGGR": cv2.COLOR_BayerRG2RGB_EA,
             "GRBG": cv2.COLOR_BayerGB2RGB_EA, "GBRG": cv2.COLOR_BayerGR2RGB_EA}
_CHANNEL = {"RGGB": [[0, 1], [1, 2]], "BGGR": [[2, 1], [1, 0]], "GRBG": [[1, 0], [2, 1]], "GBRG": [[1, 2], [0, 1]]}


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


class ColorCamera:
    """Image formation of one colour camera profile. Call it with an ideal render (from the render camera described by
    render_w, render_h, render_fovy), the camera's angular velocity and a random generator (None: no sensor noise)."""

    def __init__(self, p: dict, margin: float = 0.06):
        self.W, self.H = int(p["width"]), int(p["height"])
        self.K = np.array([[p["fx"], 0.0, p["cx"]], [0.0, p["fy"], p["cy"]], [0.0, 0.0, 1.0]], float)
        self.dist = np.asarray(p["dist"], float)
        self.exposure, self.readout = float(p["exposure_s"]), float(p["readout_s"])
        self.full_well, self.read_noise = float(p["full_well_e"]), float(p["read_noise_e"])
        self.white, self.max_dn = float(p["exposure_white"]), float(2 ** int(p["adc_bits"]) - 1)
        # where every output pixel looks: undistorted normalised coordinates
        u, v = np.meshgrid(np.arange(self.W, dtype=float), np.arange(self.H, dtype=float))
        xy = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2), self.K, self.dist,
                                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)).reshape(self.H, self.W, 2)
        x, y = xy[..., 0], xy[..., 1]
        # render camera: centred square-pixel pinhole with the stream's focal length, covering every output ray plus a
        # margin for the motion shifts (`margin` in normalised units: 0.06 = 3.4 deg)
        f = float(0.5 * (self.K[0, 0] + self.K[1, 1]))
        self.render_w = 2 * int(np.ceil((np.abs(x).max() + margin) * f))
        self.render_h = 2 * int(np.ceil((np.abs(y).max() + margin) * f))
        self.render_fovy = float(np.degrees(2.0 * np.arctan(self.render_h / 2.0 / f)))
        Kr = intrinsics_from_fovy(self.render_fovy, self.render_w, self.render_h)
        self.u0 = (Kr[0, 0] * x + Kr[0, 2]).astype(np.float32)
        self.v0 = (Kr[1, 1] * y + Kr[1, 2]).astype(np.float32)
        self.map1, self.map2 = cv2.convertMaps(self.u0, self.v0, cv2.CV_16SC2)    # static resampling (camera not turning)
        # image motion [px/s] per rad/s of camera rotation (OpenCV camera axes): rotational optical flow
        #   du = wy f (1 + x^2) - wz f y - wx f x y        dv = -wx f (1 + y^2) + wz f x + wy f x y
        fr = float(Kr[0, 0])
        self._fu = [(fr * (1 + x * x)).astype(np.float32), (-fr * y).astype(np.float32), (-fr * x * y).astype(np.float32)]
        self._fv = [(-fr * (1 + y * y)).astype(np.float32), (fr * x).astype(np.float32), (fr * x * y).astype(np.float32)]
        self._flow_gain = fr * (1.0 + float((x * x + y * y).max()))          # bound on |flow| per rad/s
        self.t_row = ((np.arange(self.H) / (self.H - 1) - 0.5) * self.readout).astype(np.float32)[:, None]
        # vignetting: relative illumination 1 / (1 + a r^2)^2, equal to `vignetting_corner` at the farthest pixel
        r2 = x * x + y * y
        a = (1.0 / np.sqrt(float(p["vignetting_corner"])) - 1.0) / float(r2.max())
        self.vignetting = (1.0 / (1.0 + a * r2) ** 2).astype(np.float32)
        # Bayer filter: the colour channel each pixel keeps, as indices into the flattened (H, W, 3) image
        pattern = str(p.get("bayer", "RGGB")).upper()
        ch = np.tile(np.array(_CHANNEL[pattern]), (self.H // 2 + 1, self.W // 2 + 1))[:self.H, :self.W]
        self._pick = (np.arange(self.H * self.W) * 3 + ch.ravel()).astype(np.int64)
        self._demosaic = _DEMOSAIC[pattern]
        self._lut_in = _srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)
        lin = np.minimum(np.arange(int(self.max_dn) + 1) / (self.max_dn * self.white), 1.0)   # ISP gain: white back to 1
        self._lut_out = np.round(255.0 * _linear_to_srgb(lin)).astype(np.uint8)

    def __call__(self, rgb: np.ndarray, omega, rng: np.random.Generator | None = None) -> np.ndarray:
        """Colour image (H, W, 3) uint8 from an ideal render (render_h, render_w, 3) uint8. `omega` is the camera's
        angular velocity [rad/s] in OpenCV camera axes (x right, y down, z forward)."""
        lin = self._lut_in[rgb]                                             # light is linear; the render is sRGB
        wx, wy, wz = (float(w) for w in omega)
        flow = self._flow_gain * float(np.sqrt(wx * wx + wy * wy + wz * wz))   # px/s at the fastest pixel
        if flow * (self.readout + self.exposure) < 0.05:                     # not turning: plain resampling
            img = cv2.remap(lin, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            du = wy * self._fu[0] + wz * self._fu[1] + wx * self._fu[2]
            dv = wx * self._fv[0] + wz * self._fv[1] + wy * self._fv[2]
            n = int(np.clip(np.ceil(flow * self.exposure / 0.75), 1, 16))    # sub-frames ~0.75 px apart along the blur
            img = None
            for k in range(n):
                t = self.t_row + ((k + 0.5) / n - 0.5) * self.exposure        # each row's own time (rolling shutter)
                s = cv2.remap(lin, self.u0 + du * t, self.v0 + dv * t, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                img = s if img is None else img + s
            img /= n
        img *= self.vignetting[..., None]
        e = img.reshape(-1)[self._pick].reshape(self.H, self.W) * (self.white * self.full_well)   # electrons, Bayer
        if rng is not None:                                                 # shot noise (Poisson, Gaussian form) + read noise
            e = (e + np.sqrt(np.maximum(e, 0.0)) * rng.standard_normal(e.shape, dtype=np.float32)
                 + self.read_noise * rng.standard_normal(e.shape, dtype=np.float32))
        dn = np.clip(np.rint(e * (self.max_dn / self.full_well)), 0, self.max_dn).astype(np.uint16)
        return self._lut_out[cv2.cvtColor(dn, self._demosaic)]


def _add(body, name, kind, size, pos, rgba, quat=None):
    """A visual-only geom (no collision, no mass: the head's inertia is set in the XML, so physics is unchanged)."""
    g = body.add_geom()
    g.name, g.type, g.size, g.pos, g.rgba = name, kind, size, pos, rgba
    if quat is not None:
        g.quat = quat
    g.contype, g.conaffinity, g.group, g.density = 0, 0, 2, 0.0


_BOX, _CYL = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
_ALONG_X = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]                  # cylinder axis (z) turned onto x
_ALONG_Y = [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0.0, 0.0]                 # cylinder axis (z) turned onto y
# ROBOTIS Dynamixel XL430-W250 (the servo family of the TurtleBot3): 28.5 x 46.5 x 34 mm; output horn ~11 mm from one end
_SERVO = (0.0465, 0.0285, 0.034)                                             # long side, short side, along the horn axis [m]
_HORN_FROM_END = 0.0113
_SERVO_RGBA, _HORN_RGBA, _BRACKET_RGBA = [0.07, 0.07, 0.08, 1], [0.72, 0.72, 0.74, 1], [0.60, 0.61, 0.64, 1]
# The Waffle Pi's top plate is 15.5 mm below the pan joint origin of the head model (plate at z = 0.110 m in the base
# frame, measured by a ray cast on the base mesh). The pan axis is vertical, so only the tilt-axis height matters for
# the kinematics: the pan servo sits on the plate and the bracket reaches up to the same tilt axis.
_PLATE_BELOW_PAN = 0.0155


def build_pan_tilt(spec: mujoco.MjSpec, cam_width: float) -> None:
    """Replace the head's placeholder post with a two-servo pan-tilt unit, around the SAME joint axes (visual only).

    Pan: a Dynamixel XL430 lying on the top plate, horn up on the pan axis, its body reaching back. On its horn, an
    aluminium U-bracket turns with the pan; its two arms rise beside the camera up to the tilt axis. Tilt: a second
    XL430 bolted to the outside of one arm, horn on the tilt axis, driving the camera through a side plate; an idler
    bearing on the other arm. The bracket width follows the camera (`cam_width`)."""
    base, pan, tilt = spec.body("base"), spec.body("head_pan"), spec.body("head_tilt")
    for g in list(pan.geoms):                                                 # the old post
        spec.delete(g)
    po = np.array(pan.pos, float)                                             # pan axis on the plate, base frame
    tz = float(np.array(tilt.pos, float)[2])                                  # tilt axis height above the pan origin
    L, S, H = _SERVO
    back = L / 2 - _HORN_FROM_END                                             # servo centre behind its horn axis
    zb = -_PLATE_BELOW_PAN                                                    # top plate, in the pan frame
    # pan servo, fixed to the robot: lying on the plate, horn axis = pan axis
    _add(base, "pan_servo", _BOX, [L / 2, S / 2, H / 2], list(po + [-back, 0.0, zb + H / 2]), _SERVO_RGBA)
    # turning with the pan: horn, U-bracket (plate + two arms), tilt servo, idler
    _add(pan, "pan_horn", _CYL, [0.0125, 0.00125, 0], [0, 0, zb + H + 0.00125], _HORN_RGBA)
    gap, t = cam_width / 2 + 0.006, 0.0025                                    # half the inner width, plate thickness
    z0, z1 = zb + H + 0.0025, tz + 0.014                                      # bracket from the horn up past the tilt axis
    _add(pan, "bracket_plate", _BOX, [0.016, gap + t, t / 2], [0, 0, z0 + t / 2], _BRACKET_RGBA)
    for i, s in enumerate((-1, 1)):
        _add(pan, f"bracket_arm_{i}", _BOX, [0.012, t / 2, (z1 - z0) / 2], [0, s * (gap + t / 2), (z0 + z1) / 2], _BRACKET_RGBA)
    _add(pan, "tilt_servo", _BOX, [L / 2, H / 2, S / 2], [-back, gap + t + H / 2, tz], _SERVO_RGBA)
    _add(pan, "tilt_servo_horn", _CYL, [0.0125, 0.00125, 0], [0, gap - 0.00125, tz], _HORN_RGBA, _ALONG_Y)
    _add(pan, "idler", _CYL, [0.006, 0.002, 0], [0, -(gap + t + 0.002), tz], [0.15, 0.15, 0.16, 1], _ALONG_Y)
    # turning with the tilt: the plates that hold the camera on the horn and on the idler
    for i, s in enumerate((-1, 1)):
        _add(tilt, f"camera_plate_{i}", _BOX, [0.012, 0.0015, 0.011], [0, s * (cam_width / 2 + 0.002), 0], _BRACKET_RGBA)


def mount_on_head(spec: mujoco.MjSpec, p: dict, render_fovy: float, render_w: int, render_h: int) -> None:
    """Replace the head's generic camera block with the camera body (front glass, colour lens, IR windows) on a
    two-servo pan-tilt unit, and make `head_cam` the colour lens: its position on the head and the render camera's
    field of view."""
    body = spec.body("head_tilt")
    for g in list(body.geoms):
        spec.delete(g)
    dx, wy, hz = (float(v) for v in p["body_size"])                         # depth, width, height [m]
    add = lambda *a: _add(body, *a)
    box, cyl, along_x = _BOX, _CYL, _ALONG_X
    add("d435i_body", box, [dx / 2, wy / 2, hz / 2], [0, 0, 0], [0.20, 0.20, 0.22, 1])
    add("d435i_glass", box, [0.0008, wy / 2 - 0.002, hz / 2 - 0.002], [dx / 2 + 0.0008, 0, 0], [0.02, 0.02, 0.025, 1])
    lens = [float(v) for v in p["lens_offset"]]
    front = dx / 2 + 0.0016
    add("d435i_rgb_lens", cyl, [0.0045, 0.0006, 0], [front + 0.0006, lens[1], lens[2]], [0.05, 0.08, 0.16, 1], along_x)
    for i, side in enumerate((-1, 1)):                                     # the two IR imagers of the stereo pair
        yi = side * float(p["stereo_baseline_m"]) / 2
        add(f"d435i_ir_{i}", cyl, [0.0055, 0.0006, 0], [front + 0.0006, yi, 0], [0.10, 0.04, 0.05, 1], along_x)
    build_pan_tilt(spec, wy)
    cam = spec.camera("head_cam")
    cam.pos, cam.fovy = lens, float(render_fovy)
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), int(render_w))
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), int(render_h))
