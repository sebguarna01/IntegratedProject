"""Figure: the head camera profiles side by side (the ideal camera and the simulated Intel RealSense D435i colour camera).

    python scripts/camera_profile_figure.py [--camera d435i] [--level large_6_pillars] [--seed 3]

Panels: the D435i on the pan-tilt head (MuJoCo view from outside the robot); the same scene through the ideal camera and
through the D435i with the head still; the D435i while the head pans at the gaze slew limit (1.5 rad/s), with motion
blur and rolling-shutter skew; a 4x crop of the dock marker (demosaicing, sensor noise); the profile's parameters.
Output: results/figures/camera_profiles.png (another camera: camera_profiles_<camera>.png)
"""
import argparse

import cv2
import numpy as np

from activenav.config import apply_overrides, load_config
from activenav.evaluation.search import make_search_session
from activenav.evaluation.viz import put
from activenav.types import Command


def label(img, text):
    img = img.copy()
    put(img, text, (10, 24), 0.6)
    return img


def marker_crop(img, half_w=70, half_h=45, zoom=4):
    """4x crop around the first ArUco marker in the image (the image centre if none is found)."""
    d = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = d.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    cx, cy = (corners[0].reshape(4, 2).mean(axis=0) if ids is not None else (img.shape[1] / 2, img.shape[0] / 2))
    x0 = int(np.clip(cx - half_w, 0, img.shape[1] - 2 * half_w))
    y0 = int(np.clip(cy - half_h, 0, img.shape[0] - 2 * half_h))
    return cv2.resize(img[y0:y0 + 2 * half_h, x0:x0 + 2 * half_w], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def spec_panel(p, cam, w, h, name="d435i"):
    img = np.full((h, w, 3), 30, np.uint8)
    lines = [f"Simulated Intel RealSense {name[:1].upper() + name[1:]} colour camera",
             f"stream {cam.width} x {cam.height} at {p['fps']} Hz, FOV {np.degrees(cam.hfov):.1f} x {np.degrees(cam.vfov):.1f} deg",
             f"fx {p['fx']} fy {p['fy']} px, principal point ({p['cx']}, {p['cy']})",
             "lens distortion k1 k2 p1 p2 k3 = " + " ".join(f"{v:g}" for v in p["dist"]),
             f"exposure {1000 * p['exposure_s']:.0f} ms (motion blur), " + (f"rolling shutter {1000 * p['readout_s']:.0f} ms" if p["readout_s"] > 0 else "global shutter"),
             f"latency {1000 * p['latency_s']:.0f} ms, frames time-stamped with head angles",
             f"Bayer {p['bayer']}, {p['adc_bits']}-bit, full well {p['full_well_e']} e-, read noise {p['read_noise_e']} e-",
             f"vignetting {100 * p['vignetting_corner']:.0f}% at the corners, sRGB",
             f"body {1000 * p['body_size'][1]:.0f} x {1000 * p['body_size'][2]:.0f} x {1000 * p['body_size'][0]:.0f} mm, "
             f"stereo baseline {1000 * p['stereo_baseline_m']:.0f} mm",
             "",
             "datasheet: FOV, frame rate, body, baseline",
             "assumed (typical): intrinsics detail, distortion, exposure,",
             "readout, latency, sensor noise, vignetting (configs/default.yaml)"]
    for i, ln in enumerate(lines):
        put(img, ln, (14, 34 + 26 * i), 0.5, (255, 255, 255) if i < 9 else (170, 170, 170))
    return img


def fit(img, w, h):
    """Letterbox an image into w x h."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 30, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def session(cfg, level, seed, pan):
    s, _ = make_search_session(cfg, {}, "B", seed, level=level, sensor="synthetic")
    for _ in range(60):
        s.env.step(Command(pan_target=pan, tilt_target=0.05))
    return s.env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="d435i", help="a profile in `cameras:`")
    ap.add_argument("--level", default="large_6_pillars")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pan", type=float, default=0.6)
    args = ap.parse_args()
    base = load_config()
    W, H = 640, 400

    ideal = session(apply_overrides(load_config(), ["camera=ideal"]), args.level, args.seed, args.pan)
    img_ideal = ideal.render(noise=True)
    ideal.close()

    cfg = apply_overrides(load_config(), [f"camera={args.camera}"])
    env = session(cfg, args.level, args.seed, args.pan)
    c = env.camera
    img_still = env.render(noise=True)
    # the D435i on the head, seen from beside the robot
    hid = env.model.body("head_tilt").id
    head = env.data.xpos[hid].copy()
    yaw = float(np.degrees(env.ground_truth()[2] + args.pan))
    close, _, _ = env.render_free(head, 0.42 * cfg["cameras"][args.camera]["body_size"][1] / 0.09, yaw + 205.0, -18.0, W, H, fovy=40.0)
    pan = args.pan
    for _ in range(12):                                   # swing the head at the gaze slew limit
        pan -= 1.5 * env.control_dt
        env.step(Command(pan_target=pan, tilt_target=0.05))
    img_moving = env.render(noise=True)
    env.close()

    crop = marker_crop(img_still)
    bgr = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    panels = [label(fit(bgr(close), W, H), f"Intel RealSense {args.camera.upper()[:1] + args.camera[1:]} on the pan-tilt head"),
              label(fit(bgr(img_ideal), W, H), "ideal camera: 640x480, 60 deg vertical, pinhole"),
              label(fit(bgr(img_still), W, H), f"{args.camera} colour: {c.width}x{c.height}, "
                                               f"{np.degrees(c.hfov):.0f}x{np.degrees(c.vfov):.0f} deg, head still"),
              label(fit(bgr(img_moving), W, H), (f"{args.camera}, head panning 1.5 rad/s: blur + rolling shutter" if cfg["cameras"][args.camera]["readout_s"] > 0 else f"{args.camera}, head panning 1.5 rad/s: blur (global shutter)")),
              label(fit(bgr(crop), W, H), f"{args.camera}, 4x crop: demosaicing, sensor noise"),
              spec_panel(cfg["cameras"][args.camera], c, W, H, args.camera)]
    fig = np.vstack([np.hstack(panels[i:i + 3]) for i in (0, 3)])
    out = base["_root"] / "results" / "figures" / ("camera_profiles.png" if args.camera == "d435i" else f"camera_profiles_{args.camera}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig)
    print("saved", out.relative_to(base["_root"]))


if __name__ == "__main__":
    main()
