"""ArUco markers: detection and planar pose (aruco), the marker library, the measurement chain from image to range, bearing and normal in the base frame, and the camera-backed marker sensor."""
from __future__ import annotations

import cv2
import numpy as np

from activenav.sensing.camera import camera_model
from activenav.sensing.camera import hom
from activenav.sensing.camera import camera_to_base
from activenav.types import MarkerDetection
from activenav.types import MarkerMeasurement
from activenav.types import SensorBundle


# -------------------- sensing/markers.py --------------------
# ArUco marker constants and image generation (generation needs OpenCV, constants do not).
DICTIONARY = "DICT_4X4_50"

# The printed plate is bigger than the black marker square: a white margin is needed for detection.
# marker side / plate side. The scene plates are 0.40 m, so the black square is 0.30 m.
MARKER_FRACTION = 0.75

def generate_marker_image(marker_id: int, px: int = 600, mirror: bool = False) -> np.ndarray:
    """Grey uint8 image of the marker plate (black marker square + white margin)."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(d, marker_id, px, borderBits=1)
    else:                                              # older OpenCV
        img = cv2.aruco.drawMarker(d, marker_id, px, borderBits=1)
    margin = int(round(px * (1.0 / MARKER_FRACTION - 1.0) / 2.0))
    img = np.pad(img, margin, mode="constant", constant_values=255)
    return np.fliplr(img).copy() if mirror else img


# -------------------- sensing/markers.py --------------------
# ArUco detection and pose estimation (needs OpenCV >= 4.7 with the aruco module).
class ArucoMarkerDetector:
    def __init__(self, K: np.ndarray, marker_size: float, dist_coeffs=None, refine: bool = True):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.zeros(5) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
        params = cv2.aruco.DetectorParameters()
        if refine:                                   # sub-pixel corner refinement
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(d, params)
        s = marker_size / 2.0
        # Marker frame: x right, y up, z toward the viewer. Corner order: TL, TR, BR, BL.
        self._obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    def detect(self, rgb: np.ndarray) -> list[MarkerDetection]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(self._obj, pts, self.K, self.dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            out.append(MarkerDetection(int(i), pts, R, tvec.reshape(3)))
        return out


# -------------------- sensing/markers.py --------------------
# Turn an ArUco pose (in the camera frame) into a planar measurement in the robot base frame.
def marker_world_pose(x: float, y: float, z: float, normal_angle: float) -> np.ndarray:
    """T_W_M for a wall marker. Marker frame: x = tangent (right, seen from the front),
    y = up, z = outward normal (toward the viewer)."""
    n = np.array([np.cos(normal_angle), np.sin(normal_angle), 0.0])
    t = np.array([-np.sin(normal_angle), np.cos(normal_angle), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return hom(R=np.column_stack([t, up, n]), t=[x, y, z])

def measure_marker(T_B_C: np.ndarray, det: MarkerDetection, base_height: float = 0.0) -> MarkerMeasurement:
    T_B_M = T_B_C @ hom(R=det.R_cm, t=det.t_cm)
    p = T_B_M[:3, 3]
    n = T_B_M[:3, 2]                       # marker z axis = outward normal
    return MarkerMeasurement(
        marker_id=det.marker_id,
        range=float(np.hypot(p[0], p[1])),
        bearing=float(np.arctan2(p[1], p[0])),
        normal_angle=float(np.arctan2(n[1], n[0])),
        height=float(p[2] + base_height),
    )


# -------------------- sensing/markers.py --------------------
# Camera image -> marker measurements in the robot base frame (the real sensing path).
class MarkerSensor:
    needs_image = True

    def __init__(self, cfg: dict):
        cam = camera_model(cfg)                                 # intrinsics and distortion of the camera profile in use
        self._det = ArucoMarkerDetector(cam.K, cfg["perception"]["marker_size_m"], dist_coeffs=cam.dist)
        self._head = cam.head
        self.last_detections = []       # corners of the most recent frame's detections (for the live viewer)

    def measure(self, sb: SensorBundle) -> list[MarkerMeasurement]:
        if sb.image is None:
            return []
        T_B_C = camera_to_base(self._head, sb.view_pan, sb.view_tilt)   # measured (noisy) head angles when the image was taken
        self.last_detections = list(self._det.detect(sb.image))
        base_h = float(self._head.get("base_height", 0.0))
        return [measure_marker(T_B_C, d, base_h) for d in self.last_detections]
