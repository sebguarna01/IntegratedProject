"""Generate the ArUco marker textures used by models/docking_scene.xml.

    python scripts/make_markers.py            # run once, before loading the scene
    python scripts/make_markers.py --mirror   # only if the check script reports mirrored markers
"""
import argparse

import cv2

from activenav.config import load_config
from activenav.sensing.markers import generate_marker_image

MARKER_IDS = [0, 1, 2, 3, 4, 5, 10]       # must match the marker_<id> geoms in the XML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    out = load_config()["_root"] / "models" / "assets" / "markers"
    out.mkdir(parents=True, exist_ok=True)
    for mid in MARKER_IDS:
        img = generate_marker_image(mid, px=600, mirror=args.mirror)
        cv2.imwrite(str(out / f"aruco_{mid}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    print(f"wrote {len(MARKER_IDS)} marker images to {out}" + ("  (mirrored)" if args.mirror else ""))


if __name__ == "__main__":
    main()
