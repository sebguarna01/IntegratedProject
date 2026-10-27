import numpy as np

from activenav.autonomy.obstacle_sensor import GroundGeometry
from activenav.config import load_config
from activenav.sensing.floor import FloorRanger

cfg = load_config()


def _wall_image(geo, wall_x, pan, tilt, floor=(184, 184, 184), wall=(235, 230, 217)):
    """What the camera sees of a wall plane x = wall_x (base frame) standing on a plain floor."""
    H, W = geo.H, geo.W
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    pts, ok = geo.pixel_to_floor(uu.ravel(), vv.ravel(), pan, tilt)
    is_floor = (ok & (pts[:, 0] < wall_x)).reshape(H, W)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = wall
    img[is_floor] = floor
    return img


def test_wall_ahead_is_ranged_correctly():
    geo = GroundGeometry(cfg)
    for wall_x, tilt in [(1.0, 0.3), (1.5, 0.3), (2.0, 0.2), (2.5, 0.35)]:
        fr = FloorRanger(cfg)
        sc = fr.scan(_wall_image(geo, wall_x, 0.0, tilt), 0.0, tilt)
        mid = np.abs(sc.points[:, 1]) < 0.15 * wall_x                    # columns looking nearly straight ahead
        assert sc.hit[mid].all()
        assert np.allclose(sc.points[mid, 0], wall_x, atol=max(0.03, 0.05 * wall_x)), (wall_x, sc.points[mid, 0])


def test_open_floor_gives_no_hit_and_stops_at_the_range_limit():
    geo = GroundGeometry(cfg)
    fr = FloorRanger(cfg)
    sc = fr.scan(_wall_image(geo, 50.0, 0.0, 0.2), 0.0, 0.2)
    assert not sc.hit.any()
    r = np.hypot(sc.points[:, 0] - sc.origin[0], sc.points[:, 1] - sc.origin[1])
    assert r.max() <= cfg["obstacles"]["max_range"] / np.cos(np.radians(40))   # rows are lines of equal depth: edge columns reach farther


def test_lit_floor_with_strong_lightness_gradient_and_dim_wall():
    """Regression for the real renders: the lamp makes the floor's lightness vary a lot across the image, and a
    wall in shade is about as light as the floor. Only the chroma tells them apart."""
    geo = GroundGeometry(cfg)
    img = _wall_image(geo, 2.0, 0.0, 0.25, floor=(150, 150, 150), wall=(140, 136, 128))
    ramp = np.linspace(1.5, 0.75, geo.H)[:, None, None]                  # bright near the robot, dark far away
    lit = np.clip(img.astype(float) * ramp, 0, 255).astype(np.uint8)
    fr = FloorRanger(cfg)
    sc = fr.scan(lit, 0.0, 0.25)
    mid = np.abs(sc.points[:, 1]) < 0.3
    assert sc.hit[mid].all()
    assert np.allclose(sc.points[mid, 0], 2.0, atol=0.15)
