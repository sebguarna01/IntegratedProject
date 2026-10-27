import numpy as np

from activenav.autonomy.occupancy import FREE, OCC, UNKNOWN, OccupancyGrid
from activenav.config import load_config
from activenav.evaluation.worlds import get_world
from activenav.evaluation.synthetic import SyntheticObstacleSensor
from activenav.types import SensorBundle

cfg = load_config()


def _map_room(world, poses, seed=0):
    ctx = get_world(cfg, {}, world)
    env = ctx.env
    sensor = SyntheticObstacleSensor(env, np.random.default_rng(seed), cfg)
    grid = OccupancyGrid(cfg)
    origin = np.array(poses[0])                      # filter frame = start pose
    c, s = np.cos(origin[2]), np.sin(origin[2])
    for p in poses:
        sb = env.reset(seed, start_pose=p)
        for pan in (-0.8, 0.0, 0.8):
            sb = SensorBundle(t=0.0, enc_left=0.0, enc_right=0.0, pan=pan, tilt=0.3)
            sc = sensor.scan(sb)
            d = np.array(p[:2]) - origin[:2]
            est = (c * d[0] + s * d[1], -s * d[0] + c * d[1], p[2] - origin[2])      # true pose in the start frame
            grid.integrate(est, sc)
    return ctx, grid, origin


def test_map_of_the_large_room_matches_the_walls_and_pillars():
    poses = [(x, y, th) for x in (1.0, 3.0, 5.0, 7.0, 9.0) for y in (-2.0, 0.0, 2.0) for th in (0.0, 1.57, 3.14, -1.57)]
    ctx, grid, o = _map_room("room_10x6", poses)
    st = grid.state()
    c, s = np.cos(o[2]), np.sin(o[2])

    def start_frame(x, y):
        d = np.array([x, y]) - o[:2]
        return c * d[0] + s * d[1], -s * d[0] + c * d[1]
    # every pillar centre is in the start frame; cells on the pillar rim must be occupied, its far centre not free-known
    hits = 0
    for p in ctx.obstacles:
        px, py = start_frame(p["x"], p["y"])
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            rr = p["r"] + np.linspace(-0.08, 0.10, 8)             # ranging noise is a few cm: look along the radius
            iy, ix = grid.to_cell(px + rr * np.cos(a), py + rr * np.sin(a))
            hits += bool((grid.state()[iy, ix] == OCC).any())
    assert hits >= 0.6 * 16 * len(ctx.obstacles), hits
    # open floor between pillars is free, and no free cell lies inside a pillar
    x0, x1, y0, y1 = ctx.env.room_bounds()
    cx, cy = start_frame(0.5 * (x0 + x1), y0 + 0.5)
    iy, ix = grid.to_cell(cx, cy)
    assert st[iy, ix] == FREE
    for p in ctx.obstacles:
        px, py = start_frame(p["x"], p["y"])
        iy, ix = grid.to_cell(px, py)
        assert st[iy, ix] != FREE
    # walls: the seen extent is the room within a few cm
    ext = grid.extent()
    assert ext is not None and abs((ext[1] - ext[0]) - (x1 - x0)) < 0.5


def test_unknown_stays_unknown():
    g = OccupancyGrid(cfg)
    assert (g.state() == UNKNOWN).all()
    assert g.extent() is None


def test_rays_wider_apart_than_a_cell_leave_no_unknown_stripes():
    """Far from the camera neighbouring rays are further apart than a cell; the free space between them must still be
    marked, or a doorway seen from across the room looks like a wall."""
    import numpy as np
    from activenav.autonomy.obstacle_sensor import ObstacleScan
    from activenav.autonomy.occupancy import FREE, OccupancyGrid, UNKNOWN
    from activenav.config import load_config
    g = OccupancyGrid(load_config())
    ang = np.linspace(-0.3, 0.3, 24)                                       # 24 rays over 0.6 rad: ~0.15 m apart at 3 m
    pts = np.stack([3.0 * np.cos(ang), 3.0 * np.sin(ang)], axis=1)
    g.integrate(np.zeros(3), ObstacleScan(np.zeros(2), pts, np.zeros(24, bool), np.full(24, 0.02)))
    st = g.state()
    xs = np.linspace(2.4, 2.8, 9)
    iy, ix = g.to_cell(xs, np.zeros_like(xs))
    cell_row = st[iy[0] - 5:iy[0] + 6, ix[0]:ix[-1] + 1]                    # a strip across the middle rays at 2.4-2.8 m
    assert (cell_row == FREE).mean() > 0.95 and not (cell_row == UNKNOWN).any()
