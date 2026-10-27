"""Generate the MuJoCo scene XML for a world: floor, walls, ArUco markers, pillars.

models/robot_template.xml holds the robot; this fills in the room around it. The generated file is
written next to the template (models/_generated_<world>.xml, git-ignored) so relative asset paths work.
"""
from __future__ import annotations

from pathlib import Path

from activenav.sim.world import WorldSpec, load_world, validate_world

_PLATE_HALF = 0.2           # marker plate half-size [m] (0.40 m plate, 0.30 m black square)
_PLATE_GAP = 0.004          # plates sit 4 mm in front of the wall


def _marker_pose(spec: WorldSpec, wall: str, s: float):
    W, D = spec.width, spec.depth
    return {"east": ((W - _PLATE_GAP, s), "0 -1 0 0 0 1"),
            "west": ((_PLATE_GAP, s), "0 1 0 0 0 1"),
            "north": ((s, D / 2 - _PLATE_GAP), "1 0 0 0 0 1"),
            "south": ((s, -D / 2 + _PLATE_GAP), "-1 0 0 0 0 1")}[wall]


def _ensure_texture(root: Path, mid: int) -> None:
    path = root / "models" / "assets" / "markers" / f"aruco_{mid}.png"
    if path.exists():
        return
    import cv2
    from activenav.sensing.markers import generate_marker_image
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(generate_marker_image(mid, px=600, mirror=False), cv2.COLOR_GRAY2BGR))


def scene_xml(spec: WorldSpec, root: Path) -> str:
    tpl = (root / "models" / "robot_template.xml").read_text()
    W, D, h = spec.width, spec.depth, spec.wall_height
    assets = []
    for m in spec.markers:
        _ensure_texture(root, m["id"])
        assets.append(f'    <texture name="aruco_{m["id"]}" type="2d" file="markers/aruco_{m["id"]}.png"/>\n'
                      f'    <material name="mat_aruco_{m["id"]}" texture="aruco_{m["id"]}" rgba="1 1 1 1" '
                      f'specular="0" shininess="0" reflectance="0"/>')
    if spec.pillars:
        assets.append('    <material name="pillar" rgba="0.55 0.62 0.72 1" specular="0" shininess="0" reflectance="0"/>')

    g = [f'<light pos="{W / 2:g} 0 {max(3.0, h + 0.6):g}" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>',
         '',
         '    <!-- Room -->',
         f'<geom name="floor" type="plane" pos="{W / 2:g} 0 0" size="{W / 2 + 1:g} {D / 2 + 1:g} 0.1" material="floor"/>',
         f'<geom name="wall_north" type="box" pos="{W / 2:g} {D / 2 + 0.05:g} {h / 2:g}" size="{W / 2 + 0.1:g} 0.05 {h / 2:g}" material="wall"/>',
         f'<geom name="wall_south" type="box" pos="{W / 2:g} {-D / 2 - 0.05:g} {h / 2:g}" size="{W / 2 + 0.1:g} 0.05 {h / 2:g}" material="wall"/>',
         f'<geom name="wall_west" type="box" pos="-0.05 0 {h / 2:g}" size="0.05 {D / 2 + 0.1:g} {h / 2:g}" material="wall"/>',
         f'<geom name="wall_east" type="box" pos="{W + 0.05:g} 0 {h / 2:g}" size="0.05 {D / 2 + 0.1:g} {h / 2:g}" material="wall"/>',
         '',
         '    <!-- ArUco markers: 0.40 m plates (0.30 m black square). Frame: x = tangent, y = up, z = outward normal. -->']
    for m in spec.markers:
        (px, py), axes = _marker_pose(spec, m["wall"], m["s"])
        g.append(f'<geom name="marker_{m["id"]}" type="box" size="{_PLATE_HALF:g} {_PLATE_HALF:g} 0.002" '
                 f'pos="{px:g} {py:g} {m["z"]:g}" xyaxes="{axes}" material="mat_aruco_{m["id"]}" contype="0" conaffinity="0"/>')
    if spec.blocks:
        g += ['', '    <!-- Wall blocks: full-height partitions and corner blocks -->']
        for i, b in enumerate(spec.blocks):
            g.append(f'<geom name="wall_block_{i}" type="box" pos="{(b["x0"] + b["x1"]) / 2:g} {(b["y0"] + b["y1"]) / 2:g} {h / 2:g}" '
                     f'size="{(b["x1"] - b["x0"]) / 2:g} {(b["y1"] - b["y0"]) / 2:g} {h / 2:g}" material="wall"/>')
    if spec.pillars:
        g += ['', '    <!-- Pillars: full-height cylinders -->']
        for i, p in enumerate(spec.pillars):
            g.append(f'<geom name="pillar_{i}" type="cylinder" pos="{p["x"]:g} {p["y"]:g} {p["h"] / 2:g}" '
                     f'size="{p["r"]:g} {p["h"] / 2:g}" material="pillar"/>')
    if spec.people:
        pp = spec.people
        r, ph = float(pp["radius"]), float(pp["height"])
        bh = (ph - 2 * 0.11) / 2
        assets.append('    <material name="person" rgba="0.90 0.45 0.12 1" specular="0" shininess="0" reflectance="0"/>')
        assets.append('    <material name="person_head" rgba="0.95 0.75 0.60 1" specular="0" shininess="0" reflectance="0"/>')
        g += ['', '    <!-- Walking people: mocap cylinders, moved by sim/people.py every control step -->']
        for i in range(int(pp["count"])):
            g.append(f'<body name="person_{i}" mocap="true" pos="{W / 2:g} 0 0">'
                     f'<geom name="person_{i}" type="cylinder" pos="0 0 {bh:g}" size="{r:g} {bh:g}" material="person"/>'
                     f'<geom name="person_{i}_head" type="sphere" pos="0 0 {ph - 0.11:g}" size="0.11" material="person_head"/></body>')
    world = "\n    ".join(g)
    return tpl.replace("<!--@SCENE_ASSETS@-->", "\n".join(assets)).replace("<!--@SCENE_WORLD@-->", world)


def build_scene(cfg: dict, name: str, markers=None, tag: str = "") -> Path:
    """Validate the layout, write models/_generated_<name><tag>.xml and return its path.
    `markers` replaces the world's marker list (random layouts); `tag` keeps those files apart."""
    root = Path(cfg["_root"])
    spec = load_world(cfg, name)
    if markers is not None:
        spec.markers = [dict(m) for m in markers]
    validate_world(spec, cfg, starts=[cfg["worlds"][name].get("start", [1.0, -0.5, 0.0])])
    out = root / "models" / f"_generated_{name}{tag}.xml"
    out.write_text(scene_xml(spec, root))
    return out
