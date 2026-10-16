"""Build an annular drive wheel with a perpendicular flywheel above it.

Each annulus is split into convex sectors so contact preserves the real hole.
The two wheel bodies are siblings attached to the chassis, not serial joints.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from .appearance import apply_appearance
except ImportError:
    from appearance import apply_appearance


ROOT = Path(__file__).resolve().parent
SECTORS = 96
OUTER_RADIUS = .12
RIM_RADIUS = .111
INNER_RADIUS = .100
RIM_HALFWIDTH = .010
TIRE_HALFWIDTH = .012
WHEEL_MASS = .312
FLYWHEEL_RADIUS = .083
BASE_HEIGHT = .370
DRIVE_CENTRE_Z = -.250
WHEEL_EDGE_GAP = .050  # drive top to flywheel bottom, upright configuration
FLYWHEEL_CENTRE_Z = DRIVE_CENTRE_Z + OUTER_RADIUS + WHEEL_EDGE_GAP + FLYWHEEL_RADIUS


def sector_vertices(inner, outer, halfwidth):
    vertices = []
    for radius in (inner, outer):
        for angle in (0., 2 * math.pi / SECTORS):
            for y in (-halfwidth, halfwidth):
                vertices.extend((radius * math.sin(angle), y,
                                 -radius * math.cos(angle)))
    return " ".join(f"{v:.12g}" for v in vertices)


def build():
    tree = ET.parse(ROOT / "validated_model.xml")
    root = tree.getroot()
    root.set("model", "wheelbot_enclosed_cargo_body")
    asset = root.find("asset")
    for name, inner, outer, width in (
        ("rim_sector", INNER_RADIUS, RIM_RADIUS, RIM_HALFWIDTH),
        ("tire_sector", RIM_RADIUS, OUTER_RADIUS, TIRE_HALFWIDTH),
    ):
        ET.SubElement(asset, "mesh", name=name,
                      vertex=sector_vertices(inner, outer, width))
    base = root.find("worldbody/body[@name='base']")
    base.set("pos", f"0 0 {BASE_HEIGHT}")
    chassis = base.find("geom[@name='body']")
    # Hollow 160 x 220 mm enclosure: six separate collision walls, never a
    # solid convex hull around the rotor. Upright bottom/deck heights 270/560 mm.
    # Masses are an empty-body simulation budget, not a fabricated load rating.
    chassis.set("size", ".004 .110 .140")
    chassis.set("pos", "-.076 0 .040")
    chassis.set("mass", ".110")
    panels = (
        ("body_front", ".004 .110 .140", ".076 0 .040", ".110"),
        ("body_left", ".072 .004 .140", "0 .106 .040", ".065"),
        ("body_right", ".072 .004 .140", "0 -.106 .040", ".065"),
        ("body_floor", ".072 .102 .004", "0 0 -.096", ".040"),
        ("bay_divider", ".072 .102 .004", "0 0 .104", ".045"),
        ("cargo_deck", ".086 .116 .005", "0 0 .185", ".100"),
        ("cargo_rail_front", ".004 .116 .010", ".082 0 .200", ".010"),
        ("cargo_rail_rear", ".004 .116 .010", "-.082 0 .200", ".010"),
        ("cargo_rail_left", ".078 .004 .010", "0 .112 .200", ".010"),
        ("cargo_rail_right", ".078 .004 .010", "0 -.112 .200", ".010"),
        ("equipment", ".045 .065 .022", "0 0 .133", ".045"),
    )
    for name, size, pos, mass in panels:
        ET.SubElement(base, "geom", name=name, type="box", size=size, pos=pos,
                      mass=mass, rgba=".82 .85 .82 1")
    ET.SubElement(base, "geom", name="spine", type="box",
                  size=".008 .017 .100", pos="-.043 0 .008", mass=".028",
                  rgba=".15 .20 .22 1")
    ET.SubElement(base, "geom", name="rotor_bearing", type="cylinder",
                  size=".014 .012", pos=f"-.032 0 {FLYWHEEL_CENTRE_Z:g}",
                  euler="0 1.57079632679 0", mass=".012", rgba=".42 .50 .53 1")
    for index, y in enumerate((-.025, .025)):
        ET.SubElement(base, "geom", name=f"support_{index}", type="capsule",
                      fromto=f"-.043 {y} -.092 -.043 {y} -.250", size=".004",
                      mass=".006", rgba=".15 .20 .22 1")
        ET.SubElement(base, "geom", name=f"fork_tip_{index}", type="capsule",
                      fromto=f"-.043 {y} -.250 0 {y} -.250", size=".004",
                      mass=".004", rgba=".15 .20 .22 1")
    front = base.find("geom[@name='front']")
    front.set("size", ".002 .042 .009")
    front.set("pos", ".080 0 .151")
    # The reference front marker stays on the upper chassis.
    # With a 0.37 m root height the drive centre remains at 0.12 m.
    drive = base.find("body[@name='drive_wheel']")
    drive.set("pos", f"0 0 {DRIVE_CENTRE_Z:g}")
    for geom in list(drive.findall("geom")):
        drive.remove(geom)
    # Analytic annulus inertias for the widened rim and tread. Retain the
    # conceptual 312 g wheel mass budget; this is not a fixed-density extrusion.
    # Polygonal contact is an approximation to these circular rings.
    ix = iy = 0.
    for mass, inner, outer, width in (
        (.270, INNER_RADIUS, RIM_RADIUS, RIM_HALFWIDTH),
        (.042, RIM_RADIUS, OUTER_RADIUS, TIRE_HALFWIDTH),
    ):
        ix += mass * (3 * (outer**2 + inner**2) + (2 * width)**2) / 12
        iy += mass * (outer**2 + inner**2) / 2
    ET.SubElement(drive, "inertial", pos="0 0 0", mass=str(WHEEL_MASS),
                  diaginertia=f"{ix:.12g} {iy:.12g} {ix:.12g}")
    for index in range(SECTORS):
        angle = 2 * math.pi * index / SECTORS
        quat = f"{math.cos(angle / 2):.12g} 0 {math.sin(angle / 2):.12g} 0"
        ET.SubElement(drive, "geom", name=f"rim_{index:03d}", type="mesh",
                      mesh="rim_sector", quat=quat, mass="0",
                      rgba=".20 .43 .48 1" if index % 24 < 3 else ".42 .48 .50 1")
        ET.SubElement(drive, "geom", name="tire" if index == 0 else f"tire_{index:03d}",
                      type="mesh", mesh="tire_sector", quat=quat, mass="0",
                      rgba=".10 .13 .15 1")
    reaction = base.find("body[@name='reaction_wheel']")
    reaction.set("pos", f"0 0 {FLYWHEEL_CENTRE_Z:.12g}")
    reaction.find("geom[@name='flywheel']").set("size", f"{FLYWHEEL_RADIUS} .011")
    # The base contact exclusion in the reference model describes the mounting.
    # No drive/reaction exclusion: tests must prove the wheels are clear.
    root.insert(0, ET.Comment(
        "Generated by generate_nested.py. Outer diameter 240 mm, clear bore "
        "200 mm, open rim width 20 mm, tire width 24 mm, upper flywheel diameter 166 mm. "
        "Upright edge-to-edge vertical gap 50 mm; flywheel centre height 373 mm. "
        "Hollow enclosed rotor bay, upper equipment bay and cargo tray; height 580 mm. "
        "Front cover hidden for display only; its mass and collisions remain active. "
        "Empty-body mass budget only; no payload or load rating included. "
        "Ideal chassis-mounted bearings; no detailed motor/shaft packaging."
    ))
    apply_appearance(root)
    ET.indent(tree, space="  ")
    with (ROOT / "nested_model.xml").open("w", encoding="utf-8", newline="\n") as output:
        tree.write(output, encoding="unicode")


if __name__ == "__main__":
    build()
