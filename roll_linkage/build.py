"""Generate and inspect a two-motor rolling wheel with a physical four-bar roll linkage."""
from pathlib import Path
import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results/roll_linkage/concept_v1"


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def link(parent, name, ends, radius, mass, color):
    return add(parent, "geom", name=name, type="capsule", fromto=ends,
               size=radius, mass=mass, rgba=color, contype=0, conaffinity=0)


def make_xml(fixture=False, level_torso=False):
    root = ET.Element("mujoco", model="roll_linkage_level_torso_v2" if level_torso else "roll_linkage_v1")
    add(root, "compiler", angle="radian")
    add(root, "option", timestep=".001", integrator="implicitfast",
        iterations="100", tolerance="1e-10")
    default = add(root, "default")
    add(default, "joint", damping=".005", armature=".0001")
    add(default, "geom", friction="1.6 .005 .0001", condim="3")
    visual = add(root, "visual")
    add(visual, "global", offwidth="1280", offheight="720")
    add(visual, "headlight", ambient=".5 .5 .5", diffuse=".7 .7 .7")
    asset = add(root, "asset")
    add(asset, "texture", name="grid", type="2d", builtin="checker",
        width="128", height="128", rgb1=".16 .19 .23", rgb2=".21 .24 .28")
    add(asset, "material", name="floor", texture="grid", texrepeat="8 8")
    world = add(root, "worldbody")
    add(world, "light", pos="1 -1 2", dir="-1 1 -2")
    add(world, "geom", name="ground", type="plane", size="4 4 .1", material="floor")
    base = add(world, "body", name="base", pos=f"0 0 {'.45' if fixture else '.42'}")
    if not fixture:
        add(base, "freejoint", name="root")
    add(base, "geom", name="chassis", type="box", size=".11 .08 .065" if level_torso else ".075 .08 .065",
        mass=".145" if level_torso else ".645", rgba=".10 .35 .45 1")
    if level_torso:
        # Reallocate 0.5 kg of the existing torso mass to rear equipment.
        # Cancels the measured nominal x/y first moment; no extra counterweight.
        add(base, "geom", name="rear_equipment", type="box", pos="-.067775 .0038 0",
            size=".032 .05 .035", mass=".5", rgba=".22 .25 .3 1", contype="0", conaffinity="0")
        add(base, "site", name="torso_imu", pos="0 0 .07", size=".004", rgba="1 .5 .1 1")
        link(base, "torso_level_marker", ".111 -.07 0 .111 .07 0",
             ".003", "0", "1 1 1 1")
    add(base, "geom", name="cargo_tray", type="box", pos="0 0 .075",
        size=".10 .11 .008", mass=".06", rgba=".65 .76 .8 1")
    # The input and output shafts are parallel to body x (forward).
    # A 55 mm input crank and output rocker are joined by a 140 mm rod.
    link(base, "bearing_spine", ".105 0 0 .105 0 -.14", ".012", ".05", ".35 .4 .45 1")
    link(base, "upper_mount", ".065 0 0 .145 0 0", ".018", ".04", ".35 .4 .45 1")
    link(base, "lower_bearing", ".08 0 -.14 .145 0 -.14", ".015", ".025", ".35 .4 .45 1")
    add(base, "geom", name="roll_motor_housing", type="cylinder",
        pos=".09 0 0", euler="0 1.57079632679 0", size=".027 .025",
        mass=".08", rgba=".18 .21 .26 1", contype="0", conaffinity="0")
    crank = add(base, "body", name="input_crank", pos=".145 0 0")
    add(crank, "joint", name="linkage_motor", axis="1 0 0",
        limited="true", range="-.523599 .523599")
    link(crank, "input_arm", "0 0 0 0 .055 0", ".008", ".025", "1 .52 .12 1")
    rod = add(crank, "body", name="connecting_rod", pos="0 .055 0")
    add(rod, "joint", name="passive_rod", axis="1 0 0")
    link(rod, "pushrod", "0 0 0 0 0 -.14", ".006", ".035", ".95 .75 .23 1")
    add(rod, "site", name="rod_end", pos="0 0 -.14", size=".003", rgba="1 1 1 1")
    carrier = add(base, "body", name="wheel_carrier", pos="0 0 -.14")
    add(carrier, "joint", name="wheel_roll", axis="1 0 0",
        limited="true", range="-.55 .55")
    link(carrier, "output_shaft", "-.04 0 0 .145 0 0", ".009", ".04", ".2 .65 .85 1")
    link(carrier, "output_arm", ".145 0 0 .145 .055 0", ".008", ".025", "1 .52 .12 1")
    add(carrier, "site", name="rocker_end", pos=".145 .055 0", size=".003", rgba="1 1 1 1")
    link(carrier, "fork_crossbar", "0 -.045 0 0 .045 0", ".009", ".025", ".2 .65 .85 1")
    for side in (-1, 1):
        y = side * .045
        link(carrier, f"fork_{side}", f"0 {y} 0 0 {y} -.16",
             ".008", ".035", ".2 .65 .85 1")
    link(carrier, "drive_axle", "0 -.06 -.16 0 .06 -.16", ".012", ".03", ".65 .7 .76 1")
    add(carrier, "geom", name="drive_motor_housing", type="cylinder",
        pos="0 -.065 -.16", euler="1.57079632679 0 0", size=".028 .018",
        mass=".08", rgba=".18 .21 .26 1", contype="0", conaffinity="0")
    # Reuse exactly the rolling wheel geometry and inertial from the earlier concept.
    old = ET.parse(ROOT.parent / "structure_rl/models/raked_steer.xml").getroot()
    wheel = copy.deepcopy(old.find(".//body[@name='drive_wheel']"))
    wheel.set("pos", "0 0 -.16")
    wheel.find("joint").set("damping", ".00002")
    wheel.find("joint").set("armature", ".00002")
    link(wheel, "wheel_index", "0 -.011 0 .085 -.011 0", ".003", "0", "1 .65 .15 1")
    carrier.append(wheel)
    equality = add(root, "equality")
    add(equality, "connect", name="four_bar_closure", site1="rod_end", site2="rocker_end",
        solref=".003 1", solimp=".999 .9999 .001")
    actuator = add(root, "actuator")
    add(actuator, "velocity", name="drive_velocity", joint="drive", kv=".25",
        ctrllimited="true", ctrlrange="-35 35", forcelimited="true", forcerange="-2.4 2.4")
    add(actuator, "position", name="roll_linkage_position", joint="linkage_motor", kp="8", kv=".25",
        ctrllimited="true", ctrlrange="-.523599 .523599",
        forcelimited="true", forcerange="-2.4 2.4")
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def inspect_and_render(render=False):
    ROOT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for fixture, name in ((False, "model.xml"), (True, "fixture.xml")):
        (ROOT / name).write_text(make_xml(fixture), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "fixture.xml"))
    data = mujoco.MjData(model)
    assert model.nu == 2 and model.neq == 1
    qin = model.joint("linkage_motor").qposadr[0]
    qout = model.joint("wheel_roll").qposadr[0]
    rod = model.site("rod_end").id
    rocker = model.site("rocker_end").id
    peak_error = peak_torque = peak_roll = 0.
    images = []
    renderer = mujoco.Renderer(model, height=720, width=1280) if render else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.02, 0, .28]
    camera.distance = 1.05
    camera.azimuth = 215
    camera.elevation = -14
    try:
        for step in range(6000):
            data.ctrl[:] = [2., math.radians(25) * math.sin(2 * math.pi * step * .001 / 3)]
            mujoco.mj_step(model, data)
            assert np.isfinite(data.qpos).all()
            error = np.linalg.norm(data.site_xpos[rod] - data.site_xpos[rocker])
            peak_error = max(peak_error, float(error))
            peak_torque = max(peak_torque, float(np.abs(data.actuator_force).max()))
            peak_roll = max(peak_roll, float(abs(data.qpos[qout])))
            if renderer and step % 40 == 0:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
        assert peak_error < .001, peak_error
        assert peak_torque <= 2.40001, peak_torque
        assert peak_roll > math.radians(15), peak_roll
    finally:
        if renderer:
            renderer.close()
    # Unsupported free-body model: only finite dynamics, no balance claim.
    free = mujoco.MjModel.from_xml_path(str(ROOT / "model.xml"))
    state = mujoco.MjData(free)
    for _ in range(2000):
        mujoco.mj_step(free, state)
        assert np.isfinite(state.qpos).all() and np.isfinite(state.qvel).all()
        assert np.all(state.xfrc_applied == 0)
    report = dict(actuators=2, linkage="physical four-bar connect constraint",
                  wheel_diameter_mm=240, wheel_width_mm=24, roll_limit_deg=30,
                  crank_length_mm=55, connecting_rod_mm=140,
                  roll_axis_height_mm=280, wheel_center_height_mm=120,
                  total_mass_kg=float(free.body_mass.sum()),
                  fixture_peak_closure_error_mm=peak_error*1000,
                  fixture_peak_motor_torque_Nm=peak_torque,
                  fixture_peak_wheel_roll_deg=math.degrees(peak_roll),
                  free_dynamics_seconds=2, balance_validated=False,
                  preview="fixed chassis, wheel suspended 30 mm; mechanism demonstration only")
    (OUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if render:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        labeled = []
        for index, frame in enumerate(images):
            canvas = Image.fromarray(frame)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 1280, 76), fill=(18, 26, 36))
            draw.text((22, 10), "ROLL LINKAGE V1 | wheel drive + four-bar tilt drive", font=font, fill="white")
            draw.text((22, 42), "FIXED-CHASSIS MECHANISM DEMO | not a balancing controller", font=font, fill="#ffcc70")
            labeled.append(np.asarray(canvas))
        imageio.mimwrite(OUT / "mechanism.mp4", labeled, fps=25, codec="libx264")
        imageio.imwrite(OUT / "preview.png", labeled[18])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    inspect_and_render(parser.parse_args().render)
