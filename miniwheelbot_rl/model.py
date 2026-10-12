"""Reuse the open annular geometry, replace only its actuation and scene ground."""

from pathlib import Path
import xml.etree.ElementTree as ET

from .settings import Settings

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "wheelbot" / "nested_model.xml"


def velocity_model_xml(settings: Settings = Settings(), *, ground: bool = False,
                       model_path: Path = MODEL) -> str:
    root = ET.parse(model_path).getroot()
    world = root.find("worldbody")
    if not ground:
        # Entity options do not propagate through MjSpec.attach(). MJLab's
        # MujocoCfg owns the global integrator, timestep and gravity instead.
        option = root.find("option")
        if option is not None:
            root.remove(option)
        # MJLab adds one terrain entity; don't import a second robot-owned plane.
        for node in list(world):
            if node.tag in ("geom", "light"):
                world.remove(node)
    actuators = root.find("actuator")
    for node in list(actuators):
        actuators.remove(node)
    # Explicit target-speed bounds are independent of unlimited joint angles.
    # XmlActuatorCfg wraps the same servo parameters during training and replay.
    for joint, limit, kv in (
        ("reaction", settings.flywheel_speed_limit, settings.flywheel_kv),
        ("drive", settings.rolling_speed_limit, settings.rolling_kv),
    ):
        ET.SubElement(actuators, "velocity", name=f"{joint}_velocity", joint=joint,
                      kv=str(kv), gear="1", ctrllimited="true",
                      ctrlrange=f"{-limit} {limit}", forcelimited="true",
                      forcerange=f"{-settings.torque_limit} {settings.torque_limit}")
    return ET.tostring(root, encoding="unicode")


def make_spec(settings: Settings = Settings(), model_path: Path = MODEL):
    import mujoco
    return mujoco.MjSpec.from_string(velocity_model_xml(settings, model_path=model_path))
