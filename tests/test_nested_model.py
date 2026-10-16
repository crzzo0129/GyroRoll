"""Physical checks for the raised, perpendicular two-wheel layout."""

from pathlib import Path
import contextlib
import io
import json
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "wheelbot/nested_model.xml"
CONFIG = json.loads((ROOT / "wheelbot/nested_controller.json").read_text())


class NestedWheelTests(unittest.TestCase):
    def test_50mm_edge_gap_perpendicular_axes_and_two_motors(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(model.nu, 2)
        self.assertEqual(model.nv, 8)
        drive = model.joint("drive").id
        reaction = model.joint("reaction").id
        np.testing.assert_allclose(data.xanchor[drive][:2], data.xanchor[reaction][:2])
        drive_top = data.xanchor[drive, 2] + CONFIG["outer_radius"]
        flywheel_bottom = data.xanchor[reaction, 2] - model.geom("flywheel").size[0]
        self.assertAlmostEqual(flywheel_bottom - drive_top, .050, places=9)
        self.assertAlmostEqual(data.xanchor[reaction, 2], .373, places=9)
        self.assertAlmostEqual(float(data.xaxis[drive] @ data.xaxis[reaction]), 0.)
        self.assertAlmostEqual(model.body("drive_wheel").mass[0], .312)

    def test_wheel_and_chassis_clearance_through_rotation(self):
        # Check real clearance even though mounting contacts are excluded in use.
        root = ET.parse(MODEL).getroot()
        for exclusion in list(root.find("contact")):
            root.find("contact").remove(exclusion)
        option = root.find("option")
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("filterparent", "disable")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        drive = model.body("drive_wheel").id
        reaction = model.body("reaction_wheel").id
        base = model.body("base").id
        for angle in np.linspace(0., 2 * np.pi, 13):
            data.qpos[model.joint("drive").qposadr[0]] = angle
            data.qpos[model.joint("reaction").qposadr[0]] = 2 * angle
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                bodies = {int(model.geom_bodyid[g]) for g in contact.geom}
                self.assertNotEqual(bodies, {drive, reaction})
                self.assertNotEqual(bodies, {base, reaction})
                self.assertNotEqual(bodies, {base, drive})
            self.assertGreaterEqual(data.ncon, 1)

    def test_annulus_has_a_physical_hole(self):
        spec = mujoco.MjSpec.from_file(str(MODEL))
        # A world-fixed probe in the bore must not touch the annular drive wheel.
        spec.worldbody.add_geom(name="cavity_probe", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[.03, 0., 0.], pos=[0., 0., CONFIG["outer_radius"]])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        probe = model.geom("cavity_probe").id
        drive = model.body("drive_wheel").id
        for contact in data.contact:
            if probe in contact.geom:
                other = next(g for g in contact.geom if g != probe)
                self.assertNotEqual(int(model.geom_bodyid[other]), drive)

    def test_hollow_enclosure_and_cargo_tray_clear_the_rotor(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        centre = model.body("drive_wheel").pos
        axial_extent = max(model.geom("flywheel").size[1], model.geom("flywheel_hub").size[1])
        chassis = model.geom("bay_divider")
        rotor_top = model.body("reaction_wheel").pos[2] + model.geom("flywheel").size[0]
        self.assertAlmostEqual(chassis.pos[2] - chassis.size[2] - rotor_top, .014)
        floor = model.geom("body_floor")
        rotor_bottom = model.body("reaction_wheel").pos[2] - model.geom("flywheel").size[0]
        self.assertAlmostEqual(rotor_bottom - (floor.pos[2] + floor.size[2]), .012)
        for name in ("body", "body_front"):
            panel = model.geom(name)
            self.assertGreater(abs(panel.pos[0]) - panel.size[0], axial_extent)
        for name in ("body_left", "body_right"):
            panel = model.geom(name)
            self.assertAlmostEqual(abs(panel.pos[1]) - panel.size[1] - .083, .019)
        rail = model.geom("cargo_rail_front")
        self.assertAlmostEqual(.370 + rail.pos[2] + rail.size[2], .580)
        self.assertAlmostEqual(2 * (rail.pos[0] - rail.size[0]), .156)
        side_rail = model.geom("cargo_rail_left")
        self.assertAlmostEqual(2 * (side_rail.pos[1] - side_rail.size[1]), .216)
        self.assertTrue(model.geom("cargo_deck").contype[0])
        for index in range(2):
            support = model.geom(f"support_{index}")
            self.assertGreater(abs(support.pos[0] - centre[0]) - support.size[0], axial_extent + .005)
        self.assertLess(CONFIG["rim_halfwidth"], model.geom("flywheel").size[0])
        spine = model.geom("spine")
        self.assertGreater(abs(spine.pos[0]) - spine.size[0], axial_extent)
        self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_rear"), -1)
        self.assertAlmostEqual(model.body("base").mass[0], .695)

    def test_nominal_one_metre_per_second_straight_motion(self):
        from sim import make_parser, run

        args = make_parser().parse_args([
            "wheelbot", "--controller", "nested", "--headless",
            "--speed", "1", "--turn", "0", "--duration", "15",
        ])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run(args)
        summary = json.loads(stream.getvalue())
        self.assertFalse(summary["fallen"])
        self.assertEqual(summary["duration"], 15.)
        self.assertLess(summary["steady_speed_rmse"], .35)
        self.assertLess(summary["turn_yaw_rmse"], .02)


if __name__ == "__main__":
    unittest.main()
