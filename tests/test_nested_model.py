"""Physical checks for the concentric, perpendicular two-wheel layout."""

from pathlib import Path
import contextlib
import io
import json
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "wheelbot/nested_model.xml"
CONFIG = json.loads((ROOT / "wheelbot/nested_controller.json").read_text())


class NestedWheelTests(unittest.TestCase):
    def test_same_centre_perpendicular_axes_and_two_motors(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(model.nu, 2)
        self.assertEqual(model.nv, 8)
        drive = model.joint("drive").id
        reaction = model.joint("reaction").id
        np.testing.assert_allclose(data.xanchor[drive], data.xanchor[reaction])
        self.assertAlmostEqual(float(data.xaxis[drive] @ data.xaxis[reaction]), 0.)
        self.assertAlmostEqual(model.body("drive_wheel").mass[0], .312)

    def test_internal_wheel_clearance_through_rotation(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        data = mujoco.MjData(model)
        drive = model.body("drive_wheel").id
        reaction = model.body("reaction_wheel").id
        for angle in np.linspace(0., 2 * np.pi, 13):
            data.qpos[model.joint("drive").qposadr[0]] = angle
            data.qpos[model.joint("reaction").qposadr[0]] = 2 * angle
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                bodies = {int(model.geom_bodyid[g]) for g in contact.geom}
                self.assertNotEqual(bodies, {drive, reaction})
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

    def test_chassis_above_wheel_and_supports_clear_of_rotor(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        centre = model.body("drive_wheel").pos
        axial_extent = max(model.geom("flywheel").size[1], model.geom("flywheel_hub").size[1])
        chassis = model.geom("body")
        self.assertGreater(chassis.pos[2] - chassis.size[2], centre[2] + CONFIG["outer_radius"] + .01)
        for index in range(4):
            support = model.geom(f"support_{index}")
            self.assertGreater(abs(support.pos[0] - centre[0]) - support.size[0], axial_extent + .005)
        self.assertLess(CONFIG["rim_halfwidth"], model.geom("flywheel").size[0])

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
