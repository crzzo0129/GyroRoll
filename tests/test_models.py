import unittest
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class RobotModelTests(unittest.TestCase):
    def test_validated_wheelbot_geometry(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "wheelbot/validated_model.xml"))
        self.assertAlmostEqual(model.geom("tire").size[1], .004)
        self.assertAlmostEqual(model.actuator("reaction_motor").gear[0], 2.4)
        self.assertEqual(model.nu, 2)

    def test_actuators_and_contact(self):
        for kind in ("wheelbot", "steerbot"):
            with self.subTest(kind=kind):
                model = mujoco.MjModel.from_xml_path(str(ROOT / kind / "model.xml"))
                data = mujoco.MjData(model)
                data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
                mujoco.mj_forward(model, data)
                self.assertEqual(model.nu, 2)
                self.assertEqual(model.nv, 8)
                self.assertEqual(data.ncon, 1)
                contact = data.contact[0]
                self.assertEqual({model.geom(int(g)).name for g in contact.geom},
                                 {"ground", "tire"})
                self.assertLess(abs(contact.dist), 1e-6)
                self.assertTrue(np.all(np.isfinite(model.body_mass)))

    def test_steering_axis_has_trail(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot" / "model.xml"))
        fork = model.body("fork").pos
        wheel = model.body("drive_wheel").pos
        self.assertGreater(fork[0], .05)
        self.assertAlmostEqual(fork[0] + wheel[0], 0., places=6)
        self.assertTrue(np.allclose(model.jnt_axis[model.joint("steer").id], [0, 0, 1]))

    def test_validated_steerbot_matches_lqr_geometry(self):
        configuration = json.loads((ROOT / "steerbot/negative_candidate.json").read_text())
        spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
        spec.body("fork").pos[0] = configuration["trail"]
        spec.body("drive_wheel").pos[0] = -configuration["trail"]
        spec.geom("body").mass = configuration["body_mass"]
        spec.geom("body").pos[2] += configuration["body_offset_z"]
        spec.geom("steering_head").mass = configuration["head_mass"]
        spec.geom("tire").size[1] = configuration["tire_halfwidth"]
        spec.actuator("steer_motor").gear[0] = configuration["steer_gear"]
        expected = spec.compile()
        actual = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot/validated_model.xml"))
        self.assertEqual(actual.nu, 2)
        self.assertEqual(actual.nv, 8)
        for name in ("base", "fork", "drive_wheel"):
            np.testing.assert_allclose(actual.body(name).pos, expected.body(name).pos)
            np.testing.assert_allclose(actual.body(name).mass, expected.body(name).mass)
            np.testing.assert_allclose(actual.body(name).inertia, expected.body(name).inertia)
        for name in ("body", "steering_head", "tire"):
            np.testing.assert_allclose(actual.geom(name).pos, expected.geom(name).pos)
            np.testing.assert_allclose(actual.geom(name).size, expected.geom(name).size)
        np.testing.assert_allclose(actual.actuator("steer_motor").gear,
                                   expected.actuator("steer_motor").gear)

    def test_hill_keeps_robot_geometry_and_level_start(self):
        flat = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot/validated_model.xml"))
        hill = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot/hill_model.xml"))
        self.assertEqual(hill.nhfield, 1)
        np.testing.assert_allclose(hill.hfield_size[0], [200, 100, .06, .1])
        self.assertAlmostEqual(float(hill.hfield_data.max()), 1.)
        for name in ("base", "fork", "drive_wheel"):
            np.testing.assert_allclose(hill.body(name).mass, flat.body(name).mass)
            np.testing.assert_allclose(hill.body(name).inertia, flat.body(name).inertia)
        np.testing.assert_allclose(hill.geom("body").pos, flat.geom("body").pos)
        self.assertAlmostEqual(hill.body("fork").pos[0], flat.body("fork").pos[0])
        data = mujoco.MjData(hill)
        data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
        mujoco.mj_forward(hill, data)
        self.assertEqual(data.ncon, 1)
        self.assertAlmostEqual(data.contact[0].dist, 0., places=5)

    def test_external_body_force_is_a_real_lateral_load(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot/validated_model.xml"))
        velocities = []
        for force in (0., 1.):
            data = mujoco.MjData(model)
            data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
            mujoco.mj_forward(model, data)
            data.xfrc_applied[model.body("base").id, 1] = force
            mujoco.mj_step(model, data)
            velocities.append(data.qvel[1])
        self.assertGreater(velocities[1] - velocities[0], 1e-4)


if __name__ == "__main__":
    unittest.main()
