"""Validate signed velocity servos and their actual finite-torque dynamics."""

import unittest
import mujoco
import numpy as np

from miniwheelbot_rl.model import velocity_model_xml
from miniwheelbot_rl.settings import Settings


class VelocityModelTests(unittest.TestCase):
    def test_geometry_and_action_order(self):
        model = mujoco.MjModel.from_xml_string(velocity_model_xml(ground=True))
        self.assertEqual(model.nu, 2)
        self.assertEqual(model.nv, 8)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in model.actuator_trnid[:, 0]]
        self.assertEqual(names, ["reaction", "drive"])
        np.testing.assert_allclose(model.actuator_ctrlrange, [[-250, 250], [-35, 35]])
        np.testing.assert_allclose(model.actuator_forcerange, [[-2.4, 2.4]] * 2)
        np.testing.assert_allclose(model.actuator_gear[:, 0], 1.0)
        for name in ("reaction", "drive"):
            self.assertFalse(model.joint(name).limited)

    def test_signed_speed_target_is_not_state_teleportation(self):
        model = mujoco.MjModel.from_xml_string(velocity_model_xml(ground=True))
        for sign in (-1, 1):
            data = mujoco.MjData(model)
            data.ctrl[:] = [sign * 100, sign * 20]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.qvel[6:], 0.0)
            np.testing.assert_allclose(data.actuator_force, sign * np.array([2.4, 2.4]))
            mujoco.mj_step(model, data)
            self.assertTrue(np.isfinite(data.qvel).all())
            self.assertLess(np.max(np.abs(data.qvel[6:])), 20.0)

    def test_zero_command_brakes_and_ground_is_separate(self):
        model = mujoco.MjModel.from_xml_string(velocity_model_xml())
        self.assertFalse(any(t == mujoco.mjtGeom.mjGEOM_PLANE for t in model.geom_type))
        data = mujoco.MjData(model)
        data.qvel[model.joint("reaction").dofadr[0]] = 10.0
        data.qvel[model.joint("drive").dofadr[0]] = -2.0
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, [-1.2, 0.5])

    def test_speed_limit_validation(self):
        with self.assertRaises(ValueError):
            Settings(flywheel_speed_limit=-1)


if __name__ == "__main__":
    unittest.main()
