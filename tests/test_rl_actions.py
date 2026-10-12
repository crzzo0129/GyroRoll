"""Exercise the MJLab action-manager/Entity boundary without GPU simulation."""

from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


@unittest.skipIf(find_spec("mjlab") is None, "requires mjlab==1.3.0")
class VelocityActionTests(unittest.TestCase):
    def test_signed_actions_reach_the_correct_joint_targets(self):
        import torch
        from mjlab.managers.action_manager import ActionManager
        from miniwheelbot_rl.task import make_env_cfg

        cfg = make_env_cfg(num_envs=2)
        robot = cfg.scene.entities["robot"].build()
        env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
        # Build the actual manager so a wrong name fails at the same boundary
        # as environment startup, even if the raw XML model compiles.
        manager = ActionManager(cfg.actions, env)
        self.assertEqual(manager.total_action_dim, 2)
        self.assertEqual(manager.get_term("flywheel").target_names, ["reaction"])
        self.assertEqual(manager.get_term("rolling").target_names, ["drive"])
        joint_ids, names = robot.find_joints(("reaction", "drive"), preserve_order=True)
        self.assertEqual(names, ["reaction", "drive"])

        for actions, expected in (
            ([[1.0, -1.0], [-1.0, 1.0]], [[250.0, -35.0], [-250.0, 35.0]]),
            ([[2.0, -2.0], [-2.0, 2.0]], [[250.0, -35.0], [-250.0, 35.0]]),
            ([[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]),
        ):
            manager.process_action(torch.tensor(actions))
            # Only capture the final write: naming, slicing, signed scaling,
            # and clipping all run through the real MJLab implementation.
            with patch.object(robot, "set_joint_velocity_target") as write:
                manager.apply_action()
            received = torch.zeros((2, len(robot.joint_names)))
            self.assertEqual(write.call_count, 2)
            for call in write.call_args_list:
                received[:, call.kwargs["joint_ids"]] = call.args[0]
            np.testing.assert_allclose(received[:, joint_ids].numpy(), expected)

    def test_scene_options_are_applied_by_mujoco_cfg(self):
        from mjlab.utils.spec import non_default_option_fields
        from miniwheelbot_rl.task import make_env_cfg

        cfg = make_env_cfg(num_envs=1)
        robot = cfg.scene.entities["robot"].build()
        self.assertEqual(non_default_option_fields(robot.spec.option), [])
        model = robot.spec.compile()
        cfg.sim.mujoco.apply(model)
        self.assertEqual(model.opt.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        self.assertAlmostEqual(model.opt.timestep, 0.002)


if __name__ == "__main__":
    unittest.main()
