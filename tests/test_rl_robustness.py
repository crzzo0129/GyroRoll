"""Force lifecycle, actual physical impulse and stop/start acceptance checks."""
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from miniwheelbot_rl.robustness import HorizontalPushes, ROBUST_TRACKING, standing_mask, standing_score


class ForceRobot:
    def __init__(self, count):
        self.forces = torch.zeros((count, 1, 3))
        self.torques = torch.zeros_like(self.forces)

    def write_external_wrench_to_sim(self, forces, torques, env_ids=None, body_ids=None):
        assert body_ids == [0]
        ids = slice(None) if env_ids is None else env_ids
        self.forces[ids], self.torques[ids] = forces, torques


class RobustnessTests(unittest.TestCase):
    def event(self, count=2):
        robot = ForceRobot(count)
        env = SimpleNamespace(num_envs=count, device="cpu", step_dt=.01, scene={"robot": robot})
        params = dict(asset_cfg=SimpleNamespace(name="robot", body_ids=[0]),
                      force_newtons=(1.5, 1.5), duration_seconds=(.2, .2),
                      cooldown_seconds=(.1, .1))
        event = HorizontalPushes(SimpleNamespace(params=params), env)
        return event, env, robot, params

    def test_pulse_delay_duration_horizontal_bound_and_partial_reset(self):
        event, env, robot, params = self.event()
        for _ in range(9):
            event(env, None, **params)
            self.assertEqual(float(robot.forces.abs().sum()), 0.)
        event(env, None, **params)
        torch.testing.assert_close(robot.forces.norm(dim=-1), torch.full((2, 1), 1.5))
        self.assertEqual(float(robot.forces[:, :, 2].abs().sum()), 0.)
        self.assertEqual(float(robot.torques.abs().sum()), 0.)
        original = robot.forces.clone()
        for _ in range(19):
            event(env, None, **params)
            torch.testing.assert_close(robot.forces, original)
        event.reset(torch.tensor([0]))
        self.assertEqual(float(robot.forces[0].abs().sum()), 0.)
        torch.testing.assert_close(robot.forces[1], original[1])
        event(env, None, **params)
        self.assertEqual(float(robot.forces.abs().sum()), 0.)
        self.assertEqual(int(event.wait[0]), 9)

    def test_force_changes_momentum_through_physics(self):
        event, env, robot, params = self.event(count=1)
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
          <worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1.465"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        impulse = np.zeros(3)
        for _ in range(30):
            event(env, None, **params)
            force = robot.forces[0, 0].numpy()
            data.xfrc_applied[1, :3] = force
            mujoco.mj_step(model, data, nstep=5)
            impulse += force * .01
        self.assertAlmostEqual(float(np.linalg.norm(impulse)), .3, places=6)
        np.testing.assert_allclose(data.qvel[:3] * 1.465, impulse, atol=1e-6)

    def test_standing_means_no_translation_or_turn_and_rejects_drift(self):
        cmd = torch.tensor([[0., 0., 0.], [0., 0., .3], [1., 0., 0.]])
        env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: cmd))
        self.assertEqual(standing_mask(env).tolist(), [True, False, False])
        score = standing_score(torch.ones(3, dtype=torch.bool),
            torch.tensor([[0., 0.], [.2, 0.], [0., 0.]]),
            torch.tensor([0., 0., .3]), torch.ones(3))
        self.assertEqual(float(score[0]), 1.)
        self.assertLess(float(score[1]), .02)
        self.assertLess(float(score[2]), .02)

    def test_stop_go_and_failure_stays_failed_in_evaluation(self):
        from miniwheelbot_rl.evaluation import evaluate_robust_tracking
        cmd = torch.zeros((2, 3))
        robot = ForceRobot(2)
        robot.data = SimpleNamespace(root_link_lin_vel_b=torch.zeros((2, 3)))
        robot.find_bodies = lambda *a, **kw: ([0], ["base"])
        elapsed = torch.zeros(2)
        env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.1, scene={"robot": robot},
            cfg=SimpleNamespace(auto_reset=True), command_manager=SimpleNamespace(get_command=lambda _: cmd))
        def reset(env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            elapsed[ids] = 0
            robot.forces[ids] = 0
        def step(_):
            elapsed.add_(.1)
            robot.data.root_link_lin_vel_b[:, 0] = cmd[:, 0]
            return None, None, torch.tensor([False, elapsed[1] >= .2]), {}
        env.reset = reset
        wrapper = SimpleNamespace(get_observations=lambda: None, step=step)
        runner = SimpleNamespace(get_inference_policy=lambda **kw: lambda obs: torch.zeros((2, 2)))
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(stage="full_robust", duration=15, checkpoint="test.pt", seed=42,
                                   output=Path(directory)/"report.json")
            report = evaluate_robust_tracking(args, env, wrapper, runner, heading_fn=lambda _: torch.zeros(2))
        for case in report["cases"]:
            self.assertTrue(case["individual"][0]["passed"], case["name"])
            self.assertFalse(case["individual"][1]["passed"], case["name"])
        self.assertEqual(float(robot.forces.abs().sum()), 0.)
        self.assertTrue(env.cfg.auto_reset)


@unittest.skipIf(find_spec("mjlab") is None, "MJLab config integration requires mjlab==1.3.0")
class RobustConfigTests(unittest.TestCase):
    def test_full_robust_uses_flat_resets_bounded_force_and_zero_twist_samples(self):
        from miniwheelbot_rl.task import make_env_cfg
        cfg = make_env_cfg(stage="full_robust", randomize=True)
        self.assertEqual(cfg.commands['twist'].rel_standing_envs, ROBUST_TRACKING.standing_fraction)
        self.assertEqual(cfg.commands['twist'].ranges.lin_vel_x, (-2., 2.))
        self.assertEqual(cfg.events['horizontal_push'].mode, 'step')
        self.assertNotIn('push', cfg.events)
        self.assertIn('friction', cfg.events)
        self.assertIn('fall', cfg.terminations)
        self.assertNotIn('recovery_timeout', cfg.terminations)
        play = make_env_cfg(stage="full_robust", play=True)
        self.assertNotIn('horizontal_push', play.events)
        self.assertEqual(make_env_cfg(stage="full").commands['twist'].rel_standing_envs, .1)


if __name__ == '__main__':
    unittest.main()
