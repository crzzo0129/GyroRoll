"""Physical spawn clearance and recovery deadline regression tests (CPU)."""

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np
import torch

from miniwheelbot_rl.curriculum import LESSONS
from miniwheelbot_rl.model import MODEL, velocity_model_xml
from miniwheelbot_rl.settings import Settings
from miniwheelbot_rl.spawn import CollisionSupport
from miniwheelbot_rl.recovery import RecoveryState


class RecoveryStateTests(unittest.TestCase):
    def test_lying_times_out_and_transient_upright_does_not_restart_deadline(self):
        state = RecoveryState(2, "cpu")
        state.reset(torch.arange(2), torch.tensor([True, False]))
        for step in range(500):
            # First robot repeatedly looks upright for 0.5 s, never for a full second.
            stable = torch.tensor([step % 100 < 50, True])
            done = state.update(~stable, stable, 0.01, LESSONS["rise"])
        self.assertTrue(done[0])
        self.assertFalse(done[1])
        self.assertEqual(state.successes.tolist(), [0, 0])
        self.assertEqual(state.attempts.tolist(), [1, 0])

    def test_recovery_needs_continuous_stability_and_new_fall_is_charged_once(self):
        state = RecoveryState(1, "cpu")
        state.reset(torch.tensor([0]), torch.tensor([True]))
        for _ in range(100):
            state.update(torch.tensor([True]), torch.tensor([False]), 0.01, LESSONS["rise"])
        for _ in range(99):
            state.update(torch.tensor([False]), torch.tensor([True]), 0.01, LESSONS["rise"])
        self.assertTrue(state.active[0])
        state.update(torch.tensor([False]), torch.tensor([True]), 0.01, LESSONS["rise"])
        self.assertFalse(state.active[0])
        self.assertEqual(state.successes.item(), 1)
        self.assertAlmostEqual(state.time_sum.item(), 2, places=4)
        state.update(torch.tensor([True]), torch.tensor([False]), 0.01, LESSONS["rise"])
        self.assertEqual(state.fall_pulse.item(), 1)
        self.assertEqual(state.attempts.item(), 2)
        state.update(torch.tensor([True]), torch.tensor([False]), 0.01, LESSONS["rise"])
        self.assertEqual(state.fall_pulse.item(), 0)

    def test_partial_reset_preserves_other_trials(self):
        state = RecoveryState(2, "cpu")
        state.elapsed[:] = torch.tensor([4.0, 3.0])
        state.successes[:] = 1
        state.reset(torch.tensor([0]), torch.tensor([True]))
        self.assertEqual(state.elapsed.tolist(), [0, 3])
        self.assertEqual(state.successes.tolist(), [0, 1])


class SpawnGeometryTests(unittest.TestCase):
    def test_mesh_and_primitive_support_matches_real_ground_contacts(self):
        support = CollisionSupport(Settings(), MODEL, "cpu")
        spec = mujoco.MjSpec.from_string(velocity_model_xml(ground=True))
        models = {}
        for slope in (0, 3, -6, 10):
            angle = math.radians(slope)
            spec.geom("ground").quat = [math.cos(angle / 2), 0, -math.sin(angle / 2), 0]
            models[slope] = spec.compile()
        cases = [(r, p, y, s) for r, p in ((0, 0), (90, 0), (-90, 0), (100, 0),
                                          (0, 90), (0, -90), (45, 20))
                 for y in (0, 90, 180) for s in (0, 3, -6, 10)]
        for roll, pitch, yaw, slope in cases:
            with self.subTest(roll=roll, pitch=pitch, yaw=yaw, slope=slope):
                angle = math.radians(slope)
                model = models[slope]
                ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
                normal = np.array([-math.sin(angle), 0, math.cos(angle)])
                quat = np.zeros(4)
                mujoco.mju_euler2Quat(quat, np.radians([roll, pitch, yaw]), "xyz")
                # MuJoCo xyz rotates intrinsic axes, but all sampled orientations
                # still independently test arbitrary support directions.
                matrix = np.zeros(9)
                mujoco.mju_quat2Mat(matrix, quat)
                body_normal = matrix.reshape(3, 3).T @ normal
                minimum = float(support.minimum(torch.tensor(body_normal[None], dtype=torch.float32))[0])
                data = mujoco.MjData(model)
                data.qpos[:3] = [0, 0, (0.002 - minimum) / normal[2]]
                data.qpos[3:7] = quat
                mujoco.mj_forward(model, data)
                contacts = [c.dist for c in data.contact if ground_id in (c.geom1, c.geom2)]
                self.assertTrue(all(d > -1e-5 for d in contacts))
                # Lower by 3 mm in normal distance: support should now touch/penetrate.
                data.qpos[2] -= 0.003 / normal[2]
                mujoco.mj_forward(model, data)
                contacts = [c.dist for c in data.contact if ground_id in (c.geom1, c.geom2)]
                self.assertTrue(contacts, "Support was too conservative: body floats above surface")
                self.assertLess(min(contacts), -0.0005)


class RecoveryEvaluationTests(unittest.TestCase):
    def test_failed_trial_never_becomes_success_after_manual_reset(self):
        from miniwheelbot_rl.evaluation import evaluate_curriculum
        command = torch.zeros((2, 3))
        monitor = RecoveryState(2, "cpu")
        elapsed = torch.zeros(2)
        data = SimpleNamespace(actuator_force=torch.zeros((2, 2)), root_link_ang_vel_b=torch.zeros((2, 3)))
        env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.1, _recovery=monitor,
            scene={"robot": SimpleNamespace(data=data)},
            cfg=SimpleNamespace(auto_reset=True, events={"reset_root": SimpleNamespace(
                params={"settings": Settings()})}),
            command_manager=SimpleNamespace(get_command=lambda _: command))

        def reset(env_ids=None):
            ids = torch.arange(2) if env_ids is None else env_ids
            elapsed[ids] = 0
            monitor.reset(ids, torch.full((len(ids),), env._evaluation_pose != "upright"))

        def step(actions):
            elapsed.add_(0.1)
            successful = elapsed >= 1
            monitor.successes[successful] = monitor.attempts[successful]
            monitor.active[successful] = False
            # Trial 1 fails on every fresh episode, then gets explicitly reset.
            done = torch.tensor([False, elapsed[1] >= 0.2])
            return None, None, done, {}

        env.reset = reset
        wrapper = SimpleNamespace(get_observations=lambda: None, step=step)
        runner = SimpleNamespace(get_inference_policy=lambda **kw: lambda obs: torch.zeros((2, 2)))
        with TemporaryDirectory() as directory, \
                patch("miniwheelbot_rl.evaluation.tangent_velocity", lambda _: (command[:, 0], torch.zeros(2))), \
                patch("miniwheelbot_rl.evaluation.upright_gate", lambda _: torch.ones(2)), \
                patch("miniwheelbot_rl.evaluation.ground_height", lambda _: torch.full((2,), 0.37)):
            args = SimpleNamespace(stage="rise", duration=10, checkpoint="test.pt", seed=42,
                                   output=Path(directory) / "evaluation.json")
            report = evaluate_curriculum(args, env, wrapper, runner, heading_fn=lambda _: command[:, 2])
        for case in report["cases"]:
            good, failed = case["individual"]
            self.assertTrue(good["passed"], case["name"])
            self.assertFalse(failed["passed"], case["name"])
            self.assertFalse(failed["survived"])
            self.assertFalse(failed["recovered"])
            self.assertEqual(failed["tracking_samples"], 0)
        self.assertTrue(env.cfg.auto_reset)


if __name__ == "__main__":
    unittest.main()
