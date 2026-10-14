"""Check motor identity, sample timing, clipping, and exclusion of reset states."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from miniwheelbot_rl.diagnostics import diagnose
from miniwheelbot_rl.settings import Settings


class DiagnosticsTests(unittest.TestCase):
    def run_diagnostic(self, directory, fail_at=None):
        directory = Path(directory)
        (directory / "task.json").write_text(json.dumps({"settings": asdict(Settings())}))
        command = torch.zeros((1, 3))
        data = SimpleNamespace(joint_vel=torch.zeros((1, 2)),
                               actuator_force=torch.zeros((1, 2)),
                               root_link_lin_vel_b=torch.zeros((1, 3)))
        # Deliberately reverse the storage order to catch wheel/actuator confusion.
        robot = SimpleNamespace(data=data,
            find_joints=lambda names, **kw: ([1, 0], list(names)),
            find_actuators=lambda names, **kw: ([1, 0], list(names)))
        state = {"steps": 0}

        def observations():
            torch.testing.assert_close(command, torch.tensor([[0.5, 0, -0.2]]))
            return command

        def step(actions):
            torch.testing.assert_close(actions, torch.tensor([[2.0, -0.5]]))
            state["steps"] += 1
            done = fail_at is not None and state["steps"] >= fail_at
            data.joint_vel[:] = torch.tensor([[999., 999.]]) if done else torch.tensor([[-10., 100.]])
            data.actuator_force[:] = torch.tensor([[-1., 2.4]])
            data.root_link_lin_vel_b[0, 0] = 0.4
            return None, None, torch.tensor([done]), {}

        env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.01,
                              scene={"robot": robot}, reset=lambda: None,
                              command_manager=SimpleNamespace(get_command=lambda _: command))
        runner = SimpleNamespace(get_inference_policy=lambda **kw: lambda obs: torch.tensor([[2., -0.5]]))
        args = SimpleNamespace(speed=0.5, turn=-0.2, duration=3.1, seed=42,
                               checkpoint=directory / "model.pt", diagnostic_dir=directory)
        report = diagnose(args, env, SimpleNamespace(get_observations=observations, step=step), runner,
                          heading_fn=lambda env: torch.tensor([-0.1]))
        with (directory / "trace.csv").open() as f:
            rows = list(csv.DictReader(f))
        return report, rows

    def test_mapping_timing_and_statistics(self):
        with TemporaryDirectory() as directory:
            report, rows = self.run_diagnostic(directory)
            self.assertEqual(len(rows), 310)
            self.assertAlmostEqual(float(rows[0]["time_s"]), 0.01)
            self.assertEqual(float(rows[0]["flywheel_target_rad_s"]), 250)
            self.assertEqual(float(rows[0]["rolling_target_rad_s"]), -17.5)
            self.assertEqual(float(rows[0]["flywheel_actual_rad_s"]), 100)
            self.assertEqual(float(rows[0]["rolling_actual_rad_s"]), -10)
            self.assertAlmostEqual(float(rows[0]["flywheel_torque_Nm"]), 2.4, places=5)
            self.assertEqual(report["all_samples"]["flywheel"]["clipped_fraction"], 1)
            self.assertEqual(report["all_samples"]["rolling"]["clipped_fraction"], 0)
            self.assertEqual(report["all_samples"]["flywheel"]["torque_saturation_fraction"], 1)
            self.assertEqual(report["all_samples"]["rolling"]["torque_saturation_fraction"], 0)
            self.assertAlmostEqual(report["all_samples"]["forward_rmse_m_s"], 0.1)
            self.assertEqual(report["after_3s"]["samples"], 10)
            self.assertIn("<svg", (Path(directory) / "report.html").read_text(encoding="utf-8"))

    def test_stop_before_reset_and_clear_previous_csv(self):
        with TemporaryDirectory() as directory:
            report, rows = self.run_diagnostic(directory, fail_at=17)
            self.assertTrue(report["terminated_early"])
            self.assertEqual(len(rows), 16)
            self.assertEqual(float(rows[-1]["flywheel_actual_rad_s"]), 100)
            self.assertEqual(report["after_3s"]["samples"], 0)
            report, rows = self.run_diagnostic(directory, fail_at=1)
            self.assertEqual(rows, [])
            self.assertEqual(report["all_samples"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
