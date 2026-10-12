"""Check video timing and that auto-reset cannot hide a failed episode."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import imageio.v2 as imageio

from miniwheelbot_rl.video import record_video


class VideoTests(unittest.TestCase):
    def record(self, directory, fail_at=None):
        command = torch.zeros((1, 3))
        state = {"steps": 0, "reset_frame": False}

        def observations():
            torch.testing.assert_close(command, torch.tensor([[0.5, 0.0, -0.2]]))
            return command.clone()

        def step(actions):
            state["steps"] += 1
            done = fail_at is not None and state["steps"] >= fail_at
            state["reset_frame"] = done
            return None, None, torch.tensor([done]), {}

        def render():
            self.assertFalse(state["reset_frame"], "must not record automatic reset")
            return np.full((32, 48, 3), state["steps"] % 255, dtype=np.uint8)

        args = SimpleNamespace(speed=0.5, turn=-0.2, duration=1.0, seed=42,
                               video=Path(directory) / "test.mp4", checkpoint=Path("model.pt"))
        env = SimpleNamespace(device="cpu", step_dt=0.01, reset=lambda: None,
                              command_manager=SimpleNamespace(get_command=lambda _: command),
                              render=render)
        wrapped = SimpleNamespace(get_observations=observations, step=step)
        runner = SimpleNamespace(get_inference_policy=lambda **_: lambda obs: obs)
        report = record_video(args, env, wrapped, runner)
        with imageio.get_reader(str(args.video)) as reader:
            self.assertEqual(reader.count_frames(), report["frames"])
            self.assertEqual(reader.get_meta_data()["fps"], 30)
        self.assertTrue(args.video.with_suffix(".json").exists())
        return report

    def test_realtime_frame_rate_and_command(self):
        with TemporaryDirectory() as directory:
            report = self.record(directory)
        self.assertEqual(report["frames"], 30)
        self.assertFalse(report["terminated_early"])
        self.assertAlmostEqual(report["simulated_duration_s"], 1.0)

    def test_failure_stops_before_reset_frame(self):
        with TemporaryDirectory() as directory:
            report = self.record(directory, fail_at=17)
        self.assertTrue(report["terminated_early"])
        self.assertAlmostEqual(report["simulated_duration_s"], 0.17)
        self.assertEqual(report["frames"], 5)


if __name__ == "__main__":
    unittest.main()
