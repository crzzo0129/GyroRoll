"""End-to-end checks of the accepted controller/geometry combinations."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModerateAcceptanceTests(unittest.TestCase):
    def replay(self, kind, controller_args):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "sim.py"), kind, "--headless", "--test",
             "--duration", "15", "--dr", "--disturbance", "--kick-scale", ".6",
             "--mirror", "--seed", "0", *controller_args],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        return json.loads(completed.stdout)

    def test_wheelbot_mirror(self):
        summary = self.replay("wheelbot", ["--controller", "thin"])
        self.assertTrue(summary["pass_2m_turn"])
        self.assertEqual(summary["duration"], 15.)

    def test_steerbot_negative_trail_mirror(self):
        summary = self.replay(
            "steerbot", ["--controller", "lqr", "--lqr-tuning",
                         str(ROOT / "steerbot/negative_candidate.json")],
        )
        self.assertTrue(summary["pass_2m_turn"])
        self.assertLess(summary["turn_yaw_rmse"], .15)


if __name__ == "__main__":
    unittest.main()
