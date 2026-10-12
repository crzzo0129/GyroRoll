"""Check turn tracking measures horizontal heading even with body tilt."""

from types import SimpleNamespace
import unittest
import mujoco
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires the MJLab training environment")
class HeadingRateTests(unittest.TestCase):
    def test_tilted_heading_derivative_matches_orientation_integration(self):
        from miniwheelbot_rl.task import heading_rate
        quat = np.array([0.9, 0.3, -0.2, 0.15])
        quat /= np.linalg.norm(quat)
        omega_w = np.array([0.4, -0.3, 0.7])
        data = SimpleNamespace(root_link_quat_w=torch.tensor(quat[None, :], dtype=torch.float32),
                              root_link_ang_vel_w=torch.tensor(omega_w[None, :], dtype=torch.float32))
        env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": SimpleNamespace(data=data)})
        dt = 1e-6
        delta = np.zeros(4)
        mujoco.mju_axisAngle2Quat(delta, omega_w / np.linalg.norm(omega_w),
                                 np.linalg.norm(omega_w) * dt)
        next_quat = np.zeros(4)
        mujoco.mju_mulQuat(next_quat, delta, quat)
        headings = []
        for q in (quat, next_quat):
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, q)
            headings.append(np.arctan2(mat.reshape(3, 3)[1, 0], mat.reshape(3, 3)[0, 0]))
        expected = (headings[1] - headings[0]) / dt
        self.assertAlmostEqual(float(heading_rate(env)[0]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
