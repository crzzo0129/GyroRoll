"""Cross-check offline score against the independent replay path."""

import json
import ctypes
import unittest
from pathlib import Path

import numpy as np

from linearize_steer import identify
from optimize_steer import evaluate
from probe_tilted_flywheel import evaluate as evaluate_tilted, model_at
import optimize_wheel_rounded as wheel_probe
from sim import joystick_axes, run


ROOT = Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    def test_joystick_axes_pointer_count_api(self):
        values = (ctypes.c_float * 3)(-.5, .75, 0.)
        self.assertEqual(list(joystick_axes((values, 3))), [-.5, .75, 0.])
        self.assertEqual(joystick_axes([-.5, .75]), [-.5, .75])

    def test_lateral_trim_controllable(self):
        result = identify()
        self.assertEqual(result["controllability_rank"], 6)
        self.assertTrue(1.9 < result["speed"] < 2.2)
        narrow = identify(tire_halfwidth=.004, steer_gear=2.4)
        self.assertLess(abs(narrow["trim"][3]), .005)

    def test_optimizer_and_replay_agree(self):
        params = json.loads((ROOT / "steerbot" / "optimized_controller.json").read_text())["params"]
        if len(params) == 11:
            params = params + [0., 0.]
        optimized = evaluate(np.asarray(params), return_metrics=True)

        class Args:
            kind = "steerbot"
            model = None
            com_offset = None
            trail = None
            tilt = None
            head_mass = None
            body_mass = None
            body_offset_z = None
            tire_halfwidth = None
            reaction_gear = None
            steer_gear = None
            rotor_friction_scale = None
            recovery_rotor_gain = None
            recovery_yaw_gain = None
            recovery_yaw_integral_gain = None
            pre_turn_yaw_gain = None
            pre_turn_rotor_gain = None
            speed_slew = None
            roll_ff_scale = 1.
            yaw_kp_scale = 1.
            controller_params = None
            lqr_tuning = None
            seed = 0
            dr = False
            controller = "optimized"
            rotor_speed = 0.
            checkpoint = None
            policy_yaw_scale = 1.
            headless = True
            video = None
            speed = 2.
            turn = 0.
            test = True
            duration = 15.
            output = None
            disturbance = False
            kick_scale = 1.
            mirror = False
            test_turn_amplitude = .35
            test_cruise_speed = 2.

        import contextlib
        import io

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run(Args())
        replay = json.loads(stream.getvalue())
        self.assertEqual(optimized["fallen"], replay["fallen"])
        self.assertAlmostEqual(optimized["speed_rmse"], replay["steady_speed_rmse"], places=6)
        self.assertAlmostEqual(optimized["yaw_rmse"], replay["turn_yaw_rmse"], places=6)
        self.assertAlmostEqual(optimized["lateral_rmse"], replay["turn_lateral_rmse"], places=6)

    def test_tilted_feedback_and_replay_agree(self):
        path = ROOT / "wheelbot/tilted_axis_probe.json"
        configuration = json.loads(path.read_text())
        optimized = evaluate_tilted(model_at(configuration["tilt"]),
                                    configuration["params"], True)

        class Args:
            kind = "wheelbot"
            model = None
            com_offset = None
            trail = None
            tilt = None
            head_mass = None
            body_mass = None
            body_offset_z = None
            tire_halfwidth = None
            reaction_gear = None
            steer_gear = None
            rotor_friction_scale = None
            recovery_rotor_gain = None
            recovery_yaw_gain = None
            recovery_yaw_integral_gain = None
            pre_turn_yaw_gain = None
            pre_turn_rotor_gain = None
            speed_slew = None
            roll_ff_scale = 1.
            yaw_kp_scale = 1.
            controller_params = str(path)
            lqr_tuning = None
            seed = 0
            dr = False
            controller = "tilted"
            rotor_speed = 0.
            checkpoint = None
            policy_yaw_scale = 1.
            headless = True
            video = None
            speed = 2.
            turn = 0.
            test = True
            duration = 15.
            output = None
            disturbance = True
            kick_scale = 1.
            mirror = False
            test_turn_amplitude = .35
            test_cruise_speed = 2.

        import contextlib
        import io

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run(Args())
        replay = json.loads(stream.getvalue())
        self.assertEqual(optimized["fallen"], replay["fallen"])
        self.assertAlmostEqual(optimized["speed_rmse"], replay["steady_speed_rmse"], delta=3e-5)
        self.assertAlmostEqual(optimized["yaw_rmse"], replay["turn_yaw_rmse"], delta=3e-5)
        self.assertAlmostEqual(optimized["lateral_rmse"], replay["turn_lateral_rmse"], delta=3e-5)

    def test_narrow_wheel_feedback_and_replay_agree(self):
        from argparse import Namespace
        import contextlib
        import io

        path = ROOT / "wheelbot/thin_turning_004_g24.json"
        params = json.loads(path.read_text())["params"]
        wheel_probe.MODEL = wheel_probe.model_at(.004, cylinder=True, reaction_gear=2.4)
        optimized = wheel_probe.evaluate(params, turning=True, metrics=True)
        args = Namespace(kind="wheelbot", model=None, com_offset=None, trail=None,
                         tilt=None, head_mass=None, body_mass=None, body_offset_z=None,
                         tire_halfwidth=None, reaction_gear=None,
                         steer_gear=None,
                         controller_params=str(path), lqr_tuning=None,
                         seed=0, dr=False,
                         controller="thin", rotor_speed=110., checkpoint=None,
                         policy_yaw_scale=1.,
                         headless=True, video=None, speed=2., turn=0., test=True,
                         duration=15., output=None, disturbance=True, mirror=False,
                         kick_scale=1.,
                         test_turn_amplitude=.35, test_cruise_speed=2.,
                         rotor_friction_scale=None, recovery_rotor_gain=0.,
                         recovery_yaw_gain=0., recovery_yaw_integral_gain=0.,
                         pre_turn_yaw_gain=0.,
                         pre_turn_rotor_gain=0.,
                         speed_slew=2.,
                         roll_ff_scale=1., yaw_kp_scale=1.)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run(args)
        replay = json.loads(stream.getvalue())
        self.assertEqual(optimized["fallen"], replay["fallen"])
        self.assertAlmostEqual(optimized["speed_rmse"], replay["steady_speed_rmse"], delta=1e-4)
        self.assertAlmostEqual(optimized["yaw_rmse"], replay["turn_yaw_rmse"], delta=1e-4)
        self.assertAlmostEqual(optimized["lateral_rmse"], replay["turn_lateral_rmse"], delta=1e-4)


if __name__ == "__main__":
    unittest.main()
