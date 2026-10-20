"""Multi-seed CPU-physics acceptance with repeatable disturbances and DR."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from result_paths import legacy_result_dir


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wheelbot", "steerbot"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--policy-yaw-scale", type=float, default=1.)
    parser.add_argument("--model", default=None)
    parser.add_argument("--com-offset", type=float, default=None)
    parser.add_argument("--trail", type=float, default=None)
    parser.add_argument("--tilt", type=float, default=None)
    parser.add_argument("--head-mass", type=float, default=None)
    parser.add_argument("--body-mass", type=float, default=None)
    parser.add_argument("--body-offset-z", type=float, default=None)
    parser.add_argument("--tire-halfwidth", type=float, default=None)
    parser.add_argument("--reaction-gear", type=float, default=None)
    parser.add_argument("--steer-gear", type=float, default=None)
    parser.add_argument("--rotor-friction-scale", type=float, default=None)
    parser.add_argument("--recovery-rotor-gain", type=float, default=None)
    parser.add_argument("--recovery-yaw-gain", type=float, default=None)
    parser.add_argument("--recovery-yaw-integral-gain", type=float, default=None)
    parser.add_argument("--pre-turn-yaw-gain", type=float, default=None)
    parser.add_argument("--pre-turn-rotor-gain", type=float, default=None)
    parser.add_argument("--speed-slew", type=float, default=None)
    parser.add_argument("--roll-ff-scale", type=float, default=None)
    parser.add_argument("--yaw-kp-scale", type=float, default=1.)
    parser.add_argument("--controller-params", default=None)
    parser.add_argument("--lqr-tuning", default=None)
    parser.add_argument("--controller", choices=("baseline", "optimized", "lqr", "tilted", "thin", "nested"), default="baseline")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Keep experimental acceptance separate from the published report")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--mirror", action="store_true", help="Test right turn before left turn")
    parser.add_argument("--test-turn-amplitude", type=float, default=.35)
    parser.add_argument("--test-cruise-speed", type=float, default=2.)
    parser.add_argument("--kick-scale", type=float, default=1.)
    parser.add_argument("--disturbance", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply the legacy velocity/roll-rate impulses")
    parser.add_argument("--side-push-force", type=float, default=0.)
    parser.add_argument("--side-push-start", type=float, default=8.5)
    parser.add_argument("--side-push-duration", type=float, default=.3)
    parser.add_argument("--rotor-speed", type=float, default=0.)
    args = parser.parse_args()
    output = args.output_dir or legacy_result_dir("wheelbot_nested_open" if args.controller == "nested" else args.kind)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        command = [sys.executable, str(ROOT / "sim.py"), args.kind,
                   "--headless", "--test", "--duration", "15",
                   "--dr", "--seed", str(seed),
                   "--controller", args.controller,
                   "--output", str(output / f"seed_{seed}.json")]
        if args.disturbance:
            command += ["--disturbance"]
        if args.side_push_force:
            command += ["--side-push-force", str(args.side_push_force),
                        "--side-push-start", str(args.side_push_start),
                        "--side-push-duration", str(args.side_push_duration)]
        if args.mirror:
            command += ["--mirror"]
        if args.kick_scale != 1.:
            command += ["--kick-scale", str(args.kick_scale)]
        if args.test_turn_amplitude != .35:
            command += ["--test-turn-amplitude", str(args.test_turn_amplitude)]
        if args.test_cruise_speed != 2.:
            command += ["--test-cruise-speed", str(args.test_cruise_speed)]
        if args.checkpoint:
            command += ["--checkpoint", str(Path(args.checkpoint).resolve())]
        if args.policy_yaw_scale != 1.:
            command += ["--policy-yaw-scale", str(args.policy_yaw_scale)]
        if args.model:
            command += ["--model", str(Path(args.model).resolve())]
        if args.com_offset is not None:
            command += ["--com-offset", str(args.com_offset)]
        if args.trail is not None:
            command += ["--trail", str(args.trail)]
        if args.tilt is not None:
            command += ["--tilt", str(args.tilt)]
        if args.head_mass is not None:
            command += ["--head-mass", str(args.head_mass)]
        if args.body_mass is not None:
            command += ["--body-mass", str(args.body_mass)]
        if args.body_offset_z is not None:
            command += ["--body-offset-z", str(args.body_offset_z)]
        if args.tire_halfwidth is not None:
            command += ["--tire-halfwidth", str(args.tire_halfwidth)]
        if args.reaction_gear is not None:
            command += ["--reaction-gear", str(args.reaction_gear)]
        if args.steer_gear is not None:
            command += ["--steer-gear", str(args.steer_gear)]
        if args.rotor_friction_scale is not None:
            command += ["--rotor-friction-scale", str(args.rotor_friction_scale)]
        if args.recovery_rotor_gain is not None:
            command += ["--recovery-rotor-gain", str(args.recovery_rotor_gain)]
        if args.recovery_yaw_gain is not None:
            command += ["--recovery-yaw-gain", str(args.recovery_yaw_gain)]
        if args.recovery_yaw_integral_gain is not None:
            command += ["--recovery-yaw-integral-gain", str(args.recovery_yaw_integral_gain)]
        if args.pre_turn_yaw_gain is not None:
            command += ["--pre-turn-yaw-gain", str(args.pre_turn_yaw_gain)]
        if args.pre_turn_rotor_gain is not None:
            command += ["--pre-turn-rotor-gain", str(args.pre_turn_rotor_gain)]
        if args.speed_slew is not None:
            command += ["--speed-slew", str(args.speed_slew)]
        if args.roll_ff_scale is not None:
            command += ["--roll-ff-scale", str(args.roll_ff_scale)]
        if args.yaw_kp_scale != 1.:
            command += ["--yaw-kp-scale", str(args.yaw_kp_scale)]
        if args.controller_params:
            command += ["--controller-params", str(Path(args.controller_params).resolve())]
        if args.lqr_tuning:
            command += ["--lqr-tuning", str(Path(args.lqr_tuning).resolve())]
        if args.rotor_speed:
            command += ["--rotor-speed", str(args.rotor_speed)]
        if args.video and seed == args.seed_start:
            command += ["--video", str(output / "rollout.mp4")]
        env = os.environ.copy()
        if args.video:
            env["MUJOCO_GL"] = "egl"
        subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL)
        results.append(json.loads((output / f"seed_{seed}.json").read_text())["summary"])
        print(f"seed {seed}: {results[-1]}", flush=True)
    def mean_of(key):
        values = [r[key] for r in results if r[key] is not None]
        return sum(values) / len(values) if values else None

    tilt = args.tilt
    rotor_speed = args.rotor_speed
    if args.controller == "tilted":
        params_path = (Path(args.controller_params) if args.controller_params else
                       ROOT / "wheelbot/tilted_axis_robust.json")
        tilt = tilt if tilt is not None else json.loads(params_path.read_text())["tilt"]
        rotor_speed = rotor_speed or 110.
    if args.controller in ("thin", "nested"):
        params_path = (Path(args.controller_params) if args.controller_params else
                       ROOT / ("wheelbot/nested_controller.json" if args.controller == "nested"
                               else "wheelbot/thin_turning_004_g24.json"))
        rotor_speed = rotor_speed or json.loads(params_path.read_text()).get("initial_rotor_speed", 110.)
    tire_halfwidth = args.tire_halfwidth
    reaction_gear = args.reaction_gear
    if args.controller in ("thin", "nested"):
        params_path = (Path(args.controller_params) if args.controller_params else
                       ROOT / ("wheelbot/nested_controller.json" if args.controller == "nested"
                               else "wheelbot/thin_turning_004_g24.json"))
        geometry = json.loads(params_path.read_text())
        tire_halfwidth = tire_halfwidth or geometry.get("tire_halfwidth")
        reaction_gear = reaction_gear or geometry["reaction_gear"]
    steer_gear = args.steer_gear
    geometry_path = (args.lqr_tuning if args.controller == "lqr"
                     else args.controller_params)
    if args.kind == "steerbot" and geometry_path:
        configuration = json.loads(Path(geometry_path).read_text())
        tire_halfwidth = tire_halfwidth or configuration.get("tire_halfwidth")
        steer_gear = steer_gear or configuration.get("steer_gear")
        trail = args.trail if args.trail is not None else configuration.get("trail")
        head_mass = args.head_mass if args.head_mass is not None else configuration.get("head_mass")
        body_mass = args.body_mass if args.body_mass is not None else configuration.get("body_mass")
        body_offset_z = (args.body_offset_z if args.body_offset_z is not None else
                         configuration.get("body_offset_z"))
    else:
        trail, head_mass = args.trail, args.head_mass
        body_mass, body_offset_z = args.body_mass, args.body_offset_z
    recovery_rotor_gain = args.recovery_rotor_gain
    recovery_yaw_gain = args.recovery_yaw_gain
    recovery_yaw_integral_gain = args.recovery_yaw_integral_gain
    if args.controller in ("thin", "nested"):
        recovery_rotor_gain = (geometry.get("recovery_rotor_gain", 0.)
                               if recovery_rotor_gain is None else recovery_rotor_gain)
        recovery_yaw_gain = (geometry.get("recovery_yaw_gain", 0.)
                             if recovery_yaw_gain is None else recovery_yaw_gain)
        recovery_yaw_integral_gain = (
            geometry.get("recovery_yaw_integral_gain", 0.)
            if recovery_yaw_integral_gain is None else recovery_yaw_integral_gain)
    aggregate = {
        "model": args.kind, "controller": args.checkpoint or args.controller,
        "policy_yaw_scale": args.policy_yaw_scale,
        "model_path": (str(Path(args.model).resolve()) if args.model else
                       str(ROOT / args.kind / ("nested_model.xml" if args.controller == "nested" else
                                               "validated_model.xml" if args.kind == "wheelbot"
                                               and args.controller == "thin" else "model.xml"))),
        "com_offset": args.com_offset,
        "trail": trail,
        "tilt": tilt,
        "head_mass": head_mass,
        "body_mass": body_mass,
        "body_offset_z": body_offset_z,
        "tire_halfwidth": tire_halfwidth,
        "reaction_gear": reaction_gear,
        "outer_radius": geometry.get("outer_radius") if args.controller == "nested" else None,
        "inner_radius": geometry.get("inner_radius") if args.controller == "nested" else None,
        "rim_halfwidth": geometry.get("rim_halfwidth") if args.controller == "nested" else None,
        "flywheel_radius": geometry.get("flywheel_radius") if args.controller == "nested" else None,
        "steer_gear": steer_gear,
        "rotor_friction_scale": args.rotor_friction_scale,
        "recovery_rotor_gain": recovery_rotor_gain,
        "recovery_yaw_gain": recovery_yaw_gain,
        "recovery_yaw_integral_gain": recovery_yaw_integral_gain,
        "pre_turn_yaw_gain": (geometry.get("pre_turn_yaw_gain", 0.)
                              if args.pre_turn_yaw_gain is None else args.pre_turn_yaw_gain)
        if args.controller in ("thin", "nested") else None,
        "pre_turn_rotor_gain": (geometry.get("pre_turn_rotor_gain", 0.)
                                if args.pre_turn_rotor_gain is None else args.pre_turn_rotor_gain)
        if args.controller in ("thin", "nested") else None,
        "speed_slew": (geometry.get("speed_slew", 2.)
                       if args.speed_slew is None else args.speed_slew)
        if args.controller in ("thin", "nested") else None,
        "roll_ff_scale": (geometry.get("roll_ff_scale", 1.)
                          if args.roll_ff_scale is None else args.roll_ff_scale)
        if args.controller in ("thin", "nested") else args.roll_ff_scale,
        "yaw_kp_scale": args.yaw_kp_scale,
        "controller_params": str(Path(args.controller_params).resolve()) if args.controller_params else None,
        "lqr_tuning": str(Path(args.lqr_tuning).resolve()) if args.lqr_tuning else None,
        "rotor_speed": rotor_speed,
        "mirror": args.mirror,
        "domain_randomization": True,
        "disturbance": args.disturbance,
        "side_push_force": args.side_push_force,
        "side_push_start": args.side_push_start,
        "side_push_duration": args.side_push_duration,
        "kick_scale": args.kick_scale,
        "test_turn_amplitude": args.test_turn_amplitude,
        "test_cruise_speed": args.test_cruise_speed,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "pass_rate": sum(r["pass_2m_turn"] for r in results) / len(results),
        "mean_steady_speed_rmse": mean_of("steady_speed_rmse"),
        "mean_straight_yaw_rmse": mean_of("straight_yaw_rmse"),
        "mean_turn_yaw_rmse": mean_of("turn_yaw_rmse"),
        "mean_recovery_yaw_rmse": mean_of("recovery_yaw_rmse"),
        "mean_turn_lateral_rmse": mean_of("turn_lateral_rmse"),
        "individual": results,
    }
    (output / "acceptance.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps({key: value for key, value in aggregate.items() if key != "individual"}, indent=2))


if __name__ == "__main__":
    main()
