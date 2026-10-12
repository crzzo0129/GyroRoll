"""Interactive MuJoCo viewer and repeatable headless smoke test.

Examples: python sim.py wheelbot --headless --duration 15
          python sim.py steerbot --speed 2 --turn .3
"""

import argparse
import json
import threading
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from control import (Baseline, LqrSteerController, OptimizedSteerController,
                     ThinWheelController, TiltedWheelController, observe)
from policy import Policy


ROOT = Path(__file__).resolve().parent


def reset(model, data, kind, controller, seed=0):
    mujoco.mj_resetData(model, data)
    # qpos of the free joint starts with (xyz, wxyz).
    data.qpos[:7] = model.qpos0[:7]
    if kind == "wheelbot" and controller.rotor_speed:
        reaction = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "reaction")
        data.qvel[model.jnt_dofadr[reaction]] = controller.rotor_speed
    if kind == "steerbot":
        data.qvel[0] = .8
        drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drive")
        data.qvel[model.jnt_dofadr[drive]] = .8 / .12
    mujoco.mj_forward(model, data)
    controller.reset()


def sample_commands(t, speed, turn, test, mirror=False,
                    turn_amplitude=.35, cruise_speed=2.):
    if not test:
        return speed, turn
    if t < 2.0:
        return 1.0, 0.0
    if t < 5.0:
        return cruise_speed, 0.0
    if t < 8.0:
        return cruise_speed, -turn_amplitude if mirror else turn_amplitude
    if t < 11.0:
        return cruise_speed, turn_amplitude if mirror else -turn_amplitude
    return 1.5, 0.0


def joystick_axes(result):
    # PyGLFW versions return either an array or a (C pointer, count) pair.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
        return result[0][:result[1]]
    return result


def run(args):
    default_xml = ("nested_model.xml" if args.controller == "nested" else
                   "validated_model.xml" if args.kind == "wheelbot" and args.controller == "thin"
                   else "model.xml")
    model_path = str(args.model or ROOT / args.kind / default_xml)
    if args.controller == "nested":
        if args.kind != "wheelbot":
            raise ValueError("Nested-wheel feedback only applies to wheelbot")
        if args.tilt is not None or args.tire_halfwidth is not None:
            raise ValueError("Regenerate the nested annulus to change its geometry")
        params_path = (Path(args.controller_params) if args.controller_params
                       else ROOT / "wheelbot/nested_controller.json")
    if args.controller == "tilted":
        if args.kind != "wheelbot":
            raise ValueError("Tilted flywheel feedback only applies to wheelbot")
        params_path = Path(args.controller_params) if args.controller_params else ROOT / "wheelbot/tilted_axis_robust.json"
        designed_tilt = json.loads(params_path.read_text())["tilt"]
        if args.tilt is None:
            args.tilt = designed_tilt
        elif not np.isclose(args.tilt, designed_tilt):
            raise ValueError("Controller and model flywheel tilts must match")
    if args.controller == "thin":
        if args.kind != "wheelbot":
            raise ValueError("Narrow flywheel feedback only applies to wheelbot")
        if args.tilt is not None:
            raise ValueError("The narrow-tire controller assumes an orthogonal flywheel axis")
        params_path = Path(args.controller_params) if args.controller_params else ROOT / "wheelbot/thin_turning_004_g24.json"
        configuration = json.loads(params_path.read_text())
        halfwidth = configuration["tire_halfwidth"]
        gear = configuration["reaction_gear"]
        if args.tire_halfwidth is not None and not np.isclose(args.tire_halfwidth, halfwidth):
            raise ValueError("Controller and tire widths must match")
        if args.reaction_gear is not None and not np.isclose(args.reaction_gear, gear):
            raise ValueError("Controller and reaction motor gears must match")
        args.tire_halfwidth = halfwidth
        args.reaction_gear = gear
    geometry_path = (args.lqr_tuning if args.controller == "lqr"
                     else args.controller_params)
    if args.kind == "steerbot" and args.controller in ("optimized", "lqr") and geometry_path:
        configuration = json.loads(Path(geometry_path).read_text())
        for key, attribute in (("tire_halfwidth", "tire_halfwidth"),
                               ("steer_gear", "steer_gear"),
                               ("trail", "trail"), ("head_mass", "head_mass"),
                               ("body_mass", "body_mass"),
                               ("body_offset_z", "body_offset_z")):
            designed = configuration.get(key)
            explicit = getattr(args, attribute)
            if designed is not None and explicit is not None and not np.isclose(designed, explicit):
                raise ValueError(f"Controller and {key} geometry must match")
            if designed is not None:
                setattr(args, attribute, designed)
    if (args.com_offset is None and args.trail is None and args.tilt is None
            and args.head_mass is None and args.tire_halfwidth is None
            and args.reaction_gear is None and args.steer_gear is None
            and args.body_mass is None and args.body_offset_z is None):
        model = mujoco.MjModel.from_xml_path(model_path)
    else:
        spec = mujoco.MjSpec.from_file(model_path)
        if args.com_offset is not None:
            if args.kind != "steerbot":
                raise ValueError("COM offset is only supported for steerbot")
            spec.geom("body").pos[0] = args.com_offset
        if args.trail is not None:
            if args.kind != "steerbot":
                raise ValueError("Trail overrides are only supported for steerbot")
            spec.body("fork").pos[0] = args.trail
            spec.body("drive_wheel").pos[0] = -args.trail
        if args.tilt is not None:
            if args.kind != "wheelbot":
                raise ValueError("Flywheel tilt overrides are only supported for wheelbot")
            spec.body("reaction_wheel").quat[:] = (np.cos(args.tilt / 2), 0.,
                                                    -np.sin(args.tilt / 2), 0.)
        if args.head_mass is not None:
            if args.kind != "steerbot" or args.head_mass <= 0:
                raise ValueError("Positive steering-head mass only applies to steerbot")
            spec.geom("steering_head").mass = args.head_mass
        if args.body_mass is not None:
            if args.kind != "steerbot" or args.body_mass <= 0:
                raise ValueError("Positive chassis mass only applies to steerbot")
            spec.geom("body").mass = args.body_mass
        if args.body_offset_z is not None:
            if args.kind != "steerbot":
                raise ValueError("Vertical chassis offset only applies to steerbot")
            base_spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
            spec.geom("body").pos[2] = base_spec.geom("body").pos[2] + args.body_offset_z
        if args.tire_halfwidth is not None:
            if args.tire_halfwidth <= 0:
                raise ValueError("Tire half-width must be positive")
            spec.geom("tire").size[1] = args.tire_halfwidth
        if args.reaction_gear is not None:
            if args.kind != "wheelbot" or args.reaction_gear <= 0:
                raise ValueError("Positive reaction gear only applies to wheelbot")
            spec.actuator("reaction_motor").gear[0] = args.reaction_gear
        if args.steer_gear is not None:
            if args.kind != "steerbot" or args.steer_gear <= 0:
                raise ValueError("Positive steering torque only applies to steerbot")
            spec.actuator("steer_motor").gear[0] = args.steer_gear
        model = spec.compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    if args.dr:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        scale = rng.uniform(.85, 1.15)
        model.body_mass[body] *= scale
        model.body_inertia[body] *= scale
        friction = rng.uniform(1.1, 1.9)
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name or ""
            if name == "tire" or name.startswith("tire_") or name.startswith("rim_"):
                model.geom_friction[geom_id, 0] = friction
        mujoco.mj_setConst(model, data)
    if args.controller in ("optimized", "lqr"):
        if args.kind != "steerbot":
            raise ValueError("The optimized and LQR controllers are only defined for steerbot")
        path = Path(args.controller_params) if args.controller_params else ROOT / "steerbot" / "optimized_controller.json"
        if args.controller == "lqr":
            controller = LqrSteerController(path, com_offset=args.com_offset, trail=args.trail,
                                            tire_halfwidth=args.tire_halfwidth,
                                            steer_gear=args.steer_gear,
                                            head_mass=args.head_mass, body_mass=args.body_mass,
                                            body_offset_z=args.body_offset_z,
                                            tuning_path=args.lqr_tuning)
        else:
            controller = OptimizedSteerController(path)
    elif args.controller == "tilted":
        controller = TiltedWheelController(params_path)
    elif args.controller in ("thin", "nested"):
        controller = ThinWheelController(params_path, model.actuator("reaction_motor").gear[0],
                                         args.rotor_friction_scale,
                                         args.recovery_rotor_gain, args.recovery_yaw_gain,
                                         args.recovery_yaw_integral_gain,
                                         args.pre_turn_yaw_gain, args.pre_turn_rotor_gain,
                                         args.speed_slew,
                                         args.roll_ff_scale, args.yaw_kp_scale)
    else:
        controller = Baseline(args.kind)
    controller.rotor_speed = args.rotor_speed or getattr(controller, "rotor_speed", 0.)
    reset(model, data, args.kind, controller)
    policy = Policy(args.checkpoint) if args.checkpoint else None
    last_action = np.zeros(2)
    manual_speed = args.speed
    manual_turn = args.turn

    def key_callback(key):
        nonlocal manual_speed, manual_turn
        if key == glfw.KEY_R:
            reset(model, data, args.kind, controller)
        elif key == glfw.KEY_UP:
            manual_speed = min(2.2 if args.kind == "steerbot" else 2., manual_speed + .2)
        elif key == glfw.KEY_DOWN:
            manual_speed = max(.6 if args.kind == "steerbot" else -1., manual_speed - .2)
        elif key == glfw.KEY_LEFT:
            manual_turn = min(.35, manual_turn + .1)
        elif key == glfw.KEY_RIGHT:
            manual_turn = max(-.35, manual_turn - .1)
        elif key == glfw.KEY_SPACE:
            manual_turn = 0.

    if not args.headless:
        threads_before_viewer = set(threading.enumerate())
        viewer_context = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        viewer_threads = set(threading.enumerate()) - threads_before_viewer
    else:
        viewer_context = None
    writer = None
    renderer = None
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 544
        renderer = mujoco.Renderer(model, height=544, width=960)
        writer = imageio.get_writer(args.video, fps=30, codec="libx264", quality=8)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = model.body("base").id
        camera.distance = getattr(args, "camera_distance", 1.35)
        camera.azimuth = 115
        camera.elevation = -15

    samples = []
    speed_errors_dense = []
    straight_yaw_errors_dense = []
    yaw_errors_dense = []
    recovery_yaw_errors_dense = []
    lateral_errors_dense = []
    fallen = False
    next_frame_t = 0.
    fired_disturbances = set()
    side_force = getattr(args, "side_push_force", 0.)
    push_start = getattr(args, "side_push_start", 8.5)
    push_duration = getattr(args, "side_push_duration", .3)
    if push_duration <= 0:
        raise ValueError("Side-push duration must be positive")
    base_id = model.body("base").id
    minimum_height = .5 * float(model.qpos0[2]) if args.controller == "nested" else .16
    try:
        while data.time < args.duration and (viewer_context is None or viewer_context.is_running()):
            tick = time.monotonic()
            if viewer_context is not None and glfw.joystick_present(glfw.JOYSTICK_1):
                axes = joystick_axes(glfw.get_joystick_axes(glfw.JOYSTICK_1))
                if axes is not None and len(axes) >= 2:
                    manual_speed = (float(.6 + np.clip(-axes[1], 0, 1) * 1.6)
                                    if args.kind == "steerbot" else
                                    float(np.clip(-axes[1], -1, 1) * 2.))
                    manual_turn = float(np.clip(-axes[0], -1, 1) * .35)
            speed, turn = sample_commands(data.time, manual_speed, manual_turn, args.test,
                                          args.mirror, args.test_turn_amplitude,
                                          args.test_cruise_speed)
            state = observe(model, data, args.kind)
            if data.time >= 3.0:
                speed_errors_dense.append((state.velocity[0] - speed) ** 2)
            if 3. < data.time < 4.8 and args.test:
                straight_yaw_errors_dense.append(state.omega[2] ** 2)
            if 5.5 < data.time < 10.8:
                yaw_errors_dense.append((state.omega[2] - turn) ** 2)
                lateral_errors_dense.append(state.velocity[1] ** 2)
            if 12. < data.time < 15. and args.test:
                recovery_yaw_errors_dense.append(state.omega[2] ** 2)
            if abs(state.roll) > .8 or abs(state.pitch) > .8 or state.height < minimum_height:
                fallen = True
                if args.headless:
                    break
                reset(model, data, args.kind, controller)
            if policy is None:
                data.ctrl[:] = controller.act(state, speed, turn, model.opt.timestep)
            elif round(data.time / model.opt.timestep) % 10 == 0:
                if len(policy.mean) == 18:
                    joint_pos = data.qpos[7:9].copy()  # Legacy experiment, for comparison only.
                else:
                    joint_pos = data.qpos[7:8].copy() if args.kind == "steerbot" else np.empty(0)
                joint_vel = data.qvel[6:8].copy()
                gravity = data.xmat[model.body("base").id].reshape(3, 3)[2, :]
                obs = np.concatenate((state.velocity, state.omega, -gravity,
                                      joint_pos, joint_vel,
                                      [speed, 0., turn * args.policy_yaw_scale], last_action))
                last_action = policy.act(obs)
                if args.kind == "steerbot":
                    data.ctrl[model.actuator("steer_motor").id] = last_action[0]
                    data.ctrl[model.actuator("drive_motor").id] = last_action[1]
                else:
                    data.ctrl[:] = last_action
            for at, sign in ((8.5, 1), (11.5, -1)):
                if args.disturbance and data.time >= at and at not in fired_disturbances:
                    data.qvel[1] += sign * .25 * args.kick_scale
                    data.qvel[3] += sign * .25 * args.kick_scale
                    fired_disturbances.add(at)
            pushing = side_force != 0. and push_start <= data.time < push_start + push_duration
            data.xfrc_applied[base_id, :] = 0.
            if pushing:
                data.xfrc_applied[base_id, 1] = side_force
            mujoco.mj_step(model, data)
            if len(samples) == 0 or data.time - samples[-1]["t"] >= .02:
                samples.append(dict(t=round(data.time, 3), v=round(state.velocity[0], 3),
                                    vy=round(state.velocity[1], 3),
                                    vy_ref=0,
                                    target=round(speed, 3), yaw_rate=round(state.omega[2], 3),
                                    turn=round(turn, 3), roll=round(state.roll, 3),
                                    roll_rate=round(state.omega[0], 3),
                                    pitch=round(state.pitch, 3), x=round(state.position[0], 3),
                                    y=round(state.position[1], 3),
                                    second_pos=round(state.second_pos, 3),
                                    second_speed=round(state.second_speed, 3),
                                    drive_action=round(float(data.ctrl[0]), 3),
                                    second_action=round(float(data.ctrl[1]), 3)))
                samples[-1]["side_force_y"] = side_force if pushing else 0.
            if viewer_context is not None:
                viewer_context.sync()
                time.sleep(max(0, model.opt.timestep - (time.monotonic() - tick)))
            if renderer is not None and data.time >= next_frame_t:
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                if pushing:
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((18, 16, 370, 66), fill=(18, 35, 41))
                    draw.text((32, 27), f"WORLD +Y BODY FORCE: {side_force:g} N",
                              fill=(255, 232, 177), font=ImageFont.load_default(size=19))
                    frame = np.asarray(image)
                writer.append_data(frame)
                next_frame_t += 1 / 30
    finally:
        if viewer_context is not None:
            viewer_context.close()
            for thread in viewer_threads:
                thread.join(timeout=3.)
        if writer is not None:
            writer.close()
            renderer.close()
    if args.headless:
        speed_rmse = float(np.sqrt(np.mean(speed_errors_dense))) if speed_errors_dense else None
        yaw_rmse = float(np.sqrt(np.mean(yaw_errors_dense))) if yaw_errors_dense else None
        straight_yaw_rmse = (float(np.sqrt(np.mean(straight_yaw_errors_dense)))
                             if straight_yaw_errors_dense else None)
        recovery_yaw_rmse = (float(np.sqrt(np.mean(recovery_yaw_errors_dense)))
                             if recovery_yaw_errors_dense else None)
        lateral_rmse = float(np.sqrt(np.mean(lateral_errors_dense))) if lateral_errors_dense else None
        report = dict(kind=args.kind, duration=round(data.time, 3), fallen=fallen,
                      samples=len(samples), speed_rmse=float(np.sqrt(np.mean([
                          (s["v"] - s["target"]) ** 2 for s in samples]))) if samples else None,
                      steady_speed_rmse=speed_rmse, turn_yaw_rmse=yaw_rmse,
                      straight_yaw_rmse=straight_yaw_rmse,
                      recovery_yaw_rmse=recovery_yaw_rmse,
                      turn_lateral_rmse=lateral_rmse,
                      side_push_force=side_force, side_push_start=push_start,
                      side_push_duration=push_duration,
                      max_abs_roll=max((abs(s["roll"]) for s in samples), default=0),
                      pass_2m_turn=bool(not fallen and data.time >= args.duration - .01
                                        and speed_rmse is not None and speed_rmse < .35
                                        and straight_yaw_rmse is not None and straight_yaw_rmse < .15
                                        and yaw_rmse is not None and yaw_rmse < .15
                                        and recovery_yaw_rmse is not None and recovery_yaw_rmse < .15
                                        and lateral_rmse is not None and lateral_rmse < .35) if args.test else None,
                      final_xy=data.qpos[:2].tolist())
        print(json.dumps(report, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(dict(summary=report, trajectory=samples), indent=2))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["wheelbot", "steerbot"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--test", action="store_true", help="Fixed speed/turn acceptance sequence")
    parser.add_argument("--mirror", action="store_true", help="Reverse the turn sequence in --test")
    parser.add_argument("--test-turn-amplitude", type=float, default=.35)
    parser.add_argument("--test-cruise-speed", type=float, default=2.)
    parser.add_argument("--duration", type=float, default=15.)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--turn", type=float, default=0.)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--video", type=str, default=None, help="Record follow-camera MP4 (use MUJOCO_GL=egl headless)")
    parser.add_argument("--camera-distance", type=float, default=1.35,
                        help="Tracking camera distance for the recorded video")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--policy-yaw-scale", type=float, default=1.,
                        help="Scale yaw command entering PPO; metrics use the unscaled command")
    parser.add_argument("--model", type=str, default=None, help="Alternative MJCF for a candidate morphology")
    parser.add_argument("--com-offset", type=float, default=None,
                        help="Steerbot chassis COM offset during morphology experiments (m)")
    parser.add_argument("--trail", type=float, default=None,
                        help="Steerbot steering-axis trail in a morphology experiment (m)")
    parser.add_argument("--tilt", type=float, default=None,
                        help="Experimental Wheelbot flywheel-axis tilt (rad)")
    parser.add_argument("--head-mass", type=float, default=None,
                        help="Experimental steerbot steering-head mass (kg)")
    parser.add_argument("--body-mass", type=float, default=None)
    parser.add_argument("--body-offset-z", type=float, default=None)
    parser.add_argument("--tire-halfwidth", type=float, default=None,
                        help="Experimental wheelbot cylindrical tire half-width (m)")
    parser.add_argument("--reaction-gear", type=float, default=None,
                        help="Experimental wheelbot reaction motor torque at full control (N m)")
    parser.add_argument("--steer-gear", type=float, default=None,
                        help="Experimental steerbot steering motor torque at full control (N m)")
    parser.add_argument("--rotor-friction-scale", type=float, default=None,
                        help="Multiplier for narrow-wheel rotor friction compensation")
    parser.add_argument("--recovery-rotor-gain", type=float, default=None)
    parser.add_argument("--recovery-yaw-gain", type=float, default=None)
    parser.add_argument("--recovery-yaw-integral-gain", type=float, default=None)
    parser.add_argument("--pre-turn-yaw-gain", type=float, default=None)
    parser.add_argument("--pre-turn-rotor-gain", type=float, default=None)
    parser.add_argument("--speed-slew", type=float, default=None)
    parser.add_argument("--roll-ff-scale", type=float, default=None)
    parser.add_argument("--yaw-kp-scale", type=float, default=1.)
    parser.add_argument("--controller-params", type=str, default=None, help="Alternative optimized controller JSON")
    parser.add_argument("--lqr-tuning", type=str, default=None, help="Alternative LQR weight JSON")
    parser.add_argument("--controller", choices=("baseline", "optimized", "lqr", "tilted", "thin", "nested"), default="baseline")
    parser.add_argument("--rotor-speed", type=float, default=0., help="Initial Wheelbot flywheel angular speed (rad/s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dr", action="store_true", help="Vary mass and tire friction")
    parser.add_argument("--disturbance", action="store_true", help="Two lateral/roll velocity kicks")
    parser.add_argument("--kick-scale", type=float, default=1.,
                        help="Multiplier for the 0.25 m/s and 0.25 rad/s test impulses")
    parser.add_argument("--side-push-force", type=float, default=0.,
                        help="World +Y force applied to chassis (N); separate from velocity kicks")
    parser.add_argument("--side-push-start", type=float, default=8.5)
    parser.add_argument("--side-push-duration", type=float, default=.3)
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
