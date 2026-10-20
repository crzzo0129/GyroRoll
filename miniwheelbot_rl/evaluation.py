"""Deterministic recovery and ramp acceptance, preserving terminal trajectories."""

import json
import math
import torch
from .curriculum import LESSONS, slope_angles
from .recovery import state, tangent_velocity, upright_gate, ground_height


def evaluate_robust_tracking(args, env, wrapped, runner, *, heading_fn=None):
    """Flat tracking, stop/start transitions and reproducible 1.5 N force pulses."""
    if args.duration < 15:
        raise ValueError("Robust tracking evaluation requires --duration >= 15")
    if args.stage in LESSONS:
        raise ValueError("Robust tracking evaluates upright tracking stages, not recovery stages")
    if heading_fn is None:
        from .task import heading_rate
        heading_fn = heading_rate
    policy = runner.get_inference_policy(device=env.device)
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(("base",), preserve_order=True)
    cases = [("stop", 0., (0., 0.)), ("stop_go", 1., (0., 0.)),
             ("move_push_forward", 1., (1.5, 0.)),
             ("move_push_backward", 1., (-1.5, 0.)),
             ("move_push_left", 1., (0., 1.5)),
             ("move_push_right", 1., (0., -1.5)),
             ("stop_push_left", 0., (0., 1.5)),
             ("stop_push_right", 0., (0., -1.5))]
    report = {"checkpoint": str(args.checkpoint), "stage": args.stage,
              "suite": "robust", "seed": args.seed, "duration_s": args.duration,
              "action_smoothing": args.action_smoothing,
              "push_window_s": [5., 5.2], "force_frame": "world, at base body CoM",
              "stop_go_segments": [[0, 1.], [4, 0.], [8, -1.], [12, 0.]],
              "stop_planar_rmse_limit_m_s": .10, "cases": []}
    old_auto = env.cfg.auto_reset
    env.cfg.auto_reset = False
    zeros = torch.zeros((env.num_envs, 1, 3), device=env.device)
    try:
        for name, speed, push in cases:
            env.reset()
            previous_action = None
            alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            sums = torch.zeros((env.num_envs, 3), device=env.device)
            stop_sums = torch.zeros((env.num_envs, 2), device=env.device)
            counts = torch.zeros(env.num_envs, device=env.device)
            stop_counts = torch.zeros_like(counts)
            for step in range(round(args.duration / env.step_dt)):
                t = step * env.step_dt
                target = speed
                if name == "stop_go":
                    target = 1. if t < 4 else 0. if t < 8 else -1. if t < 12 else 0.
                env.command_manager.get_command("twist")[:] = torch.tensor(
                    [target, 0., 0.], device=env.device)
                forces = zeros.clone()
                if 5. <= t < 5.2:
                    forces[alive, 0, :2] = torch.tensor(push, device=env.device)
                robot.write_external_wrench_to_sim(forces, zeros, body_ids=body_ids)
                with torch.inference_mode():
                    actions = policy(wrapped.get_observations())
                    if previous_action is not None:
                        actions = (args.action_smoothing * actions +
                                   (1 - args.action_smoothing) * previous_action)
                    previous_action = actions
                    _, _, done, _ = wrapped.step(actions)
                alive &= ~done.bool()
                transition = name == "stop_go" and any(s <= t < s + 1 for s in (4, 8, 12))
                if t >= 3 and not transition:
                    vx, vy = robot.data.root_link_lin_vel_b[:, :2].unbind(dim=-1)
                    yaw = heading_fn(env)
                    errors = torch.stack((vx - target, yaw, vy), dim=-1)
                    sums += torch.where(alive[:, None], errors.square(), 0.)
                    counts += alive
                    if target == 0:
                        stop_sums += torch.where(alive[:, None],
                            torch.stack((vx.square() + vy.square(), yaw.square()), dim=-1), 0.)
                        stop_counts += alive
                failed_ids = done.nonzero(as_tuple=False).flatten()
                if len(failed_ids):
                    # Continue the batch, but never restore a failed trial's alive flag.
                    env.reset(env_ids=failed_ids)
            rmse = (sums / counts.clamp_min(1)[:, None]).sqrt()
            stop_rmse = (stop_sums / stop_counts.clamp_min(1)[:, None]).sqrt()
            needs_stop = speed == 0 or name == "stop_go"
            passed = alive & (counts * env.step_dt >= 1.)
            passed &= (rmse[:, 0] < .35) & (rmse[:, 1] < .15) & (rmse[:, 2] < .35)
            if needs_stop:
                passed &= (stop_counts * env.step_dt >= 1.) & (stop_rmse[:, 0] < .10) & (stop_rmse[:, 1] < .15)
            report["cases"].append({"name": name, "push_xy_N": list(push),
                "pass_rate": float(passed.float().mean()),
                "individual": [{"survived": bool(alive[i]), "passed": bool(passed[i]),
                    "rmse_forward": float(rmse[i, 0]) if counts[i] else None,
                    "rmse_heading_rate": float(rmse[i, 1]) if counts[i] else None,
                    "rmse_lateral": float(rmse[i, 2]) if counts[i] else None,
                    "stop_planar_rmse": float(stop_rmse[i, 0]) if stop_counts[i] else None,
                    "stop_heading_rmse": float(stop_rmse[i, 1]) if stop_counts[i] else None,
                    "stop_samples": int(stop_counts[i])} for i in range(env.num_envs)]})
            print(f"{name}: pass={float(passed.float().mean()):.2f}")
    finally:
        robot.write_external_wrench_to_sim(zeros, zeros, body_ids=body_ids)
        env.cfg.auto_reset = old_auto
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Robust tracking evaluation saved: {args.output}")
    return report


def evaluate_curriculum(args, env, wrapped, runner, *, heading_fn=None):
    if heading_fn is None:
        from .task import heading_rate
        heading_fn = heading_rate
    lesson = LESSONS[args.stage]
    if args.duration < 10:
        raise ValueError("Curriculum evaluation requires --duration >= 10 for recovery + tracking")
    policy = runner.get_inference_policy(device=env.device)
    # name, initial pose, terrain column, yaw, forward command, turn, push roll
    cases = [("flat_stop", "upright", 0, 0, 0, 0, 0),
             ("flat_forward", "upright", 0, 0, 0.5, 0, 0),
             ("flat_reverse", "upright", 0, 0, -0.5, 0, 0),
             ("flat_turn_left", "upright", 0, 0, 0.4, 0.2, 0),
             ("flat_turn_right", "upright", 0, 0, 0.4, -0.2, 0),
             ("tilt_left", "tilt_left", 0, 0, 0.3, 0, 0),
             ("tilt_right", "tilt_right", 0, 0, 0.3, 0, 0)]
    if args.stage != "rise_tilt":
        cases += [("rise_left", "left", 0, 0, 0.3, 0, 0),
                  ("rise_right", "right", 0, 0, 0.3, 0, 0)]
    if lesson.pitch_fraction:
        cases += [("rise_front", "front", 0, 0, 0.3, 0, 0),
                  ("rise_back", "back", 0, 0, 0.3, 0, 0)]
    if lesson.push:
        cases += [("push_left", "upright", 0, 0, 0.3, 0, 2.0),
                  ("push_right", "upright", 0, 0, 0.3, 0, -2.0)]
    if lesson.slope_degrees:
        for column, angle in enumerate(slope_angles(lesson)):
            if column == 0:
                continue
            for direction, yaw in (("up", 0), ("down", math.pi),
                                   ("cross_left", math.pi / 2), ("cross_right", -math.pi / 2)):
                # Positive patch rises in +x; negative patches swap up/down.
                adjusted_yaw = yaw + (math.pi if angle < 0 else 0)
                cases.append((f"slope_{angle:g}_{direction}", "upright", column,
                              adjusted_yaw, 0.5, 0, 0))
        cases += [("slope_rise_left", "left", 3, 0, 0.3, 0, 0),
                  ("slope_rise_right", "right", 3, 0, 0.3, 0, 0)]
    report = {"checkpoint": str(args.checkpoint), "stage": args.stage,
              "seed": args.seed, "duration_s": args.duration, "suite": "curriculum",
              "action_smoothing": args.action_smoothing,
              "speed_definition": "surface-tangent forward; yaw = horizontal heading rate",
              "recovery_deadline_s": lesson.recovery_seconds,
              "stable_duration_s": lesson.stable_seconds,
              "pose_jitter_deg": {"roll": 2, "pitch": 2, "yaw": 5}, "cases": []}
    old_auto = env.cfg.auto_reset
    env.cfg.auto_reset = False
    env._evaluation_jitter = True
    try:
        for name, pose, column, yaw, speed, turn, push in cases:
            env._evaluation_pose, env._evaluation_column, env._evaluation_yaw = pose, column, yaw
            env.reset()
            previous_action = None
            alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            ever_fell = torch.zeros_like(alive)
            recovered = torch.zeros_like(alive)
            first_time = torch.full((env.num_envs,), float("nan"), device=env.device)
            sums = torch.zeros((env.num_envs, 3), device=env.device)
            counts = torch.zeros(env.num_envs, device=env.device)
            boundary = torch.zeros((env.num_envs, 2), device=env.device)
            saturated = torch.zeros_like(boundary)
            steps = torch.zeros(env.num_envs, device=env.device)
            for step in range(round(args.duration / env.step_dt)):
                t = step * env.step_dt
                env.command_manager.get_command("twist")[:] = torch.tensor([speed, 0, turn], device=env.device)
                if push and step == round(3 / env.step_dt):
                    robot = env.scene["robot"]
                    velocity = robot.data.root_link_vel_w.clone()
                    velocity[:, 3] += push
                    robot.write_root_link_velocity_to_sim(velocity)
                    env.sim.forward()
                with torch.inference_mode():
                    actions = policy(wrapped.get_observations())
                    if previous_action is not None:
                        actions = (args.action_smoothing * actions +
                                   (1 - args.action_smoothing) * previous_action)
                    previous_action = actions
                    _, _, done, _ = wrapped.step(actions)
                    monitor = state(env)
                    ever_fell |= (monitor.attempts > 0) & alive
                    newly = (monitor.successes > 0) & alive & ~recovered
                    first_time[newly] = (step + 1) * env.step_dt
                    recovered |= newly
                    boundary += (actions.abs() >= 0.99) * alive[:, None]
                    robot = env.scene["robot"]
                    saturated += (robot.data.actuator_force.abs() >=
                        0.99 * env.cfg.events["reset_root"].params["settings"].torque_limit) * alive[:, None]
                    steps += alive
                    # Measure every sample in the last window, including new falls.
                    # A late recovery cannot pass using a handful of favorable samples.
                    window_start = 8.0 if push else 6.0
                    valid = alive & ~done.bool() & (t >= window_start)
                    vx, vy = tangent_velocity(env)
                    errors = torch.stack((vx - speed, heading_fn(env) - turn, vy), dim=-1)
                    sums += torch.where(valid[:, None], errors.square(), 0.0)
                    counts += valid
                    alive &= ~done.bool()
                    ids = done.nonzero(as_tuple=False).flatten()
                    if len(ids):
                        # Snapshot above precedes reset; dead trials never re-enter statistics.
                        env.reset(env_ids=ids)
                if not alive.any():
                    break
            rmse = (sums / counts.clamp_min(1)[:, None]).sqrt()
            stable = (upright_gate(env) > 0.99) & (ground_height(env) > 0.30) & ~state(env).active
            stable &= env.scene["robot"].data.root_link_ang_vel_b[:, :2].norm(dim=-1) < 1.5
            required = pose != "upright"
            passed = alive & stable & (counts * env.step_dt >= 1.5)
            passed &= (rmse[:, 0] < 0.35) & (rmse[:, 1] < 0.15) & (rmse[:, 2] < 0.35)
            passed &= (~ever_fell | recovered) & (recovered if required else torch.ones_like(alive))
            results = []
            for i in range(env.num_envs):
                results.append({"survived": bool(alive[i]), "needed_recovery": bool(ever_fell[i]),
                    "recovered": bool(recovered[i]), "stable_at_end": bool(stable[i] & alive[i]),
                    "first_recovery_time_s": float(first_time[i]) if recovered[i] else None,
                    "tracking_rmse": rmse[i].tolist() if counts[i] else None,
                    "tracking_samples": int(counts[i]),
                    "action_boundary_fraction": (boundary[i] / steps[i].clamp_min(1)).tolist(),
                    "torque_saturation_fraction": (saturated[i] / steps[i].clamp_min(1)).tolist(),
                    "passed": bool(passed[i])})
            report["cases"].append({"name": name, "pose": pose,
                "slope_degrees": slope_angles(lesson)[column], "heading_rad": yaw,
                "command": [speed, turn], "pass_rate": passed.float().mean().item(),
                "survival_rate": alive.float().mean().item(), "individual": results})
            print(f"{name}: pass={passed.float().mean().item():.2f}", flush=True)
    finally:
        env.cfg.auto_reset = old_auto
        env._evaluation_jitter = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Curriculum evaluation saved: {args.output.resolve()}")
    return report
