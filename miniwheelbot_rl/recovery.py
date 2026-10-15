"""Shared resets, recovery deadline and metrics for one continuous policy.

Only physical state gates rewards. Recovery bookkeeping is used for episode
termination/evaluation, never to select a controller or hide/change the command.
"""

import math
import torch
from .curriculum import slope_angles
from .spawn import CollisionSupport


class RecoveryState:
    def __init__(self, num_envs, device):
        for name in ("elapsed", "stable", "attempts", "successes", "time_sum", "fall_pulse",
                     "resumed_count", "resumed_forward_square", "resumed_yaw_square"):
            setattr(self, name, torch.zeros(num_envs, device=device))
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_step = -1

    def reset(self, ids, needs_recovery):
        for name in ("elapsed", "stable", "attempts", "successes", "time_sum", "fall_pulse",
                     "resumed_count", "resumed_forward_square", "resumed_yaw_square"):
            getattr(self, name)[ids] = 0
        self.active[ids] = needs_recovery
        self.attempts[ids] = needs_recovery.float()

    def update(self, fallen, stable, dt, lesson):
        new_fall = fallen & ~self.active
        self.fall_pulse[:] = new_fall.float()
        self.attempts += new_fall
        self.active |= new_fall
        self.elapsed += self.active * dt
        self.stable[:] = torch.where(stable & self.active, self.stable + dt, 0.0)
        success = self.active & (self.stable >= lesson.stable_seconds - 1e-5)
        self.successes += success
        self.time_sum += success * self.elapsed
        self.active &= ~success
        self.elapsed *= self.active
        return self.active & (self.elapsed >= lesson.recovery_seconds - 1e-5)


def state(env):
    if not hasattr(env, "_recovery"):
        env._recovery = RecoveryState(env.num_envs, env.device)
    return env._recovery


def surface_normal(env):
    if hasattr(env, "_surface_normal"):
        return env._surface_normal
    normal = torch.zeros((env.num_envs, 3), device=env.device)
    normal[:, 2] = 1
    return normal


def ground_height(env):
    """Signed distance from root to its local ramp plane, not global root z."""
    relative = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    return (relative * surface_normal(env)).sum(dim=-1)


def tangent_velocity(env):
    from mjlab.utils.lab_api.math import quat_apply
    robot = env.scene["robot"]
    axis = torch.zeros((env.num_envs, 3), device=env.device)
    axis[:, 0] = 1
    forward = quat_apply(robot.data.root_link_quat_w, axis)
    normal = surface_normal(env)
    forward -= (forward * normal).sum(dim=-1, keepdim=True) * normal
    forward /= forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    lateral = torch.cross(normal, forward, dim=-1)
    velocity = robot.data.root_link_lin_vel_w
    return (velocity * forward).sum(dim=-1), (velocity * lateral).sum(dim=-1)


def upright_gate(env):
    up = -env.scene["robot"].data.projected_gravity_b[:, 2]
    # Continuous physical-state gate: off beyond 60 deg, fully on below 20 deg.
    return ((up - 0.5) / (math.cos(math.radians(20)) - 0.5)).clamp(0, 1)


def reset_robot(env, env_ids, lesson, settings, model_path, start_pose="random"):
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_apply_inverse
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count = len(env_ids)
    robot = env.scene["robot"]
    if not hasattr(env, "_collision_support"):
        env._collision_support = CollisionSupport(settings, model_path, env.device)
        env._surface_normal = torch.zeros((env.num_envs, 3), device=env.device)
        env._surface_normal[:, 2] = 1
    angles = slope_angles(lesson)
    if lesson.slope_degrees:
        terrain = env.scene.terrain
        # Re-sample columns for training coverage, including when num_envs=1.
        weights = torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15], device=env.device)
        columns = torch.multinomial(weights, count, replacement=True)
        if hasattr(env, "_evaluation_column"):
            columns[:] = env._evaluation_column
        terrain.terrain_types[env_ids] = columns
        terrain.env_origins[env_ids] = terrain.terrain_origins[0, columns]
        angle = torch.tensor(angles, device=env.device)[columns] * (math.pi / 180)
        env._surface_normal[env_ids, 0] = -angle.sin()
        env._surface_normal[env_ids, 2] = angle.cos()
    roll = (torch.rand(count, device=env.device) * 2 - 1) * 0.025
    pitch = torch.zeros_like(roll)
    yaw = (torch.rand_like(roll) * 2 - 1) * math.pi
    selected = torch.rand_like(roll) < lesson.fallen_fraction
    magnitude = torch.empty_like(roll).uniform_(*lesson.roll_degrees) * (math.pi / 180)
    sign = torch.where(torch.rand_like(roll) < 0.5, -1.0, 1.0)
    roll = torch.where(selected, magnitude * sign, roll)
    pitch_case = selected & (torch.rand_like(roll) < lesson.pitch_fraction)
    pitch = torch.where(pitch_case, magnitude * sign, pitch)
    roll = torch.where(pitch_case, roll * 0.2, roll)
    start_pose = getattr(env, "_evaluation_pose", start_pose)
    if start_pose != "random":
        poses = {"upright": (0, 0), "left": (90, 0), "right": (-90, 0),
                 "front": (0, 90), "back": (0, -90), "tilt_left": (45, 0),
                 "tilt_right": (-45, 0)}
        r, p = poses[start_pose]
        roll[:] = math.radians(r)
        pitch[:] = math.radians(p)
        selected[:] = start_pose != "upright"
        yaw[:] = getattr(env, "_evaluation_yaw", 0.0)
        if getattr(env, "_evaluation_jitter", False):
            # Distinct, seeded trials instead of N identical deterministic copies.
            roll += (torch.rand_like(roll) * 2 - 1) * math.radians(2)
            pitch += (torch.rand_like(pitch) * 2 - 1) * math.radians(2)
            yaw += (torch.rand_like(yaw) * 2 - 1) * math.radians(5)
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    normal = env._surface_normal[env_ids]
    minimum = env._collision_support.minimum(quat_apply_inverse(quat, normal))
    root = torch.zeros((count, 13), device=env.device)
    root[:, :3] = env.scene.env_origins[env_ids]
    # Translate vertically so every collision geom clears the surface by 2 mm.
    root[:, 2] += (0.002 - minimum) / normal[:, 2]
    root[:, 3:7] = quat
    joints = torch.zeros((count, len(robot.joint_names)), device=env.device)
    robot.write_joint_state_to_sim(joints, joints, env_ids=env_ids)
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    state(env).reset(env_ids, selected)


def recovery_timeout(env, lesson):
    monitor = state(env)
    if monitor.last_step != env.common_step_counter:
        data = env.scene["robot"].data
        up = -data.projected_gravity_b[:, 2]
        height = ground_height(env)
        fallen = (up < math.cos(0.7)) | (height < 0.185)
        stable = (up > math.cos(math.radians(20))) & (height > 0.30)
        stable &= data.root_link_ang_vel_b[:, :2].norm(dim=-1) < 1.5
        monitor.update(fallen, stable, env.step_dt, lesson)
        resumed = (monitor.successes > 0) & ~monitor.active & stable
        monitor.resumed_count += resumed
        for channel in ("forward", "yaw"):
            getattr(monitor, f"resumed_{channel}_square").add_(
                torch.where(resumed, tracking_error(env, channel).square(), 0.0))
        monitor.last_step = env.common_step_counter
    return monitor.active & (monitor.elapsed >= lesson.recovery_seconds - 1e-5)


def out_of_patch(env):
    from .terrain import SAFE_RADIUS
    xy = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return xy.abs().amax(dim=-1) > SAFE_RADIUS


def invalid_state(env):
    data = env.scene["robot"].data
    return (~torch.isfinite(data.root_link_pos_w).all(dim=-1)
            | ~torch.isfinite(data.joint_vel).all(dim=-1) | (ground_height(env) < -0.10))


def posture_reward(env):
    # Nonzero gradient around lying poses, including slightly past 90 degrees.
    return (1 - env.scene["robot"].data.projected_gravity_b[:, 2]) * 0.5


def fall_cost(env):
    # Reward manager multiplies by dt: charge a fixed cost per new fall.
    return state(env).fall_pulse / env.step_dt


def recovery_metric(env, name):
    monitor = state(env)
    if name == "success_fraction":
        return monitor.successes / monitor.attempts.clamp_min(1)
    if name == "mean_recovery_seconds":
        return monitor.time_sum / monitor.successes.clamp_min(1)
    if name == "slope_degrees":
        return torch.acos(surface_normal(env)[:, 2].clamp(-1, 1)) * (180 / math.pi)
    if name.startswith("resumed_") and name.endswith("_rmse"):
        square = getattr(monitor, name.replace("_rmse", "_square"))
        return (square / monitor.resumed_count.clamp_min(1)).sqrt()
    return getattr(monitor, name).float()


def tracking_error(env, channel):
    from .task import heading_rate, command
    actual = tangent_velocity(env)[0] if channel == "forward" else heading_rate(env)
    target = command(env)[:, 0 if channel == "forward" else 1]
    # True time-mean absolute error, not command-manager lifetime accumulation.
    return (actual - target).abs()


def snapshot(env):
    """Single-environment replay summary, called before any terminal reset."""
    monitor = state(env)
    return {"attempts": int(monitor.attempts[0]), "successes": int(monitor.successes[0]),
            "recovering_at_end": bool(monitor.active[0]),
            "mean_recovery_seconds": (float(monitor.time_sum[0] / monitor.successes[0])
                                      if monitor.successes[0] else None),
            "ground_relative_height_m": float(ground_height(env)[0])}
