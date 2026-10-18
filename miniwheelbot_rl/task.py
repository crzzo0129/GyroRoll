"""A two-action MDP: signed wheel-speed targets, without a balance controller."""

from copy import deepcopy
from functools import partial

import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.viewer import ViewerConfig

from .model import make_spec, MODEL
from .settings import Settings, STAGES
from .policy_config import POLICY_VERSION, distribution_config
from .curriculum import LESSONS


def command(env):
    return env.command_manager.get_command("twist")[:, (0, 2)]


def wheel_speeds(env, settings: Settings):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(("reaction", "drive"), preserve_order=True)
    scales = torch.tensor([settings.flywheel_speed_limit, settings.rolling_speed_limit],
                          device=env.device)
    return robot.data.joint_vel[:, ids] / scales


def heading_rate(env):
    """Derivative of horizontal body-forward heading, including tilt effects."""
    robot = env.scene["robot"]
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    forward = quat_apply(robot.data.root_link_quat_w, forward)
    forward_dot = torch.cross(robot.data.root_link_ang_vel_w, forward, dim=-1)
    return ((forward[:, 0] * forward_dot[:, 1] - forward[:, 1] * forward_dot[:, 0])
            / forward[:, :2].square().sum(dim=-1).clamp_min(1e-6))


def tracking(env, channel: str, sigma: float):
    robot = env.scene["robot"]
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0, 1).square()
    cmd = command(env)
    actual, target = ((robot.data.root_link_lin_vel_b[:, 0], cmd[:, 0])
                      if channel == "forward" else (heading_rate(env), cmd[:, 1]))
    if "recovery_timeout" in env.cfg.terminations:
        from .recovery import upright_gate, tangent_velocity
        upright = upright_gate(env)
        if channel == "forward":
            actual = tangent_velocity(env)[0]
    return upright * torch.exp(-((actual - target) / sigma).square())


def upright_reward(env):
    return (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0, 1).square()


def lateral_penalty(env):
    if "recovery_timeout" in env.cfg.terminations:
        from .recovery import upright_gate, tangent_velocity
        return upright_gate(env) * tangent_velocity(env)[1].square()
    return env.scene["robot"].data.root_link_lin_vel_b[:, 1].square()


def roll_penalty(env):
    return recovery_regularization(env) * env.scene["robot"].data.root_link_ang_vel_b[:, 0].square()


def roll_angle_penalty(env):
    """Squared ZYX roll in radians, independent of pitch for upright poses."""
    gravity = env.scene["robot"].data.projected_gravity_b
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    return roll.square()


def pitch_penalty(env):
    return recovery_regularization(env) * env.scene["robot"].data.root_link_ang_vel_b[:, 1].square()


def recovery_regularization(env):
    if "recovery_timeout" in env.cfg.terminations:
        from .recovery import upright_gate
        return 0.2 + 0.8 * upright_gate(env)
    return 1.0


def action_rate_penalty(env):
    return recovery_regularization(env) * (env.action_manager.action.clamp(-1, 1)
            - env.action_manager.prev_action.clamp(-1, 1)).square().sum(dim=1)


def torque_penalty(env):
    return recovery_regularization(env) * env.scene["robot"].data.actuator_force.square().sum(dim=1)


def overspeed_penalty(env, settings: Settings):
    return (wheel_speeds(env, settings).abs() - 1).clamp_min(0).square().sum(dim=1)


def failure_penalty(env):
    return env.termination_manager.terminated.float()


def make_env_cfg(num_envs: int = 256, stage: str = "warmup", *,
                 settings: Settings = Settings(), play: bool = False,
                 randomize: bool = False, model_path=MODEL,
                 start_pose: str = "random") -> ManagerBasedRlEnvCfg:
    max_speed, max_yaw = STAGES[stage]
    robot = EntityCfg(
        spec_fn=partial(make_spec, settings, model_path),
        articulation=EntityArticulationInfoCfg(actuators=(
            XmlActuatorCfg(target_names_expr=("reaction", "drive"), command_field="velocity"),
        )),
        init_state=EntityCfg.InitialStateCfg(pos=(0, 0, settings.initial_height),
                                            joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    )
    terms = {
        "linear_velocity": ObservationTermCfg(func=mdp.base_lin_vel),
        "angular_velocity": ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25),
        "gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "wheel_velocity": ObservationTermCfg(func=wheel_speeds, params={"settings": settings}),
        "command": ObservationTermCfg(func=command),
        "previous_action": ObservationTermCfg(func=mdp.last_action),
    }
    events = {
        "reset_root": EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={
            "pose_range": {"roll": (-0.025, 0.025), "pitch": (-0.025, 0.025),
                           "yaw": (-3.14159, 3.14159)},
            "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot")}),
        "reset_wheels": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", params={
            "position_range": (-3.14159, 3.14159), "velocity_range": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot")}),
    }
    if randomize and not play:
        events["friction"] = EventTermCfg(func=dr.geom_friction, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=("tire", "tire_.*", "rim_.*")),
            "operation": "abs", "ranges": (1.1, 1.9), "shared_random": True})
        events["push"] = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=(5.0, 8.0), params={"asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1),
                                   "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15)}})
    # Two separate terms enforce flywheel-first action order regardless of XML
    # joint declaration order. Targets are absolute signed speeds, with no bias.
    actions = {}
    for key, joint, limit in (("flywheel", "reaction", settings.flywheel_speed_limit),
                              ("rolling", "drive", settings.rolling_speed_limit)):
        actions[key] = JointVelocityActionCfg(entity_name="robot",
            # MJLab 1.3.0 resolves this field against actuated joint names,
            # although its public name is actuator_names.
            actuator_names=(joint,), scale=limit, offset=0.0,
            use_default_offset=False, clip={".*": (-limit, limit)})
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=num_envs, env_spacing=4.0, entities={"robot": robot},
                       terrain=TerrainEntityCfg(terrain_type="plane")),
        observations={"actor": ObservationGroupCfg(terms=terms, enable_corruption=False),
                      "critic": ObservationGroupCfg(terms=deepcopy(terms), enable_corruption=False)},
        actions=actions,
        commands={"twist": UniformVelocityCommandCfg(entity_name="robot",
            resampling_time_range=(3.0, 6.0), heading_command=False,
            rel_standing_envs=0.1, rel_forward_envs=0.0, rel_world_envs=0.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-max_speed, max_speed), lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-max_yaw, max_yaw)))},
        events=events,
        rewards={
            "forward": RewardTermCfg(func=tracking, weight=4.0,
                                     params={"channel": "forward", "sigma": 0.4}),
            "yaw": RewardTermCfg(func=tracking, weight=2.0,
                                 params={"channel": "yaw", "sigma": 0.25}),
            "upright": RewardTermCfg(func=upright_reward, weight=0.5),
            "lateral": RewardTermCfg(func=lateral_penalty, weight=-0.5),
            "roll": RewardTermCfg(func=roll_penalty, weight=-0.05),
            "pitch": RewardTermCfg(func=pitch_penalty, weight=-0.01),
            "action_rate": RewardTermCfg(func=action_rate_penalty, weight=-0.02),
            "torque": RewardTermCfg(func=torque_penalty, weight=-0.005),
            "overspeed": RewardTermCfg(func=overspeed_penalty, weight=-1.0,
                                       params={"settings": settings}),
            "failure": RewardTermCfg(func=failure_penalty, weight=-10.0),
        },
        terminations={
            "fall": TerminationTermCfg(func=mdp.bad_orientation,
                                       params={"limit_angle": settings.maximum_tilt}),
            "height": TerminationTermCfg(func=mdp.root_height_below_minimum,
                                         params={"minimum_height": settings.minimum_height}),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="robot", body_name="base", distance=1.5),
        sim=SimulationCfg(nconmax=96, njmax=384,
                         mujoco=MujocoCfg(timestep=settings.timestep, integrator="implicitfast", iterations=30,
                                          ls_iterations=10, cone="elliptic")),
        decimation=settings.decimation, episode_length_s=settings.episode_seconds,
    )
    if stage not in LESSONS:
        # Penalize sustained sideways lean separately from angular motion.
        cfg.rewards["roll_angle"] = RewardTermCfg(func=roll_angle_penalty, weight=-2.0)
        # Smooth normalized wheel-speed commands while retaining balance corrections.
        cfg.rewards["action_rate"].weight = -0.10
    if stage == "full_robust":
        from mjlab.managers.metrics_manager import MetricsTermCfg
        from .robustness import ROBUST_TRACKING, HorizontalPushes, standing_reward, standing_fraction
        robust = ROBUST_TRACKING
        cfg.commands["twist"].rel_standing_envs = robust.standing_fraction
        cfg.rewards["standing"] = RewardTermCfg(func=standing_reward, weight=1.0)
        cfg.metrics["standing/command_fraction"] = MetricsTermCfg(func=standing_fraction)
        # Use finite force pulses instead of stacking velocity jumps from --randomize.
        cfg.events.pop("push", None)
        if not play:
            cfg.events["horizontal_push"] = EventTermCfg(func=HorizontalPushes, mode="step",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                        "force_newtons": robust.force_newtons,
                        "duration_seconds": robust.duration_seconds,
                        "cooldown_seconds": robust.cooldown_seconds})
    if stage in LESSONS:
        from mjlab.managers.metrics_manager import MetricsTermCfg
        from . import recovery
        from .terrain import terrain_cfg
        lesson = LESSONS[stage]
        cfg.scene.terrain = terrain_cfg(lesson)
        # This reset owns both root pose and wheel angles so support geometry agrees.
        cfg.events.pop("reset_wheels")
        cfg.events["reset_root"] = EventTermCfg(func=recovery.reset_robot, mode="reset",
            params={"lesson": lesson, "settings": settings, "model_path": model_path,
                    "start_pose": start_pose})
        if lesson.push and not play:
            cfg.events["push"] = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval",
                interval_range_s=(7.0, 10.0), params={"asset_cfg": SceneEntityCfg("robot"),
                    "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                        "roll": (-lesson.push, lesson.push), "pitch": (-lesson.push, lesson.push)}})
        cfg.terminations = {
            "recovery_timeout": TerminationTermCfg(func=recovery.recovery_timeout,
                                                     params={"lesson": lesson}),
            "invalid_state": TerminationTermCfg(func=recovery.invalid_state),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        }
        if lesson.slope_degrees:
            cfg.terminations["patch_boundary"] = TerminationTermCfg(func=recovery.out_of_patch)
        cfg.rewards["upright"] = RewardTermCfg(func=recovery.posture_reward, weight=4.0)
        cfg.rewards["new_fall"] = RewardTermCfg(func=recovery.fall_cost, weight=-2.0)
        # No repeatable get-up bonus: stable tracking is the sustained incentive.
        for name in ("attempts", "successes", "success_fraction", "mean_recovery_seconds",
                     "resumed_count", "resumed_forward_rmse", "resumed_yaw_rmse"):
            cfg.metrics[f"recovery/{name}"] = MetricsTermCfg(func=recovery.recovery_metric,
                params={"name": name}, reduce="last")
        for name in ("active", "slope_degrees"):
            cfg.metrics[f"recovery/{name}"] = MetricsTermCfg(func=recovery.recovery_metric,
                params={"name": name})
        for channel in ("forward", "yaw"):
            cfg.metrics[f"tracking/{channel}_mae"] = MetricsTermCfg(func=recovery.tracking_error,
                params={"channel": channel})
    elif start_pose != "random":
        raise ValueError("--start-pose requires a recovery curriculum stage")
    return cfg


def make_agent_cfg(iterations: int = 3000, *, policy_version: str = POLICY_VERSION):
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128, 128, 128), obs_normalization=True,
            distribution_cfg=distribution_config(policy_version)),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, num_learning_epochs=4,
            num_mini_batches=4, entropy_coef=0.0, gamma=0.99, lam=0.95, desired_kl=0.015),
        num_steps_per_env=32, max_iterations=iterations, clip_actions=1.0,
        experiment_name="nested_velocity_mjlab130", save_interval=50,
        logger="tensorboard", upload_model=False,
    )
