"""Train either two-actuator unicycle with the installed mjlab + RSL-RL.

python train.py steerbot --stage 1 --iterations 300 --num-envs 512
python train.py steerbot --stage 2 --iterations 500 --resume '.*'
python train.py steerbot --stage 3 --iterations 1000 --resume '.*'
"""

import argparse
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig


ROOT = Path(__file__).resolve().parent
ROBOT = SceneEntityCfg("robot")


def command(env):
    return env.command_manager.get_command("twist")


def tracking_reward(env):
    robot: Entity = env.scene["robot"]
    vel = robot.data.root_link_lin_vel_b
    ang = robot.data.root_link_ang_vel_b
    gravity = robot.data.projected_gravity_b
    cmd = command(env)
    # Positive reward only when upright; velocity alone must not reward falling.
    upright = torch.clamp(-gravity[:, 2], min=0.0) ** 4
    linear = torch.exp(-((vel[:, 0] - cmd[:, 0]) / 0.55) ** 2)
    yaw = torch.exp(-((ang[:, 2] - cmd[:, 2]) / 0.40) ** 2)
    lateral = torch.exp(-((vel[:, 1]) / 0.5) ** 2)
    return upright * (1.0 + 4.0 * linear + 1.2 * yaw + 0.2 * lateral)


def action_penalty(env):
    return torch.sum(env.action_manager.action ** 2, dim=1)


def roll_penalty(env):
    robot: Entity = env.scene["robot"]
    return robot.data.projected_gravity_b[:, 1] ** 2


def yaw_reward(env):
    robot: Entity = env.scene["robot"]
    actual = robot.data.root_link_ang_vel_b[:, 2]
    target = command(env)[:, 2]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], min=0.0) ** 4
    return upright * torch.exp(-((actual - target) / .25) ** 2)


def rotor_speed_reward(env):
    robot: Entity = env.scene["robot"]
    return torch.exp(-((robot.data.joint_vel[:, 1] - 110.) / 55.) ** 2)


def lateral_reward(env):
    robot: Entity = env.scene["robot"]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], min=0.) ** 4
    return upright * torch.exp(-(robot.data.root_link_lin_vel_b[:, 1] / .2) ** 2)


def motion_error_penalty(env):
    robot: Entity = env.scene["robot"]
    desired = command(env)
    linear = robot.data.root_link_lin_vel_b
    angular = robot.data.root_link_ang_vel_b
    return (2. * (linear[:, 0] - desired[:, 0]) ** 2
            + 5. * (angular[:, 2] - desired[:, 2]) ** 2
            + 2. * linear[:, 1] ** 2)


def steer_angle(env):
    robot: Entity = env.scene["robot"]
    return robot.data.joint_pos[:, :1]


def make_config(kind: str, stage: int, num_envs: int, trail: float | None = None,
                tilt: float | None = None,
                head_mass: float | None = None,
                tire_halfwidth: float | None = None,
                steer_gear: float | None = None) -> ManagerBasedRlEnvCfg:
    second = "reaction" if kind == "wheelbot" else "steer"
    path = ROOT / kind / "model.xml"

    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(path))
        if trail is not None:
            if kind != "steerbot":
                raise ValueError("Trail overrides only apply to steerbot")
            spec.body("fork").pos[0] = trail
            spec.body("drive_wheel").pos[0] = -trail
        if tilt is not None:
            if kind != "wheelbot":
                raise ValueError("Flywheel tilt overrides only apply to wheelbot")
            spec.body("reaction_wheel").quat[:] = (np.cos(tilt / 2), 0.,
                                                    -np.sin(tilt / 2), 0.)
        if head_mass is not None:
            if kind != "steerbot" or head_mass <= 0:
                raise ValueError("Positive steering-head mass only applies to steerbot")
            spec.geom("steering_head").mass = head_mass
        if tire_halfwidth is not None:
            if tire_halfwidth <= 0:
                raise ValueError("Tire half-width must be positive")
            spec.geom("tire").size[1] = tire_halfwidth
        if steer_gear is not None:
            if kind != "steerbot" or steer_gear <= 0:
                raise ValueError("Positive steering torque only applies to steerbot")
            spec.actuator("steer_motor").gear[0] = steer_gear
        return spec

    robot = EntityCfg(
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlMotorActuatorCfg(target_names_expr=("drive", second)),)
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, .28),
            lin_vel=(.8, 0, 0) if kind == "steerbot" else (0, 0, 0),
            joint_pos={".*": 0.0},
            joint_vel={"drive": 6.67 if kind == "steerbot" else 0.0,
                       second: 110.0 if kind == "wheelbot" and stage >= 3 else 0.0},
        ),
    )
    terms = {
        "lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "gravity": ObservationTermCfg(func=mdp.projected_gravity),
    }
    if kind == "steerbot":
        terms["steer_angle"] = ObservationTermCfg(func=steer_angle)
    terms.update({
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "command": ObservationTermCfg(func=command),
        "action": ObservationTermCfg(func=mdp.last_action),
    })
    speed_ranges = {1: (.6, 1.3), 2: (1.3, 2.2), 3: (1.2, 2.3),
                    4: (1.8, 2.15), 5: (1.8, 2.2), 6: (1.8, 2.2),
                    7: (1.8, 2.2), 8: (1.7, 2.2), 9: (1.7, 2.2)}
    yaw_ranges = {1: (0.0, 0.0), 2: (-.2, .2), 3: (-.65, .65),
                  4: (-.4, .4), 5: (-.4, .4), 6: (-.4, .4),
                  7: (-.4, .4), 8: (-.4, .4), 9: (-.4, .4)}
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={"pose_range": {"roll": (-.025, .025), "pitch": (-.025, .025),
                                   "yaw": (-3.14, 3.14)},
                    "velocity_range": {"x": (-.15, .15)}, "asset_cfg": ROBOT},
        ),
    }
    if stage >= 2:
        events["push"] = EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(3., 6.),
            params={"velocity_range": {"y": (-.15, .15), "roll": (-.25, .25),
                                       "pitch": (-.2, .2), "yaw": (-.2, .2)}},
        )
        events["friction"] = EventTermCfg(
            func=dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=("tire",)),
                    "operation": "abs", "ranges": (1.1, 1.9), "shared_random": False},
        )
        events["base_mass"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                    "alpha_range": (.5 * math.log(.85), .5 * math.log(1.15))},
        )
    rewards = {
        "tracking": RewardTermCfg(func=tracking_reward, weight=1.0),
        "effort": RewardTermCfg(func=action_penalty, weight=-.012),
        "roll": RewardTermCfg(func=roll_penalty, weight=-.5),
    }
    if stage >= 3:
        rewards["yaw"] = RewardTermCfg(func=yaw_reward, weight=30.0 if stage >= 8 else 18.0 if stage == 7 else 12.0 if stage == 4 else 8.0 if stage >= 5 else 4.0)
        if stage >= 4:
            rewards["roll"].weight = -.1
            rewards["effort"].weight = -.005
        if stage >= 8:
            rewards["roll"].weight = -.05
            rewards["effort"].weight = -.002
        if stage >= 5:
            rewards["lateral"] = RewardTermCfg(func=lateral_reward, weight=2.0 if stage >= 6 else 8.0)
        if stage >= 6:
            rewards["errors"] = RewardTermCfg(func=motion_error_penalty,
                                                weight=-2.5 if stage >= 8 else -1.5 if stage == 7 else -1.0)
        if kind == "wheelbot":
            rewards["rotor_speed"] = RewardTermCfg(func=rotor_speed_reward, weight=2.0)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={"robot": robot}, num_envs=num_envs, env_spacing=5.),
        observations={"actor": ObservationGroupCfg(terms),
                      "critic": ObservationGroupCfg(dict(terms))},
        actions={"effort": JointEffortActionCfg(
            entity_name="robot", actuator_names=("drive", second), scale=1.0)},
        commands={"twist": UniformVelocityCommandCfg(
            entity_name="robot", resampling_time_range=(1.5, 3.0) if stage >= 7 else (2.5, 5.0),
            heading_command=False, rel_standing_envs=0.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=speed_ranges[stage], lin_vel_y=(0.0, 0.0),
                ang_vel_z=yaw_ranges[stage]))},
        events=events,
        rewards=rewards,
        terminations={
            "fall": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": .75}),
            "height": TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": .15}),
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY,
                            entity_name="robot", body_name="base", distance=2.4),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=.002)),
        decimation=10, episode_length_s=15. if stage >= 8 else 12.,
    )


def make_agent(kind, stage, iterations):
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution",
                                              "init_std": .4, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=3e-4, num_learning_epochs=4, num_mini_batches=4,
            entropy_coef=.008, gamma=.99, lam=.95, desired_kl=.015),
        experiment_name=f"{kind}_v2", run_name=f"stage{stage}" if stage > 1 else "",
        save_interval=20,
        logger="tensorboard", upload_model=False,
        num_steps_per_env=32, max_iterations=iterations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wheelbot", "steerbot"))
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9), default=1)
    parser.add_argument("--trail", type=float, default=None)
    parser.add_argument("--tilt", type=float, default=None,
                        help="Experimental flywheel mounting angle in radians")
    parser.add_argument("--head-mass", type=float, default=None,
                        help="Experimental steerbot steering-head mass (kg)")
    parser.add_argument("--tire-halfwidth", type=float, default=None)
    parser.add_argument("--steer-gear", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--resume", type=str, default=None, help="Previous stage run name or regex")
    parser.add_argument("--load-checkpoint", type=str, default="model_.*.pt",
                        help="Checkpoint file or regex within the resumed run")
    args = parser.parse_args()

    from mjlab.scripts.train import TrainConfig, launch_training
    from mjlab.tasks.registry import register_mjlab_task

    env_cfg = make_config(args.kind, args.stage, args.num_envs, args.trail,
                          args.tilt, args.head_mass, args.tire_halfwidth,
                          args.steer_gear)
    agent_cfg = make_agent(args.kind, args.stage, args.iterations)
    if args.trail is not None:
        agent_cfg.experiment_name = f"{args.kind}_trail{round(args.trail * 100):03d}"
    if args.tilt is not None:
        agent_cfg.experiment_name = f"{args.kind}_tilt{round(args.tilt * 100):03d}"
    if args.head_mass is not None:
        agent_cfg.experiment_name += f"_head{round(args.head_mass * 1000):03d}"
    if args.tire_halfwidth is not None:
        agent_cfg.experiment_name += f"_tire{round(args.tire_halfwidth * 1000):03d}"
    if args.steer_gear is not None:
        agent_cfg.experiment_name += f"_steer{round(args.steer_gear * 10):02d}"
    if args.resume:
        agent_cfg.resume = True
        agent_cfg.load_run = args.resume
        agent_cfg.load_checkpoint = args.load_checkpoint
    task_name = (f"Wheelbot-{args.kind}-{args.stage}-{args.trail}-{args.tilt}"
                 f"-{args.head_mass}-{args.tire_halfwidth}-{args.steer_gear}")
    register_mjlab_task(task_name, env_cfg,
                        make_config(args.kind, args.stage, 1, args.trail,
                                    args.tilt, args.head_mass,
                                    args.tire_halfwidth, args.steer_gear), agent_cfg)
    os.chdir(ROOT)
    launch_training(task_name, TrainConfig.from_task(task_name))
