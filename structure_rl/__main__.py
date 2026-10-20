"""Train and compare independent PPO policies for the two concept structures."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from .model import MORPHOLOGIES, PHASES, VERSION, SYNC_TOLERANCE_RAD_S, MotorSettings, load_manifest, source_model, observation_dimension


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'smoke', 'evaluate', 'video'))
    p.add_argument('--morphology', choices=MORPHOLOGIES)
    p.add_argument('--phase', choices=PHASES)
    p.add_argument('--profile', choices=('baseline','steer_v2','linkage_v1'))
    p.add_argument('--mirror-coeff', type=float, default=None,
                   help='Optional steering mirror loss coefficient; steer_v2 defaults to 1')
    p.add_argument('--warm-start', action='store_true', help='Load policy/value only; reset optimizer, iteration and exploration')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--run-dir', type=Path, help='New directory for this training run')
    p.add_argument('--duration', type=float, default=15.)
    p.add_argument('--extended-evaluation', action='store_true', help='Also evaluate resets at -0.8 m/s')
    p.add_argument('--output', type=Path, help='Default: results/<model>/rl/evaluation.json')
    p.add_argument('--video', type=Path, help='Default: results/<model>/rl/video.mp4')
    p.add_argument('--speed', type=float, default=1.)
    p.add_argument('--turn', type=float, default=0.)
    p.add_argument('--initial-speed', type=float, default=0., help='Video reset speed; 0 tests autonomous start')
    return p


def build(args):
    if version('mjlab') != '1.3.0':
        raise RuntimeError('Requires mjlab==1.3.0; see requirements-mjlab.txt')
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from miniwheelbot_rl.task import make_agent_cfg
    from miniwheelbot_rl.policy_config import POLICY_VERSION
    from .task import configuration

    settings = MotorSettings()
    model_path = None
    if args.checkpoint:
        manifest = load_manifest(args.checkpoint, args.morphology)
        if manifest['policy_version'] != POLICY_VERSION or manifest['mjlab'] != '1.3.0':
            raise ValueError('Incompatible policy or MJLab contract')
        args.morphology = manifest['morphology']
        args.phase = args.phase or manifest['phase']
        args.profile = args.profile or manifest.get('training_profile','baseline')
        if args.mirror_coeff is None:
            args.mirror_coeff = manifest.get('mirror_loss_coefficient')
        settings = MotorSettings(**manifest['motors'])
        model_path = args.checkpoint.parent / 'geometry.xml'
    if args.morphology is None:
        raise ValueError('--morphology is required when starting from scratch')
    family = 'twin_flywheel' if args.morphology == 'twin_sync' else args.morphology
    result_dir = Path('results') / family / 'rl'
    args.output = args.output or result_dir / 'evaluation.json'
    args.video = args.video or result_dir / 'video.mp4'
    args.phase = args.phase or ('moving' if args.morphology == 'raked_steer' else 'warmup')
    args.profile = args.profile or ('linkage_v1' if args.morphology=='roll_linkage' else 'baseline')
    args.mirror_coeff = args.mirror_coeff if args.mirror_coeff is not None else (1. if args.profile=='steer_v2' else 0.)
    if not math.isfinite(args.mirror_coeff) or args.mirror_coeff < 0:
        raise ValueError('Mirror coefficient must be finite and nonnegative')
    if args.mirror_coeff and (args.profile!='steer_v2' or args.morphology!='raked_steer'):
        raise ValueError('Mirror loss is only supported for steer_v2')
    if args.warm_start and (args.checkpoint is None or args.mode!='train'):
        raise ValueError('--warm-start requires train --checkpoint')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    wp.config.kernel_cache_dir = str(Path('.cache/warp_structure').resolve())
    configure_torch_backends()
    replay = args.mode in ('evaluate', 'video')
    cfg = configuration(args.morphology, args.phase, args.num_envs, settings, model_path, replay, args.profile)
    cfg.seed = args.seed
    if replay:
        cfg.auto_reset = False
        cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.)
        cfg.commands['twist'].resampling_time_range = (1e8, 1e8)
        cfg.commands['twist'].rel_standing_envs = 0.
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode='rgb_array' if args.mode == 'video' else None)
    try:
        if replay:
            env._fixed_reset = True
            env._initial_speed = args.initial_speed
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.)
        agent = make_agent_cfg(args.iterations)
        agent.seed = args.seed
        agent.experiment_name = 'structure_' + args.morphology
        if args.profile=='steer_v2':
            agent.algorithm.learning_rate=1e-4
            agent.algorithm.schedule='fixed'
        agent_config=asdict(agent)
        if args.mirror_coeff:
            agent_config['algorithm']['symmetry_cfg']={
                'use_data_augmentation':False,'use_mirror_loss':True,
                'mirror_loss_coeff':args.mirror_coeff,
                'data_augmentation_func':'structure_rl.steer_training:mirror'}
        log_dir = None
        if args.mode == 'train':
            log_dir = args.run_dir or Path('logs/structure_rl') / args.morphology / (
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + args.phase)
            log_dir.mkdir(parents=True, exist_ok=False)
            geometry = (model_path or source_model(args.morphology)).read_bytes()
            (log_dir / 'geometry.xml').write_bytes(geometry)
            manifest = {
                'task_version': VERSION, 'mjlab': '1.3.0', 'policy_version': POLICY_VERSION,
                'runtime_versions': {name: version(name) for name in ('mjlab', 'torch', 'mujoco', 'mujoco-warp', 'warp-lang')},
                'morphology': args.morphology, 'phase': args.phase, 'motors': asdict(settings),
                'training_profile':args.profile,'mirror_loss_coefficient':args.mirror_coeff,
                'warm_start':args.warm_start,'exploration_reset_std':.08 if args.warm_start else None,
                'seed': args.seed, 'num_envs': args.num_envs, 'iterations_requested': args.iterations,
                'source_checkpoint': str(args.checkpoint) if args.checkpoint else None,
                'observation_dimension': observation_dimension(args.morphology), 'action_dimension': 2,
                'action_order': ['drive_rad_s', {'twin_sync':'shared_rotor_rad_s',
                    'raked_steer':'steer_rad','roll_linkage':'linkage_motor_rad'}[args.morphology]],
                'command_order': ['forward_m_s', 'heading_rate_rad_s'],
                'geometry_sha256': hashlib.sha256(geometry).hexdigest(),
                'reward_weights': {k: v.weight for k, v in cfg.rewards.items()},
                'commands': {'forward': cfg.commands['twist'].ranges.lin_vel_x,
                             'yaw_rate': cfg.commands['twist'].ranges.ang_vel_z,
                             'standing_fraction': cfg.commands['twist'].rel_standing_envs},
                'physics_dt': cfg.sim.mujoco.timestep, 'policy_dt': env.step_dt,
                'rotor_sync_validation_tolerance_rad_s': SYNC_TOLERANCE_RAD_S,
                'algorithm': 'PPO; native finite-torque motor servos; no expert balance controller',
                'initialization': ('half zero, quarter +0.3..0.9, quarter -0.3..0.9 m/s' if args.profile in ('steer_v2','linkage_v1') and args.phase!='warmup' else
                                   'moving: 0.5 to 1 m/s; tracking: half zero, half 0.8 m/s; warmup: zero'),
            }
            if args.profile in ('steer_v2','linkage_v1') and args.phase=='tracking':
                from .steer_training import TRAINING_REVISION
                manifest['training_revision']=TRAINING_REVISION
                manifest['commands'].update(sampling='20% stop, 20% forward straight, 20% reverse straight, 15% left, 15% right, 10% mixed maneuvers',
                                            resampling_seconds=[4,8],turn_speed_sign='equal forward/reverse probability',
                                            mixed_maneuvers='25% each forward straight/reverse straight/left/right; independent of actual velocity',
                                            nonzero_speed_magnitude=[.3,1.])
            if args.morphology=='roll_linkage':
                manifest.update(model_variant='level_torso_v2',training_revision='linkage_v1_20261020',
                    passive_observation_order=['wheel_roll_rad/limit','passive_rod_rad/limit',
                                               'wheel_roll_rad_s/10','passive_rod_rad_s/10'],
                    mechanical_constraint='physical site-connect four-bar; free torso; two actuators')
            (log_dir / 'task.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            (log_dir / 'agent.json').write_text(json.dumps(agent_config, indent=2), encoding='utf-8')
        runner = None
        if args.mode != 'smoke':
            runner = MjlabOnPolicyRunner(wrapped, agent_config, str(log_dir) if log_dir else None, args.device)
            if args.checkpoint:
                load_cfg={'actor':True,'critic':True,'optimizer':False,'iteration':False} if args.warm_start else None
                runner.load(str(args.checkpoint),load_cfg=load_cfg)
                if args.warm_start:
                    from .steer_training import reset_exploration
                    reset_exploration(runner.alg.actor)
                    env.common_step_counter=0
                    print('Warm start: policy/value retained; optimizer/iteration reset; std=0.08',flush=True)
        return env, wrapped, runner, log_dir
    except BaseException:
        env.close()
        raise


def smoke(args, env, wrapped):
    import torch
    from .task import joint_ids,metric
    robot = env.scene['robot']
    ids = joint_ids(env, args.morphology)
    assert wrapped.num_actions == 2
    maximum_sync = 0.
    maximum_closure = 0.
    for step in range(100):
        # Exercise both signs and deliberately exceed policy bounds.
        actions = torch.zeros((env.num_envs, 2), device=env.device)
        actions[:, 0] = .1 if step % 20 < 10 else -.1
        actions[:, 1] = 1.5 if step % 20 < 10 else -1.5
        obs, reward, _, _ = wrapped.step(actions)
        assert obs['actor'].shape == (env.num_envs, observation_dimension(args.morphology))
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        assert robot.data.actuator_force.abs().max() <= 2.4001
        if args.morphology=='roll_linkage':
            maximum_closure=max(maximum_closure,metric(env,'closure_error',args.morphology).max().item())
        if args.morphology == 'twin_sync':
            assert robot.data.actuator_force[:, 1:].abs().max() <= 1.2001
            maximum_sync = max(maximum_sync, (robot.data.joint_vel[:, ids[1]] - robot.data.joint_vel[:, ids[2]]).abs().max().item())
    if args.morphology == 'twin_sync':
        maximum_sync = max(maximum_sync, env._sync_peak.max().item())
    assert maximum_sync < SYNC_TOLERANCE_RAD_S, f'Rotor synchronization violation: {maximum_sync}'
    assert maximum_closure < .001, f'Four-bar closure exceeds 1 mm: {maximum_closure}'
    print(json.dumps({'smoke_passed': True, 'morphology': args.morphology, 'steps': 100,
                      'observation_dimension': observation_dimension(args.morphology), 'action_dimension': 2,
                      'max_sampled_closure_error_m': maximum_closure,
                      'max_physics_substep_sync_error_rad_s': maximum_sync}, indent=2), flush=True)


def main():
    args = parser().parse_args()
    args.num_envs = args.num_envs if args.num_envs is not None else {'train': 256, 'smoke': 4, 'evaluate': 8, 'video': 1}[args.mode]
    if args.num_envs < 1 or args.iterations < 1 or not math.isfinite(args.duration) or args.duration <= 3:
        raise ValueError('Positive num-envs/iterations and finite duration > 3 s are required')
    if not all(math.isfinite(x) for x in (args.speed, args.turn, args.initial_speed)):
        raise ValueError('Commands and initial speed must be finite')
    if args.mode in ('evaluate', 'video') and args.checkpoint is None:
        raise ValueError('Replay requires --checkpoint from structure_rl')
    if args.mode == 'video' and (args.num_envs != 1 or (args.video is not None and args.video.suffix.lower() != '.mp4')):
        raise ValueError('Video requires --num-envs 1 and an .mp4 path')
    if os.name == 'nt':
        os.environ['MUJOCO_GL'] = 'glfw'
    elif args.mode == 'video':
        os.environ.setdefault('MUJOCO_GL', 'egl')
    env, wrapped, runner, log_dir = build(args)
    try:
        if args.mode == 'smoke':
            smoke(args, env, wrapped)
        elif args.mode == 'train':
            print(f'Run directory: {log_dir.resolve()}', flush=True)
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            (log_dir / 'complete.json').write_text(json.dumps({'completed': True, 'iterations_requested': args.iterations}), encoding='utf-8')
        elif args.mode == 'evaluate':
            from .evaluation import evaluate
            evaluate(args, env, wrapped, runner)
        else:
            from miniwheelbot_rl.video import record_video
            args.action_smoothing = 1.
            report = record_video(args, env, wrapped, runner)
            report.update(morphology=args.morphology, initial_speed_m_s=args.initial_speed)
            args.video.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    main()
