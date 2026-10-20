"""Matched tests with no auto reset; moving starts are reported separately."""
import html
import json
import math


def heading(quat):
    import torch
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def evaluate(args, env, wrapped, runner):
    import torch
    from miniwheelbot_rl.task import heading_rate
    from .task import joint_ids
    robot = env.scene['robot']
    ids = joint_ids(env, args.morphology)
    policy = runner.get_inference_policy(device=env.device)
    env._fixed_reset = True
    env._evaluation_roll = (torch.linspace(-math.pi / 180, math.pi / 180, env.num_envs, device=env.device)
                            if env.num_envs > 1 else torch.zeros(1, device=env.device))
    cases = [('stop', 0., 0.), ('forward', 1., 0.), ('reverse', -1., 0.),
             ('left', .8, .3), ('right', .8, -.3), ('stop_start', 1., 0.)]
    report = {'checkpoint': str(args.checkpoint), 'morphology': args.morphology,
              'duration_s': args.duration, 'seed': args.seed, 'action_smoothing': 1.,
              'initial_roll_deg': env._evaluation_roll.rad2deg().tolist(),
              'settling_time_s': 3., 'notes': 'RMSE includes only live samples after 3 s; failed trials never pass. Moving starts are imposed initial states, not learned starts.',
              'thresholds': {'forward_rmse_m_s': .35, 'heading_rate_rmse_rad_s': .15,
                             'lateral_rmse_m_s': .35}, 'cases': []}
    for initial_speed in ((0., .8, -.8) if getattr(args,'extended_evaluation',False) else (0., .8)):
        env._initial_speed = initial_speed
        for name, speed, turn in cases:
            env.reset()
            alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            counts = torch.zeros(env.num_envs, device=env.device)
            sums = torch.zeros((env.num_envs, 3), device=env.device)
            survival = torch.full_like(counts, args.duration)
            previous_heading = heading(robot.data.root_link_quat_w)
            heading_error = torch.zeros_like(counts)
            maximum_heading = torch.zeros_like(counts)
            maximum_roll = torch.zeros_like(counts)
            max_sync = torch.zeros_like(counts)
            action_delta = torch.zeros_like(counts)
            action_count = torch.zeros_like(counts)
            saturation = torch.zeros_like(counts)
            previous_action = torch.zeros((env.num_envs, 2), device=env.device)
            for step in range(math.ceil(args.duration / env.step_dt)):
                t = step * env.step_dt
                target_speed = (0. if t < args.duration / 3 or t >= 2 * args.duration / 3 else speed) if name == 'stop_start' else speed
                env.command_manager.get_command('twist')[:] = torch.tensor([target_speed, 0., turn], device=env.device)
                with torch.inference_mode():
                    obs = wrapped.get_observations()
                    action = policy(obs).clamp(-1, 1)
                    finite = torch.isfinite(obs['actor']).all(dim=1) & torch.isfinite(action).all(dim=1)
                    invalid = alive & ~finite
                    survival[invalid] = t
                    alive &= finite
                    errors = torch.stack((robot.data.root_link_lin_vel_b[:, 0] - target_speed,
                                          heading_rate(env) - turn,
                                          robot.data.root_link_lin_vel_b[:, 1]), dim=1)
                    if t >= 3.:
                        sums += torch.where(alive[:, None], errors.square(), 0.)
                        counts += alive
                    if step:
                        action_delta += torch.where(alive, (action - previous_action).square().sum(dim=1), 0.)
                        saturation += torch.where(alive, (action.abs() >= .98).float().mean(dim=1), 0.)
                        action_count += alive
                    previous_action = action
                    _, _, done, _ = wrapped.step(torch.nan_to_num(action))
                    now = heading(robot.data.root_link_quat_w)
                    delta = torch.atan2(torch.sin(now - previous_heading), torch.cos(now - previous_heading))
                    heading_error += torch.where(alive, delta - turn * env.step_dt, 0.)
                    previous_heading = now
                    maximum_heading = torch.maximum(maximum_heading, heading_error.abs())
                    gravity = robot.data.projected_gravity_b
                    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                    maximum_roll = torch.maximum(maximum_roll, torch.where(alive, roll, 0.))
                    if args.morphology == 'twin_sync':
                        diff = (robot.data.joint_vel[:, ids[1]] - robot.data.joint_vel[:, ids[2]]).abs()
                        max_sync = torch.maximum(max_sync, torch.where(alive, diff, 0.))
                    failed = alive & done.bool()
                    survival[failed] = min((step + 1) * env.step_dt, args.duration)
                    alive &= ~done.bool()
                if not bool(alive.any()):
                    break
                # MJLab disallows another step with a pending manual reset.
                # Reset only failed slots; alive stays false permanently, so
                # neither their later motion nor survival enters the report.
                reset_ids = done.bool().nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids):
                    env.reset(env_ids=reset_ids)
            rmse = (sums / counts.clamp_min(1)[:, None]).sqrt()
            passed = alive & (counts > 0) & (rmse[:, 0] < .35) & (rmse[:, 1] < .15) & (rmse[:, 2] < .35)
            case = {'name': name, 'initial_speed_m_s': initial_speed, 'command': [speed, turn],
                    'survival_rate': alive.float().mean().item(), 'pass_rate': passed.float().mean().item(),
                    'individual': []}
            for i in range(env.num_envs):
                case['individual'].append({'survived': bool(alive[i]), 'passed': bool(passed[i]),
                    'survival_s': float(survival[i]), 'samples': int(counts[i]),
                    'forward_rmse': float(rmse[i, 0]) if counts[i] else None,
                    'heading_rate_rmse': float(rmse[i, 1]) if counts[i] else None,
                    'lateral_rmse': float(rmse[i, 2]) if counts[i] else None,
                    'final_heading_error_deg': float(heading_error[i].rad2deg()),
                    'max_heading_error_deg': float(maximum_heading[i].rad2deg()),
                    'max_roll_deg': float(maximum_roll[i].rad2deg()),
                    'mean_action_delta_squared': float(action_delta[i] / action_count[i].clamp_min(1)),
                    'action_saturation_fraction': float(saturation[i] / action_count[i].clamp_min(1)),
                    'max_sampled_rotor_sync_error_rad_s': float(max_sync[i]) if args.morphology == 'twin_sync' else None})
            report['cases'].append(case)
            print(f"{name} initial={initial_speed}: survival={case['survival_rate']:.3f}, pass={case['pass_rate']:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write_html(report, args.output.with_suffix('.html'))
    print(f'Evaluation: {args.output.resolve()}', flush=True)
    return report


def write_html(report, path):
    rows = []
    def mean(case, key):
        vals = [x[key] for x in case['individual'] if x[key] is not None]
        return f'{sum(vals)/len(vals):.3f}' if vals else '—'
    for c in report['cases']:
        cells = [html.escape(c['name']), str(c['initial_speed_m_s']), f"{c['survival_rate']:.0%}",
                 f"{c['pass_rate']:.0%}", mean(c, 'forward_rmse'), mean(c, 'heading_rate_rmse'),
                 mean(c, 'max_heading_error_deg'), mean(c, 'max_roll_deg'), mean(c, 'mean_action_delta_squared')]
        rows.append('<tr>' + ''.join('<td>' + x + '</td>' for x in cells) + '</tr>')
    headers = ['任务', '初始速度 m/s', '存活率', '跟踪通过率', '速度 RMSE m/s', '航向角速度 RMSE rad/s', '最大航向误差 °', '最大 roll °', '动作变化平方均值']
    path.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>结构策略评估</title>'
        '<style>body{font:16px system-ui;margin:40px;color:#172638;background:#f5f7fa}table{border-collapse:collapse;background:white}th,td{padding:12px;border:1px solid #d9e0e8;text-align:right}th{background:#e8eff8}p{max-width:1000px;line-height:1.7}</style>'
        f'<h1>{html.escape(report["morphology"])} · 速度跟踪评估</h1><p>{html.escape(report["checkpoint"])}</p>'
        '<p>初始速度 0 表示静止起步；0.8 m/s 表示直接赋予初速度，不能视为自主起步。无动作滤波；失败环境仅为推进批量仿真而复位，此后样本全部排除。'
        '每例前 3 秒不计入 RMSE；失败样本不会通过。通过要求全程存活，速度 / 航向角速度 / 侧向速度 RMSE 分别小于 0.35 / 0.15 / 0.35。'
        '表中误差取各环境均值，最大航向误差基于目标角速度积分；完整逐环境数据见同名 JSON。stop_start 按三等分执行停—走—停。</p>'
        '<table><thead><tr>' + ''.join('<th>' + x + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></html>', encoding='utf-8')
