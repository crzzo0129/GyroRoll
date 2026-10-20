"""Deterministic motor diagnostics, exported without plotting dependencies."""

import csv
from html import escape
import json
import math

from .settings import Settings


def summarize(rows, settings):
    """Summaries use successful control endpoints, never automatic reset states."""
    if not rows:
        return {"samples": 0}
    result = {"samples": len(rows)}
    for motor in ("flywheel", "rolling"):
        raw = [r[f"{motor}_raw"] for r in rows]
        action = [r[f"{motor}_action"] for r in rows]
        torque = [r[f"{motor}_torque_Nm"] for r in rows]
        error = [r[f"{motor}_target_rad_s"] - r[f"{motor}_actual_rad_s"] for r in rows]
        delta = [b - a for a, b in zip(action, action[1:])]
        result[motor] = {
            "clipped_fraction": sum(abs(v) > 1 for v in raw) / len(rows),
            "action_boundary_fraction": sum(abs(v) >= 0.99 for v in action) / len(rows),
            "torque_saturation_fraction": sum(abs(v) >= 0.99 * settings.torque_limit
                                               for v in torque) / len(rows),
            "torque_rms_Nm": math.sqrt(sum(v * v for v in torque) / len(rows)),
            "speed_rmse_rad_s": math.sqrt(sum(v * v for v in error) / len(rows)),
            "action_delta_rms": (math.sqrt(sum(v * v for v in delta) / len(delta))
                                 if delta else None),
        }
    for name, actual, target in (("forward_rmse_m_s", "forward_m_s", "command_m_s"),
                                 ("heading_rmse_rad_s", "heading_rad_s", "command_rad_s")):
        result[name] = math.sqrt(sum((r[actual] - r[target]) ** 2 for r in rows) / len(rows))
    return result


def chart(rows, title, fields, unit, bounds=()):
    """Render all samples, including rapid alternation, as an offline SVG."""
    if not rows:
        return ""
    colors = ("#2563eb", "#ea580c")
    values = [r[key] for r in rows for key, _ in fields] + list(bounds)
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, 0.01)
    low, high = low - padding, high + padding
    end = max(rows[-1]["time_s"], 0.01)
    x = lambda t: 62 + 716 * t / end
    y = lambda v: 212 - 186 * (v - low) / (high - low)
    parts = [f'<section><h2>{escape(title)}</h2><svg viewBox="0 0 800 250" '
             f'role="img" aria-label="{escape(title)}">']
    for i in range(5):
        value = low + (high - low) * i / 4
        parts.append(f'<path d="M62 {y(value):.2f}H778" stroke="#e2e8f0"/>'
                     f'<text x="55" y="{y(value)+4:.2f}" text-anchor="end">{value:.3g}</text>')
        t = end * i / 4
        parts.append(f'<text x="{x(t):.2f}" y="235" text-anchor="middle">{t:.2f}</text>')
    for bound in bounds:
        parts.append(f'<path d="M62 {y(bound):.2f}H778" stroke="#dc2626" stroke-dasharray="5 4"/>')
    for (key, _), color in zip(fields, colors):
        points = " ".join(f'{x(r["time_s"]):.2f},{y(r[key]):.2f}' for r in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1"/>')
    parts.append(f'<text x="62" y="15">{escape(unit)}</text>'
                 '<text x="778" y="248" text-anchor="end">时间 / s</text></svg><p>')
    for (_, label), color in zip(fields, colors):
        parts.append(f'<span style="color:{color}">━ {escape(label)}　</span>')
    parts.append('</p></section>')
    return "".join(parts)


def write_outputs(directory, rows, report, settings):
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["time_s", "command_m_s", "command_rad_s", "forward_m_s", "heading_rad_s"]
    fields += [f"{motor}_{suffix}" for motor in ("flywheel", "rolling")
               for suffix in ("raw", "action", "target_rad_s", "actual_rad_s", "torque_Nm")]
    if rows and "tilt_deg" in rows[0]:
        fields += ["tilt_deg", "height_m", "recovering", "recoveries"]
    with (directory / "trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["all_samples"] = summarize(rows, settings)
    report["after_3s"] = summarize([r for r in rows if r["time_s"] > 3.0], settings)
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plots = []
    for motor, label, limit in (("flywheel", "飞轮", settings.flywheel_speed_limit),
                                ("rolling", "滚动轮", settings.rolling_speed_limit)):
        plots.append(chart(rows, f"{label}：目标与实际转速",
                           ((f"{motor}_target_rad_s", "目标"), (f"{motor}_actual_rad_s", "实际")),
                           "rad/s", (-limit, limit)))
        plots.append(chart(rows, f"{label}：电机力矩", ((f"{motor}_torque_Nm", "力矩"),),
                           "N·m", (-settings.torque_limit, settings.torque_limit)))
        plots.append(chart(rows, f"{label}：策略输出与执行动作",
                           ((f"{motor}_raw", "裁剪前的策略均值"), (f"{motor}_action", "裁剪后")),
                           "归一化动作", (-1, 1)))
    plots.append(chart(rows, "机体前进速度", (("command_m_s", "指令"), ("forward_m_s", "实际")), "m/s"))
    plots.append(chart(rows, "水平航向变化率", (("command_rad_s", "指令"), ("heading_rad_s", "实际")), "rad/s"))
    if rows and "tilt_deg" in rows[0]:
        plots.append(chart(rows, "起身：机体倾斜角", (("tilt_deg", "相对竖直方向"),), "deg", (20,)))
        plots.append(chart(rows, "起身：离地高度", (("height_m", "地面法向高度"),), "m", (0.30,)))
        plots.append(chart(rows, "恢复状态", (("recovering", "正在恢复"), ("recoveries", "累计成功")), ""))
    status = (f'在 {report["simulated_duration_s"]:.2f} 秒发生回合终止，已停止采样。'
              if report["terminated_early"] else "已完成指定时长。")
    table = []
    for window, label in (("all_samples", "含起步"), ("after_3s", "3 秒后")):
        stats = report[window]
        for motor, name in (("flywheel", "飞轮"), ("rolling", "滚动轮")):
            if motor in stats:
                m = stats[motor]
                table.append(f'<tr><td>{label}</td><td>{name}</td>'
                             f'<td>{m["clipped_fraction"]:.1%}</td>'
                             f'<td>{m["action_boundary_fraction"]:.1%}</td>'
                             f'<td>{m["torque_saturation_fraction"]:.1%}</td>'
                             f'<td>{m["speed_rmse_rad_s"]:.3f}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>策略电机诊断</title>
<style>body{font-family:system-ui,sans-serif;margin:32px auto;padding:0 20px;max-width:1200px;color:#17233b;background:#f6f8fc}
h1{font-size:28px}h2{font-size:18px}p{line-height:1.7}section{background:white;padding:18px;border-radius:12px;min-width:0}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px;margin-top:24px}
svg{width:100%}svg text{font-size:11px;fill:#475569}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:12px;text-align:left;border-bottom:1px solid #e2e8f0}code{overflow-wrap:anywhere}</style>
<h1>策略电机诊断 · 确定性回放</h1>'''
    page += f'<p>模型：<code>{escape(report["checkpoint"])}</code><br>前进指令 {report["command"][0]:g} m/s；'
    page += f'转向指令 {report["command"][1]:g} rad/s。{status}</p>'
    page += '<p>使用策略均值，不加入探索噪声。红色虚线表示限制；先看目标是否跳变，再看实际转速是否跟随、力矩是否持续限幅。'
    page += '统计同时给出含起步和 3 秒后的结果；过早终止时，短时间误差不能代表成功跟踪。</p>'
    page += '<p>均值有界的策略可能无需裁剪却仍长期贴近 ±1，请同时看动作贴边比例和力矩换向情况。</p>'
    if "recovery" in report:
        r = report["recovery"]
        page += (f'<p>起身次数：{r["successes"]} / {r["attempts"]}；'
                 f'结束时仍在恢复：{"是" if r["recovering_at_end"] else "否"}。'
                 '前进速度按地面切向计算；起身过程的全程误差不代表站稳后的跟踪误差，'
                 '请用 curriculum 评估检查恢复后继续跟踪。</p>')
    page += '<table><tr><th>时段</th><th>电机</th><th>动作被裁剪比例</th><th>|动作| ≥0.99 比例</th><th>力矩 ≥99% 上限比例</th><th>转速 RMSE / rad/s</th></tr>'
    page += ''.join(table) + '</table><div class="plots">' + ''.join(plots) + '</div>'
    page += '<p>每个控制周期末采样（100 Hz 默认配置）：目标为刚执行的指令，实际转速和力矩为周期末读数。'
    page += '力矩数据不包含周期内部峰值；终止后自动重置的状态不进入曲线。完整数据见 trace.csv，统计见 summary.json。</p></html>'
    (directory / "report.html").write_text(page, encoding="utf-8")


def diagnose(args, env, wrapped, runner, *, heading_fn=None):
    import torch
    if heading_fn is None:
        from .task import heading_rate
        heading_fn = heading_rate
    if env.num_envs != 1 or not all(math.isfinite(v) for v in (args.speed, args.turn, args.duration)):
        raise ValueError("Diagnostics require one environment and finite command/duration values")
    manifest = json.loads((args.checkpoint.parent / "task.json").read_text(encoding="utf-8"))
    settings = Settings(**manifest["settings"])
    robot = env.scene["robot"]
    joints, _ = robot.find_joints(("reaction", "drive"), preserve_order=True)
    actuators, _ = robot.find_actuators(("reaction_velocity", "drive_velocity"), preserve_order=True)
    scales = torch.tensor([settings.flywheel_speed_limit, settings.rolling_speed_limit], device=env.device)
    policy = runner.get_inference_policy(device=env.device)
    env.reset()
    action_smoothing = args.action_smoothing
    previous_action = None
    command = torch.tensor([args.speed, 0.0, args.turn], device=env.device)
    rows, terminated, elapsed = [], False, 0.0
    for step in range(math.ceil(args.duration / env.step_dt)):
        env.command_manager.get_command("twist")[:] = command
        with torch.inference_mode():
            raw = policy(wrapped.get_observations()).clone()
            if not torch.isfinite(raw).all():
                raise RuntimeError("Policy produced a non-finite action")
            action = raw.clamp(-1, 1)
            if previous_action is not None:
                action = action_smoothing * action + (1 - action_smoothing) * previous_action
            previous_action = action
            target = action * scales
            _, _, done, _ = wrapped.step(action)
            elapsed = (step + 1) * env.step_dt
            if bool(done.any()):
                terminated = True
                break  # Returned state has already reset; do not log it.
            row = {"time_s": elapsed, "command_m_s": args.speed, "command_rad_s": args.turn,
                   "forward_m_s": float(robot.data.root_link_lin_vel_b[0, 0]),
                   "heading_rad_s": float(heading_fn(env)[0])}
            if hasattr(env, "_recovery"):
                from .recovery import tangent_velocity, ground_height, state
                row["forward_m_s"] = float(tangent_velocity(env)[0][0])
                row.update(tilt_deg=math.degrees(math.acos(max(-1, min(1,
                    float(-robot.data.projected_gravity_b[0, 2]))))),
                    height_m=float(ground_height(env)[0]),
                    recovering=float(state(env).active[0]), recoveries=float(state(env).successes[0]))
            for i, motor in enumerate(("flywheel", "rolling")):
                row.update({f"{motor}_raw": float(raw[0, i]),
                            f"{motor}_action": float(action[0, i]),
                            f"{motor}_target_rad_s": float(target[0, i]),
                            f"{motor}_actual_rad_s": float(robot.data.joint_vel[0, joints[i]]),
                            f"{motor}_torque_Nm": float(robot.data.actuator_force[0, actuators[i]])})
            if not all(math.isfinite(v) for v in row.values()):
                raise RuntimeError("Simulation produced non-finite diagnostic data")
            rows.append(row)
    report = {"checkpoint": str(args.checkpoint), "policy": "deterministic_mean",
              "command": [args.speed, args.turn], "seed": args.seed,
              "action_smoothing": action_smoothing,
              "settings": manifest["settings"], "step_dt_s": env.step_dt,
              "requested_duration_s": args.duration, "simulated_duration_s": elapsed,
              "terminated_early": terminated, "sampling": "control_endpoint; auto-reset states excluded"}
    if hasattr(env, "_recovery"):
        from .recovery import snapshot
        report["recovery"] = snapshot(env)
        report["stage"] = args.stage
    write_outputs(args.diagnostic_dir, rows, report, settings)
    print(f"Diagnostic report: {(args.diagnostic_dir / 'report.html').resolve()}")
    print(f"Terminated early: {terminated}; recorded {len(rows)} samples")
    return report
