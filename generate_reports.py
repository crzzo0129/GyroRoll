"""Render self-contained experiment reports from measured acceptance rollouts."""

import html
import json
from pathlib import Path


from result_paths import legacy_result_dir

ROOT = Path(__file__).resolve().parent
INFO = {
    "wheelbot": {
        "title": "正交飞轮独轮车",
        "subtitle": "接地轮驱动俯仰与前进；水平反作用轮直接调节横滚，俯仰与转子角动量的耦合产生偏航。",
        "second": "反作用轮：x 轴，质量 0.44 kg，半径 0.083 m，力矩上限约 0.48 N·m。",
        "mechanism": "航向不是独立关节。偏航响应依赖反作用轮转速、俯仰速度与轮地约束，零飞轮转速附近难以利用陀螺项。stage 3 使用初始 110 rad/s 飞轮转速作为探索条件；报告不将它表述为从静止自旋启动的能力。",
    },
    "steerbot": {
        "title": "转向平衡独轮车",
        "subtitle": "接地轮驱动前进与俯仰；竖直转向关节改变轮子的相对方向，借行驶中的接触力调节横滚和航向。",
        "second": "竖直转向关节：机械拖距 0.12 m，关节限位 ±0.8 rad，电机力矩上限约 0.6 N·m。",
        "mechanism": "这是单个接地点，不使用自行车转角—曲率公式。转向轴位于接地点前方，拖距提供接触力臂。此构型不能靠转向在零速稳住横滚，所有训练与测试均从非零前进速度起步。",
    },
}


def value(x, units=""):
    return "未测得" if x is None else f"{x:.3f}{units}"


def render(kind):
    info = INFO[kind]
    results_name = "steerbot_moderate" if kind == "steerbot" else kind
    results = legacy_result_dir(results_name)
    results_relative = results.relative_to(ROOT).as_posix()
    acceptance = json.loads((results / "acceptance.json").read_text())
    trajectory = json.loads((results / "seed_0.json").read_text())["trajectory"]
    measured = acceptance["individual"]
    seed_zero = measured[0]
    recording_label = (f"{seed_zero['duration']:.2f} 秒跌倒前录屏"
                       if seed_zero["fallen"] else "完整 15 秒录屏")
    passed = acceptance["pass_rate"] == 1.
    verdict = "通过全部随机种子" if passed else "尚未通过"
    status = "pass" if passed else "fail"
    checkpoint = html.escape(acceptance["controller"])
    chosen = acceptance["controller"]
    if chosen == "optimized":
        controller_description = "13/11 参数状态反馈；在完整 15 s MuJoCo 轨迹上差分进化调参（非 PPO 策略）。"
    elif chosen == "thin":
        controller_description = ("窄胎飞轮七参数状态反馈，转向指令回正后加入偏航恢复项；"
                                  "在 15 s MuJoCo 轨迹上优化（非 PPO 策略）。")
    elif chosen == "lqr":
        controller_description = "约 2 m/s 直行工作点有限差分线性化，LQR 横向反馈 + 偏航积分（非 PPO 策略）。"
    elif chosen == "baseline":
        controller_description = "手调的俯仰/横滚状态反馈基线（非 PPO 策略）。"
    else:
        controller_description = "RSL-RL PPO 两层 128 单元 ELU MLP，50 Hz 推理。"
    frequency = "50 Hz" if chosen.endswith(".pt") else "500 Hz"
    offset = acceptance.get("com_offset")
    morphology = (f"机身主质量相对轮轴前移 {offset:.3f} m（候选几何；PPO 课程使用原始 0 m 几何）。"
                  if offset is not None else "机身主质量位于轮轴上方（原始几何）。")
    mainwheel_description = "半径 0.12 m、质量 0.30 kg，圆柱几何与平面接触；主驱动力矩上限约 2.4 N·m。"
    second_description = info["second"]
    mechanism = info["mechanism"]
    if chosen == "thin":
        halfwidth = acceptance["tire_halfwidth"]
        mainwheel_description = (f"半径 0.12 m、半宽 {halfwidth:.3f} m、质量 0.30 kg；"
                                 "窄圆柱轮胎与平面接触，主驱动力矩上限约 2.4 N·m。")
        second_description = (f"正交 x 轴反作用轮：质量 0.44 kg、半径 0.083 m，"
                              f"电机力矩上限 {acceptance['reaction_gear']:.1f} N·m。"
                              "这与原始概念模型的 0.48 N·m 上限不同。")
        morphology = ("机身主质量位于轮轴上方；窄胎/高力矩是单独调过参数的概念变型，"
                      f"起始转子预旋转 {acceptance['rotor_speed']:.0f} rad/s。")
    if kind == "steerbot" and acceptance.get("trail") is not None:
        mainwheel_description = (f"半径 0.12 m、半宽 {acceptance['tire_halfwidth']:.3f} m、"
                                 "质量 0.30 kg，窄圆柱接触；主驱动力矩 2.4 N·m。")
        second_description = (f"竖直转向轴相对轮轴拖距 {acceptance['trail']:.3f} m；"
                              f"转向头质量 {acceptance['head_mass']:.3f} kg，"
                              f"电机上限 {acceptance['steer_gear']:.1f} N·m。")
        morphology = (f"机身主质量 {acceptance['body_mass']:.3f} kg，"
                      f"相对原模型竖直移动 {acceptance['body_offset_z']:+.3f} m；"
                      "该候选不同于最初的正拖距模型。")
        mechanism = ("转向轴在轮轴后方，是主动控制的负拖距结构；依靠前进时轮地接触力"
                     "同时调节横滚和偏航，零速不具备转向平衡能力。强冲击下存在方向不对称，"
                     "不能把中等扰动验收外推到任意冲击。")
    controller_file = (acceptance.get("lqr_tuning") or acceptance.get("controller_params") or
                       (str(ROOT / "wheelbot/thin_turning_004_g24.json")
                        if chosen == "thin" else None))
    controller_artifact = (f" 参数文件 <code>{html.escape(controller_file)}</code>。"
                           if controller_file else "")
    alternate = ""
    alternate_path = ROOT / 'results/steerbot_legacy/steerbot_trail050' / "acceptance.json"
    if kind == "steerbot" and alternate_path.exists():
        alt = json.loads(alternate_path.read_text())
        alternate = (f'<h2>长拖距对照</h2><div class="lead"><div>'
                     f'<video controls muted playsinline preload="metadata" '
                     f'src="../results/steerbot_legacy/steerbot_trail050/rollout.mp4"></video>'
                     f'</div><div><p>转向轴机械拖距扩至 0.50 m，采用单独训练的 PPO 策略。'
                     f'五种子扰动/DR 验收：完整通过 {alt["pass_rate"]:.0%}，'
                     f'速度 RMSE {value(alt["mean_steady_speed_rmse"], " m/s")}，'
                     f'偏航 RMSE {value(alt["mean_turn_yaw_rmse"], " rad/s")}，'
                     f'侧向速度 RMSE {value(alt["mean_turn_lateral_rmse"], " m/s")}。'
                     f'该长叉臂目前为理想刚体运动学，没有建模实体叉臂的额外质量和柔度；'
                     f'仍未通过偏航验收。</p></div></div>')
    lightfork_path = ROOT / 'results/steerbot_legacy/steerbot_lightfork' / "acceptance.json"
    if kind == "steerbot" and lightfork_path.exists():
        alt = json.loads(lightfork_path.read_text())
        alternate += (f'<h2>轻量长叉对照</h2><div class="lead"><div>'
                      f'<video controls muted playsinline preload="metadata" '
                      f'src="../results/steerbot_legacy/steerbot_lightfork/rollout.mp4"></video>'
                      f'</div><div><p>0.50 m 拖距、转向头质量减至 0.005 kg，'
                      f'以完整左右转 15 s 名义轨迹差分进化调参。名义偏航 RMSE 曾达到 '
                      f'0.164 rad/s，但五种子扰动/DR 验收完整通过 {alt["pass_rate"]:.0%}，'
                      f'速度 RMSE {value(alt["mean_steady_speed_rmse"], " m/s")}，'
                      f'偏航 RMSE {value(alt["mean_turn_yaw_rmse"], " rad/s")}，'
                      f'侧向速度 RMSE {value(alt["mean_turn_lateral_rmse"], " m/s")}。'
                      f'轻量化提高名义灵敏度，但目前控制器对质量/摩擦变化敏感；'
                      f'长叉臂本身的质量与柔度也尚未建模。</p></div></div>')
    negative_path = ROOT / "results/steerbot_legacy/steerbot_negative015/acceptance.json"
    if kind == "steerbot" and negative_path.exists():
        alt = json.loads(negative_path.read_text())
        alternate += (f'<h2>负拖距转向候选</h2><div class="lead"><div>'
                      f'<video controls muted playsinline preload="metadata" '
                      f'src="../results/steerbot_legacy/steerbot_negative015/rollout.mp4"></video>'
                      f'</div><div><p>转向轴在轮轴后 0.15 m（负拖距），窄圆柱轮胎半宽 '
                      f'0.004 m，转向电机 2.4 N·m，车身主质量 0.645 kg、下移 0.062 m。'
                      f'对称直行工作点线性化后用 LQR 和积分偏航控制；无扰动名义回放'
                      f'转弯偏航 RMSE 曾达到 0.136 rad/s，但两次侧向/横滚速度扰动下'
                      f'五种子完整通过 {alt["pass_rate"]:.0%}，'
                      f'速度 RMSE {value(alt["mean_steady_speed_rmse"], " m/s")}，'
                      f'偏航 RMSE {value(alt["mean_turn_yaw_rmse"], " rad/s")}，'
                      f'侧向速度 RMSE {value(alt["mean_turn_lateral_rmse"], " m/s")}。'
                      f'负拖距并非被动稳定；当前控制器不能充分抑制扰动后的偏航振荡。'
                      f'</p></div></div>')
    candidate_path = ROOT / "results/steerbot_legacy/steerbot_negative_candidate/acceptance.json"
    if kind == "steerbot" and candidate_path.exists():
        alt = json.loads(candidate_path.read_text())
        mirrored = json.loads((ROOT / "results/steerbot_legacy/steerbot_negative_mirror/acceptance.json").read_text())
        holdout = json.loads((ROOT / "results/steerbot_legacy/steerbot_negative_holdout/acceptance.json").read_text())
        alternate = (f'<h2>强冲击压力测试</h2><div class="lead"><div>'
                     f'<video controls muted playsinline preload="metadata" '
                     f'src="../results/steerbot_legacy/steerbot_negative_candidate/rollout.mp4"></video>'
                     f'</div><div><p>转向轴在轮轴后约 0.145 m，主轮半宽 4 mm，'
                     f'车身主质量 0.737 kg；对称直行工作点 LQR，单个转向电机。'
                     f'同一负拖距候选，冲击升至 0.25 m/s 与 0.25 rad/s。'
                     f'正向五种子含 DR 通过 {alt["pass_rate"]:.0%}，'
                     f'速度 RMSE {value(alt["mean_steady_speed_rmse"], " m/s")}，'
                     f'转弯偏航 RMSE {value(alt["mean_turn_yaw_rmse"], " rad/s")}，'
                     f'侧向速度 RMSE {value(alt["mean_turn_lateral_rmse"], " m/s")}。'
                     f'留出 5–14 号种子通过 '
                     f'{sum(r["pass_2m_turn"] for r in holdout["individual"])}/10；'
                     f'镜像转弯顺序通过 '
                     f'{sum(r["pass_2m_turn"] for r in mirrored["individual"])}/5，'
                     f'偏航 RMSE {value(mirrored["mean_turn_yaw_rmse"], " rad/s")}。'
                     f'强冲击下未达到双向操纵验收。'
                     f'</p></div></div>' + alternate)
    original_steer_path = ROOT / "results/steerbot_legacy/steerbot/acceptance.json"
    if kind == "steerbot" and original_steer_path.exists():
        original = json.loads(original_steer_path.read_text())
        alternate += (f'<h2>原正拖距模型</h2><div class="lead"><div>'
                      f'<video controls muted playsinline preload="metadata" '
                      f'src="../results/steerbot_legacy/steerbot/rollout.mp4"></video>'
                      f'</div><div><p>原始转向轴在轮轴前 0.12 m，转向电机 0.6 N·m、'
                      f'轮胎半宽 0.016 m。五种子强冲击/DR 通过 '
                      f'{original["pass_rate"]:.0%}，偏航 RMSE '
                      f'{value(original["mean_turn_yaw_rmse"], " rad/s")}，'
                      f'侧向速度 RMSE {value(original["mean_turn_lateral_rmse"], " m/s")}。'
                      f'该结果不能代表上方负拖距构型。</p></div></div>')
    tilted_path = ROOT / 'results/wheelbot_legacy/wheelbot_tilt040' / "acceptance.json"
    if kind == "wheelbot" and tilted_path.exists():
        alt = json.loads(tilted_path.read_text())
        alternate = (f'<h2>倾斜飞轮轴对照</h2><div class="lead"><div>'
                     f'<video controls muted playsinline preload="metadata" '
                     f'src="../results/wheelbot_legacy/wheelbot_tilt040/rollout.mp4"></video>'
                     f'</div><div><p>将原水平飞轮轴倾斜 {value(alt["tilt"], " rad")}，'
                     f'保留两电机和单轮接触；起始飞轮预旋转 110 rad/s。'
                     f'这是探索性变型，不是已发表 Wheelbot 原构型。'
                     f'差分进化状态反馈的五种子扰动/DR 验收：完整通过 {alt["pass_rate"]:.0%}，'
                     f'速度 RMSE {value(alt["mean_steady_speed_rmse"], " m/s")}，'
                     f'偏航 RMSE {value(alt["mean_turn_yaw_rmse"], " rad/s")}，'
                     f'侧向速度 RMSE {value(alt["mean_turn_lateral_rmse"], " m/s")}。'
                     f'单独的 GPU PPO 阶段 6 在抽样 checkpoint 的名义回放中最佳偏航 RMSE '
                     f'约 0.423 rad/s，没有计入上述反馈验收；不同指令幅值和镜像顺序也未达到泛化要求。'
                     f'</p></div></div>')
    original_path = ROOT / 'results/wheelbot_legacy/wheelbot_original' / "acceptance.json"
    if kind == "wheelbot" and original_path.exists():
        alt = json.loads(original_path.read_text())
        alternate += (f'<h2>原始宽胎/低力矩对照</h2><div class="lead"><div>'
                      f'<video controls muted playsinline preload="metadata" '
                      f'src="../results/wheelbot_legacy/wheelbot_original/rollout.mp4"></video>'
                      f'</div><div><p>原模型轮胎半宽 0.016 m、飞轮电机 0.48 N·m。'
                      f'直行基线在五种子扰动/DR 的整段转向验收中通过 {alt["pass_rate"]:.0%}，'
                      f'速度 RMSE {value(alt["mean_steady_speed_rmse"], " m/s")}，'
                      f'偏航 RMSE {value(alt["mean_turn_yaw_rmse"], " rad/s")}。'
                      f'该基线未闭环跟踪偏航指令，不能把窄胎候选的通过结果'
                      f'归于原宽胎几何。</p></div></div>')
    if kind == "wheelbot":
        mirror_path = ROOT / "results/wheelbot_legacy/wheelbot_thin_mirror/acceptance.json"
        holdout_path = ROOT / "results/wheelbot_legacy/wheelbot_holdout/acceptance.json"
        strong_mirror_ext_path = ROOT / "results/wheelbot_legacy/wheelbot_mirror_holdout/acceptance.json"
        fresh_paths = [legacy_result_dir(name) / "acceptance.json" for name in
                       ("wheelbot_fresh_holdout", "wheelbot_fresh_mirror_holdout")]
        moderate_paths = [legacy_result_dir(name) / "acceptance.json" for name in
                          ("wheelbot_moderate_mirror", "wheelbot_moderate_holdout",
                           "wheelbot_moderate_mirror_holdout")]
        if mirror_path.exists() and holdout_path.exists():
            mirror = json.loads(mirror_path.read_text())
            holdout = json.loads(holdout_path.read_text())
            moderate = ""
            if all(path.exists() for path in moderate_paths):
                m, h, hm = [json.loads(path.read_text()) for path in moderate_paths]
                moderate = (f'中等冲击 0.15 m/s + 0.15 rad/s：镜像 0–4 号 '
                            f'{sum(r["pass_2m_turn"] for r in m["individual"])}/5；'
                            f'留出 5–14 号正向 '
                            f'{sum(r["pass_2m_turn"] for r in h["individual"])}/10、镜像 '
                            f'{sum(r["pass_2m_turn"] for r in hm["individual"])}/10。')
            strong_mirror_ext = ""
            if strong_mirror_ext_path.exists():
                sm = json.loads(strong_mirror_ext_path.read_text())
                strong_mirror_ext = (f'；镜像 5–14 号 '
                                     f'{sum(r["pass_2m_turn"] for r in sm["individual"])}/10')
            fresh = ""
            if all(path.exists() for path in fresh_paths):
                f, fm = [json.loads(path.read_text()) for path in fresh_paths]
                fresh = (f'固定参数后，未参与选择的 15–29 号种子中等冲击正向 '
                         f'{sum(r["pass_2m_turn"] for r in f["individual"])}/15、镜像 '
                         f'{sum(r["pass_2m_turn"] for r in fm["individual"])}/15；'
                         '正向 29 号回正偏航超标。')
            alternate = (f'<h2>扩展鲁棒性</h2><p>强冲击相反转向顺序的 0–4 号种子通过 '
                         f'{sum(r["pass_2m_turn"] for r in mirror["individual"])}/5；'
                         f'扩展的 5–14 号正向种子通过 '
                         f'{sum(r["pass_2m_turn"] for r in holdout["individual"])}/10'
                         f'{strong_mirror_ext}。{moderate}{fresh}'
                         f'飞轮预旋 {acceptance["rotor_speed"]:.0f} rad/s；'
                         '2 m/s 直行 20 秒测试不跌倒，'
                         '但从起步持续单向 ±0.35 rad/s 转弯在测试中约 10–17 秒跌倒。'
                         f'主序列种子 0 在 3 秒后的最大瞬时偏航角速度达到 '
                         f'{max(abs(s["yaw_rate"]) for s in trajectory if s["t"] > 3):.2f} rad/s；'
                         '分段 RMSE 通过不表示瞬时响应始终平顺。'
                         '这些结果不应外推至无限时长、任意手柄轨迹或真实硬件。</p>' + alternate)
    references = ('''<h2>参考与模型差异</h2><p>Hose、Weisgerber、Trimpe，
<a href="https://arxiv.org/abs/2502.04582">The Mini Wheelbot (ICRA 2025)</a>；
<a href="https://github.com/wheelbot/Mini-Wheelbot">官方动力学与硬件仓库</a>；
<a href="https://github.com/hshose/mini-wheelbot-ampc">偏航近似 MPC 代码</a>。
官方 Mini Wheelbot 动力学参数中主轮半径为 0.032 m，机身质量 0.312 kg，
并以单点滚动约束及单独的偏航接触摩擦建模；本页 0.12 m 刚性圆柱轮胎、0.68 kg 机身
是独立概念模型，并非论文参数复现。因此公开 MPC/模仿学习结果不能直接当成本模型的验收结论。</p>'''
                  if kind == "wheelbot" else "")
    rows = "".join(
        f"<tr><td>{i}</td><td>{r['duration']:.2f} s</td><td>{value(r['steady_speed_rmse'], ' m/s')}</td>"
        f"<td>{value(r.get('straight_yaw_rmse'), ' rad/s')}</td>"
        f"<td>{value(r['turn_yaw_rmse'], ' rad/s')}</td>"
        f"<td>{value(r.get('recovery_yaw_rmse'), ' rad/s')}</td>"
        f"<td>{value(r.get('turn_lateral_rmse'), ' m/s')}</td><td>{value(r['max_abs_roll'], ' rad')}</td>"
        f"<td>{'通过' if r['pass_2m_turn'] else '未通过'}</td></tr>"
        for i, r in enumerate(measured)
    )
    data = json.dumps(trajectory, ensure_ascii=False)
    kick_magnitude = .25 * acceptance.get("kick_scale", 1.)
    robustness_note = ""
    steer_details = ""
    demo_note = ""
    if kind == "steerbot":
        demo_note = ("<p class=\"controller-note\"><strong>本页主演示使用 LQR，不是强化学习策略。</strong>"
                     "首页验收、机身 +y 外力和缓丘视频均运行 <code>--controller lqr</code>。"
                     "下方历史对照视频的控制器各不相同：长拖距对照使用 PPO checkpoint，"
                     "其余注明优化反馈的对照不是 PPO。训练过 PPO 不等于该策略已通过本构型验收。</p>")
        force_result = json.loads((ROOT / "results/steerbot_legacy/steerbot_side_force/acceptance.json").read_text())
        force_trace = json.loads((ROOT / "results/steerbot_legacy/steerbot_side_force/seed_0.json").read_text())["trajectory"]
        hill_result = json.loads((ROOT / "results/steerbot_legacy/steerbot_hill_accept/acceptance.json").read_text())
        hill_trace = json.loads((ROOT / "results/steerbot_legacy/steerbot_hill.json").read_text())["trajectory"]
        hill_turns = [json.loads((ROOT / f"results/steerbot_legacy/steerbot_hill_{side}.json").read_text())["summary"]
                      for side in ("left", "right")]
        mirrored = json.loads((ROOT / "results/steerbot_legacy/steerbot_moderate_mirror/acceptance.json").read_text())
        held = json.loads((ROOT / "results/steerbot_legacy/steerbot_moderate_holdout/acceptance.json").read_text())
        held_mirror = json.loads((ROOT / "results/steerbot_legacy/steerbot_moderate_mirror_holdout/acceptance.json").read_text())
        strong = json.loads(candidate_path.read_text())
        strong_mirror = json.loads((ROOT / "results/steerbot_legacy/steerbot_negative_mirror/acceptance.json").read_text())
        fresh = ""
        fresh_paths = [legacy_result_dir(name) / "acceptance.json" for name in
                       ("steerbot_fresh_holdout", "steerbot_fresh_mirror_holdout")]
        if all(path.exists() for path in fresh_paths):
            f, fm = [json.loads(path.read_text()) for path in fresh_paths]
            fresh = (f'固定参数后的 15–29 号中等冲击留出种子正向 '
                     f'{sum(r["pass_2m_turn"] for r in f["individual"])}/15、镜像 '
                     f'{sum(r["pass_2m_turn"] for r in fm["individual"])}/15。')
        robustness_note = (f'<p class="warning">中等冲击：镜像 0–4 号种子 '
                           f'{sum(r["pass_2m_turn"] for r in mirrored["individual"])}/5；'
                           f'留出 5–14 号种子正向 '
                           f'{sum(r["pass_2m_turn"] for r in held["individual"])}/10、镜像 '
                           f'{sum(r["pass_2m_turn"] for r in held_mirror["individual"])}/10。'
                           f'0.25 m/s + 0.25 rad/s 强冲击：正向 '
                           f'{sum(r["pass_2m_turn"] for r in strong["individual"])}/5，镜像 '
                           f'{sum(r["pass_2m_turn"] for r in strong_mirror["individual"])}/5；'
                           f'不能宣称强扰动双向通过。{fresh}额外固定指令测试：2 m/s、'
                           f'持续 ±0.35 rad/s 转弯各 20 秒，种子 0 的中等冲击回放均不跌倒；'
                           f'未覆盖任意手柄轨迹。</p>')
        tuning = json.loads((ROOT / "steerbot/negative_candidate.json").read_text())["params"]
        speed_gains = json.loads((ROOT / "steerbot/optimized_controller.json").read_text())["params"]
        steer_details = f'''
<h2>构件识别</h2>
<p><a href="validated_model.xml">打开验收构型 XML</a>。该文件将 <code>model.xml</code> 和
<code>negative_candidate.json</code> 的几何参数合并，方便直接查看；仿真程序仍从原始 XML 加载并施加同一组覆盖参数。</p>
<div class="table-wrap"><table><thead><tr><th>画面颜色</th><th>MJCF 名称</th><th>结构与作用</th></tr></thead><tbody>
<tr><td>蓝色长方体</td><td><code>body</code> / <code>base</code></td><td>主机身，蓝色部分质量 {acceptance['body_mass']:.3f} kg；自由关节提供空间位置和姿态。俯仰与横滚来自整个机身运动，不是独立电机。</td></tr>
<tr><td>黄色小方块</td><td><code>front</code></td><td>位于机身 +x 一侧的前向标记，质量 0.015 kg；不是摄像头、传感器或第三个电机。</td></tr>
<tr><td>黑色薄圆轮</td><td><code>tire</code> / <code>drive_wheel</code></td><td>唯一接地轮，半径 0.12 m、半宽 0.004 m、质量 0.30 kg；绕局部 y 轴的 <code>drive</code> 关节由 <code>drive_motor</code> 驱动，力矩上限 2.4 N·m。</td></tr>
<tr><td>轮心银色圆盘</td><td><code>hub</code></td><td>轮毂可视及惯性部件，质量 0.012 kg，不参与碰撞。</td></tr>
<tr><td>轮旁白/浅灰圆柱</td><td><code>steering_head</code> / <code>fork</code></td><td>转向轴位置标记和转向头质量 {acceptance['head_mass']:.4f} kg；不接地。竖直 <code>steer</code> 关节限位 ±0.8 rad，由第二个电机施加至多 2.4 N·m。转向轴在轮轴后方 {abs(acceptance['trail']):.3f} m；中间的刚性连接没有画出实体叉臂，因此画面看似分离。</td></tr>
</tbody></table></div>
<p>机身 +x 为前方、+y 为左、+z 向上。模型有一个机身自由关节（6 DoF）和两个受控转动关节（驱动轮、转向）；只有轮胎与地面接触。黑轮自身的偏航不是另一个关节，而是转向和接触力带动整个机身改变航向。</p>
<h2>控制器实现</h2>
<p><a href="lqr_explained.html">独立图解：从动力学到 LQR 控制器</a>。下面是简要实现摘要，详细推导、直观示意和实验边界请见独立页面。</p>
<p>验收使用 <code>LqrSteerController</code>，不是 PPO。下图分成两个闭环；箭头指信号/物理作用方向，机身状态通过理想仿真观测反馈。轮驱动控制前后俯仰与速度，转向电机同时承担左右平衡与偏航，<strong>没有第三个横滚电机</strong>。</p>
<div class="lane"><div class="lane-title">前进 / 俯仰</div><div class="flow"><div>手柄速度目标<br><small>v<sub>cmd</sub></small></div><span>→</span><div>限速变化 + 速度误差<br><small>产生目标俯仰角</small></div><span>→</span><div>俯仰 PD<br><small>轮驱动力矩</small></div><span>→</span><div>车轮与机身<br><small>前向速度 v<sub>x</sub>、俯仰角</small></div></div><div class="feedback">↳ 测量 v<sub>x</sub> / 俯仰，反馈至前两级</div></div>
<div class="lane lateral-lane"><div class="lane-title">侧向平衡 / 航向</div><div class="flow"><div>手柄偏航目标<br><small>r<sub>cmd</sub></small></div><span>→</span><div>低通 + 前馈<br><small>期望转弯状态</small></div><span>→</span><div>LQR 状态反馈 + 积分<br><small>转向力矩</small></div><span>→</span><div>轮地接触<br><small>侧向力 F<sub>y</sub></small></div><span>→</span><div>机身<br><small>横滚、侧滑、偏航</small></div></div><div class="feedback">↳ 测量横滚角/速度、侧向速度、偏航速度、转向角/速度，全部反馈至 LQR；世界 +y 外力直接加在机身，不进入电机</div></div>
<h3>为什么仅靠转向可以平衡横滚？</h3>
<ol class="steps"><li>机身质心在轮地接触点上方，失去侧向支撑时，微小横滚偏差会被重力放大。转向电机<strong>不能直接</strong>给横滚关节加力矩。</li>
<li>前进时改变轮的朝向，轮地接触约束产生横向力和偏航运动；该接触力作用点在质心下方，具有横滚力臂，其横滚力矩量级约为 <code>h |F_y|</code>。力矩方向取决于力的方向；完整运动还受重力、轮胎滑移和转向几何影响。这里的 <code>h</code> 是质心到接触点的高度。</li>
<li>于是同一个转向动作会改变横滚、侧向速度和偏航。控制器在不同误差下选择力矩的方向和幅度，相当于借<strong>行进中的地面反力</strong>间接平衡左右；转向不是独立于平衡的“纯航向自由度”。零速时这一途径大幅退化，当前控制器不支持零速站立。</li></ol>
<h3>从运动方程到当前反馈</h3>
<ol class="steps"><li>先闭合前进环：速度指令以 2 m/s² 斜率变化；速度误差生成限幅 ±0.6 rad 的目标俯仰角，再用俯仰角和俯仰角速度算轮电机命令，最终限制到 ±2.4 N·m。实际增益分别为 {speed_gains[0]:.3f}（速度误差到俯仰）、{speed_gains[1]:.3f}（俯仰误差）和 {speed_gains[2]:.3f}（俯仰角速度）。</li>
<li>在这个前进环运行且速度约 2 m/s 时，对转向电机做很小的正负输入，并分别轻微改变状态，用 MuJoCo 有限差分求 20 ms 后的状态变化，得到局部模型 <code>下一步状态 ≈ A·当前状态 + B·转向命令</code>。六个状态是<strong>横滚角、横滚角速度、机身侧向速度、偏航角速度、转向角、转向角速度</strong>；该局部模型的可控矩阵秩为 6。</li>
<li>给横滚、侧滑、偏航误差和转向动作代价赋权，用离散 Riccati 方程求一个 6 状态反馈增益；再增加偏航误差的有界积分，防止持续偏航误差。权重矩阵对角为 <code>[{tuning[0]:.2f}, 2, {tuning[6]:.2f}, {tuning[1]:.2f}, 2, 0.2, {tuning[2]:.2f}]</code>，电机代价为 <code>{tuning[3]:.2f}</code>。这些是局部 LQR 的权重，<strong>不是额外传感器或执行器</strong>。</li>
<li>对手柄转弯目标先低通。通过模型求稳态转向输入及对应的横滚/侧向/转向角状态作为参考值。电机输出可读成：<code>稳态转弯前馈 − 状态偏差反馈 − 偏航误差积分</code>，最终限幅到 ±1；转向关节的实际最大力矩为 2.4 N·m。每 2 ms 用新观测执行一次。</li></ol>
<p>这是一套 2 m/s 附近的局部反馈，不是全速域的非线性稳定性证明。偏航角速度归零只表示不再转动；受到横向力后航向角可能已经改变，若想回到世界坐标的原路线还需外层航向/路径闭环。仿真使用无噪声的理想状态，未验证硬件传感和延迟。</p>
<h2>机身 +y 外力实验</h2>
<div class="media-pair"><video controls muted playsinline preload="metadata" src="../results/steerbot_legacy/steerbot_side_force/rollout.mp4"></video><div><p>在<strong>世界坐标 +y</strong> 方向，将 {force_result['side_push_force']:.1f} N 直接施加到机身 <code>base</code> 质心附近，8.5–8.7 s 持续 0.2 s，冲量约 0.2 N·s。视频左上角只在施力帧显示标记；此实验<strong>关闭</strong>了原有的侧向速度/横滚角速度瞬时冲击。直行序列 0–2 s 为 1 m/s、2–11 s 为 2 m/s、11–15 s 为 1.5 m/s。</p><p>质量 ±15%、轮胎摩擦 1.1–1.9 的五个种子均行驶满 15 s，无跌倒；峰值横滚 {max(r['max_abs_roll'] for r in force_result['individual']):.3f} rad。包含受力瞬态的严格速度/偏航/侧滑综合阈值通过 {sum(r['pass_2m_turn'] for r in force_result['individual'])}/5，主要超标是瞬时偏航：转弯评价窗口（实际为直行）平均偏航 RMSE {force_result['mean_turn_yaw_rmse']:.3f} rad/s。种子 0 在 10–11 s 横滚与偏航角速度绝对值分别不超过 {max(abs(s['roll']) for s in force_trace if 10<s['t']<11):.3f} rad、{max(abs(s['yaw_rate']) for s in force_trace if 10<s['t']<11):.3f} rad/s。横滚恢复并不意味着原有横向位置恢复。</p></div></div>
<h2>手柄越过缓丘</h2>
<div class="media-pair"><video controls muted playsinline preload="metadata" src="../results/steerbot_legacy/steerbot_hill.mp4"></video><div><p>独立 <a href="hill_model.xml">缓丘 MJCF</a> 使用同一负拖距机器人的关节、质量和控制器，仅将平地换成高度场；起点平地，沿世界 x=2–6 m 为 <code>sin²</code> 连续凸坡，峰高 0.06 m（最大坡度约 2.7°），x=4 m 达峰。两侧不碰撞的小球标出坡起、坡顶和坡尾；地形横向范围 ±100 m。</p><p>无外力、直行指令的五个 DR 种子均通过 15 s 越丘测试，速度稳态 RMSE 均值 {hill_result['mean_steady_speed_rmse']:.3f} m/s，横滚峰值 {max(r['max_abs_roll'] for r in hill_result['individual']):.3f} rad；视频为名义 2 m/s 指令、{hill_trace[-1]['t']:.0f} s 回放。额外的种子 0、2 m/s 持续左/右各 0.1 rad/s 转向 15 s 回放均未跌倒，偏航跟踪 RMSE 分别为 {hill_turns[0]['turn_yaw_rmse']:.3f} / {hill_turns[1]['turn_yaw_rmse']:.3f} rad/s。结果<strong>未</strong>覆盖坡上急转、任意手柄动作或加外力的组合。</p></div></div>
<p><strong>手柄操纵缓丘</strong>：在项目根目录、有桌面图形环境的终端运行以下命令（视频本身是固定指令回放，不是交互界面）。</p>
<pre><code>PY=/home/legged-ma/miniforge3/envs/zhu_mjlab/bin/python
$PY sim.py steerbot --model steerbot/hill_model.xml --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2 --turn 0 --duration 60</code></pre>
<h2>打开模型与手柄操纵</h2>
<p>在有桌面图形环境的终端，从项目根目录执行：</p>
<pre><code>cd {html.escape(str(ROOT))}
PY=/home/legged-ma/miniforge3/envs/zhu_mjlab/bin/python
$PY -m mujoco.viewer --mjcf steerbot/validated_model.xml
$PY sim.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2 --turn 0 --duration 60
$PY -m mujoco.viewer --mjcf steerbot/hill_model.xml
$PY sim.py steerbot --model steerbot/hill_model.xml --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2 --turn 0 --duration 60</code></pre>
<p>第一条只打开合并后的平地 MJCF 供检查；它不会加载控制器，直接播放会倒。第二条运行平地闭环仿真，可插入标准手柄 1。它从 <code>model.xml</code> 加载并按调参 JSON 覆盖几何，与上方独立 XML 等价；不要省略 <code>--controller lqr</code> 和 <code>--lqr-tuning</code>，否则会运行原正拖距基线。</p>
<p>第三条单独打开缓丘几何，同样没有控制器。第四条加载缓丘和控制器，可以用手柄或键盘越丘；丘位于前方 2–6 m，默认持续 60 s，关闭窗口可提前退出。该命令即使场景 XML 已是验收形态，仍用参数 JSON 检查几何并加载匹配的控制增益，不会重复移动机身质心。</p>
<div class="table-wrap"><table><thead><tr><th>输入</th><th>作用</th></tr></thead><tbody>
<tr><td>左摇杆上下</td><td>前向速度：中位 0.6 m/s，上推至 2.2 m/s，下推仍为 0.6 m/s；约上推 87.5% 对应 2.0 m/s。</td></tr>
<tr><td>左摇杆左右</td><td>目标偏航角速度 −0.35 至 +0.35 rad/s；左推为正（向左转），右推为负。</td></tr>
<tr><td>键盘 ↑ / ↓</td><td>每次增加/减少 0.2 m/s，范围 0.6–2.2 m/s。</td></tr>
<tr><td>键盘 ← / →</td><td>每次增加/减少 0.1 rad/s 偏航指令，限幅 ±0.35 rad/s。</td></tr>
<tr><td>空格 / R</td><td>空格把转向指令归零；R 重置机器人位置、速度和控制器内部状态，但保留当前手动指令。</td></tr>
</tbody></table></div>
<p>手柄一旦连接，其左摇杆每个物理步覆盖键盘设定的速度/转向；拔掉手柄可单独使用键盘。启动时模型给予 0.8 m/s 前进初速度和匹配的轮速，不是零速站立控制。<code>--test</code> 模式使用固定验收序列，会忽略手动速度与转向输入。</p>
<p>复现实测中等冲击的五种子回放：<code>$PY accept.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --kick-scale .6 --seeds 5 --video --output-dir results/steerbot_legacy/steerbot_moderate</code>。该命令在无桌面时使用 EGL 录制视频，不能用作实时手柄模式。</p>
<p>复现外力和坡道：<code>$PY accept.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --no-disturbance --test-turn-amplitude 0 --side-push-force 1 --side-push-duration .2 --seeds 5 --video --output-dir results/steerbot_legacy/steerbot_side_force</code>；<code>$PY accept.py steerbot --model steerbot/hill_model.xml --controller lqr --lqr-tuning steerbot/negative_candidate.json --no-disturbance --test-turn-amplitude 0 --seeds 5 --output-dir results/steerbot_legacy/steerbot_hill_accept</code>。手动场景若也要施力，可在第四条命令加 <code>--side-push-force 1 --side-push-start 8.5 --side-push-duration .2</code>。</p>
'''
    training_heading = ("历史强化学习尝试（非本页主演示）"
                        if kind == "steerbot" else "训练设置")
    training_lead = ("此前确实在 GPU 上训练过 PPO，并保存了策略 checkpoint；"
                     "但本页主验收、外力与缓丘演示均运行 LQR。PPO 候选尚未达到"
                     "本页负拖距构型的完整双向操纵验收，不能把上述结果归功于 PPO。"
                     if kind == "steerbot" else
                     "以下 PPO 课程在 GPU 运行；本次验收控制器单列，不把优化反馈结果冒充强化学习结果。")
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{info['title']} | MuJoCo 实验报告</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, "Noto Sans CJK SC", sans-serif; background: #f4f7f8; color: #17272c; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; }} header {{ background: #143c44; color: #f5fcfd; padding: 18px max(20px, calc((100% - 1280px)/2)); display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
header strong {{ font-size: 17px; }} nav {{ display:flex; gap:18px; }} a {{ color: inherit; }} nav a {{ opacity:.82; text-decoration:none; }}
main {{ max-width: 1280px; padding: 26px 20px 80px; margin: auto; }} h1 {{ font-size: 30px; line-height: 1.2; margin: 0 0 10px; }} h2 {{ font-size: 20px; border-bottom: 1px solid #d3dcde; padding-bottom: 8px; margin: 32px 0 14px; }} p {{ line-height: 1.68; max-width: 88ch; }}
.eyebrow {{ color: #53737a; font-size: 12px; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }}
.lead {{ display:grid; grid-template-columns: minmax(0, 1.3fr) minmax(280px, .7fr); gap: 22px; align-items: start; }}
video {{ display:block; background: #111; width:100%; aspect-ratio: 960 / 544; border-radius: 4px; }}
.overview {{ border-left: 4px solid #d46e45; padding: 0 18px; }} .overview.pass {{ border-color: #168e72; }}
.verdict {{ font-size: 21px; font-weight: 750; color: #aa412e; }} .overview.pass .verdict {{ color: #116a55; }}
.metrics {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:24px; }} .metric {{ border-top:1px solid #cbd7da; padding-top:12px; }} .metric span {{ display:block; font-size:12px; color:#547079; }} .metric b {{ display:block; font-size:20px; font-variant-numeric: tabular-nums; margin-top:4px; }}
.charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }} .plot {{ min-width:0; }} canvas {{ background:#fff; width:100%; height:240px; border:1px solid #d5e0e2; border-radius:4px; }} .plot figcaption {{ font-size:13px; color:#48626a; margin-top:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:30px; }} dl {{ display:grid; grid-template-columns:145px 1fr; gap:9px 14px; line-height:1.5; }} dt {{ color:#49676f; }} dd {{ margin:0; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }} th,td {{ text-align:left; padding:11px 8px; border-bottom:1px solid #d4dfe0; }} th {{ background:#e6eef0; }} .table-wrap {{ overflow-x:auto; }} code {{ overflow-wrap:anywhere; background:#e5ecee; padding:2px 4px; }}
pre {{ max-width:100%; overflow-x:auto; padding:16px; background:#e5ecee; border:1px solid #d4dfe0; border-radius:4px; font-size:13px; line-height:1.55; }} pre code {{ background:transparent; padding:0; overflow-wrap:normal; }} h3 {{ font-size:16px; margin:22px 0 8px; }}
.warning {{ padding:12px 16px; border-left:3px solid #d46e45; background:#f6ede9; }}
.controller-note {{ border-left:3px solid #168e72; background:#e9f2ee; padding:12px 16px; max-width:none; }}
.lane {{ border-top:2px solid #168e72; padding:12px 0 10px; margin:22px 0; }} .lateral-lane {{ border-color:#c25b37; }}
.lane-title {{ font-weight:750; margin-bottom:12px; }} .flow {{ display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; }} .flow div {{ border:1px solid #c9d7d9; background:#fff; padding:13px; min-width:142px; flex:1; line-height:1.5; }} .flow small {{ color:#49676f; }} .flow span {{ align-self:center; color:#647b80; font-size:24px; }} .feedback {{ color:#49676f; font-size:13px; margin-top:10px; line-height:1.6; }}
.steps {{ padding-left:24px; line-height:1.7; max-width:100ch; }} .steps li {{ margin-bottom:9px; }} .media-pair {{ display:grid; grid-template-columns:minmax(0,1.2fr) minmax(280px,1fr); gap:22px; align-items:start; }} .media-pair p:first-child {{ margin-top:0; }}
@media (max-width: 840px) {{ .lead,.grid,.charts,.media-pair {{ grid-template-columns:1fr; }} header {{ flex-wrap:wrap; }} h1 {{ font-size:25px; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; }} pre code {{ overflow-wrap:anywhere; }} .flow {{ display:grid; grid-template-columns:1fr; }} .flow span {{ justify-self:center; transform:rotate(90deg); line-height:1; }} }}
</style></head><body>
<header><strong>独轮车 · MuJoCo 实验记录</strong><nav><a href="../wheelbot/report.html">飞轮构型</a><a href="../steerbot/report.html">转向构型</a></nav></header>
<main><div class="eyebrow">{kind.upper()} / 两执行器 / 单轮接触</div><h1>{info['title']}</h1><p>{info['subtitle']}</p>
<section class="lead"><div><video controls muted playsinline preload="metadata" src="../{results_relative}/rollout.mp4"></video><p class="eyebrow">种子 0：{recording_label}；2 m/s 阶跃、左右转与扰动按实际持续时间执行</p></div>
<div class="overview {status}"><div class="eyebrow">验收结论 · 冲击 {kick_magnitude:.2f} m/s + {kick_magnitude:.2f} rad/s</div><div class="verdict">{verdict}</div><p>通过 {sum(r['pass_2m_turn'] for r in measured)} / {len(measured)} 个随机种子。目标是完整 15 秒不跌倒、稳态速度 RMSE &lt; 0.35 m/s、直行/转弯/回正偏航 RMSE 各 &lt; 0.15 rad/s、机身侧向速度 RMSE &lt; 0.35 m/s。</p>
<div class="metrics"><div class="metric"><span>稳态速度 RMSE · 平均</span><b>{value(acceptance['mean_steady_speed_rmse'], ' m/s')}</b></div><div class="metric"><span>直行偏航 RMSE · 平均</span><b>{value(acceptance.get('mean_straight_yaw_rmse'), ' rad/s')}</b></div><div class="metric"><span>转向偏航 RMSE · 平均</span><b>{value(acceptance['mean_turn_yaw_rmse'], ' rad/s')}</b></div><div class="metric"><span>回正偏航 RMSE · 平均</span><b>{value(acceptance.get('mean_recovery_yaw_rmse'), ' rad/s')}</b></div><div class="metric"><span>转向侧滑 RMSE · 平均</span><b>{value(acceptance.get('mean_turn_lateral_rmse'), ' m/s')}</b></div></div></div></section>
{demo_note}
{robustness_note}
<h2>时间序列</h2><div class="charts"><figure class="plot"><canvas id="speed"></canvas><figcaption>机身前向速度与手柄指令（m/s）</figcaption></figure><figure class="plot"><canvas id="yaw"></canvas><figcaption>机身偏航角速度与转向指令（rad/s）</figcaption></figure><figure class="plot"><canvas id="lateral"></canvas><figcaption>机身侧向速度（m/s），指令为 0</figcaption></figure></div>
<h2>建模与环境</h2><div class="grid"><div><dl><dt>机身与自由度</dt><dd>浮动机身 6 DoF；驱动轮、第二执行器各 1 个转动关节。前向 x、左向 y、竖直 z。</dd><dt>MJCF</dt><dd><code>{html.escape(Path(acceptance['model_path']).name)}</code>，实验几何参数见下文。</dd><dt>主轮</dt><dd>{mainwheel_description}</dd><dt>质心配置</dt><dd>{morphology}</dd><dt>第二执行器</dt><dd>{second_description}</dd><dt>轮地摩擦</dt><dd>MuJoCo 摩擦系数 1.6（滑动）；验收逐种子采样 1.1–1.9。</dd><dt>积分与控制</dt><dd>物理积分步长 0.002 s；本次控制每秒更新 {frequency} 次；误差按物理步统计。</dd></dl></div><div><p>{mechanism}</p><p class="warning">限制：圆柱轮胎和点/线接触是概念验证模型；未包含轮胎形变、电机温升、传感噪声或硬件执行器时延。训练成绩不能直接推断实物性能。</p></div></div>
{steer_details}
<h2>{training_heading}</h2><p>{training_lead}</p><div class="grid"><div><dl><dt>运行时</dt><dd>mjlab 1.2.0 + MuJoCo Warp 3.6.0 + RSL-RL 5.0.1，RTX 5090。</dd><dt>课程阶段 1</dt><dd>0.6–1.3 m/s，直线行驶与平衡。</dd><dt>课程阶段 2</dt><dd>1.3–2.2 m/s，偏航 ±0.2 rad/s；每 3–6 s 随机速度扰动。</dd><dt>课程阶段 3</dt><dd>1.2–2.3 m/s，偏航 ±0.65 rad/s；增加偏航奖励。</dd><dt>课程阶段 4</dt><dd>1.8–2.15 m/s，偏航 ±0.4 rad/s；聚焦精修。两版均未达到完整转向标准。</dd><dt>课程阶段 5+</dt><dd>{'转向版增加低侧向速度奖励；另行探索 0.50 m 长拖距的阶段 6–9 和 4 mm 窄胎/2.4 N·m 转向电机的独立课程。' if kind == 'steerbot' else '原构型未运行；另行探索倾斜飞轮轴的阶段 6。'}</dd></dl></div><div><dl><dt>PPO 结构</dt><dd>两层 128 单元 ELU MLP；输入机身速度、角速度、重力方向、关节速度、指令、上一动作{'' if kind == 'wheelbot' else '及转向角'}。</dd><dt>域随机化</dt><dd>机身质量 ±15%；轮胎滑动摩擦 1.1–1.9；俯仰/横滚初始姿态 ±0.025 rad。</dd><dt>验收扰动</dt><dd>8.5 s、11.5 s：车身侧向速度 ±{kick_magnitude:.2f} m/s、横滚角速度 ±{kick_magnitude:.2f} rad/s。</dd><dt>本页主控制器</dt><dd>{controller_description} <code>{checkpoint}</code>。{controller_artifact}</dd></dl></div></div>
<h2>逐种子验收</h2><div class="table-wrap"><table><thead><tr><th>种子</th><th>持续时间</th><th>速度 RMSE</th><th>直行偏航</th><th>转弯偏航</th><th>回正偏航</th><th>侧滑 RMSE</th><th>最大横滚</th><th>结论</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>验收序列：0–2 s 为 1 m/s；2–5 s 为 2 m/s 直线；5–8 s、8–11 s 分别为 2 m/s 下 ±0.35 rad/s 转向；11–15 s 为 1.5 m/s 回正。速度误差自 3 s 起统计，直行偏航在 3–4.8 s、转弯偏航与侧滑在 5.5–10.8 s、回正偏航在 12–15 s 统计。侧向速度约束避免只转动机身、实际却横向滑走。</p>
{alternate}
{references}
</main><script>
const samples = {data};
function plot(id, measured, target, range) {{
  const canvas=document.getElementById(id),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=240;
  canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);
  const left=44,top=16,right=12,bottom=32,iw=w-left-right,ih=h-top-bottom;
  const x=t=>left+iw*t/15,y=v=>top+ih*(range[1]-v)/(range[1]-range[0]);
  c.strokeStyle='#d8e2e4';c.lineWidth=1;c.font='11px system-ui';c.fillStyle='#617980';
  for(let t=0;t<=15;t+=3){{c.beginPath();c.moveTo(x(t),top);c.lineTo(x(t),top+ih);c.stroke();c.fillText(t+' s',x(t)-7,h-9);}}
  for(let i=0;i<=4;i++){{let v=range[0]+(range[1]-range[0])*i/4;c.beginPath();c.moveTo(left,y(v));c.lineTo(w-right,y(v));c.stroke();c.fillText(v.toFixed(1),4,y(v)+3);}}
  [[target,'#c25b37',true],[measured,'#137b80',false]].forEach(([field,color,dashed])=>{{c.beginPath();c.strokeStyle=color;c.lineWidth=2;c.setLineDash(dashed?[6,4]:[]);
    samples.forEach((s,i)=>{{let px=x(s.t),py=y(s[field]);i?c.lineTo(px,py):c.moveTo(px,py);}});c.stroke();}});
  c.setLineDash([]);c.fillStyle='#137b80';c.fillText('实测',left+8,top+12);c.fillStyle='#c25b37';c.fillText('指令',left+60,top+12);
}}
function draw() {{plot('speed','v','target',[-.2,2.8]);plot('yaw','yaw_rate','turn',[-.8,.8]);plot('lateral','vy','vy_ref',[-1.6,1.6]);}}
window.addEventListener('resize',draw);draw();
</script></body></html>'''
    destination = ROOT / kind / "report.html"
    destination.write_text(page)
    print(destination)


if __name__ == "__main__":
    for model in INFO:
        render(model)
