"""Render measured free-body LQR trajectories and a Chinese evidence report."""
from html import escape
from pathlib import Path
import json
import math
import numpy as np
import mujoco
from .validate_lqr import FOLDER


def chart(z):
    series = [("躯干 Roll", np.degrees(z[:, 1]), "#176b9a"),
              ("轮架相对躯干 Roll", np.degrees(z[:, 8]), "#d67a18")]
    extent = max(5., *(float(abs(v).max()) for _, v, _ in series))
    last = max(.1, z[-1, 0])
    paths = []
    for label, values, color in series:
        points = " ".join(f"{45+680*t/last:.2f},{120-90*v/extent:.2f}"
                          for t, v in zip(z[::3, 0], values[::3]))
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    return f'''<svg viewBox="0 0 760 250" role="img" aria-label="躯干与轮架角度曲线">
<path d="M45 25V215H730M45 120H730" fill="none" stroke="#bccbd6"/>
<text x="4" y="35">{extent:.0f}°</text><text x="4" y="215">−{extent:.0f}°</text>
<text x="45" y="240">0 s</text><text x="675" y="240">{last:.2f} s</text>{"".join(paths)}</svg>
<p class="small"><span style="color:#176b9a">蓝：躯干相对地面的 Roll</span>；
<span style="color:#d67a18">橙：轮架相对躯干的 Roll</span>。轮架世界姿态还包含躯干姿态。</p>'''


def main():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    report = json.loads((FOLDER/"evaluation.json").read_text(encoding="utf-8"))
    cases = report["cases"]
    nominal = {c["case"]: c for c in cases if c["timestep"] == .001}
    examples = [
        ("stand_roll_+0.1", "STAND | initial torso roll +0.1 deg", "静止，小偏差 +0.1°", 8.),
        ("stand_roll_+0.5", "STAND | initial torso roll +0.5 deg", "静止，偏差 +0.5°", 8.),
        ("stand_roll_+1.0", "STAND | initial torso roll +1.0 deg", "静止，偏差 +1°", 15.),
        ("start_+0.5", "FORWARD START | target +0.5 m/s", "从静止起步至 +0.5 m/s", 15.),
        ("moving_-1.0_roll_+0.1", "REVERSE | initial and target -1.0 m/s", "已有倒车初速度 −1 m/s", 15.),
    ]
    model = mujoco.MjModel.from_xml_path(str(FOLDER/"model.xml"))
    # Plane collision is infinite; extend only its displayed rectangle so a
    # long reverse rollout does not visually leave the floor.
    model.geom_size[model.geom("ground").id, :2] = 50.
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 1.1; camera.azimuth = 215.; camera.elevation = -14.
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    video_cases = []
    with mujoco.Renderer(model, height=640, width=960) as renderer:
        with imageio.get_writer(str(FOLDER/"lqr_feasibility.mp4"), fps=25, codec="libx264",
                                quality=7, macro_block_size=1) as writer:
            for index, (name, label, chinese, cutoff) in enumerate(examples, 1):
                result = nominal[name]
                saved = np.load(FOLDER/result["trajectory"])
                poses, rows = saved["qpos"], saved["frame_telemetry"]
                used = np.flatnonzero(rows[:, 0] <= cutoff)
                for j in used:
                    row = rows[j]
                    data.qpos[:] = poses[j]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.qpos[:3] + [0, 0, -.14]
                    renderer.update_scene(data, camera=camera)
                    canvas = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 960, 87), fill="#152333")
                    draw.text((18, 8), f"{index}/5  {label}", font=font, fill="white")
                    draw.text((18, 40), f"FREE BODY | LQR | t={row[0]:.2f}s | v={row[5]:+.2f} m/s",
                              font=small, fill="#bbd5e6")
                    draw.rectangle((0, 570, 960, 640), fill="#152333")
                    draw.text((18, 578), f"Torso roll {math.degrees(row[1]):+.2f} deg    Pitch {math.degrees(row[2]):+.2f} deg",
                              font=font, fill="#7dc9ee")
                    draw.text((18, 609), f"Wheel relative roll {math.degrees(row[8]):+.2f} deg    motors limited to 2.4 Nm",
                              font=small, fill="#ffd078")
                    writer.append_data(np.asarray(canvas))
                    if index == 2 and j == min(8, used[-1]):
                        canvas.save(FOLDER/"preview.png")
                failed_in_excerpt = not result["survived"] and result["duration_s"] <= cutoff
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((40, 280, 920, 365), fill="#8b2830" if failed_in_excerpt else "#1d504b")
                text = f"FAILED at {result['duration_s']:.2f}s | final frame held" if failed_in_excerpt else (
                    "SURVIVED 15s; torso-level criterion " + ("PASSED" if result["level_pass"] else "FAILED"))
                draw.text((60, 305), text, font=font, fill="white")
                for _ in range(40):
                    writer.append_data(np.asarray(canvas))
                video_cases.append(dict(case=name, display_name=chinese, excerpt_seconds=float(rows[used[-1], 0]),
                                        failed_in_excerpt=failed_in_excerpt, held_final_frame_seconds=1.6))
    (FOLDER/"video_manifest.json").write_text(json.dumps(video_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    grouped = []
    for timestep in (.001, .0005):
        for speed in (0., .5, 1., -.5, -1.):
            group = [c for c in cases if c["timestep"] == timestep and c["target_speed"] == speed
                     and c["case"].startswith(("stand_roll", "moving"))]
            grouped.append(f'<tr><td>{timestep*1000:g}</td><td>{speed:+g}</td><td>{sum(c["survived"] for c in group)}/{len(group)}</td>'
                           f'<td>{sum(c["level_pass"] for c in group)}/{len(group)}</td></tr>')
    detail = []
    for c in cases:
        fmt = lambda key: "—" if c[key] is None else f'{c[key]:.3f}'
        detail.append("<tr>"+"".join(f"<td>{cell}</td>" for cell in [
            escape(c["case"]), f'{c["timestep"]*1000:g}', fmt("duration_s"), "是" if c["survived"] else "否",
            "通过" if c["level_pass"] else "未通过", fmt("max_torso_roll_deg"), fmt("torso_roll_rms_deg"),
            fmt("max_torso_pitch_deg"), fmt("speed_rmse"), fmt("max_relative_wheel_roll_deg"),
            fmt("heading_final_deg")])+"</tr>")
    plots = []
    for name, _, label, _ in examples:
        z = np.load(FOLDER/nominal[name]["trajectory"])["telemetry"]
        plots.append(f"<section><h3>{label}</h3>{chart(z)}</section>")
    small_case, medium_case, large_case = [nominal[f"stand_roll_+{v}"] for v in ("0.1", "0.5", "1.0")]
    push_group = [c for c in cases if "push" in c["case"]]
    start_group = [c for c in cases if c["case"].startswith("start")]
    text = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>连杆摆轮 · 躯干水平 LQR 验证</title>
<style>*{{box-sizing:border-box}}body{{background:#eff3f7;color:#1a3043;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;margin:0}}
main{{max-width:1200px;margin:auto;padding:40px 24px}}h1{{font-size:32px;line-height:1.4}}h2{{margin-top:32px}}
section,.card{{background:white;padding:22px;border-radius:12px;border:1px solid #d8e1e8;margin:18px 0}}
.notice{{border-left:5px solid #c16e24;background:#fff3df;padding:20px}}.small{{font-size:14px;color:#526c7c}}
video,img,svg{{width:100%}}svg{{font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}
td,th{{padding:10px;border-bottom:1px solid #dae3e8;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}
.scroll{{overflow:auto}}a{{color:#126982}}code{{overflow-wrap:anywhere}}</style></head><body><main>
<p class="small">2026-10-20 · 自由机身 · 有限力矩 · 平地空载 · <a href="../../index.html">连杆结果目录</a></p>
<h1>轮架可独立摆动，但目前 LQR 未达到“躯干基本不晃”</h1>
<p class="notice"><strong>结论：局部静止恢复可行，实用的水平躯干运动尚未验证通过。</strong>
机构自由度允许躯干 0°、轮架相对倾斜 20°；但在自由动力学中，电机反作用力、重力和接触仍会让躯干摆动。
本轮测试不能证明所有控制器都不可行，也不能把“最终站住”当作“全程水平”。</p>
<section><h2>针对上方躯干的修改</h2>
<p>保留两个电机及原有连杆自由度，没有增加姿态锁定或第三电机。将原躯干质量中的 0.5 kg 重新分配为后置设备块，
位置为 x=−67.775 mm、y=+3.8 mm；躯干前后长度扩大为 220 mm 以容纳设备。总质量仍为 {report["checks"]["mass_kg"]:.3f} kg。
直立总重心原先向前偏约 22 mm、向右偏约 1.23 mm，新版 x/y 偏移接近零。</p>
<p>LQR 的躯干 Roll/Pitch 参考均为 0；允许下方轮架相对转动。滚轮目标转速限制 ±35 rad/s，
连杆目标摆角限制 ±30°，两个电机力矩均限制在 ±2.4 Nm。</p></section>
<h2>实际运动视频</h2><p>蓝色读数是躯干相对地面的姿态；黄色读数是轮架相对躯干的摆角。
前两段为 15 秒测试的前 8 秒；失败段保留跌倒过程。片段末尾定格 1.6 秒并明确标注。
倒车段赋予了 −1 m/s 初速度，不能视为学会自主倒车起步。</p>
<video controls preload="metadata" poster="preview.png" src="lqr_feasibility.mp4"></video>
<section><h2>最关键的三组静止结果（1 ms 步长）</h2><ul>
<li>初始偏差 +0.1°：存活 15 s，但躯干最大 Roll 为 <strong>{small_case["max_torso_roll_deg"]:.2f}°</strong>，
轮架最大相对摆角 {small_case["max_relative_wheel_roll_deg"]:.2f}°。</li>
<li>初始偏差 +0.5°：{'存活 15 s' if medium_case["survived"] else '未完整存活'}，
躯干最大 Roll 达 <strong>{medium_case["max_torso_roll_deg"]:.2f}°</strong>；
即使最终收敛，也不符合载物躯干基本水平的要求。</li>
<li>初始偏差 +1°：{'存活' if large_case["survived"] else '失败'}，
持续 {large_case["duration_s"]:.2f} s，轮架最大相对摆角 {large_case["max_relative_wheel_roll_deg"]:.2f}°。
关节限位为柔性约束，实际角度可能短暂越过命令限幅。</li></ul></section>
<h2>匹配工况与步长复核</h2>
<p>五种初始/目标速度，每种包含初始 Roll ±0.1°、±0.5°、±1°，分别以 1 ms 和 0.5 ms 步长运行。
两种步长使用同一套在 1 ms 下设计的增益，不重新调参。移动工况均有相应初速度。</p>
<table><tr><th>物理步长 ms</th><th>速度 m/s</th><th>完整存活</th><th>水平＋跟踪通过</th></tr>{"".join(grouped)}</table>
<p>另测静止左右推力 0.05 / 0.2 / 1 N（5～5.2 s）及从静止起步至 ±0.5 m/s。
推力测试通过 {sum(c["level_pass"] for c in push_group)}/{len(push_group)}；
起步通过 {sum(c["level_pass"] for c in start_group)}/{len(start_group)}。</p>
<p class="small">水平＋跟踪通过要求：完整 15 s；前 3 s 后躯干 Roll/Pitch RMS 均小于 3°；
全程躯干 Roll/Pitch 峰值均小于 5°；速度 RMSE 小于 0.15 m/s；
轮架峰值小于 31°、闭环误差小于 1 mm。跌倒或提前停止一律失败。
这个容差仅用于第一轮筛查，不等于运输水杯或光滑货物已通过。</p>
<h2>躯干与轮架的实际摆动</h2>{"".join(plots)}
<section><h2>方法与边界</h2>
<p>基于完整 MuJoCo 切空间状态的离散 LQR，保留柔性闭环、被动关节和轮地接触动态；
轮胎一段胶囊节距内取 8 个相位平均线性化，控制周期 10 ms。各速度使用独立局部增益，
采用 0.999 折扣处理近中性模态；并未声称完整模型所有模态严格渐近稳定。</p>
<p>有限差分扰动从 10⁻⁵ 改为 10⁻⁴ 时，静止 A/B 矩阵相对变化为
{report["checks"]["finite_difference_A_relative_change"]:.3g} /
{report["checks"]["finite_difference_B_relative_change"]:.3g}。
所有正式回放只通过有限力矩电机和明确标注的外力推进，没有逐帧覆盖机身姿态、隐藏支撑或状态重置。</p>
<p>尚未验证载荷、实际电机曲线、延迟、噪声及机械干涉；轮架和连杆仍按概念版关闭自碰撞。
没有独立 yaw 电机，本轮没有把航向保持列为通过条件。完整航向变化仍保存在逐例结果中。</p>
<p><strong>下一步判断：</strong>当前问题主要是有限摆角下的动态恢复和姿态瞬态。
如目标是运输水杯，建议先研究能在躯干小倾角下更有效地移动接地点的连杆几何或额外姿态稳定自由度，
再比较非线性控制；不能只凭一次小偏差恢复就开始宣称载物可行。</p></section>
<details><summary>全部 {len(cases)} 个测试</summary><div class="scroll"><table>
<tr><th>工况</th><th>步长 ms</th><th>时长 s</th><th>存活</th><th>水平跟踪</th><th>Roll 峰值 °</th>
<th>Roll RMS °</th><th>Pitch 峰值 °</th><th>速度 RMSE</th><th>轮架峰值 °</th><th>末端航向 °</th></tr>
{"".join(detail)}</table></div></details>
<p><a href="evaluation.json">完整指标 JSON</a> · <a href="controllers.json">LQR 增益与线性模型</a> ·
<a href="model.xml">对应模型快照</a> · <a href="video_manifest.json">视频片段说明</a></p>
</main></body></html>'''
    # This report lives four directory levels below results/index.html.
    text = text.replace('href="../../index.html"', 'href="../../../index.html"')
    (FOLDER/"report.html").write_text(text, encoding="utf-8")
    print(FOLDER/"report.html")


if __name__ == "__main__":
    main()
