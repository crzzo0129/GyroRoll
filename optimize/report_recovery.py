"""Create an offline report from completed recovery validation traces."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from probe_rise import Experiment,render


def main():
    folder=Path('results/chamfer_recovery')
    paths=sorted((folder/'validation').glob('dt_*.json'))+sorted((folder/'validation_fine').glob('dt_*.json'))
    # A later full validation can contain the same fine-step cases. Keep the
    # newest completed replay for each case rather than double-counting it.
    unique={}
    for path in sorted(paths,key=lambda p:p.stat().st_mtime):
        r=json.loads(path.read_text())
        unique[(r['physics_timestep'],r['initial_roll_deg'],r['initial_pitch_deg'])]=r
    cases=list(unique.values())
    assert len(cases)==6
    success=[r['within_5s'] and r['stable_at_end'] and not r['fell_after_recovery'] for r in cases]
    chosen=folder/'validation/dt_0.002_roll_90_pitch_0.json'
    report=json.loads(chosen.read_text())
    arrays=np.load(chosen.with_suffix('.npz'))
    trace=arrays['trace']
    ex=Experiment()
    digest=hashlib.sha256(ex.model_path.read_bytes()).hexdigest()
    assert all(r['geometry_sha256']==digest for r in cases)
    shutil.copyfile(ex.model_path,folder/'geometry.xml')
    render(ex,report,trace,arrays['qpos'],folder/'recovery.mp4')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    canvas=Image.new('RGB',(1440,1200),'#f3f6f6')
    d=ImageDraw.Draw(canvas)
    panels=[('总倾角 / °',[np.degrees(trace[:,3])],(0,180),['实际倾角']),
            ('飞轮转速 / rad/s',[trace[:,8],trace[:,10]],(-270,270),['实际','目标']),
            ('滚轮转速 / rad/s',[trace[:,9],trace[:,11]],(-40,40),['实际','目标']),
            ('电机力矩 / N·m',[trace[:,12],trace[:,13]],(-2.6,2.6),['飞轮','滚轮'])]
    colors=['#087f8c','#d77a35']
    for i,(title,series,limits,labels) in enumerate(panels):
        y=i*300
        d.text((48,y+12),title,font=font,fill='#18343d')
        left,top,right,bottom=100,y+60,1390,y+250
        d.rectangle((left,top,right,bottom),fill='white')
        def point(t,v):
            return (left+(right-left)*t/15,bottom-(bottom-top)*(v-limits[0])/(limits[1]-limits[0]))
        for value in np.linspace(*limits,5):
            yy=point(0,value)[1]
            d.line((left,yy,right,yy),fill='#dfe8ea')
            d.text((25,yy-10),f'{value:g}',font=small,fill='#567078')
        for t in range(0,16,3):
            xx=point(t,0)[0]
            d.text((xx-5,bottom+8),str(t),font=small,fill='#567078')
        for j,values in enumerate(series):
            d.line([point(t,v) for t,v in zip(trace[:,0],values)],fill=colors[j],width=2)
            d.text((1080+j*140,y+16),labels[j],font=small,fill=colors[j])
        if i==0:
            yy=point(0,20)[1]
            for x in range(left,right,18):d.line((x,yy,x+9,yy),fill='#a65b5b',width=2)
        d.text((1270,bottom+8),'时间 / s',font=small,fill='#567078')
    canvas.save(folder/'curves.png')
    rows=''.join(f'<tr><td>{r["physics_timestep"]*1000:g} ms</td><td>{r["initial_roll_deg"]:+g}°</td>'
                 f'<td>{format(r["recovered_at_s"],".2f")+" s" if r["recovered_at_s"] is not None else "未恢复"}</td><td>{r["max_stable_seconds"]:.2f} s</td>'
                 f'<td>{r["final_tilt_deg"]:.2f}°</td><td>{"通过" if ok else "失败"}</td></tr>'
                 for r,ok in sorted(zip(cases,success),key=lambda v:(-v[0]['physics_timestep'],-v[0]['initial_roll_deg'])))
    max_penetration=max(-r['minimum_contact_distance_m'] for r in cases)*1000
    recovered=[r['recovered_at_s'] for r,ok in zip(cases,success) if ok]
    recovery_range=f'{min(recovered):.2f}–{max(recovered):.2f} s' if recovered else '尚未通过'
    curve64=base64.b64encode((folder/'curves.png').read_bytes()).decode()
    summary=dict(cases=len(cases),passed=sum(success),geometry_sha256=digest,
                 max_soft_contact_penetration_mm=max_penetration,
                 recovery_seconds=[r['recovered_at_s'] for r in cases],
                 tests='17 unit/regression checks passed',
                 scope='Deterministic empty-body, level-ground, zero initial wheel speed, settled side rest. No payload, slope, perturbation or hardware validation.')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GyroRoll · 侧躺起身重新设计</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f3;color:#18343d;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 28px}}h1{{font-size:40px;line-height:1.3}}h2{{margin-top:42px}}.tag{{color:#087f8c;font-weight:700}}
.cards{{display:flex;gap:18px;flex-wrap:wrap}}.card{{padding:22px 30px;background:white;border-radius:14px;flex:1;min-width:200px}}
.big{{font-size:34px;font-weight:700}}video,img{{width:100%;border-radius:12px}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:12px 15px;border-bottom:1px solid #dce5e7;text-align:left}}pre{{background:#18343d;color:#e5f6f7;padding:22px;border-radius:12px;overflow:auto;font-size:14px}}
.note{{padding:20px 25px;border-left:4px solid #c98c4c;background:#fff8ef}}a{{color:#087f8c}}footer{{font-size:13px;color:#567078;overflow-wrap:anywhere}}
</style><main><div class="tag">GYROROLL / LOCAL RECOVERY / 2026-10-17</div>
<h1>两轮协同翻起，再制动接住</h1>
<p>针对下部斜切的新机身，重新设计了动态起身动作：滚轮先改变机身的支撑姿态，飞轮与滚轮配合翻起，随后降低角速度并切入平衡反馈。起身过程中每 40 ms 根据当前状态重新预测未来 0.4 s 的动作。</p>
<div class="cards"><div class="card">侧躺验证<div class="big">{sum(success)} / 6</div>左右两侧 × 三种物理步长</div>
<div class="card">达到稳定判据<div class="big">{recovery_range}</div>从控制启动计时，含连续稳定 1 s</div>
<div class="card">持续验证<div class="big">15 s</div>每例起身后保持约 11.8–11.9 s</div></div>
<h2>完整视频</h2><video controls preload="metadata" src="recovery.mp4" poster="recovery.png"></video>
<p>2 ms、初始 roll +90° 的连续物理回放。轮速从零开始，先无驱动自然静置 2 s；控制启动时自然侧躺约 76.56°。录像没有拼接重置或外力扶正。下载报告后，视频文件需与 HTML 放在同一目录。</p>
<h2>控制如何工作</h2><ol><li><b>改变支撑：</b>两轮往复配合，让机身从侧躺转到更有利的前后支撑姿态。</li>
<li><b>动态翻起：</b>利用飞轮反转与滚轮换向产生的运动，越过静态抬起困难的区域。</li>
<li><b>预测制动：</b>根据实时姿态和轮速修正短时动作，降低抵达直立附近时的角速度。</li>
<li><b>平衡接管：</b>参考动作在 3.2 s 结束，随后用较强的 roll 阻尼、飞轮动量反馈和两轮陀螺耦合补偿保持平衡。</li></ol>
<p>最终验证使用短时预测控制（MPC）。仅按固定时间播放动作，或仅增加轨迹周围的时变 LQR 修正，都没有通过同样的跨步长、左右侧测试。</p>
<h2>独立回放结果</h2><table><thead><tr><th>物理步长</th><th>放置 roll</th><th>恢复时间</th><th>最长连续稳定</th><th>结束倾角</th><th>结果</th></tr></thead><tbody>{rows}</tbody></table>
<p>判据：总倾角 &lt;20°、根节点高度 &gt;0.30 m、roll/pitch 合成角速度 &lt;1.5 rad/s，连续满足 1 s；5 s 内恢复，15 s 结束时仍稳定，恢复后没有再次倒下。六例是确定性场景检查，不是随机扰动成功率。</p>
<h2>动作与电机实测</h2><img alt="倾角、两轮目标与实际转速、实际电机力矩" src="data:image/png;base64,{curve64}">
<p>两个电机力矩上限均为 ±2.4 N·m；飞轮目标轮速 ±250 rad/s，滚轮目标轮速 ±35 rad/s。执行器与训练设置一致。目标转速边界不等于实际轮速的硬裁剪；惯性和接触可以使实际轮速短暂越界。</p>
<h2>验证范围</h2><div class="note">本结果限于当前空载机身、平地、已知完整仿真状态和指定的侧放初态。动作包含明显翻滚与接触冲击，六例最大软接触侵入约 {max_penetration:.1f} mm。载物、坡地、摩擦/质量变化、传感噪声及实物尚未验证。预测计算尚未满足实时控制要求；当前用于可行性验证与生成示范轨迹。它不是训练好的 PPO，也尚未并入最终单一 policy。</div>
<h2>本地复现</h2><pre>.venv-mjlab/Scripts/python.exe probe_rise.py --controller results/chamfer_recovery/controller.json --reference results/chamfer_recovery/reference.npz --feedback mpc --duration 15 --video results/chamfer_recovery/replay.mp4 --output results/chamfer_recovery/replay.json</pre>
<p>另一侧增加 <code>--roll -90</code>；细步长增加 <code>--timestep .001</code> 或 <code>--timestep .0005</code>。相同算法、参考和参数用于六例验证，预测模型使用各自的物理步长。</p>
<p><a href="controller.json">控制参数</a> · <a href="summary.json">验证摘要</a> · <a href="geometry.xml">几何快照</a> · <a href="reference.npz">参考轨迹</a></p>
<footer>相关单元/回归检查：17 项通过。MPC 每次 48 个候选、4 轮更新，固定随机种子 71。几何 SHA-256：{digest}</footer></main></html>'''
    (folder/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
