"""Summarize the saved evaluation evidence and embed the local demonstration videos."""
from pathlib import Path
import html
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/raked_steer/rl/steer_v2_final'
BASELINE=ROOT/'results/raked_steer/rl/2026-10-19_compare01'
baseline=json.loads((BASELINE/'tracking_evaluation.json').read_text())
(OUT/'baseline').mkdir(parents=True,exist_ok=True)
for extension in ('html','json'):
    shutil.copy2(BASELINE/f'tracking_evaluation.{extension}',
                 OUT/f'baseline/tracking_evaluation.{extension}')
records=[('原斜轴策略',baseline,None,None)]
for name,label in [('balanced','普通优化'),('balanced_mirror','对称约束')]:
    for stage in (1,2):
        r=json.loads((OUT/name/f'stage{stage}_evaluation.json').read_text())
        records.append((f'{label} · {stage*2000} 次',r,name,stage))

def trials(r,speeds=(0.,.8),case=None):
    return [x for c in r['cases'] if c['initial_speed_m_s'] in speeds and (case is None or c['name']==case)
            for x in c['individual']]
def ratio(ts,key='passed'):
    return f'{sum(x[key] for x in ts)}/{len(ts)}' if ts else '—'
def mean(ts,key):
    vals=[x[key] for x in ts if x[key] is not None]
    return sum(vals)/len(vals) if vals else None

rows=[];detail=[];data=[]
for label,r,name,stage in records:
    matched=trials(r);extra=trials(r,(-.8,));all_trials=matched+extra
    reverse=trials(r,case='reverse')
    cells=[label,ratio(matched,'survived'),ratio(matched),ratio(reverse),ratio(extra),ratio(all_trials) if extra else '—']
    link=f'{name}/stage{stage}_evaluation.html' if name else 'baseline/tracking_evaluation.html'
    cells.append(f'<a href="{link}">原始报告</a>')
    rows.append('<tr'+(' class="selected"' if name=='balanced' and stage==1 else '')+'>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
    fwd=trials(r,(0.,),'forward');rev=trials(r,(0.,),'reverse')
    values=[label,f'{mean(fwd,"forward_rmse"):.3f}',f'{mean(fwd,"heading_rate_rmse"):.3f}',
            f'{mean(fwd,"final_heading_error_deg"):+.1f}°',f'{mean(rev,"forward_rmse"):.3f}',
            f'{mean(rev,"heading_rate_rmse"):.3f}',f'{mean(rev,"mean_action_delta_squared"):.5f}']
    detail.append('<tr>'+''.join(f'<td>{c}</td>' for c in values)+'</tr>')
    data.append({'label':label,'matched_survived':sum(x['survived'] for x in matched),'matched_passed':sum(x['passed'] for x in matched),
                 'extra_passed':sum(x['passed'] for x in extra) if extra else None,'all_passed':sum(x['passed'] for x in all_trials) if extra else None,
                 'reverse_matched_passed':sum(x['passed'] for x in reverse),'source_checkpoint':r['checkpoint']})

def table(headers,body):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'

document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>斜轴转向 · 训练完成总结</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#1b293a;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:auto;padding:40px 24px 70px}h1{font-size:34px;line-height:1.3;margin:12px 0}h2{font-size:23px;margin:30px 0 12px}
p{margin:10px 0}.badge{font-size:13px;color:#107257;letter-spacing:.08em}.lead{font-size:19px;max-width:980px}.card{background:white;border:1px solid #dde4ed;border-radius:14px;padding:22px;margin:20px 0}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}th,td{padding:12px;text-align:right;border-bottom:1px solid #dfe6ed}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}.selected{background:#eaf7f2}a{color:#17599c}video{display:block;width:100%;background:#0f1724;border-radius:10px}.note{font-size:14px;color:#52647a}.callout{border-left:4px solid #c88024;padding-left:18px}code{word-break:break-all;white-space:normal;font-size:13px}li{margin:8px 0}
</style><main><div class="badge">训练与评估已完成 · 两组各续训 4000 次 · 两个任务均正常退出</div>
<h1>斜轴转向：更能保持平衡，但航向与倒车仍有短板</h1>
<p class="lead">当前优先保留<strong>普通优化 2000 次检查点</strong>作为后续改进起点。它在 144 个测试中通过 70 个；继续训练至 4000 次后通过数降至 63。对称约束最终通过 52 个，尚未显示整体优势。</p>
<div class="card"><h2>相同条件下的结果</h2><p class="note">“原测试”包含静止起步和 +0.8 m/s 初始速度，共 96 个测试。“新增测试”以 −0.8 m/s 初始化，共 48 个。倒车列只统计原测试中的 16 个倒车任务，不能等同于从倒车状态执行所有命令的新增测试。</p>'''
document+=table(['策略','原测试存活','原测试通过','倒车任务通过','新增倒车初始通过','全部通过','详情'],rows)
document+='''</div><h2>最终两组视频对比</h2><p>左侧：普通优化 4000 次；右侧：对称约束 4000 次。两侧使用相同命令与初始状态。</p>
<video controls preload="metadata" src="videos/final_comparison.mp4"></video>
<p class="note">0–10 秒前进；10–22 秒倒车；22–37 秒连续换向（27 秒从 +1 切换到 −1 m/s）；37–45 秒静止命令。各片段开头独立复位，连续换向片段内部不复位。失败会标注 TERMINATED 并保持最后画面，不用复位伪装持续运动。</p>
<h2>建议保留的中期检查点视频</h2><p>普通优化 2000 次，采用同一套 45 秒演示；倒车任务通过 15/16，最终普通优化仅为 2/16。</p>
<video controls preload="metadata" src="videos/balanced_stage1.mp4"></video>
<p class="note">视频使用云端原策略实际仿真轨迹，单独在物理 GPU 1 渲染，没有重新模拟运动。均为确定性策略、无动作平滑、零初速和零 roll 的单次演示。它们帮助观察行为，不能代替多初态评估；均未施加外力、坡度或负载。</p>
<div class="card"><h2>误差与动作变化</h2><p class="note">下表为静止起步的 8 个初始 roll 条件的均值。速度单位 m/s，航向角速度单位 rad/s。航向误差记录全程累计值，RMSE 排除前 3 秒。动作变化为两维归一化动作相邻时刻差的平方和均值。</p>'''
document+=table(['策略','前进速度 RMSE','前进航向角速度 RMSE','前进最终航向误差','倒车速度 RMSE','倒车航向角速度 RMSE','倒车动作变化'],detail)
document+='''</div><h2>结论与下一步</h2><ul>
<li><strong>平衡能力提高：</strong>普通优化中期和最终都在原 96 个测试中全程存活，原策略为 88/96。新增的倒车初始测试仍会失败，不能称为全面稳定。</li>
<li><strong>最新检查点不是最优：</strong>普通优化的倒车任务通过数从 2000 次的 15/16 降到 4000 次的 2/16；对称约束最终为 0/16。后续应按独立评估选择检查点，而非直接使用最后一次保存。</li>
<li><strong>航向问题尚未解决：</strong>前进任务虽能通过现有速度阈值，最终普通优化和对称约束在 15 秒内的平均航向偏差约 −46° 和 −47°。原策略约 +15°。通过率没有包含累计航向限制，因此不能据此声称直行更准。</li>
<li><strong>对称约束目前不优先：</strong>它要求镜像状态输出镜像动作，是软损失；这一组结果不足以证明该方法普遍无效，也未证明当前系数合适。此次仅一个训练随机种子。</li>
<li><strong>建议下一轮：</strong>从普通优化 2000 次继续，先加入航向误差观测/航向保持任务，并把累计航向偏差纳入验收；平衡前进、倒车、停走和换向能力，定期评估并保留最佳检查点，再研究更弱的对称约束。</li>
</ul><p class="callout">测试边界：平地、每例 15 秒、初始 roll 为 −1° 至 +1°。原通过条件：全程存活，前进速度 / 航向角速度 / 侧向速度 RMSE 分别低于 0.35 / 0.15 / 0.35。没有验证坡地、推力、载物或实机。这里只优化了训练策略，几何与执行器参数未修改。</p>
<h2>建议保留的策略</h2><p><code>'''
document+=html.escape(records[1][1]['checkpoint'])+'</code></p><p class="note">同目录的 task.json、agent.json、geometry.xml 应与 checkpoint 一起保留。</p></main></html>'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'summary.html').write_text(document,encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT/'summary.html')
