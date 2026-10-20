"""Build an offline model-based result catalogue; no experiment execution."""
from pathlib import Path
import html
import json
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
GROUPS = [
    ("roll_linkage", "连杆摆轮", "滚动驱动＋四连杆 Roll 摆动；已完成自由机身 LQR 首轮验证，躯干水平鲁棒性尚未通过。",
     "../../roll_linkage/README.md",
     [("躯干水平 LQR 报告", "lqr/level_torso_v2/report.html"),
      ("自由机身验证视频", "lqr/level_torso_v2/lqr_feasibility.mp4"),
      ("V1 机构演示视频", "concept_v1/mechanism.mp4")]),
    ("raked_steer", "斜轴转向", "无飞轮，滚动驱动＋斜轴转向；包含传统控制和 PPO 训练。",
     "../../structure_rl/README.md",
     [("PPO 优化总结", "rl/steer_v2_final/summary.html"),
      ("普通优化 2000 次视频（含倒车）", "rl/steer_v2_final/videos/balanced_stage1.mp4")]),
    ("twin_flywheel", "双飞轮", "滚轮左右各一飞轮；同向等速 RL、固定速比验证与早期独立电机对照分开保存。",
     "../../structure_rl/README.md",
     [("同向等速策略评估", "rl/2026-10-19_compare01/tracking_evaluation.html"),
      ("固定速比验证", "coupling/README.md")]),
    ("wheelbot_cargo", "单飞轮 · 载物机身与历史套轮", "包括上置飞轮载物机身、起身验证，以及 nested 系列历史几何。",
     "../../miniwheelbot_rl/README.md",
     [("起身验证报告", "chamfer_recovery/report.html"), ("起身视频", "chamfer_recovery/recovery.mp4")]),
    ("wheelbot_legacy", "单飞轮 · 早期模型", "早期 wheelbot、飞轮尺寸、扰动、镜像和控制器调参记录。",
     "../../wheelbot/report.html", []),
    ("steerbot_legacy", "转向轮 · 早期模型", "早期 steerbot 的拖距、轮胎、坡地、扰动及控制器验证。",
     "../../steerbot/report.html", []),
    ("comparisons", "跨模型对照", "仅保存跨模型总表和原始下载归档，各模型的数据与视频存放在各自目录。",
     "../../structure_rl/README.md",
     [("双飞轮与斜轴：本地对照", "structure_concepts/report.html")]),
    ("_shared", "共享记录", "部署包、云端运行记录和本次整理的路径映射。历史记录中的远程路径保持原样。",
     "../../tools/README.md", []),
]


def main():
    cards = []
    markdown = [
        "# 实验结果导航", "",
        "直接打开 [index.html](index.html)，按机器人模型查找报告和视频。", "",
        "同一模型内部保留原实验名称；`local_lqr/` 为传统控制验证，`rl/` 为强化学习评估。",
        "当前模型源码仍在项目原模块内，结果目录中的 XML 是实验快照。",
        "旧 JSON/CSV/log/ZIP 按原字节保留，其中的路径可能是历史云端路径；",
        "本地旧→新路径映射见 [_shared/organization_20261020.json](_shared/organization_20261020.json)。",
        "", "| 目录 | 模型 / 用途 |", "|---|---|",
    ]
    catalogue = []
    for slug, title, description, source, featured in GROUPS:
        folder = RESULTS / slug
        folder.mkdir(parents=True, exist_ok=True)
        files = [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and "archived_cache" not in p.parts and p.name != "README.md"]
        sources = (folder / source).resolve()
        if not sources.exists():
            # The production task is documented in the root cloud guide.
            source = "../../CLOUD_README.md"
        links = []
        for label, relative in featured:
            path = folder / relative
            if path.exists():
                links.append(f'<a class="pill" href="{quote(path.relative_to(RESULTS).as_posix())}">{html.escape(label)}</a>')
        available = [p for p in files if p.suffix in (".html", ".mp4", ".png")]
        details = []
        for p in sorted(available):
            relative = p.relative_to(folder).as_posix()
            href = quote(p.relative_to(RESULTS).as_posix())
            details.append(f'<li><span>{html.escape(p.suffix[1:].upper())}</span><a href="{href}">{html.escape(relative)}</a></li>')
        size = sum(p.stat().st_size for p in files) / 1024**2
        cards.append(f'''<article data-search="{html.escape(title + " " + slug + " " + description)}">
<div class="eyebrow">{html.escape(slug)}</div><h2>{title}</h2><p>{description}</p>
<div class="links">{"".join(links)}</div>
<p class="meta">{len(files)} 个文件 · {size:.1f} MB · <a href="{quote(slug+'/README.md')}">目录说明</a></p>
<details><summary>展开报告、视频与图片（{len(available)}）</summary><ul>{"".join(details)}</ul></details></article>''')
        markdown.append(f"| [{slug}/]({slug}/README.md) | {title} |")
        readme = [f"# {title}", "", description, "", f"[返回结果导航](../index.html) · [模型与任务说明]({source})", ""]
        for label, relative in featured:
            if (folder / relative).exists():
                readme.append(f"- [{label}]({relative})")
        readme += ["", "## 实验目录", ""]
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                readme.append(f"- `{p.name}/`")
        readme += ["", "原实验数据保留。历史日志和 JSON 中的旧路径用于追溯，不代表当前本地目录。", ""]
        (folder / "README.md").write_text("\n".join(readme), encoding="utf-8")
        catalogue.append(dict(family=slug, title=title, files=len(files), size_mb=round(size, 2)))
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GyroRoll · 模型与实验结果</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#183044;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1240px;margin:auto;padding:48px 24px}h1{font-size:34px;margin:0 0 12px}header p{max-width:900px;color:#526677}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;border:1px solid #dce5eb;border-radius:14px;padding:24px;min-width:0}
.eyebrow{font-size:12px;letter-spacing:.08em;color:#547785}h2{font-size:23px;margin:6px 0 12px}p{margin:10px 0}
a{color:#12647a;text-decoration:none}a:hover{text-decoration:underline}.links{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
.pill{border:1px solid #cce0e5;background:#edf7f8;border-radius:7px;padding:6px 10px;font-size:14px}
.meta{font-size:13px;color:#647786}summary{cursor:pointer;color:#365567}ul{list-style:none;padding:0;max-height:330px;overflow:auto}
li{display:flex;gap:12px;padding:7px 0;font-size:13px;border-top:1px solid #edf1f4;overflow-wrap:anywhere}
li span{font-size:10px;color:#6a8191;min-width:34px;padding-top:3px}input{width:100%;max-width:500px;padding:12px 16px;border:1px solid #cbd9e2;border-radius:8px;font:inherit;margin:15px 0 28px}
.note{padding:16px 20px;border-left:3px solid #679aa5;background:#e7eff2;font-size:14px;margin:25px 0}
[hidden]{display:none!important}@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:28px 16px}h1{font-size:28px}}
</style></head><body><main><header><div class="eyebrow">GYROROLL / EXPERIMENT LIBRARY</div>
<h1>按模型查找实验结果</h1><p>模型结构、控制验证、训练报告和视频按类别归档。每个目录保留实验原名，方便追溯。</p>
<p><a href="../README.md">项目说明</a> · <a href="README.md">目录规范</a></p></header>
<input id="search" aria-label="筛选模型" placeholder="筛选模型：连杆 / 斜轴 / 双飞轮…">
<div class="grid">''' + "".join(cards) + '''</div>
<p class="note">数据整理不改变实验结论。固定机身机构演示、传统控制、PPO 回放分别标注；历史通过率不能直接转移到新几何。
连杆摆轮已完成首轮 LQR 验证，尚未达到稳定载物所需的水平姿态。</p>
<script>document.querySelector('#search').addEventListener('input',function(){
const q=this.value.trim().toLowerCase();document.querySelectorAll('article').forEach(a=>{
a.hidden=!a.dataset.search.toLowerCase().includes(q)&&!a.textContent.toLowerCase().includes(q);});});</script>
</main></body></html>'''
    (RESULTS / "index.html").write_text(page, encoding="utf-8")
    markdown += ["", "刷新导航：`python tools/maintenance/index_results.py`。", "",
                 "后续连杆摆轮 LQR 输出放在 `roll_linkage/lqr/`；检查点继续保存在 `logs/`。", ""]
    (RESULTS / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(catalogue, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
