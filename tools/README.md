# 项目辅助工具

| 目录 | 用途 |
|---|---|
| `cloud/` | 启智登录、持久连接，以及历史双 GPU 启动脚本 |
| `reporting/` | 已完成实验的报告汇总 |
| `maintenance/` | 结果目录迁移与导航生成 |

从项目根目录执行：

```powershell
.\.venv-mjlab\Scripts\python.exe tools/cloud/qzcli_local.py login
.\.venv-mjlab\Scripts\python.exe tools/maintenance/index_results.py
.\.venv-mjlab\Scripts\python.exe tools/reporting/build_steer_summary.py
```

`cloud/launch_steer_v2.py` 是 2026-10-19 云端实验的历史启动器，
含当时的绝对实验目录。当前续训使用 `python -m structure_rl train`，
具体命令见 [结构训练说明](../structure_rl/README.md)。

`maintenance/organize_results.py` 是一次性迁移工具，迁移已完成后会拒绝重复执行。
移动前校验路径、拒绝覆盖、移动后校验文件哈希；映射与原文件哈希保存在
[迁移记录](../results/_shared/organization_20261020.json)。
