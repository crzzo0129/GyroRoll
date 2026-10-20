# 实验结果导航

直接打开 [index.html](index.html)，按机器人模型查找报告和视频。

同一模型内部保留原实验名称；`local_lqr/` 为传统控制验证，`rl/` 为强化学习评估。
当前模型源码仍在项目原模块内，结果目录中的 XML 是实验快照。
旧 JSON/CSV/log/ZIP 按原字节保留，其中的路径可能是历史云端路径；
本地旧→新路径映射见 [_shared/organization_20261020.json](_shared/organization_20261020.json)。

| 目录 | 模型 / 用途 |
|---|---|
| [roll_linkage/](roll_linkage/README.md) | 连杆摆轮 |
| [raked_steer/](raked_steer/README.md) | 斜轴转向 |
| [twin_flywheel/](twin_flywheel/README.md) | 双飞轮 |
| [wheelbot_cargo/](wheelbot_cargo/README.md) | 单飞轮 · 载物机身与历史套轮 |
| [wheelbot_legacy/](wheelbot_legacy/README.md) | 单飞轮 · 早期模型 |
| [steerbot_legacy/](steerbot_legacy/README.md) | 转向轮 · 早期模型 |
| [comparisons/](comparisons/README.md) | 跨模型对照 |
| [_shared/](_shared/README.md) | 共享记录 |

刷新导航：`python tools/maintenance/index_results.py`。

后续连杆摆轮 LQR 输出放在 `roll_linkage/lqr/`；检查点继续保存在 `logs/`。
