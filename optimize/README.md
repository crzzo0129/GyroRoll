# 优化脚本

所有 `optimize_*.py` 集中在本目录，用于控制器参数和机器人几何的离线搜索。它们与 `miniwheelbot_rl/` 的直接强化学习任务分别运行。

从项目根目录启动，例如：

```bash
python optimize/optimize_nested.py --help
python optimize/optimize_nested.py --iterations 12
python -m optimize.optimize_nested --iterations 12
```

其他脚本同样支持 `python optimize/脚本名.py` 或 `python -m optimize.脚本名`。部分早期脚本没有命令行参数，启动后会直接开始优化；先阅读脚本再运行。模型与配置仍读取根目录下的 `wheelbot/`、`steerbot/`，输出仍保存在各脚本原有的配置或结果位置。

| 脚本 | 用途 |
|---|---|
| `optimize_nested.py` | 当前开放滚动轮和飞轮构型的反馈参数搜索 |
| `optimize_wheel*.py`、`optimize_tilted_robust.py` | 飞轮构型的反馈、轮胎和陀螺步态探索 |
| `optimize_steer*.py`、`optimize_raked_steer.py` | 转向机构的控制器与几何搜索 |
| `optimize_lqr*.py` | 局部线性模型的 LQR 权重与鲁棒参数搜索 |
| `optimize_morph*.py`、`optimize_lightfork.py` | 机身、前叉与质量布局搜索 |
| `optimize_longtrail*.py` | 长拖距概念构型的转弯与反向转弯搜索 |
