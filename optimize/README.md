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
| `plan_recovery.py` | 原生 MuJoCo 批量轨迹搜索，生成有界两轮起身参考；搜索分数不等于通过验收 |
| `tune_recovery_capture.py` | 对参考动作结束后的平衡反馈调参 |
| `linearize_recovery.py` | 保存起身参考状态，并计算时变 LQR 对照；最终通过验证的是 MPC |
| `validate_recovery.py` | 同一预测控制器的左右侧、跨步长、15 s 独立回放 |
| `report_recovery.py` | 从已完成的六项回放生成视频、曲线和离线 HTML |

## 当前侧躺起身实验

先按根目录 README 的命令回放已保存的控制器。重新执行六项验证：

```powershell
.venv-mjlab/Scripts/python.exe optimize/validate_recovery.py --controller results/chamfer_recovery/controller.json
```

初次实验将 2 / 1 ms 数据保存在 `results/chamfer_recovery/validation/`，0.5 ms 数据保存在 `validation_fine/`。`report_recovery.py` 读取这两个目录；报告中的通过数字必须来自完整回放，不能引用优化器的最小瞬时倾角。

`plan_recovery.py` 的早期 `swing_plan` / `capture_plan` 搜索分数仅用于探索，曾对滚轮而非飞轮施加动量代价；当前代码已按真实关节 DOF 修正。最终 MPC 的六例验收直接使用真实倾角、高度、角速度和持续时间，不依赖上述搜索分数。固定动作与时变 LQR 的失败回放也保留在结果目录供比较。
