# GyroRoll 云端训练 / MJLab 1.3.0

上传 `miniwheelbot_mjlab130_cloud.zip` 到云端并解压。所有命令在解压后的 `miniwheelbot_mjlab130/` 目录执行。包内包含当前 24 cm 开放滚动轮模型、两维轮速动作 PPO 任务、评估与浏览器回放入口，不包含已训练策略。

本地没有启动 PPO 训练。已完成四项 MuJoCo 速度执行器测试，以及四个并行世界的 MuJoCo-Warp GPU 物理步进检查；完整 MJLab 任务、PPO 更新、浏览器回放仍需在云端按下文验证。

## 1. 配置云端环境

以下使用 Linux、Python 3.12 和 NVIDIA CUDA GPU。首次从 256 并行环境开始，根据显存和实际吞吐调整；当前轮圈和胎面共有 192 个凸扇区网格。云端驱动需要支持所安装的 CUDA 12.8 PyTorch；CPU 镜像不能替代 CUDA PyTorch。

```bash
cd miniwheelbot_mjlab130
python3.12 -m venv .venv-mjlab
source .venv-mjlab/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-mjlab.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

MJLab 固定为 1.3.0，其他主要依赖见 `requirements-mjlab.txt`。使用独立环境可避免旧版项目的 MuJoCo / MJLab 依赖混用。纯物理训练和评估不需要桌面窗口；浏览器回放使用 Viser。

## 2. 先做云端检查

```bash
python -m unittest discover -s tests -p 'test_rl_*.py' -v
python -m miniwheelbot_rl smoke --num-envs 4
python -m miniwheelbot_rl train --num-envs 4 --iterations 2 --stage warmup --log-root logs/cloud_check
```

`smoke` 检查 15 维观测、两维动作顺序、有限数值和电机力矩上限。最后一条只进行两次 PPO 更新，确认训练和保存通路。首次 Warp 编译会比后续启动慢。确认没有报错且 `logs/cloud_check/` 保存 checkpoint 后再开始正式训练；这些检查不代表机器人已经学会平衡或转向。

单元检查也包含无需 GPU 仿真的真实 MJLab `Entity` / `ActionManager` 集成测试，验证动作关节匹配、顺序、正反转缩放、饱和与零目标速度。MJLab 1.3.0 的 `JointVelocityActionCfg.actuator_names` 实际匹配被驱动关节名（`reaction`、`drive`），不是 XML 电机名（`reaction_velocity`、`drive_velocity`）。仿真的积分器、步长和重力由全局 `MujocoCfg` 指定，机器人实体不单独携带这些选项。

## 3. 正式训练与续训

```bash
python -m miniwheelbot_rl train --stage warmup --num-envs 256 --iterations 3000
```

策略从随机初始化直接学习，不使用旧控制器、MPC 专家或模仿数据。输出顺序固定为 `[飞轮目标角速度, 滚轮目标角速度]`，两路均支持正反转，目标范围分别为 ±250 rad/s、±35 rad/s；有限力矩的速度伺服执行动作。指令是 `[前进/后退速度, 转向角速度]`。

| 阶段 | 前进/后退速度 m/s | 转向角速度 rad/s |
|---|---|---|
| warmup | ±0.5 | ±0.20 |
| tracking | ±1.2 | ±0.40 |
| full | ±2.0 | ±0.60 |

也可以用 `--stage full` 从头训练完整范围。分阶段训练时，将下列 `RUN/model_XXX.pt` 替换成实际 checkpoint 路径；文件名以运行目录实际输出为准。`--iterations` 是本次继续训练的更新次数。

```bash
python -m miniwheelbot_rl train --stage tracking --num-envs 256 --iterations 3000 --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt
python -m miniwheelbot_rl train --stage full --randomize --num-envs 256 --iterations 5000 --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt
```

随机化当前包括轮胎/轮圈摩擦和间歇速度冲击，不包含质量、电机参数或延迟随机化。所有电机参数是仿真起始值，尚未按实物标定。动作定义、观测、奖励和物理频率详见 [任务说明](discuss/直接强化学习训练.md)。

每次运行保存在 `logs/rsl_rl/nested_velocity_mjlab130/日期时间_阶段/`，包含 checkpoint、`task.json`、`agent.json`、`geometry.xml` 和 TensorBoard 曲线。下载或搬迁时保留整个运行目录；续训/评估会恢复保存的几何和电机参数。旧力矩动作策略不能直接续训为本任务。

## 4. 看训练曲线和运动效果

训练曲线在云端启动：

```bash
tensorboard --logdir logs/rsl_rl/nested_velocity_mjlab130 --host 127.0.0.1 --port 6006
```

使用 checkpoint 在云端启动浏览器回放，默认仅一个机器人，默认采用 checkpoint 保存的指令阶段：

```bash
python -m miniwheelbot_rl play --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt --viewer viser --viewer-port 8080
```

在你的电脑打开另一个终端，填入云端用户名、地址和实际 SSH 端口建立转发：

```bash
ssh -p 22 -L 8080:127.0.0.1:8080 -L 6006:127.0.0.1:6006 用户名@云端地址
```

浏览器访问 `http://localhost:8080` 看运动，访问 `http://localhost:6006` 看训练曲线。回放面板提供暂停、重置和速度指令控件。云平台若提供 Web 端口代理，可用 `--viewer-host 0.0.0.0` 启动回放并按平台方式映射 8080；TensorBoard 同理改 `--host 0.0.0.0`。远程回放的环境和策略在云端运行，本地浏览器只显示画面。

## 5. 独立评估

```bash
python -m miniwheelbot_rl evaluate --num-envs 8 --duration 15 --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt --output results/nested_rl/evaluation.json
```

评估覆盖静止、1 m/s 前进、−1 m/s 后退，以及速度 ±0.8 m/s、转向角速度 ±0.35 rad/s 的四种转弯组合。每个初始状态必须全程不倒；去除前三秒启动阶段后，前进速度 RMSE <0.35 m/s、转向角速度 RMSE <0.15 rad/s、横向速度 RMSE <0.35 m/s 才计通过。报告分别给出每类存活率、通过率和各初始状态误差，跌倒后重置不会清除失败记录。可增加环境数、改变种子重复评估；建议每类通过率至少 95% 再认为该范围跟踪基本达标。

这是名义平地评估，不替代后续质量/电机不确定性、延迟、外力和更长时间转弯的鲁棒性验收。warmup 范围外的指令是泛化检查，需看后续阶段策略表现。训练奖励升高不能单独证明验收通过。

## 常见处理

- 显存不足：减小 `--num-envs`，如 256 → 128 → 64。
- CUDA 不可用：先修复驱动与 CUDA PyTorch，再运行任务。
- 报 `reaction_velocity: []`：更新到仓库最新版，再重跑单元检查和 `smoke`；早期版本错误地使用了 XML 电机名。
- 模型载入/动作检查/PPO 更新报错：保留完整报错和安装版本，先排除环境问题。
- 未安装 MJLab 1.3.0：入口会拒绝运行其他版本，按固定依赖重新安装。
- 打包更新：修改代码后运行 `python package_cloud.py`，重新上传 `dist/` 中的 zip。

参考：[MJLab v1.3.0](https://github.com/mujocolab/mjlab/tree/v1.3.0)、[Viser 1.0.26 浏览器服务实现](https://github.com/nerfstudio-project/viser/blob/v1.0.26/src/viser/_viser.py)。
