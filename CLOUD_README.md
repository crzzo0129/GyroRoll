# GyroRoll 云端训练 / MJLab 1.3.0

当前结构为宽体封闭飞轮舱、上方设备舱和顶部载物托盘，整机高度 580 mm，主机身厚 160 mm、宽 220 mm，托盘可用面积 156 × 216 mm。滚轮轴心离地 120 mm、飞轮轴心离地 373 mm，两轮上/下沿净距保持 50 mm。滚轮胎面加宽至 24 mm，轮圈宽 20 mm，双叉间距同步加宽；滚轮沿用 312 g 概念质量预算，惯量按新宽度更新。空载仿真质量预算为 1.465 kg，重心离地约 350.4 mm；质量分布、惯量和碰撞结构已经更新，需要针对新结构训练。机身下部 140 mm 高度已斜切，底端宽 140 mm、上端宽 220 mm；自然侧躺约 76.6° 时滚轮与机身同时接地。左右两侧及三个物理步长均已验证接触，但尚未验证可靠起身。当前未添加货物质量，载物能力及载物策略尚未验证。

旧 checkpoint 的续训/回放仍恢复其保存的 `geometry.xml`；训练当前新结构请先不带 `--checkpoint` 运行 `train --stage warmup`。不要覆盖旧 checkpoint 的几何文件或哈希。后续起身课程从新结构的 warmup checkpoint 续训。XML 包含所有外观网格和材质，无需额外图片资源。当前 XML 默认以 Cover Removed 方式显示，隐藏正面盖板和饰面，飞轮正面保留一条白色刻度；隐藏盖板的质量和碰撞仍然生效。旧 checkpoint 的几何快照不会自动采用新外观。

上传 `miniwheelbot_mjlab130_cloud.zip` 到云端并解压。所有命令在解压后的 `miniwheelbot_mjlab130/` 目录执行。包内包含当前 24 cm 开放滚动轮模型、两维轮速动作 PPO 任务、评估与浏览器回放入口，不包含已训练策略。

新增单策略课程：`rise_tilt → rise → recover → slope3 → slope6 → slope10 → mixed`，按“先起身，再坡地”的顺序续训。课程设置、云端训练命令、起身录像和分场景验收见 [起身恢复与坡地课程](discuss/起身恢复与坡地课程.md)。现有有界策略可以直接作为第一阶段的 checkpoint。

本地没有启动 PPO 训练。已完成四项 MuJoCo 速度执行器测试，以及四个并行世界的 MuJoCo-Warp GPU 物理步进检查；完整 MJLab 任务、PPO 更新、浏览器回放仍需在云端按下文验证。

有界策略另通过六项 CPU 单元/集成检查：均值和标准差约束、优化更新后的边界、真实 RSL-RL PPO 更新、保存恢复、确定性导出以及旧模型回放兼容。测试使用合成观测和奖励，不代表机器人训练效果。

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
python -m miniwheelbot_rl train --stage warmup --num-envs 256 --iterations 3000 --log-root logs/rsl_rl/cargo_roll005
```

策略从随机初始化直接学习，不使用旧控制器、MPC 专家或模仿数据。输出顺序固定为 `[飞轮目标角速度, 滚轮目标角速度]`，两路均支持正反转，目标范围分别为 ±250 rad/s、±35 rad/s；有限力矩的速度伺服执行动作。指令是 `[前进/后退速度, 转向角速度]`。

当前新训练使用 `bounded_mean_gaussian_v1`：策略均值经 tanh 限制在 ±1；两个通道各自的探索标准差初始为 0.1，使用 sigmoid 参数化限制在 0.02～0.2；熵系数为 0。训练仍加入高斯探索噪声，采样后仍可能越界，由执行端裁剪；回放/评估使用同一个 tanh 均值。当前横滚角速度惩罚权重已提高到 -0.05；俯仰为 -0.01，转速倍率和电机力矩上限沿用原设置。

第一轮建议先从头训练 500 次更新检查数值和回放，再决定是否继续：

```bash
python -m miniwheelbot_rl train --stage warmup --num-envs 256 --iterations 500
```

新运行的 `task.json` 保存 `policy_version`。只有这一新版本的 checkpoint 可继续当前训练；没有该字段的旧模型仍按原来的无界高斯策略用于 `play/evaluate/video/diagnose`，保持输出含义不变，但不能直接续训为新版本。开始新版本时不传 `--checkpoint`。新版本之间可以正常续训，加载时恢复已学习的标准差，而非重新设为 0.1。

| 阶段 | 前进/后退速度 m/s | 转向角速度 rad/s |
|---|---|---|
| warmup | ±0.5 | ±0.20 |
| tracking | ±1.2 | ±0.40 |
| full | ±2.0 | ±0.60 |
| full_robust | ±2.0 | ±0.60 |

也可以用 `--stage full` 从头训练完整范围。分阶段训练时，将下列 `RUN/model_XXX.pt` 替换成实际 checkpoint 路径；文件名以运行目录实际输出为准。`--iterations` 是本次继续训练的更新次数。

```bash
python -m miniwheelbot_rl train --stage tracking --num-envs 256 --iterations 3000 --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt
python -m miniwheelbot_rl train --stage full --randomize --num-envs 256 --iterations 5000 --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt
```

普通阶段的 `--randomize` 包括轮胎/轮圈摩擦和间歇速度冲击，不包含质量、电机参数或延迟随机化。`full_robust` 使用下述有限时长外力替代速度冲击，带 `--randomize` 时另加摩擦随机化。所有电机参数是仿真起始值，尚未按实物标定。动作定义、观测、奖励和物理频率详见 [任务说明](discuss/直接强化学习训练.md)。

### full 之后：运动抗推与静止，不训练起身

已有 full 策略可以接着训练 `full_robust`，仍然只有一个 15 维观测、两维轮速动作的 policy。平地直立重置，保留跌倒终止；加载 checkpoint 时继续使用它自己的几何与电机参数。

- 每次重采样有 **25% 概率**选择完整静止指令 `(v, yaw_rate)=(0,0)`；其余命令保持 full 范围。每段持续 3–6 s，机器人会经历行走、停车和再启动。原 full 的静止概率是 10%。
- 静止额外奖励权重为 1，约束机身水平速度和航向角速度，允许飞轮转动维持平衡。速度尺度为 0.10 m/s，航向角速度尺度为 0.15 rad/s；这不等于锁定世界坐标位置。
- 每个环境独立等待 **4–8 s**，在机身 base 的质心施加随机水平方向 **0.5–1.5 N** 外力，持续 **0.10–0.25 s** 后清零。首次重置后也先等待；重新开始回合会清除该环境的推力。推力既可出现在运动中，也可出现在静止指令期间。
- `--randomize` 可额外打开摩擦随机化，不叠加旧的瞬时速度修改。普通 play/video/diagnose 关闭随机推力；抗推效果用下面的专项评估测量。

把 `FULL_RUN/model_XXXX.pt` 替换为已训好的 full checkpoint，旁边保留 `task.json` 和 `geometry.xml`：

```bash
python -m miniwheelbot_rl smoke --stage full_robust --num-envs 4
python -m miniwheelbot_rl train --stage full_robust --num-envs 256 --iterations 3000 \
  --checkpoint FULL_RUN/model_XXXX.pt --log-root logs/rsl_rl/full_robust
```

3000 是首轮追加更新次数，最终看评估效果。观察 `Episode_Reward/standing`、`Episode_Metrics/standing/command_fraction`、原 forward/yaw 奖励、roll 惩罚和回合长度。25% 是采样概率；提前跌倒或走停难度差异会影响实测回合统计比例。

```bash
python -m miniwheelbot_rl evaluate --suite robust --num-envs 16 --duration 15 \
  --checkpoint ROBUST_RUN/model_XXXX.pt --output results/full_robust_evaluation.json
```

专项评估涵盖静止、走停切换、行进中前后左右推力、静止中左右推力；第 5 s 施加 1.5 N 外力、持续 0.2 s。要求全程不跌倒，静止采样窗口的水平合成速度 RMSE <0.10 m/s、航向角速度 RMSE <0.15 rad/s。走停切换每次留 1 s 过渡，失败后重置的轨迹不会恢复成功资格。`full_robust` checkpoint 默认 evaluate 选择此套件；可另用 `--suite tracking` 复查原速度跟踪。

本地已验证推力定时、单环境重置、力—冲量关系和专项评估统计。完整 MJLab 配置集成、GPU 训练和真实策略抗推效果需要在云端运行上述检查；本地 CPU 测试不代表策略已经学会抗推。

每次运行保存在 `logs/rsl_rl/nested_velocity_mjlab130/日期时间_阶段/`，包含 checkpoint、`task.json`、`agent.json`、`geometry.xml` 和 TensorBoard 曲线。下载或搬迁时保留整个运行目录；续训/评估会恢复保存的几何和电机参数。旧力矩动作策略不能直接续训为本任务。

## 4. 看训练曲线和运动效果

无需网页访问时，可以在已配置训练环境的 GPU 实例直接导出 MP4，再通过平台文件管理器下载播放。更新项目代码后，在训练用 Python 环境安装录像依赖：

```bash
python -m pip install imageio==2.37.0 imageio-ffmpeg==0.6.0
MUJOCO_GL=egl python -m miniwheelbot_rl video --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt --duration 15 --speed 0.5 --turn 0 --video results/nested_rl/forward.mp4
```

把 checkpoint 路径换成实际文件，并保留旁边的 `task.json`、`geometry.xml` 和 `agent.json`。默认单环境、960×544、30 fps、跟随相机，不启动网页服务器。`--speed` 单位为 m/s，`--turn` 单位为 rad/s；warmup 建议先看 0.5 m/s 直行，后续可测试 `--speed 0.8 --turn 0.3`。发生跌倒等回合终止时录像提前停止，不把自动重置后的动作拼成连续成功运动；同名 JSON 记录实际时长与终止状态。依赖已安装后，录像本身不需要外网或 TensorBoard。Linux 无窗口渲染需要 NVIDIA EGL 支持。

横滚与俯仰角速度惩罚分别记录为 `Episode_Reward/roll` 和 `Episode_Reward/pitch`，当前 `roll` 权重为 -0.05（原来的 5 倍），`pitch` 权重为 -0.01。恢复课程仍按姿态减弱两项惩罚。更新代码后启动或续训时生效，正在运行的训练需要重启才能采用新代码。

运动阶段（warmup/tracking/full/full_robust）另加 `Episode_Reward/roll_angle`，公式为 `-2.0 × roll²`，roll 单位是弧度，由机体坐标下重力计算 `atan2(-gravity_y, -gravity_z)`。此项约束持续侧倾，独立于原角速度惩罚；起身/坡地课程暂不加入此项。新运行的 `task.json` 保存奖励权重以便核对。续训使用更新后的代码构建奖励，checkpoint 恢复策略及原几何、电机参数，不会覆盖新奖励。建议先追加 1500 次更新，同时检查直行侧倾、转弯跟踪、抗推和静止效果；转弯时可能需要合理侧倾，不以 roll 始终为零作为目标。

运动阶段的 `action_rate` 权重进一步从 -0.02 增强到 -0.10，惩罚为 `-0.10 × sum((clip(a_t, -1, 1) - clip(a_previous, -1, 1))²)`，两个动作均为归一化轮速目标。这里是相邻控制步差值，不除以控制周期；默认周期为 0.01 s。恢复课程保持原权重。建议从已训练的 `full_robust_roll` 模型追加 1500 次更新，同时比较相同工况下 `summary.json` 的 `after_3s` 中 `action_delta_rms`、速度跟踪误差及力矩饱和比例，并复查 robust 评估，避免平滑后抗推响应变慢。权重增大后原始 reward 曲线不能直接与旧运行比较。

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

## 6. 目标转速、实际转速和力矩诊断

更新代码后，在训练用 Python 环境运行（将 checkpoint 替换为实际路径）：

```bash
python -m miniwheelbot_rl diagnose \
  --checkpoint logs/rsl_rl/nested_velocity_mjlab130/RUN/model_XXX.pt \
  --duration 20 --speed 1.0 --turn 0 \
  --diagnostic-dir results/nested_rl/diagnostic_forward
```

默认单环境，使用策略均值，不添加探索噪声，不修改 checkpoint。保留模型旁边的 `task.json` 和 `geometry.xml`；按模型保存的转速、力矩和几何参数运行。无需额外绘图库、外网或图形窗口。

输出 `report.html`（可下载到电脑离线打开）、`trace.csv`（完整 100 Hz 默认控制频率记录）和 `summary.json`。包含两个轮子的裁剪前策略均值、裁剪后归一化动作、目标/实际转速、力矩，以及机体前进速度和水平航向变化率。统计分别覆盖全程和前三秒之后，包括动作裁剪比例、力矩达到 99% 上限的比例、转速误差等。终止时立即停止，不将自动重置后的状态作为有效数据；力矩为控制周期末采样，不代表周期内峰值或能耗。

建议分别测试直行 `--speed 1 --turn 0`、转弯 `--speed 0.8 --turn 0.35`、倒车转弯 `--speed -0.8 --turn 0.35`，每次指定不同输出目录；换种子可重复初始状态测试。目标频繁跳变且力矩换向，优先检查策略抖动；目标平稳、实际长期追不上且力矩持续同向饱和，再检查动力和负载限制。确定性结果不能代替训练随机采样的诊断，也不能凭单次回放判断所有指令都已通过。

## 常见处理

- 显存不足：减小 `--num-envs`，如 256 → 128 → 64。
- CUDA 不可用：先修复驱动与 CUDA PyTorch，再运行任务。
- 报 `reaction_velocity: []`：更新到仓库最新版，再重跑单元检查和 `smoke`；早期版本错误地使用了 XML 电机名。
- 模型载入/动作检查/PPO 更新报错：保留完整报错和安装版本，先排除环境问题。
- 未安装 MJLab 1.3.0：入口会拒绝运行其他版本，按固定依赖重新安装。
- 打包更新：修改代码后运行 `python package_cloud.py`，重新上传 `dist/` 中的 zip。

参考：[MJLab v1.3.0](https://github.com/mujocolab/mjlab/tree/v1.3.0)、[Viser 1.0.26 浏览器服务实现](https://github.com/nerfstudio-project/viser/blob/v1.0.26/src/viser/_viser.py)。
