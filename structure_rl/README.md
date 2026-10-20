# 两种概念结构的速度跟踪强化学习

入口：`python -m structure_rl`。这是独立的结构比较任务，不能加载原先单飞轮策略。
训练依赖 `requirements-mjlab.txt` 中的 MJLab 1.3.0 与 CUDA PyTorch。

## 模型与控制

| 项目 | twin_sync | raked_steer |
|---|---|---|
| 结构 | 滚轮两侧各一个飞轮，转轴均为机身 +x | 无飞轮；转向轴相对竖直倾斜 15°，地面拖距 −160 mm |
| 动作 0 | 滚轮目标转速 ±35 rad/s | 滚轮目标转速 ±35 rad/s |
| 动作 1 | 两飞轮共用目标转速 ±250 rad/s | 转向目标角 ±0.7 rad |
| 力矩上限 | 滚轮 2.4 Nm，飞轮各 1.2 Nm | 滚轮 2.4 Nm，转向 2.4 Nm |
| 速度/位置伺服 | 滚轮 kv=0.25；飞轮各 kv=0.06 | 滚轮 kv=0.25；转向 kp=8，kd=0.25 |
| 总质量 | 1.465 kg | 1.007 kg |

两飞轮只有一个独立控制量，目标转速严格相同；保留 `q_left=q_right` 的物理等式约束。
MuJoCo 等式是柔性数值约束，不能承诺实际速度逐 bit 相等。每个物理子步记录实际转速差，
GPU smoke 检查最大差值小于 0.01 rad/s。本地无控制倒地碰撞测试最大约 0.002 rad/s。
不通过改写 qvel 伪造同步。仅测试同向等速，不包含反向等速。

采用之前概念验证的简化结构与窄圆截面轮胎，不等同于旧生产机器人外壳。
两结构质量不同；这轮比较的是各自完整结构，不是等质量条件下的单变量实验。

## 观测、奖励和训练

策略/价值网络各 16 维观测：机身速度 3、角速度 3、重力方向 3、机构状态 3、速度命令 2、上一动作 2。
双飞轮机构状态为三轮转速；转向机构状态为滚轮转速、转向角、转向角速度。
动作均裁剪至 [-1,1] 再映射为物理目标，策略 100 Hz、物理仿真 500 Hz。
无 LQR、无专家控制、无隐蔽支撑或持续外力。

奖励权重：前进跟踪 4、航向角速度跟踪 2、直立 1；侧向速度 −0.5、roll 角速度 −0.05、
pitch 角速度 −0.01、roll 角度 −2、动作变化 −0.10、力矩 −0.005、超速 −1、失败 −10。
跟踪为高斯误差奖励，前进 sigma=0.4 m/s，航向角速度 sigma=0.25 rad/s。
目标是速度/航向角速度跟踪，当前没有绝对航向锁定控制。

PPO 使用 128×128×128 网络，每环境 32 步，每次更新 4 epochs，学习率 3e-4，
有界均值高斯策略，标准差限定 0.02–0.2。每结构独立从零训练：

1. 双飞轮 `warmup`：静止初始化，速度 ±0.5 m/s、角速度 ±0.15 rad/s，25% 静止命令。
2. 斜轴 `moving`：赋予 0.5–1 m/s 初速度及相符滚轮速度，命令 0.5–1 m/s、±0.3 rad/s。
3. 各自接着 `tracking`：速度 ±1 m/s、角速度 ±0.3 rad/s，25% 静止命令；一半静止初始化，一半 0.8 m/s 初始化。

默认每结构先 2500 次迭代，再 5000 次，256 并行环境。每 50 次迭代保存模型。
分阶段只是初始化/命令课程变化，每种结构最终各一个策略。
斜轴带初速度的训练不代表已经学会静止起步，必须看独立评估结果。

## 云端运行

在部署目录中用 `/opt/mjlab-venv/bin/python` 执行。启动器检查选定两张卡没有其他计算进程；
只设置子进程的 GPU 可见性，不修改实例配置，不停止其他任务。

```bash
/opt/mjlab-venv/bin/python -u -m structure_rl.launch --gpus 0,1
```

启动器分别为两结构运行 smoke → 第一阶段训练 → 评估 → tracking 训练 → 评估。
日志位于 `logs/structure_rl/launch_<时间>/`，包含两个 console log 和 PID/GPU 记录 `jobs.json`。
每结构的 `status.json` 记录当前阶段、最新模型与错误；训练目录的 `complete.json` 仅在正常结束后写入。
可单独执行任一结构：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/mjlab-venv/bin/python -u -m structure_rl.pipeline --morphology twin_sync
CUDA_VISIBLE_DEVICES=1 /opt/mjlab-venv/bin/python -u -m structure_rl.pipeline --morphology raked_steer
```

上面两行默认前台运行；qz-cli 使用 detached job 启动总入口并记录 job ID。
首次云端 smoke 可能需要编译 Warp 内核，日志暂时不刷新不一定是卡死。

```bash
/opt/mjlab-venv/bin/tensorboard --logdir logs/structure_rl --host 0.0.0.0 --port 6006
```

## 评估与视频

将下面的 `CHECKPOINT` 替换为某个新训练目录里的 `model_*.pt`。必须同时保留其 `task.json` 和 `geometry.xml`。

```bash
python -m structure_rl evaluate --checkpoint CHECKPOINT --duration 15 --output results/comparison/evaluation.json
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m structure_rl video --checkpoint CHECKPOINT --duration 15 --speed 1 --turn 0 \
  --initial-speed 0 --video results/comparison/forward.mp4
```

评估自动生成 JSON 和同名 HTML，覆盖静止、正向、反向、左右转、停—走—停；
分别使用 0 与 0.8 m/s 初始化，并在 8 个环境中覆盖 ±1° 的初始 roll。
报告存活率、跟踪通过率、速度误差、累计航向误差、最大倾角、动作变化与饱和率。
不平滑动作；失败样本不会判为通过。失败环境仅为推进批量仿真而单独复位，此后样本全部排除。
视频在首次失败时停止。RMSE 只统计 3 秒之后仍存活的采样，
须结合存活率看，不能拿提前倒地后的低误差证明成功。
GPU1 是参考聊天中已验证可渲染的卡；训练期间不额外占用 GPU 录视频。

## 检查状态

本地 CPU MuJoCo 和动作映射测试：

```bash
python -m unittest tests.test_structure_rl tests.test_structure_concepts -v
```

13 项检查通过，包括实际限幅、同速约束、场景 attachment 后等式保留、关节索引映射、模型校验、失败环境的评估排除。
这不替代云端 MJLab/Warp/GPU smoke，也不代表训练收敛。
# Steering optimization profile (steer_v2)

The original baseline remains the default. The compatible 16-observation / 2-action
`steer_v2` profile preserves the geometry and finite-torque servos. It samples
20% stop, 20% forward straight, 20% reverse straight, 15% left turns, 15% right
turns, and 10% mixed maneuver commands every 4–8 seconds. The mixed group
samples forward straight, reverse straight, left and right with 25% probability
each, independent of actual velocity. Turns retain equal forward/reverse speed
probability. This covers straight/turn transitions, same-direction speed changes
and reversals. Overall probabilities are 20% stop, 22.5% forward straight,
22.5% reverse straight, 17.5% left and 17.5% right.
All nonzero speed magnitudes remain uniform over 0.3–1.0 m/s. Turn speeds have
equal forward/reverse probability. Resets are 50% stationary and 25% each moving
forward/reverse at 0.3–0.9 m/s. Command changes never overwrite robot velocity.

Yaw exponential reward weight increases from 2 to 4; a smooth absolute-error
cost with scale 0.3 rad/s and weight -0.5 distinguishes large yaw errors.
An upright, zero-speed/zero-yaw standing reward has weight 1. The revision
`mixed_maneuvers_pitch_20261020` increases pitch angular-velocity weight from -0.01
to -0.05 and adds squared pitch angle (radians) with weight -0.5. Pitch is computed
as atan2(gravity_x, norm(gravity_yz)), separating forward lean from roll.
These terms are also multiplied by policy dt (0.01 s). Action-rate, torque,
and roll penalties retain their weights. Newly started steer_v2 runs, including
checkpoint continuation, use this revision and record it in task.json. Previously
completed training used the older reversal distribution and pitch weight.
This profile does not add
absolute heading memory, so cumulative heading drift remains an evaluation metric.

Warm start loads the actor, critic and observation normalizers, resets optimizer
and iteration, restores bounded exploration std to 0.08 (bounds 0.02–0.2), and
uses a fixed 1e-4 learning rate. Optional sagittal mirror loss has coefficient 1;
coefficient 0 is the matched ablation. No augmented PPO likelihoods are used.
Raw observations are reflected before the model's own normalization.

`python -m structure_rl.optimize --checkpoint BASELINE_MODEL --output NEW_DIRECTORY`
runs GPU smoke, 2000 fine-tuning iterations, evaluation, 2000 more, and evaluation.
Pass `--mirror-coeff 0` for the ablation. Extended evaluation preserves all 96
original trials, then adds 48 trials starting at -0.8 m/s. Old and new reward
totals are not directly comparable; compare matched-case survival, tracking,
heading drift, roll and action variation instead.

### Continue the recommended ordinary policy with mixed maneuvers and pitch penalties

From the updated repository root, use the existing cloud checkpoint (keep its
sibling task.json and geometry.xml). This starts 2000 additional updates with no
mirror loss; optimizer/iteration/exploration are reset by --warm-start.

```bash
CUDA_VISIBLE_DEVICES=0 /opt/mjlab-venv/bin/python -u -m structure_rl train \
  --checkpoint /inspire/qb-ilm2/project/leverage-robot/ky26210/GyroRoll/experiments/structure_rl_steer_v2_20261019/logs/steer_v2/balanced/stage1/model_1999.pt \
  --morphology raked_steer --phase tracking --profile steer_v2 \
  --mirror-coeff 0 --warm-start --num-envs 256 --iterations 2000 \
  --run-dir logs/steer_v2/mixed_maneuvers_pitch
```

The output directory must not already exist. Evaluate the resulting policy before
extending training; the earlier experiment degraded with additional iterations.

## Rendering evaluated policies without mixing CUDA simulation and EGL

The cloud instance's physical EGL device 1 can render frames, but mixing live
Warp stepping and EGL in one process can abort. Record the actual GPU rollout
first, then render saved poses in a separate process. Rendering does not run
physics again or filter the policy actions. These demonstrations contain forward,
reverse, continuous forward-to-reverse, and stop commands, with explicit resets
between independent cases and visible termination markers on failures.

```bash
CUDA_VISIBLE_DEVICES=0 python -m structure_rl.record_trajectory --name balanced --stage 1
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=1 \
  python -m structure_rl.render_trajectory --name balanced --stage 1
```

Use `--stage 2` for the final model, or `--name balanced_mirror --stage 2`
for the final symmetry model. Run from the isolated `structure_rl_steer_v2_20261019`
experiment directory containing `logs/steer_v2`. Outputs go to
`results/steer_v2_trajectories` and `results/steer_v2_final`.
