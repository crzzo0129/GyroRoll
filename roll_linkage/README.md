# 连杆驱动滚轮 Roll 摆动：概念模型 V1

滚动电机保留，第二个电机通过平行四连杆驱动轮架绕机身前后轴（x）摆动。
坐标：x 向前、y 向左、z 向上。没有飞轮，也没有独立 yaw 转向关节。
保留 V1 机构演示，V2 躯干水平目标模型已接入 LQR 验证和强化学习速度跟踪。

## 强化学习速度跟踪

### 当前实验：linkage_v4，初始化后继续 PPO

2026-10-20 的迭代发现：纯 PPO 的 V1/V2 都没有学会站立；V3 将局部 LQR
反馈初始化为普通 MLP 权重后，GPU 默认 TF32 精度会破坏高增益反馈。
关闭 TF32、使用完整 FP32 后，同一网络在 ±0.1° 初始 Roll/Pitch 的 16 个
独立站立测试中全部维持 12 秒。这个结果来自初始化，不能称为 RL 已学会速度跟踪。

V4 使用 ReLU 的正负通道表示线性反馈，避免 V3 的大偏置相消。
全部网络参数仍参与 PPO 更新；运行时只有神经网络，没有外接 LQR、隐藏力、
姿态锁定或额外执行器。2 次 PPO 更新、学习率 1e-6 后，站立测试仍 16/16 通过。
同一模型在 ±0.3 m/s 起步测试中仍失败，说明运动能力尚未验证成功。

- 保持 V2 的奖励与真实物理模型。探索 std 初始 0.01、范围 0.005–0.04，
  熵系数 0.001、gamma 0.995、lambda 0.99、每次采集 128 步。
- 只在 reset 时按轮胎最低点放置机器人，补偿约 0.579 mm 静态接触压缩；
  运动过程中没有高度修正。另有 `--airborne-reset` 检查离地释放的鲁棒性。
- `--graded-moving` 将运动课程拆为 10%、25%、50%、100%：训练速度范围分别
  ±0.04/0.10/0.20/0.40 m/s；独立测试为 ±0.03/0.075/0.15/0.30 m/s。
  初始倾角从 ±0.12° 逐步到 ±0.3°，速度误差阈值同比缩小。
  小课程通过只记作 `lesson_passed`，不算最终运动阶段 `promote`。
- 每轮评估退化则回退到上一份保留模型（包含优化器状态）并将学习率减半；
  接受新模型前还复查所有已通过课程。达到预算或学习率下限会停为
  `needs_revision`，不会自动把失败策略推进下一阶段。
- tracking 最终仍测试停车、前进、倒车、前后行驶时的左右转，以及前进切倒车；
  每条命令 16 次、15 秒，验收标准不因早期课程而降低。

可复现命令（新目录，GPU0）：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/mjlab-venv/bin/python -u -m structure_rl train \
  --morphology roll_linkage --profile linkage_v4 --phase warmup \
  --seed-only --num-envs 256 --run-dir logs/linkage_v4_seed
CUDA_VISIBLE_DEVICES=0 /opt/mjlab-venv/bin/python -u -m structure_rl.linkage_curriculum \
  --profile linkage_v4 --checkpoint logs/linkage_v4_seed/model_0.pt \
  --output logs/linkage_v4_gated --graded-moving --num-envs 256 \
  --learning-rate 1e-6 --block-iterations 10 --max-blocks 24
```

V3 使用 ELU，V4 使用 ReLU；不能把 V3 checkpoint 改配置名后当 V4 加载。
`--seed-only` 保存的是零次 PPO 更新的初始化模型，不生成训练完成标记。

### 历史实验：linkage_v2，按评估结果升级

V1 在 2500 次 warmup 后独立评估全部失败，但旧流程仍按迭代数进入 tracking。
故已停下 V1 作为对照；其诊断显示输入摇臂目标接近 −30° 后侧倒，
不是电机力矩饱和。V2 保持几何、2.4 Nm 电机、20 维观测和两个动作不变：

- warmup 仅零速度命令，初始 Roll/Pitch ±0.1°；moving 从静止学 ±0.4 m/s，
  初始倾角 ±0.3°；tracking 才启用完整混合命令和移动初始状态。
- 增加直立附近敏感的姿态奖励 +4、存活奖励 +2、摆角超出 65% 范围的软惩罚 −2。
  摔倒惩罚提高到 −100，动作变化惩罚先降低至 −0.02；其他姿态惩罚保留。
- 探索标准差初始 0.08，范围 0.04–0.15；熵系数 0.005，折扣 0.995，
  每次采集 64 步；新策略输出层缩小至接近零目标，没有 LQR 或额外稳定器。
- 每 300 次迭代做独立确定性评估，每条命令 16 次从静止起步；
  站立评估 12 秒，运动评估 15 秒。每条命令存活率 ≥90%、通过率 ≥80%、
  闭环误差 <1 mm 才升级；单次通过同时要求速度/转向误差及 Roll/Pitch
  RMS <5°、峰值 <20°。训练回报不作为升级依据。
- 每阶段最多 8 个区间。未通过则以 `needs_revision` 停下并保留模型，
  不会继续扩大任务。通过课程不等于完成载物或现实可行性验证。

```bash
CUDA_VISIBLE_DEVICES=0 /opt/mjlab-venv/bin/python -u -m structure_rl.linkage_curriculum \
  --output logs/roll_linkage_v2 --num-envs 256 \
  --block-iterations 300 --max-blocks 8
```

每个区间保存 `task.json`、模型、`gate.json`；根目录 `status.json` 保存升级历史。
旧 `structure_rl.pipeline --morphology roll_linkage` 入口已阻止无门槛升级。

### 历史基线：linkage_v1

使用 `model_level_torso.xml`，配置 `linkage_v1`。机身自由运动，
物理四连杆 connect 约束保持不变，只有滚轮速度和输入摇臂角度两个主动输入。
动作裁剪到 ±1 后分别映射为 ±35 rad/s 和 ±30°，两电机力矩上限均为 2.4 Nm。
策略观测为 20 维：机身线速度 3、角速度 3、重力方向 3、
主动电机状态 3、被动轮架和连杆角度/角速度 4、速度命令 2、上次动作 2。
主动状态为滚轮转速、输入摇臂角度和角速度。
没有 LQR 辅助、隐藏外力或机身姿态锁定，也不加载其他结构的策略。

物理步长 1 ms、策略周期 10 ms、求解器 100 次迭代。
先 2500 次低速 warmup（±0.5 m/s、±0.15 rad/s、25% 停车，从静止初始化），
再接同一策略 5000 次 tracking。tracking 命令每 4–8 秒重新采样：
20% 停车、22.5% 前进直行、22.5% 倒车直行、17.5% 左转、17.5% 右转；
非零速度幅值 0.3–1 m/s，转向幅值 0.1–0.3 rad/s。
转弯命令同样覆盖前进和倒车，换向不依赖机器人当前实际速度。
tracking 初始速度为 50% 静止、25% 前进、25% 倒车；这是初始条件，
不代表策略已学会起步。阶段结束分别评估静止起步与移动起步。

奖励权重：速度跟踪 +4、转向跟踪 +4、直立 +1、停车稳定 +1；
侧向速度平方 −0.5、Roll 角速度平方 −0.05、Pitch 角速度平方 −0.05、
Roll 角度平方 −2、Pitch 角度平方 −0.5、动作变化平方 −0.1、
电机力矩平方 −0.005、超速 −1、摔倒 −10、转向误差软惩罚 −0.5。
沿用环境按策略步长积分奖励的实现；Roll/Pitch 目标作用在上方躯干。
记录速度误差、转向误差、躯干姿态和闭环端点距离。

旧实验先做 GPU 冒烟检查，再串行训练两阶段并评估，训练状态保存在
`logs/structure_rl/roll_linkage/level_torso_v2_tracking/status.json`。
“已接入”不代表训练或可行性验证已经成功；以评估报告和视频为准。

## 躯干水平版 V2 与 LQR 验证

**[完整报告与视频](../results/roll_linkage/lqr/level_torso_v2/report.html)**。

`model_level_torso.xml` 保留轮架相对躯干的 Roll 自由度，躯干没有世界姿态约束。
LQR 参考躯干 Roll/Pitch 为 0。为避免前置机构要求长期前倾，从原躯干质量中划出
0.5 kg 作为后置设备块（x=−67.775 mm、y=+3.8 mm），外壳前后长度改为 220 mm。
整机质量仍为 1.542 kg，直立总重心的 x/y 偏移接近零。

两个物理步长、五种速度、左右初始倾斜、起步与侧推共 78 个测试：
20 个完整存活，12 个满足本轮水平姿态和速度阈值。两种步长的通过情况一致。
静止 +0.1° 偏差可以恢复，但躯干瞬态达到约 4.15°；+0.5° 偏差虽可恢复，
瞬态达到约 23°；+1° 约 0.75 s 失败。少数倒车样本通过姿态/速度阈值，
但存在明显航向漂移，不能当作稳定直行已通过。

结论：运动学上可以“下方摆、上方不跟随”，但动力学上仍有反作用力；
当前有限摆角机构和局部 LQR 尚未达到载物所需的稳定水平姿态。
这不构成对其他控制器或几何设计的不可行性证明。

复现（项目根目录）：

```powershell
.\.venv-mjlab\Scripts\python.exe -m roll_linkage.validate_lqr
.\.venv-mjlab\Scripts\python.exe -m roll_linkage.report_lqr
```

正式结果在 `results/roll_linkage/lqr/level_torso_v2/`；
`lqr.py` 是控制和仿真模块，`validate_lqr.py` 保存逐例轨迹与指标，
`report_lqr.py` 只读取轨迹生成视频和报告，不重新仿真物理。
下文描述 V1 的原始机构尺寸及演示。

## 结构

- 上方：机身安装的 Roll 电机，输出轴沿 x，驱动 55 mm 输入摇臂。
- 中间：140 mm 连接杆，两端通过转动副/闭环约束连接。
- 下方：55 mm 输出摇臂与轮架刚性相连，带动滚轮倾斜。
- 输入、输出摇臂等长，两固定轴间距 140 mm，形成平行四边形。
- Roll 转轴位于直立地面上方 280 mm；轮轴位于其下方 160 mm。
- 滚轮直径 240 mm、胎宽 24 mm，复用斜轴概念模型的轮胎几何与惯量。
- 顶部保留载物托盘，模型总质量约 1.542 kg；质量为概念估计。

轮架不是只在原地改变外倾角：因 Roll 转轴高于轮轴，轮心也会左右移动。
30° 摆角时，轮心相对机身横移约 80 mm、上移约 21.4 mm。
这可能提供接地点调整能力，但平衡能力、转弯能力需进一步验证。

## 两个主动输入

| 电机 | 控制量 | 范围 | 力矩上限 |
|---|---|---|---|
| 滚动驱动 | 滚轮目标转速 | ±35 rad/s | 2.4 Nm |
| 连杆驱动 | 输入摇臂目标角度 | ±30° | 2.4 Nm |

滚动速度伺服 kv=0.25；连杆位置伺服 kp=8、kd=0.25。
输出轮架的 ±0.55 rad 限位略宽于输入限位，避免重复限位数值冲突。
连接杆关节、轮架 Roll 关节均被动，只有两个 actuator。
闭环通过 MuJoCo connect 约束传力，不通过逐帧改写关节角或伪造同步实现。

## 文件与检查

- `model.xml`：自由机身、重力、轮胎地面接触，可用于后续控制验证。
- `fixture.xml`：固定机身，轮胎直立离地 30 mm，仅用于机构检查。
- [机构视频](../results/roll_linkage/concept_v1/mechanism.mp4) / [预览图](../results/roll_linkage/concept_v1/preview.png)：固定机身演示，±25° 正弦目标，滚轮同时转动。
- [validation.json](../results/roll_linkage/concept_v1/validation.json)：实际检查结果。
- `build.py`：可复现的模型生成、检查与渲染。

在项目根目录运行：

```powershell
.\.venv-mjlab\Scripts\python.exe roll_linkage/build.py --render
```

已检查 6 秒固定机身机构运动的闭环误差、电机力矩和输出摆角，以及 2 秒
自由机身动力学是否保持有限数值。固定机身演示不是平衡或行驶成功的证据。
连杆、轮架、轴承和电机壳目前关闭碰撞，用于隔离闭环机构动力学；
机身和滚轮有碰撞。实体干涉、轴承尺寸、强度和实际装配需要下一轮机械设计检查。
当前预览是概念外形，不是制造图。
