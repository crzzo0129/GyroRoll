# GyroRoll

GyroRoll 是一个使用滚动轮与正交飞轮的两电机独轮机器人仿真项目。当前构型为 240 mm 开放滚动轮和上置的 166 mm 飞轮，滚轮上沿到飞轮下沿的竖直净距为 50 mm；基于 MJLab 1.3.0 的直接强化学习任务使用两路有符号目标轮速，学习前进/后退速度与转向角速度跟踪。训练在云端运行，配置与启动步骤见 [云端说明](CLOUD_README.md)。

项目同时保留早期模型、传统控制器、转向机构对照与实验记录。当前新构型的 RL 策略尚未训练，旧控制器结果与新 RL 任务分开评估。

离线优化脚本统一放在 [`optimize/`](optimize/)，用途与运行方法见 [优化脚本说明](optimize/README.md)。从项目根目录使用 `python optimize/optimize_nested.py --help`，或 `python -m optimize.optimize_nested --help`。

## 早期对照构型

本目录有两个独立构型：`wheelbot/validated_model.xml`（接地驱动轮 + 横滚反作用轮）；`steerbot/model.xml` 加上 `steerbot/negative_candidate.json` 中的几何覆盖（接地驱动轮 + 负拖距竖直转向关节）。`wheelbot/model.xml` 保留最初的宽胎/低力矩对照，单独的 `steerbot/model.xml` 则是最初的正拖距对照。都使用一个自由机身、一个圆柱轮胎/平面接触、两个有限力矩电机；不是预设轨迹曲率的运动学动画。

## 云端直接强化学习（MJLab 1.3.0）

新增 `miniwheelbot_rl/`：PPO 直接输出 `[飞轮目标角速度, 滚轮目标角速度]`，两路都支持正反转，学习前进/后退速度和转向角速度跟踪。轮速经有限力矩的原生速度伺服执行；身体平衡与转向由策略学习。

上传 [云端训练包](dist/miniwheelbot_mjlab130_cloud.zip)，按照 [云端部署与运行说明](CLOUD_README.md) 安装、检查、训练、续训、浏览器回放和评估。动作单位、奖励与物理设置见 [任务说明](discuss/直接强化学习训练.md)，观测维度、动作映射、奖励公式与 PPO 参数见 [训练配置 HTML](discuss/rl_training_spec.html)。本地未启动 PPO，完整任务仍需在云端验证，不能引用旧反馈控制器的通过率。

```bash
python -m miniwheelbot_rl smoke --num-envs 4
python -m miniwheelbot_rl train --stage warmup --num-envs 256 --iterations 3000
```

full 之后可用 `--stage full_robust --checkpoint FULL_RUN/model_XXXX.pt` 继续训练平地运动抗扰与静止：25% 的整段零速/零转向命令，加 0.5–1.5 N、0.10–0.25 s 的随机水平外力。该阶段从直立开始，跌倒结束回合；训练和专项评估命令见 [云端说明](CLOUD_README.md)。

## 本机运行

### 宽体封闭飞轮舱＋顶部载物托盘（两轮边缘净距 50 mm）

`wheelbot/nested_model.xml` 目前采用 **240 mm 加宽滚动轮、166 mm 上置飞轮**。滚轮绕 y 轴、飞轮绕 x 轴转动，轴线保持正交。直立时滚轮轴心离地 120 mm、上沿 240 mm；飞轮下沿 290 mm、轴心 373 mm、上沿 456 mm，两轮边缘净距为 50 mm。飞轮相对机身根节点 z = 3 mm。开放轮圈宽 20 mm、内孔直径 200 mm、外径 222 mm；接地胎面宽 24 mm、外径 240 mm。96 个凸扇区构成轮圈和胎面，保留真实空腔。轮环与飞轮没有互相禁用碰撞；机身与外轮的安装接触排除沿用原版。

机身采用宽体外壳，飞轮包在下部舱内，上部隔出设备舱，顶部为浅载物托盘。下部两侧削去直角：离地 270–410 mm 的 140 mm 高度内，外宽由底端 140 mm 渐扩至 220 mm；前后盖板及两侧碰撞壁板同步斜切，保留内部空腔。主外壳厚 160 mm（x）× 宽 220 mm（y），离地 270–550 mm；托盘外包络 172 × 232 mm，承载面离地 560 mm，围边高 20 mm，整机总高 580 mm。托盘内部可用面积为 156 × 216 mm。托盘标记仅表示预留绑带位置，尚未建模实体固定孔。

飞轮舱由独立碰撞壁板围成，不使用包住飞轮的实心碰撞盒。飞轮底边到舱底内表面留 12 mm，上边到隔板留 14 mm，上部直壁处左右各留 19 mm，下部斜壁到飞轮圆周的最小法向净距约 4.9 mm。上方设备舱的净空约 144 × 204 × 72 mm，与旋转件分隔；设备块是概念占位。后立柱与双叉连接轴承和滚轮。双叉中心间距为 50 mm，叉杆内侧距胎面侧边各 9 mm。

空载机身质量预算为 695 g：前后盖各 110 g、侧壁各 65 g、底板 40 g、隔板 45 g、托盘底板 100 g、四周围边共 40 g、设备占位 45 g、立柱 28 g、轴承座 12 g、双叉 20 g、前向标记 15 g。滚轮沿用 312 g 概念质量预算（加宽后按新宽度重算惯量，并非按原材料密度直接拉伸）、飞轮总成 458 g，模型总质量为 1.465 kg，直立全机重心离地约 350.4 mm。这些是仿真质量预算，未由材料厚度、实物电机和加工件核算，也未添加货物或指定额定载重。

[结构预览](wheelbot/structure_preview.png)直接渲染实际模型，默认采用 Cover Removed 展示，依次为拆盖立体视图、正面和侧面。飞轮正面只保留一条白色径向刻度。生成的 XML 默认隐藏正面盖板及其饰面，普通查看和新模型录像也可直接看到飞轮；这只是显示设置，盖板质量和碰撞仍然生效。`wheelbot/appearance.py` 定义无质量、无碰撞的饰面；外壳、隔板、载物托盘、轴承支架均在生成器内定义真实质量和碰撞几何。盖板外观倒角，碰撞采用保守盒形。灯带、散热纹和托盘标记仅用于外观表达。

[侧放接地对比](results/chamfer_rise/side_rest_comparison.png)：机器人从左右 90° 侧放自然静置 2 s 后，均停在约 76.56°，轮胎承担约 2.85 N（总重的 19.8%）。修改前自然侧躺约 88.77°，轮胎离地约 92.2 mm；修改后有真实轮地接触。2 / 1 / 0.5 ms 物理步长下均验证了接地。这里的侧躺指自然支撑姿态，并非强制维持精确 90°。沿用旧起身控制参数仍未成功；针对新支撑状态重新设计的控制验证见下文。

### 新机身的本地起身控制验证

采用“两轮协同改变支撑 → 动态翻起 → 预测制动 → 平衡接管”。`wheelbot/recovery_control.py` 每 40 ms 用真实有限力矩速度伺服模型预测未来 0.4 s，修正两轮目标转速；3.2 s 参考动作结束后切入强 roll 阻尼、飞轮动量反馈和陀螺耦合补偿。电机上限为 2.4 N·m，目标速度为飞轮 ±250 rad/s、滚轮 ±35 rad/s。

左右侧放 × 2 / 1 / 0.5 ms 物理步长，六项确定性回放均在 **4.09–4.20 s 达到连续稳定 1 s 的判据，并保持至 15 s 结束**。只播放固定动作或使用轨迹周围的时变 LQR 均未通过同样的六项检查。完整 [起身报告](results/chamfer_recovery/report.html)、[视频](results/chamfer_recovery/recovery.mp4) 和 [参数](results/chamfer_recovery/controller.json) 已保存。相关单元及回归检查共 17 项通过。

这是空载、平地、完整仿真状态观测的局部可行性实验；动作含明显翻滚和接触冲击，未验证载物、坡地或扰动随机化。预测计算尚未达到实时要求，也没有替换或训练 PPO。最终单一 policy 仍需独立训练与验收。

```powershell
.venv-mjlab/Scripts/python.exe probe_rise.py --controller results/chamfer_recovery/controller.json --reference results/chamfer_recovery/reference.npz --feedback mpc --duration 15 --output results/chamfer_recovery/replay.json --video results/chamfer_recovery/replay.mp4
```

另一侧增加 `--roll -90`；细步长增加 `--timestep .001` 或 `--timestep .0005`。没有 `--reference ... --feedback mpc` 时只是固定动作回放，不代表最终控制方案。报告保存几何哈希与快照，旧几何不能直接套用这条轨迹。

当前空载布局通过 15 项相关检查，包含轮间/轮壳间隙、倾倒与坡地初始接触、速度伺服，以及传统控制器 15 s、1 m/s 名义直行回归。尚未验证承载能力，也未训练该结构的载物 RL 策略。后续需要加入货物质量和重心偏移随机化，再验证速度跟踪、坡地与起身；载物时的货物固定方式也需要明确。

旧 checkpoint 仍加载运行目录内的 `geometry.xml`，不能通过替换项目 XML 改变旧策略回放的机器人。新布局从不带 `--checkpoint` 的 warmup 训练开始；不要覆盖旧运行目录的几何快照。

几何由 `wheelbot/generate_nested.py` 生成，控制器参数在 `wheelbot/nested_controller.json`；运行时使用 `--controller nested` 自动加载这套模型。该变型的验收结果与原版分开保存，原版通过率不能转移到新模型。电机、轴承和支架仍是理想刚体安装，尚未完成硬件布置设计。

**观看运动：**在文件夹中双击 `run_nested.cmd` 打开当前模型的交互窗口。上/下方向键调整速度，左/右方向键输入实验转向指令，空格将转向归零，`R` 重置。建议先观察默认 1 m/s 直行。历史 [12 秒直行视频](wheelbot/nested_open_motion.mp4) 和 [旧三视图](wheelbot/nested_open_preview.png) 属于改动前的同中心布局。

本机仿真依赖已安装。其他 Windows 机器配置和运行示例：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-sim.txt
./run_nested.ps1 -Speed 1 -Duration 60
./run_nested.ps1 -Headless -Speed 1 -Duration 12 -Video wheelbot/nested_open_motion.mp4
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe accept.py wheelbot --controller nested --seeds 5
.venv/Scripts/python.exe wheelbot/preview_nested.py
```

`run_nested.ps1` 打开带控制器的交互仿真；加 `-Headless` 可直接进行无窗口回放。反馈控制与几何测试不需要 PyTorch，回放 PPO checkpoint 和训练仍需要原有训练依赖。没有已训练的新布局 PPO 策略。

历史同中心布局使用 20 rad/s 飞轮预旋转，1 m/s 无扰动直行回放 60 秒未跌倒，3–60 秒速度 RMSE 0.027 m/s，采样最大横滚角 0.004 rad；完整轨迹在 `results/wheelbot_nested_open/straight_60.json`。这些历史数据不代表当前上置布局。视频对应的 12 秒回放和轨迹在同目录的 `straight_12.json`。偏航/侧向误差按 `sim.py` 原有的 5.5–10.8 秒窗口计算。

历史同中心布局的 2 m/s 左右转向验收在质量/摩擦随机化下五种子均在约 3.7–3.9 秒加速阶段跌倒，完整验收 **0/5**。结果在 `results/wheelbot_nested_open/acceptance.json`。键盘/手柄转向入口用于实验；几何改变、直行不跌倒和完整操纵验收必须分开判断。`optimize/optimize_nested.py` 可通过统一的 `sim.py` 回放继续调参，候选先保存在 `results/nested_tuning/`，不会自动替换控制配置。

早期 48 cm 布局留作历史对照：模型、配置和图片分别是 `wheelbot/nested_48cm_model.xml`、`wheelbot/nested_48cm_controller.json`、`wheelbot/nested_48cm_preview.png`；`results/wheelbot_nested/` 与 `results/nested_straight_60_summary.json` 均属于该旧布局，不是当前 24 cm 模型的结果。

上一个 24 cm 宽轮筒布局留存在 `wheelbot/nested_24cm_enclosed_model.xml`、`wheelbot/nested_24cm_enclosed_controller.json`、`wheelbot/nested_24cm_enclosed_preview.png`、`wheelbot/nested_24cm_enclosed_motion.mp4`；`results/wheelbot_nested_24cm/` 对应该旧布局。

以下是原 Linux 实验机的运行方式，`zhu_mjlab` 环境含 MuJoCo 3.6 / mjlab 1.2 / RSL-RL 5.0.1。所有命令从本目录执行：

```bash
PY=/home/legged-ma/miniforge3/envs/zhu_mjlab/bin/python
$PY -m unittest discover -s tests -v
$PY -m mujoco.viewer --mjcf steerbot/validated_model.xml
$PY sim.py wheelbot --controller thin
$PY sim.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2
$PY -m mujoco.viewer --mjcf steerbot/hill_model.xml
$PY sim.py steerbot --model steerbot/hill_model.xml --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2 --turn 0 --duration 60
$PY sim.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2 --side-push-force 1 --side-push-start 8.5 --side-push-duration .2 --duration 20
$PY sim.py wheelbot --controller thin --headless --test --duration 15 --dr --disturbance
```

viewer 命令只打开合并后的负拖距 XML 用于查看，没有平衡控制器；直接播放会倒。仿真命令则加载原始 `steerbot/model.xml` 并按 `negative_candidate.json` 覆盖几何，和该独立 XML 等价。MuJoCo 交互窗口：手柄 1 的左摇杆上下为速度（飞轮版最高 2.0 m/s，转向版最高约 2.2 m/s），左右为目标偏航角速度（限幅 ±0.35 rad/s）；也可用上/下方向键每次增减 0.2 m/s，左/右方向键每次增减 0.1 rad/s，空格归零转向，`R` 重置。转向版的手柄中位和键盘减速下限是 0.6 m/s 的慢速巡航，因为它不能零速横滚平衡。`--speed` 和 `--turn` 是默认手柄指令，注意转向版重置时仍给予 0.8 m/s 初速度，所以 `--speed 0` 不构成零初速站立测试。**两版均须显式指定上面的验收控制器**；无参数时是原始对照基线。飞轮版验收配置预旋转 100 rad/s；持续同向打满转向会积累飞轮转速，不能保证无限时长圆周行驶。转向版通过 0.15 m/s 与 0.15 rad/s 冲击的正反转向测试，但 0.25 强冲击镜像尚未达标。转向版各颜色部件、精确控制律和手柄中位行为详见 `steerbot/report.html`。

独立缓丘场景的坡段沿世界 x=2–6 m，峰高 6 cm；带控制器的 `sim.py` 命令才可用手柄操纵。外力选项直接对机身施加世界 +y 方向 1 N、0.2 s 的载荷，不是旧的 `--disturbance` 速度冲击。坡道直行五个 DR 种子 5/5 严格通过；外力实验五个种子都未跌倒，但包含受力瞬态的严格偏航阈值只通过 3/5。转向靠轮地侧向反力间接稳定横滚，没有直接横滚电机；受力后偏航角速度恢复并不保证回到原路径。通俗的完整推导见 [LQR 控制器图解](steerbot/lqr_explained.html)，模型、录屏和逐种子验收见 [转向版报告](steerbot/report.html)。

## 训练与回放

```bash
$PY train.py wheelbot --stage 1 --num-envs 256 --iterations 250
$PY train.py wheelbot --stage 2 --num-envs 256 --iterations 350 --resume '上一阶段运行目录名'
$PY train.py wheelbot --stage 3 --num-envs 256 --iterations 800 --resume '上一阶段运行目录名'
$PY train.py wheelbot --stage 4 --num-envs 256 --iterations 1500 --resume '第三阶段运行目录名'

$PY train.py steerbot --stage 1 --num-envs 256 --iterations 500
$PY train.py steerbot --stage 2 --num-envs 256 --iterations 500 --resume '上一阶段运行目录名'
$PY train.py steerbot --stage 3 --num-envs 256 --iterations 800 --resume '上一阶段运行目录名'
$PY train.py steerbot --stage 4 --num-envs 256 --iterations 1800 --resume '第三阶段运行目录名'
$PY train.py steerbot --stage 5 --num-envs 256 --iterations 1800 --resume '第三阶段运行目录名' --load-checkpoint model_1080.pt
$PY train.py steerbot --stage 6 --trail .5 --num-envs 256 --iterations 1800 --resume source_stage3 --load-checkpoint model_1080.pt

$PY sim.py steerbot --checkpoint logs/rsl_rl/steerbot_v2/RUN/model_XXX.pt
$PY sim.py wheelbot --checkpoint logs/rsl_rl/wheelbot_v2/RUN/model_XXX.pt --rotor-speed 110
$PY sim.py steerbot --controller optimized --speed 2 --turn .3
$PY sim.py steerbot --controller lqr --speed 2 --turn .3
$PY sim.py wheelbot --controller tilted --controller-params wheelbot/tilted_axis_probe.json --speed 2 --turn .3
$PY sim.py steerbot --controller optimized --trail .5 --head-mass .005 --controller-params steerbot/lightfork.json --speed 2 --turn .3
$PY sim.py steerbot --controller lqr --lqr-tuning steerbot/lqr_negative_015.json --trail -.15 --tire-halfwidth .004 --steer-gear 2.4 --body-mass .645 --body-offset-z -.062 --head-mass .019 --speed 2 --turn .3
$PY sim.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --speed 2 --turn .3
```

`stage 1` 是 0.6–1.3 m/s 直行；`stage 2` 是 1.3–2.2 m/s，加入质量/摩擦 DR 与定时速度扰动；`stage 3` 扩展到偏航指令 ±0.65 rad/s；`stage 4` 聚焦 1.8–2.15 m/s、±0.4 rad/s，强化偏航奖励并放宽横滚惩罚；`stage 5` 对转向版进一步奖励低侧向速度。后续 stage 7–9 缩短指令切换间隔、延长回合并强化偏航奖励。Wheelbot 第三、四阶段从反作用轮 110 rad/s 的预旋转状态开始，不能据此宣称零速自旋启动。训练结果保存于 `logs/rsl_rl/`，不会覆盖旧 checkpoint；训练奖励不等于独立验收。第五至九阶段抽样 checkpoint 回放尚未达到完整验收。

`optimize/optimize_steer.py` 是 CPU 物理轨迹上的差分进化反馈调参；`linearize_steer.py` 在 2 m/s 工作点数值线性化，`optimize/optimize_lqr.py` 搜索 LQR 权重。这两种控制器与 PPO 分开验收；不能把它们的性能计入 PPO。`optimize/optimize_wheel.py` 的陀螺步态探索目前未达到转向标准。

额外实验包括 `probe_tilted_flywheel.py`（将飞轮轴倾斜 0.4 rad）与 `optimize/optimize_tilted_robust.py`，以及 `optimize/optimize_longtrail_sustained.py`、`optimize/optimize_lightfork.py`（0.50 m 拖距、5 g 转向头）。这些都是探索性概念变型，不是原发布 Wheelbot 的几何。飞轮轴倾斜反馈在名义转向下接近验收，但多指令和 DR 下仍不可靠；轻量长叉的名义偏航 RMSE 0.164 rad/s 在 DR 下退化至 0.406 rad/s。飞轮变型还运行了第六阶段 PPO，抽样 checkpoint 名义回放并未学出转向；轻量长叉第七阶段 PPO 加快左右指令切换并使用物理一致的质量/惯量和逐环境轮胎摩擦 DR。后者仍须独立验收。

第六阶段是独立的 0.5 m 拖距概念变型，以前向、偏航和侧向速度平方误差作为附加代价。运行在 `steerbot_trail050` 实验目录；恢复第三阶段 checkpoint 时，该目录下的 `source_stage3` 是指向原第三阶段运行目录的符号链接。长拖距只有理想刚体关节，未建模实体叉臂质量/柔度，不应视作已定型硬件。

## 验收与报告

```bash
$PY accept.py wheelbot --checkpoint logs/rsl_rl/wheelbot_v2/RUN/model_XXX.pt --rotor-speed 110 --seeds 5 --video
$PY accept.py wheelbot --controller thin --seeds 5 --video
$PY accept.py wheelbot --controller thin --seeds 5 --mirror --kick-scale .6 --output-dir results/wheelbot_moderate_mirror
$PY accept.py steerbot --checkpoint logs/rsl_rl/steerbot_v2/RUN/model_XXX.pt --seeds 5 --video
$PY accept.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --kick-scale .6 --seeds 5 --video --output-dir results/steerbot_moderate
$PY accept.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --kick-scale .6 --mirror --seeds 5 --output-dir results/steerbot_moderate_mirror
$PY accept.py steerbot --controller lqr --lqr-tuning steerbot/negative_candidate.json --no-disturbance --test-turn-amplitude 0 --side-push-force 1 --side-push-duration .2 --seeds 5 --video --output-dir results/steerbot_side_force
$PY accept.py steerbot --model steerbot/hill_model.xml --controller lqr --lqr-tuning steerbot/negative_candidate.json --no-disturbance --test-turn-amplitude 0 --seeds 5 --output-dir results/steerbot_hill_accept
$PY accept.py steerbot --controller optimized --seeds 5 --video
$PY accept.py wheelbot --controller tilted --controller-params wheelbot/tilted_axis_probe.json --seeds 5 --video --output-dir results/wheelbot_tilt040
$PY accept.py steerbot --controller optimized --trail .5 --head-mass .005 --controller-params steerbot/lightfork.json --seeds 5 --video --output-dir results/steerbot_lightfork
$PY generate_reports.py
```

无显示器录视频时脚本自动设置 `MUJOCO_GL=egl`；也可单独执行 `MUJOCO_GL=egl $PY sim.py wheelbot --controller thin --headless --test --video /tmp/rollout.mp4`。验收按 15 s 指令序列测 2 m/s 阶跃、左右 ±0.35 rad/s 转向、两次侧向/横滚速度扰动，并逐种子随机化机身质量 ±15% 和轮胎摩擦 1.1–1.9。默认强扰动每次对机身侧向速度和横滚角速度分别施加 ±0.25 m/s、±0.25 rad/s；`--kick-scale .6` 为对应的 ±0.15。**通过标准**：整段不跌倒，3 s 后前向速度 RMSE < 0.35 m/s，3–4.8 s 直行、5.5–10.8 s 转向和 12–15 s 回正的偏航角速度 RMSE 各 < 0.15 rad/s，转向侧向速度 RMSE < 0.35 m/s。可加 `--mirror` 测相反转向顺序；候选几何可加 `--com-offset`，实验输出用 `--output-dir` 与正式报告隔离。结果文件在 `results/`；两个 HTML 报告在 `wheelbot/report.html` 与 `steerbot/report.html`，数字由实测 JSON 生成。

模型使用刚性圆柱轮胎、理想电机力矩与理想状态观测，不等同于实物动力学。报告中未通过的项目会明确标为未通过。

## 当前结论

除标注为中等扰动的最后一行外，均为五种子、15 秒、强扰动与 DR 的独立 CPU 回放；下表依次为稳态前向速度、转弯偏航、转向侧向速度 RMSE。

| 构型与回放控制 | 速度 m/s | 偏航 rad/s | 侧向 m/s | 通过 |
| --- | ---: | ---: | ---: | ---: |
| 飞轮版直行反馈，初始转子 110 rad/s | 0.185 | 0.402 | 0.038 | 0/5 |
| 飞轮版窄胎/高力矩反馈，初始转子 100 rad/s，强扰动 | 0.234 | 0.112 | 0.017 | 5/5（镜像 5/5） |
| 0.12 m 拖距转向版优化反馈 | 0.140 | 0.271 | 0.528 | 0/5 |
| 0.50 m 拖距转向版第六阶段 PPO | 0.334 | 0.295 | 0.096 | 0/5 |
| 0.4 rad 倾斜飞轮轴反馈，预旋转 110 rad/s | 0.118 | 0.254 | 0.045 | 0/5 |
| 0.50 m 拖距、5 g 转向头反馈 | 0.159 | 0.406 | 0.113 | 0/5 |
| -0.15 m 拖距、下移质心与窄胎 LQR | 0.077 | 0.273 | 0.141 | 0/5 |
| -0.145 m 拖距、窄胎 LQR 抗扰候选 | 0.074 | 0.137 | 0.228 | 5/5（镜像 0/5） |
| -0.145 m 拖距、同一 LQR，中等扰动 | 0.073 | 0.130 | 0.223 | 5/5（镜像 5/5） |

**在定义的 15 秒 2 m/s 双向操纵任务上，两构型均通过了指定范围内的扰动/DR 验收；任意时长操纵与真实硬件未得到保证。** 飞轮窄胎版在强冲击下预定、镜像转向各 5/5，扩展 5–14 号正向 10/10、镜像 9/10；中等冲击的 0–14 号两种顺序各 15/15。选择参数后才运行的 15–29 号中等冲击留出种子正向 14/15、镜像 15/15；29 号虽未跌倒，回正偏航 RMSE 达 1.253 rad/s，超过 0.15 阈值。飞轮版从 100 rad/s 预旋开始；主轮半宽 4 mm、飞轮电机上限 2.4 N·m，与最初 16 mm/0.48 N·m 模型不同。原转向版存在横向斜行；长拖距与窄胎变型仍未通过偏航验收。单向转弯的名义调参结果不能替代反向切换与扰动验收。

早期 -0.15 m 负拖距候选在不加扰动的名义轨迹上可以 15 秒稳定完成左右转（转弯偏航 RMSE 0.136 rad/s），但 8.5 秒的侧向/横滚冲击会引发大幅偏航振荡；上述 0/5 是含 DR 与扰动的结果。`steerbot/report.html` 提供其对照视频，正拖距模型的原始验收仍单列。

在负拖距附近进一步筛选几何和重调指令前馈后，`steerbot/negative_candidate.json` 在中等冲击下，对预定和镜像顺序各五种子达到 5/5，两种顺序 5–14 号种子各 10/10，15–29 号真正留出种子各 15/15。强冲击预定顺序五种子 5/5、扩展十种子 10/10，但镜像 0/5（偏航 RMSE 约 0.170 rad/s）；双向强冲击优化版独立回放仍仅正向 3/5、镜像 4/5。此构型 2 m/s 下单向保持 ±0.35 rad/s 的额外 20 秒测试对种子 0 可持续；这并不代表随机手柄轨迹或所有 DR 种子都通过。飞轮版 2 m/s 直线 20 秒可持续，但从起步一直保持同向 ±0.35 rad/s 可能在 10–17 秒后因飞轮动量积累跌倒；主验收种子 0 在 3 秒后的瞬时偏航峰值约 1.35 rad/s，不能由分段 RMSE 合格推断全程平顺。两版都是理想状态、刚性轮胎的仿真概念验证，不应外推实物。
