"""Train, smoke-test, view and evaluate the direct RL wheel-speed policy."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from importlib.metadata import version
import json
import hashlib
import os
from pathlib import Path

from .settings import Settings, STAGES
from .policy_config import POLICY_VERSION, checkpoint_policy_version


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "smoke", "play", "evaluate", "video", "diagnose"))
    p.add_argument("--stage", choices=STAGES, help="default: checkpoint stage, otherwise warmup")
    p.add_argument("--num-envs", type=int, help="default: train 256, smoke 4, evaluate 8, otherwise 1")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--output", type=Path, default=Path("results/nested_rl/evaluation.json"))
    p.add_argument("--diagnostic-dir", type=Path, default=Path("results/nested_rl/diagnostic"))
    p.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/nested_velocity_mjlab130"))
    p.add_argument("--flywheel-limit", type=float, default=250.0, help="signed rad/s")
    p.add_argument("--rolling-limit", type=float, default=35.0, help="signed rad/s")
    p.add_argument("--viewer", choices=("viser", "native"), default="viser")
    p.add_argument("--viewer-host", default="127.0.0.1")
    p.add_argument("--viewer-port", type=int, default=8080)
    p.add_argument("--video", type=Path, default=Path("results/nested_rl/rollout.mp4"))
    p.add_argument("--speed", type=float, default=0.5, help="video/diagnose forward command, m/s")
    p.add_argument("--turn", type=float, default=0.0, help="video/diagnose heading-rate command, rad/s")
    return p


def build(args):
    if version("mjlab") != "1.3.0":
        raise RuntimeError("This task requires mjlab==1.3.0. Install requirements-mjlab.txt.")
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from .task import make_env_cfg, make_agent_cfg
    from .model import MODEL

    wp.config.kernel_cache_dir = str(Path(__file__).resolve().parents[1] / ".cache" / "warp")

    if args.num_envs < 1 or args.iterations < 1 or args.duration <= 3:
        raise ValueError("num-envs/iterations must be positive and duration must exceed 3 s")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch and check NVIDIA driver.")
    configure_torch_backends()
    settings = Settings(flywheel_speed_limit=args.flywheel_limit,
                        rolling_speed_limit=args.rolling_limit)
    model_path = MODEL
    policy_version = POLICY_VERSION
    # A checkpoint's physical action mapping must be reproduced in evaluation.
    if args.checkpoint:
        manifest_path = args.checkpoint.parent / "task.json"
        if not manifest_path.exists():
            raise ValueError("Checkpoint needs its sibling task.json; old torque policies are incompatible")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        policy_version = checkpoint_policy_version(manifest, training=args.mode == "train")
        if manifest["action_order"] != ["reaction", "drive"] or manifest["mjlab"] != "1.3.0":
            raise ValueError("Incompatible action contract or MJLab version")
        settings = Settings(**manifest["settings"])
        if args.stage is None:
            args.stage = manifest["stage"]
        model_path = args.checkpoint.parent / "geometry.xml"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["geometry_sha256"]:
            raise ValueError("Checkpoint geometry snapshot does not match task.json")
    args.stage = args.stage or "warmup"
    env_settings = (replace(settings, episode_seconds=max(settings.episode_seconds, args.duration + 1))
                    if args.mode in ("evaluate", "video", "diagnose") else settings)
    cfg = make_env_cfg(args.num_envs, args.stage, settings=env_settings,
                       play=args.mode in ("evaluate", "play", "video", "diagnose"), randomize=args.randomize,
                       model_path=model_path)
    if args.mode in ("evaluate", "video", "diagnose"):
        cfg.commands["twist"].resampling_time_range = (1e8, 1e8)
        cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.seed = args.seed
    if args.mode == "video":
        cfg.viewer.width, cfg.viewer.height = 960, 544
        cfg.viewer.azimuth, cfg.viewer.elevation = 135.0, -20.0
    env = ManagerBasedRlEnv(cfg, device=args.device,
                           render_mode="rgb_array" if args.mode == "video" else None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent = make_agent_cfg(args.iterations, policy_version=policy_version)
    agent.seed = args.seed
    log_dir = None
    if args.mode == "train":
        log_dir = args.log_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.stage}")
        log_dir.mkdir(parents=True, exist_ok=False)
        geometry = model_path.read_bytes()
        (log_dir / "geometry.xml").write_bytes(geometry)
        (log_dir / "task.json").write_text(json.dumps({
            "mjlab": "1.3.0", "settings": asdict(settings), "stage": args.stage,
            "randomize": args.randomize, "seed": args.seed,
            "action_order": ["reaction", "drive"], "action_units": "rad/s",
            "command_order": ["forward_m_s", "heading_rad_s"],
            "policy_version": policy_version,
            "algorithm": "PPO from scratch, no expert controller",
            "geometry": "wheelbot/nested_model.xml",
            "geometry_sha256": hashlib.sha256(geometry).hexdigest(),
        }, indent=2), encoding="utf-8")
        (log_dir / "agent.json").write_text(json.dumps(asdict(agent), indent=2), encoding="utf-8")
    runner = None
    if args.mode != "smoke":
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent),
                                    str(log_dir) if log_dir else None, args.device)
        if args.checkpoint:
            runner.load(str(args.checkpoint))
    return env, wrapped, runner, log_dir


def smoke(env, wrapped):
    import torch
    obs = wrapped.get_observations()
    assert obs["actor"].shape == (env.num_envs, 15), obs["actor"].shape
    assert wrapped.num_actions == 2
    assert env.action_manager.get_term("flywheel").target_names == ["reaction"]
    assert env.action_manager.get_term("rolling").target_names == ["drive"]
    for i in range(20):
        actions = torch.zeros((env.num_envs, 2), device=env.device)
        if i >= 5:
            actions[:, 0] = 0.05 if i % 2 else -0.05
            actions[:, 1] = -0.05 if i % 2 else 0.05
        obs, reward, done, _ = wrapped.step(actions)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(reward).all()
        assert env.scene["robot"].data.actuator_force.abs().max() <= 2.40001
    print(json.dumps({"mjlab": version("mjlab"), "device": str(env.device),
        "environments": env.num_envs, "steps": 20, "observation_dimension": 15,
        "action_dimension": 2, "action_order": ["reaction", "drive"],
        "finite": True, "torque_limit_checked": True}, indent=2))


def evaluate(args, env, wrapped, runner):
    import torch
    from .task import command, heading_rate
    policy = runner.get_inference_policy(device=env.device)
    # Nominal checks, not the legacy high-speed DR/impulse acceptance sequence.
    cases = (("stop", 0.0, 0.0), ("forward", 1.0, 0.0), ("reverse", -1.0, 0.0),
             ("left", 0.8, 0.35), ("right", 0.8, -0.35),
             ("reverse_left", -0.8, 0.35), ("reverse_right", -0.8, -0.35))
    report = {"mjlab": "1.3.0", "checkpoint": str(args.checkpoint),
              "duration_s": args.duration, "seed": args.seed, "cases": []}
    for name, speed, yaw in cases:
        env.reset()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        error_sums = torch.zeros((env.num_envs, 3), device=env.device)
        counts = torch.zeros(env.num_envs, device=env.device)
        for step in range(round(args.duration / env.step_dt)):
            # Set command before recomputing observations, so policy sees it now.
            env.command_manager.get_command("twist")[:] = torch.tensor(
                [speed, 0.0, yaw], device=env.device)
            obs = wrapped.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                robot = env.scene["robot"]
                errors = torch.stack((robot.data.root_link_lin_vel_b[:, 0] - speed,
                                      heading_rate(env) - yaw,
                                      robot.data.root_link_lin_vel_b[:, 1]), dim=1)
                if step * env.step_dt >= 3.0:
                    error_sums += errors.square() * alive[:, None]
                    counts += alive
                wrapped.step(actions)
                alive &= ~env.termination_manager.terminated
        rmse = (error_sums / counts.clamp_min(1)[:, None]).sqrt()
        passed = alive & (counts > 0) & (rmse[:, 0] < 0.35) & (rmse[:, 1] < 0.15) & (rmse[:, 2] < 0.35)
        report["cases"].append({"name": name, "command": [speed, yaw],
            "survival_rate": alive.float().mean().item(), "pass_rate": passed.float().mean().item(),
            "individual": [{"survived": bool(alive[i]),
                "rmse_forward": float(rmse[i, 0]) if counts[i] else None,
                "rmse_heading_rate": float(rmse[i, 1]) if counts[i] else None,
                "rmse_lateral": float(rmse[i, 2]) if counts[i] else None,
                "valid_samples": int(counts[i]), "passed": bool(passed[i])}
                for i in range(env.num_envs)]})
        print(name, report["cases"][-1]["survival_rate"], report["cases"][-1]["pass_rate"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation saved: {args.output.resolve()}")


def main():
    args = parser().parse_args()
    if args.num_envs is None:
        args.num_envs = {"train": 256, "smoke": 4, "play": 1, "evaluate": 8,
                         "video": 1, "diagnose": 1}[args.mode]
    if args.mode in ("play", "evaluate", "video", "diagnose") and args.checkpoint is None:
        raise ValueError("play/evaluate/video/diagnose requires --checkpoint")
    if args.mode == "diagnose" and args.num_envs != 1:
        raise ValueError("diagnose requires --num-envs 1")
    if args.mode == "video":
        if args.num_envs != 1:
            raise ValueError("video requires --num-envs 1")
        if args.video.suffix.lower() != ".mp4":
            raise ValueError("--video must end in .mp4")
        if os.name != "nt":
            os.environ.setdefault("MUJOCO_GL", "egl")
    # EGL is a Linux renderer. MJLab's launch_training sets it unconditionally;
    # construct its runner directly to support native Windows.
    if os.name == "nt":
        os.environ["MUJOCO_GL"] = "glfw"
    env, wrapped, runner, log_dir = build(args)
    try:
        if args.mode == "smoke":
            smoke(env, wrapped)
        elif args.mode == "train":
            print(f"Run directory: {log_dir.resolve()}", flush=True)
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        elif args.mode == "evaluate":
            evaluate(args, env, wrapped, runner)
        elif args.mode == "diagnose":
            from .diagnostics import diagnose
            diagnose(args, env, wrapped, runner)
        elif args.mode == "video":
            from .video import record_video
            record_video(args, env, wrapped, runner)
        else:
            policy = runner.get_inference_policy(device=env.device)
            if args.viewer == "native":
                from mjlab.viewer import NativeMujocoViewer
                NativeMujocoViewer(wrapped, policy).run()
            else:
                import viser
                from mjlab.viewer import ViserPlayViewer
                server = viser.ViserServer(host=args.viewer_host, port=args.viewer_port,
                                           label="Miniwheelbot — MJLab 1.3.0")
                try:
                    ViserPlayViewer(wrapped, policy, viser_server=server).run()
                finally:
                    server.stop()
    finally:
        env.close()


if __name__ == "__main__":
    main()
