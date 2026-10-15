"""Record a fixed-command policy rollout without a browser or display server."""

import json
import math


def record_video(args, env, wrapped, runner):
    import imageio.v2 as imageio
    import torch

    if not math.isfinite(args.speed) or not math.isfinite(args.turn):
        raise ValueError("Video commands must be finite")
    fps = 30
    policy = runner.get_inference_policy(device=env.device)
    env.reset()
    target = torch.tensor([args.speed, 0.0, args.turn], device=env.device)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    terminated = False
    elapsed = 0.0
    with imageio.get_writer(str(args.video), fps=fps, codec="libx264", quality=8) as writer:
        for step in range(math.ceil(args.duration / env.step_dt)):
            # Update before observation computation so the policy sees the command.
            env.command_manager.get_command("twist")[:] = target
            with torch.inference_mode():
                obs = wrapped.get_observations()
                if step * env.step_dt + 1e-9 >= frames / fps:
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("Video requires rgb_array rendering")
                    writer.append_data(frame)
                    frames += 1
                _, _, done, _ = wrapped.step(policy(obs))
            elapsed = (step + 1) * env.step_dt
            if bool(done.any()):
                # MJLab automatically resets on failure. Never show that reset as
                # uninterrupted successful motion; end this clip and report it.
                terminated = True
                break
    report = {"checkpoint": str(args.checkpoint), "command": [args.speed, args.turn],
              "seed": args.seed, "requested_duration_s": args.duration,
              "simulated_duration_s": elapsed, "terminated_early": terminated,
              "fps": fps, "frames": frames, "video": str(args.video)}
    if hasattr(env, "_recovery"):
        from .recovery import snapshot
        report["recovery"] = snapshot(env)
        report["stage"] = args.stage
        report["start_pose"] = args.start_pose or "upright"
    args.video.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Video saved: {args.video.resolve()}")
    if terminated:
        print(f"Episode terminated at {elapsed:.2f} s; recording stopped before automatic reset.")
    return report
