"""Select a checkpoint by independent MuJoCo rollouts, not training reward."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("wheelbot", "steerbot"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rotor-speed", type=float, default=0.)
    parser.add_argument("--trail", type=float, default=None)
    parser.add_argument("--tilt", type=float, default=None)
    parser.add_argument("--head-mass", type=float, default=None)
    parser.add_argument("--tire-halfwidth", type=float, default=None)
    parser.add_argument("--steer-gear", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=1, help="Only evaluate every Nth checkpoint")
    args = parser.parse_args()
    checkpoints = sorted(args.run_dir.glob("model_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))[::args.stride]
    if not checkpoints:
        raise SystemExit("No checkpoints found")
    ranked = []
    for checkpoint in checkpoints:
        command = [sys.executable, str(ROOT / "sim.py"), args.kind,
                   "--headless", "--test", "--checkpoint", str(checkpoint.resolve())]
        if args.rotor_speed:
            command += ["--rotor-speed", str(args.rotor_speed)]
        if args.trail is not None:
            command += ["--trail", str(args.trail)]
        if args.tilt is not None:
            command += ["--tilt", str(args.tilt)]
        if args.head_mass is not None:
            command += ["--head-mass", str(args.head_mass)]
        if args.tire_halfwidth is not None:
            command += ["--tire-halfwidth", str(args.tire_halfwidth)]
        if args.steer_gear is not None:
            command += ["--steer-gear", str(args.steer_gear)]
        result = json.loads(subprocess.check_output(command, text=True))
        speed = result["steady_speed_rmse"] or 3.
        yaw = result["turn_yaw_rmse"] or 3.
        recovery = result["recovery_yaw_rmse"] or 3.
        lateral = result["turn_lateral_rmse"] or 3.
        score = (speed + 2. * yaw + recovery + lateral + (15. - result["duration"])
                 + (2. if result["fallen"] else 0.))
        ranked.append(dict(checkpoint=str(checkpoint), score=score, **result))
        print(f"{checkpoint.name}: {score:.3f} (speed {speed:.3f}, yaw {yaw:.3f}, "
              f"lateral {lateral:.3f}, {result['duration']:.1f}s)", flush=True)
    ranked.sort(key=lambda r: r["score"])
    from result_paths import legacy_result_dir
    destination = args.output or legacy_result_dir(args.kind) / "checkpoint_ranking.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(ranked, indent=2))
    print(f"Best: {ranked[0]['checkpoint']} score={ranked[0]['score']:.3f}")


if __name__ == "__main__":
    main()
