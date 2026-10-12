"""Tune nested-wheel feedback using the same replay and metrics as sim.py."""

import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from sim import make_parser, run


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--drive-only", action="store_true")
    args = parser.parse_args()
    output = ROOT / "results/nested_tuning"
    output.mkdir(exist_ok=True, parents=True)
    config = json.loads((ROOT / "wheelbot/nested_controller.json").read_text())
    trial_path = output / "trial_controller.json"
    replay = make_parser().parse_args([
        "wheelbot", "--controller", "nested", "--controller-params", str(trial_path),
        "--headless", "--test", "--duration", "15",
    ])
    if args.drive_only:
        replay.test_turn_amplitude = 0.
    best = {"score": float("inf")}

    def evaluate(params):
        trial = dict(config, params=list(params))
        trial_path.write_text(json.dumps(trial))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run(replay)
        summary = json.loads(stream.getvalue())
        speed = summary["steady_speed_rmse"] or 0.
        yaw = summary["turn_yaw_rmse"] or 0.
        straight = summary["straight_yaw_rmse"] or 0.
        recovery = summary["recovery_yaw_rmse"] or 0.
        score = (speed + 2 * yaw + straight + recovery
                 + 3 * (15 - summary["duration"]) + 5 * summary["fallen"])
        if score < best["score"]:
            best.update(score=score, summary=summary, configuration=trial)
            (output / "best.json").write_text(json.dumps(best, indent=2))
            print(f"score={score:.4f} speed={speed:.4f} yaw={yaw:.4f} "
                  f"fallen={summary['fallen']} params={list(params)}", flush=True)
        return score

    if args.drive_only:
        for gain, kp, kd in itertools.product((.1, .25, .5), (1., 3., 6.), (.08, .2, .5)):
            params = config["params"].copy()
            params[:3] = (gain, kp, kd)
            evaluate(params)
    else:
        bounds = [(.05, .7), (.5, 9.), (.04, .7), (1., 12.), (.1, 2.), (-1., .5), (-.7, .7)]
        lower, upper = np.array(bounds).T
        rng = np.random.default_rng(91)
        initial = rng.uniform(lower, upper, (28, 7))
        initial[0] = np.clip(config["params"], lower, upper)
        for index in range(1, 15):
            initial[index] = np.clip(initial[0] + rng.normal(0., .08, 7) * (upper - lower), lower, upper)
        differential_evolution(evaluate, bounds, init=initial, seed=91,
                               maxiter=args.iterations, polish=False)
    (output / "candidate_controller.json").write_text(json.dumps(
        dict(best["configuration"], status="tuned on nominal replay; independent validation required"),
        indent=2))


if __name__ == "__main__":
    main()
