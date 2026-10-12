"""Optimize a bicycle-like raked steering axis without adding an actuator."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import optimize_steer


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "steerbot/model.xml"
BASE = np.array(json.loads((ROOT / "steerbot/optimized_controller.json").read_text())["params"])
INDICES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BOUNDS = [(-3., 3.), (-5., 5.), (-3., 3.), (-18., 18.), (-5., 5.),
          (.1, 12.), (.03, 3.), (.01, 1.), (-5., 5.), (-5., 5.)]


def model_at(rake, trail):
    spec = mujoco.MjSpec.from_file(str(SOURCE))
    spec.body("fork").quat[:] = (np.cos(rake / 2), 0, np.sin(rake / 2), 0)
    spec.body("fork").pos[:] = (trail * np.cos(rake), 0,
                                -.16 - trail * np.sin(rake))
    spec.body("drive_wheel").pos[0] = -trail
    return spec.compile()


def expand(x):
    params = BASE.copy()
    params[INDICES] = x
    return params


def score(x, metrics=False):
    return optimize_steer.evaluate(expand(x), metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rake", type=float, default=.4)
    parser.add_argument("--trail", type=float, default=.12)
    parser.add_argument("--iterations", type=int, default=70)
    args = parser.parse_args()
    optimize_steer.MODEL = model_at(args.rake, args.trail)
    lower, upper = np.array(BOUNDS).T
    rng = np.random.default_rng(113)
    population = rng.uniform(lower, upper, size=(80, len(BOUNDS)))
    population[0] = np.clip(BASE[INDICES], lower, upper)
    for i in range(1, 35):
        population[i] = np.clip(population[0] + rng.normal(0, .07, len(BOUNDS))
                                * (upper - lower), lower, upper)
    path = ROOT / "steerbot" / f"raked_{round(args.rake * 100):03d}_trail{round(args.trail * 100):03d}.json"
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = score(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = expand(x).tolist()
            best["rake"] = args.rake
            best["trail"] = args.trail
            path.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    result = differential_evolution(score, BOUNDS, init=population,
                                    maxiter=args.iterations, polish=False,
                                    seed=113, callback=progress)
    progress(result.x, 0)
