"""Explore a 5 g long-trail steering head under the full bidirectional sequence."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import optimize_steer


ROOT = Path(__file__).resolve().parent
BASE = np.array(json.loads((ROOT / "steerbot/lightfork.json").read_text())["params"])
INDICES = [3, 4, 5, 6, 7, 8, 9, 10, 11]
BOUNDS = [(-3., 3.), (-5., 5.), (-3., 3.), (-25., 15.), (-6., 6.),
          (.1, 12.), (.03, 3.), (.01, 1.), (-5., 5.)]


def expand(x):
    gains = BASE.copy()
    gains[INDICES] = x
    return gains


def score(x, metrics=False):
    return optimize_steer.evaluate(expand(x), metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-z", type=float, default=.018)
    parser.add_argument("--iterations", type=int, default=70)
    args = parser.parse_args()
    spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
    spec.body("fork").pos[0] = .5
    spec.body("drive_wheel").pos[0] = -.5
    spec.geom("steering_head").mass = .005
    spec.geom("body").pos[2] = args.body_z
    optimize_steer.MODEL = spec.compile()
    lower, upper = np.array(BOUNDS).T
    rng = np.random.default_rng(107)
    population = rng.uniform(lower, upper, size=(72, len(BOUNDS)))
    population[0] = np.clip(BASE[INDICES], lower, upper)
    prior = (ROOT / "steerbot/lightfork.json" if np.isclose(args.body_z, .018)
             else ROOT / "steerbot" / f"lightfork_lowcom_{round(-args.body_z * 1000):03d}.json")
    if prior.exists():
        previous = json.loads(prior.read_text())
        population[0] = np.clip(np.array(previous["params"])[INDICES], lower, upper)
    for i in range(1, 30):
        population[i] = np.clip(population[0] + rng.normal(0, .07, len(BOUNDS))
                                * (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = score(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = expand(x).tolist()
            best["trail"] = .5
            best["head_mass"] = .005
            best["body_z"] = args.body_z
            prior.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    result = differential_evolution(score, BOUNDS, seed=107, init=population,
                                    maxiter=args.iterations, polish=False, callback=progress)
    progress(result.x, 0)
