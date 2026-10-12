"""Optimize the two-motor steerbot with a narrow tire and higher steering torque."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import optimize_steer


ROOT = Path(__file__).resolve().parent
SPEC = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
SPEC.geom("tire").size[1] = .004
SPEC.actuator("steer_motor").gear[0] = 2.4
optimize_steer.MODEL = SPEC.compile()
BASE = np.array(json.loads((ROOT / "steerbot/optimized_controller.json").read_text())["params"])
BASE[8:10] *= .25
INDICES = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BOUNDS = [(-4., 4.), (-5., 5.), (-4., 4.), (-20., 20.), (-6., 6.),
          (.05, 12.), (.02, 4.), (.01, 1.2), (-5., 5.), (-5., 5.)]


def expand(x):
    params = BASE.copy()
    params[INDICES] = x
    return params


def score(x, metrics=False):
    return optimize_steer.evaluate(expand(x), metrics)


if __name__ == "__main__":
    lower, upper = np.array(BOUNDS).T
    rng = np.random.default_rng(127)
    population = rng.uniform(lower, upper, size=(80, len(BOUNDS)))
    population[0] = np.clip(BASE[INDICES], lower, upper)
    path = ROOT / "steerbot/thin_tire_004_g24.json"
    if path.exists():
        prior = np.array(json.loads(path.read_text())["params"])
        population[0] = np.clip(prior[INDICES], lower, upper)
    for i in range(1, 35):
        population[i] = np.clip(population[0] + rng.normal(0, .08, len(BOUNDS))
                                * (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = score(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = expand(x).tolist()
            best["tire_halfwidth"] = .004
            best["steer_gear"] = 2.4
            path.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    result = differential_evolution(score, BOUNDS, init=population, seed=127,
                                    maxiter=90, polish=False, callback=progress)
    progress(result.x, 0)
