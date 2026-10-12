"""Extend the feasible low-crab turn to bidirectional 15-second tracking."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from optimize import optimize_steer


spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
spec.body("fork").pos[0] = .5
spec.body("drive_wheel").pos[0] = -.5
optimize_steer.MODEL = spec.compile()
base = np.array(json.loads((ROOT / "steerbot" / "longtrail_controller.json").read_text())["params"])
indices = [3, 4, 5, 6, 7, 8, 9, 10, 11]
bounds = [(-2., 3.), (-3., 3.), (-2., 2.), (-24., 12.), (-5., 5.),
          (.2, 10.), (.05, 3.), (.01, 1.), (-4., 4.)]


def expand(x):
    params = base.copy()
    params[indices] = x
    return params


def evaluate(x):
    return optimize_steer.evaluate(expand(x))


if __name__ == "__main__":
    rng = np.random.default_rng(61)
    lower, upper = np.array(bounds).T
    population = rng.uniform(lower, upper, size=(81, len(bounds)))
    population[0] = np.clip(base[indices], lower, upper)
    population[1] = [.6666488242555164, -.008804424021216395, 0.,
                     -10.661306794346984, .4837090013809431,
                     .6636773964367926, .43996916534486896, .2799758733805381, 0.]
    for i in range(2, 35):
        population[i] = np.clip(population[0] + rng.normal(0, .07, len(bounds)) * (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = optimize_steer.evaluate(expand(x), True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = expand(x).tolist()
            best["trail"] = .5
            (ROOT / "steerbot" / "longtrail_full.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    result = differential_evolution(evaluate, bounds, seed=61, init=population,
                                    maxiter=70, polish=False, callback=progress)
    progress(result.x, 0.)
