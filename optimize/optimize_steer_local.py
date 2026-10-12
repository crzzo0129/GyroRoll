"""Concentrated yaw-loop search around the stable 2 m/s steering controller."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import differential_evolution

from optimize.optimize_steer import evaluate


PATH = ROOT / "steerbot" / "optimized_controller.json"
original = np.array(json.loads(PATH.read_text())["params"])
if len(original) == 11:
    original = np.r_[original, 0., 0.]
indices = [3, 4, 5, 6, 10]
bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-5., 5.), (.02, 1.)]


def expand(values):
    params = original.copy()
    params[indices] = values
    return params


def score(values):
    return evaluate(expand(values))


if __name__ == "__main__":
    rng = np.random.default_rng(37)
    lower, upper = np.array(bounds).T
    init = rng.uniform(lower, upper, size=(60, len(indices)))
    init[0] = original[indices]
    for i in range(1, 25):
        init[i] = np.clip(init[0] + rng.normal(0, .12, len(indices)) * (upper - lower), lower, upper)
    best = evaluate(original, True)
    best["params"] = original.tolist()

    def progress(x, convergence):
        measured = evaluate(expand(x), True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = expand(x).tolist()
            PATH.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} speed={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f}", flush=True)

    result = differential_evolution(score, bounds, maxiter=75, seed=37, init=init,
                                    polish=False, callback=progress)
    progress(result.x, 0.)
