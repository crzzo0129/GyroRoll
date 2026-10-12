"""Optimize a genuinely sustained, low-slip 2 m/s steering turn."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from probe_longtrail_turn import evaluate


ROOT = Path(__file__).resolve().parent
SOURCE = json.loads((ROOT / "steerbot/longtrail_turn.json").read_text())
BOUNDS = [(-3., 3.), (-4., 4.), (-24., 12.), (-5., 5.),
          (.2, 10.), (.05, 3.), (.01, 1.)]


def score(x, metrics=False):
    return evaluate(x, metrics, duration=15.)


if __name__ == "__main__":
    low, high = np.array(BOUNDS).T
    rng = np.random.default_rng(103)
    population = rng.uniform(low, high, size=(70, len(BOUNDS)))
    population[0] = SOURCE["params"]
    for i in range(1, 25):
        population[i] = np.clip(population[0] + rng.normal(0, .08, len(BOUNDS)) * (high - low), low, high)
    path = ROOT / "steerbot/longtrail_sustained.json"
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = score(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = x.tolist()
            best["trail"] = .5
            path.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    result = differential_evolution(score, BOUNDS, init=population, maxiter=70,
                                    polish=False, seed=103, callback=progress)
    progress(result.x, 0)
