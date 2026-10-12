"""Tune narrow-tire steering feedback on nominal and randomized pushed rollouts."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from optimize_steer_thin import BOUNDS, INDICES, optimize_steer


ROOT = Path(__file__).resolve().parent
ANCHOR = np.array(json.loads((ROOT / "steerbot/thin_tire_004_g24.json").read_text())["params"])
PATH = ROOT / "steerbot/thin_tire_004_g24_robust.json"


def expand(x):
    params = ANCHOR.copy()
    params[INDICES] = x
    return params


def evaluate(x, metrics=False):
    params = expand(x)
    results = [optimize_steer.evaluate(params, True),
               optimize_steer.evaluate(params, True, dr_seed=0, disturbance=True),
               optimize_steer.evaluate(params, True, dr_seed=1, disturbance=True)]
    score = float(np.mean([r["score"] for r in results])
                  + .5 * max(r["score"] for r in results))
    return dict(score=score, cases=results) if metrics else score


if __name__ == "__main__":
    lower, upper = np.array(BOUNDS).T
    rng = np.random.default_rng(131)
    population = rng.uniform(lower, upper, size=(70, len(BOUNDS)))
    population[0] = np.clip(ANCHOR[INDICES], lower, upper)
    if PATH.exists():
        population[0] = np.clip(np.array(json.loads(PATH.read_text())["params"])[INDICES],
                                lower, upper)
    for i in range(1, 30):
        population[i] = np.clip(population[0] + rng.normal(0, .06, len(BOUNDS))
                                * (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = evaluate(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = expand(x).tolist()
            best["tire_halfwidth"] = .004
            best["steer_gear"] = 2.4
            PATH.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} "
              f"yaw={[round(r['yaw_rmse'], 3) for r in best['cases']]} "
              f"lat={[round(r['lateral_rmse'], 3) for r in best['cases']]}", flush=True)

    result = differential_evolution(evaluate, BOUNDS, init=population,
                                    maxiter=65, polish=False, seed=131,
                                    callback=progress)
    progress(result.x, 0)
