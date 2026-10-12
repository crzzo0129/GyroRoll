"""Optimize steering morphology across nominal and randomized pushed rollouts."""

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


SOURCE = ROOT / "steerbot" / "model.xml"
START = json.loads((ROOT / "steerbot" / "morph_candidate.json").read_text())
PARAMS = np.array(START["params"])
BOUNDS = [(.015, .065), (-.45, -.10), (-1., 1.5), (-1.5, 1.5),
          (-2., 1.), (-5., 1.), (.01, .9)]


def evaluate(x, return_metrics=False):
    offset, pitch_trim, ff, kp, ki, roll, tau = x
    spec = mujoco.MjSpec.from_file(str(SOURCE))
    spec.geom("body").pos[0] = offset
    optimize_steer.MODEL = spec.compile()
    params = PARAMS.copy()
    params[[3, 4, 5, 6, 10, 13]] = ff, kp, ki, roll, tau, pitch_trim
    nominal = optimize_steer.evaluate(params, True)
    # Use two training seeds; hold out seeds 1, 2, 4 for genuine validation.
    perturbed = [optimize_steer.evaluate(params, True, dr_seed=seed, disturbance=True)
                 for seed in (0, 3)]
    score = .2 * nominal["score"] + .4 * perturbed[0]["score"] + .4 * perturbed[1]["score"]
    if return_metrics:
        return dict(score=score, nominal=nominal, train_seeds=perturbed,
                    params=params.tolist(), design=list(x))
    return score


if __name__ == "__main__":
    anchor = np.array(START["design"])
    rng = np.random.default_rng(47)
    lower, upper = np.array(BOUNDS).T
    population = rng.uniform(lower, upper, size=(56, 7))
    population[0] = np.clip(anchor, lower, upper)
    for i in range(1, 25):
        population[i] = np.clip(population[0] + rng.normal(0, .08, 7) * (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = evaluate(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            (ROOT / "steerbot" / "morph_robust.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} nominal_yaw={best['nominal']['yaw_rmse']:.3f} "
              f"pushed_yaw={best['train_seeds'][0]['yaw_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, BOUNDS, maxiter=40, init=population,
                                    seed=47, polish=False, callback=progress)
    progress(result.x, 0.)
