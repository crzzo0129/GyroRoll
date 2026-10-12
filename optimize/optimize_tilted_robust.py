"""Require a tilted flywheel controller to turn both ways at multiple commands."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import differential_evolution

from probe_tilted_flywheel import evaluate, model_at


SOURCE = json.loads((ROOT / "wheelbot/tilted_axis_probe.json").read_text())
MODEL = model_at(SOURCE["tilt"])
BOUNDS = [(1., 14.), (.15, 2.5), (-5., 5.), (-7., 7.),
          (-4., 4.), (-1., 1.), (.05, .6)]


def evaluate_all(x, metrics=False):
    cases = [(.35, False), (.35, True), (.2, False), (.2, True)]
    results = [evaluate(MODEL, x, True, amplitude, mirror)
               for amplitude, mirror in cases]
    scores = [r["speed_rmse"] + 8 * r["yaw_rmse"] + 2 * r["lateral_rmse"]
              + 8 * (15 - r["duration"]) + (8 if r["fallen"] else 0)
              for r in results]
    score = float(np.mean(scores) + .5 * max(scores))
    return dict(score=score, cases=results) if metrics else score


if __name__ == "__main__":
    anchor = np.array(SOURCE["params"])
    low, high = np.array(BOUNDS).T
    rng = np.random.default_rng(97)
    population = rng.uniform(low, high, size=(49, len(BOUNDS)))
    population[0] = anchor
    for i in range(1, 20):
        population[i] = np.clip(anchor + rng.normal(0, .06, len(BOUNDS)) * (high - low), low, high)
    path = ROOT / "wheelbot/tilted_axis_robust.json"
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = evaluate_all(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = x.tolist()
            best["tilt"] = SOURCE["tilt"]
            path.write_text(json.dumps(best, indent=2))
        print(f"score={best['score']:.3f} "
              f"yaw={[round(r['yaw_rmse'], 3) for r in best['cases']]} "
              f"fallen={[r['fallen'] for r in best['cases']]}", flush=True)

    result = differential_evolution(evaluate_all, BOUNDS, init=population,
                                    maxiter=40, seed=97, polish=False, callback=progress)
    progress(result.x, 0)
