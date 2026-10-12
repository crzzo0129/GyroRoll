"""Search narrow steerbot feedback against straight, turning, and recovery yaw."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import differential_evolution

from optimize.optimize_steer_thin import BOUNDS, INDICES, optimize_steer


ANCHOR = np.array(json.loads((ROOT / "steerbot/thin_tire_004_g24_robust.json").read_text())["params"])
PATH = ROOT / "steerbot/thin_tire_004_g24_full.json"


def expand(x):
    params = ANCHOR.copy()
    params[INDICES] = x
    return params


def evaluate(x, metrics=False):
    params = expand(x)
    cases = [optimize_steer.evaluate(params, True, full_yaw=True),
             optimize_steer.evaluate(params, True, dr_seed=0, disturbance=True, full_yaw=True),
             optimize_steer.evaluate(params, True, dr_seed=1, disturbance=True, full_yaw=True)]
    score = float(np.mean([r["score"] for r in cases]) + .5 * max(r["score"] for r in cases))
    return dict(score=score, cases=cases) if metrics else score


if __name__ == "__main__":
    lower, upper = np.array(BOUNDS).T
    rng = np.random.default_rng(137)
    population = rng.uniform(lower, upper, size=(80, len(BOUNDS)))
    anchors = [ANCHOR]
    for name in ("thin_tire_004_g24.json", "optimized_controller.json"):
        anchors.append(np.array(json.loads((ROOT / "steerbot" / name).read_text())["params"]))
    if PATH.exists():
        anchors.insert(0, np.array(json.loads(PATH.read_text())["params"]))
    for i, params in enumerate(anchors):
        population[i] = np.clip(params[INDICES], lower, upper)
    for i in range(len(anchors), 40):
        parent = anchors[i % len(anchors)]
        population[i] = np.clip(parent[INDICES] + rng.normal(0, .09, len(BOUNDS)) *
                                (upper - lower), lower, upper)
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
              f"turn={[round(r['yaw_rmse'], 3) for r in best['cases']]} "
              f"straight={[round(r['straight_yaw_rmse'], 3) for r in best['cases']]} "
              f"recovery={[round(r['recovery_yaw_rmse'], 3) for r in best['cases']]}", flush=True)

    result = differential_evolution(evaluate, BOUNDS, init=population,
                                    maxiter=85, polish=False, seed=137,
                                    callback=progress)
    progress(result.x, 0)
