"""Co-optimize a small chassis COM offset and the yaw feedback loop."""

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


MODEL_PATH = ROOT / "steerbot" / "model.xml"
BASE = np.array(json.loads((ROOT / "steerbot" / "optimized_controller.json").read_text())["params"])
if len(BASE) == 11:
    BASE = np.r_[BASE, 0., 0.]
BOUNDS = [(.01, .055), (-.35, -.12), (-.8, .8), (-1.2, 1.2),
          (-1.2, 1.2), (-4., 1.), (.03, 1.)]


def evaluate(x, metrics=False):
    offset, pitch_trim, ff, kp, ki, roll, tau = x
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec.geom("body").pos[0] = offset
    optimize_steer.MODEL = spec.compile()
    params = BASE.copy()
    params[[3, 4, 5, 6, 10]] = ff, kp, ki, roll, tau
    return optimize_steer.evaluate(np.r_[params, pitch_trim], metrics)


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    lower, upper = np.array(BOUNDS).T
    init = rng.uniform(lower, upper, size=(70, len(BOUNDS)))
    anchor = np.array((.03, -.2, *BASE[[3, 4, 5, 6, 10]]))
    init[0] = np.clip(anchor, lower, upper)
    for i in range(1, 30):
        init[i] = np.clip(init[0] + rng.normal(0, .10, len(BOUNDS)) * (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = evaluate(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["design"] = x.tolist()
            best["params"] = np.r_[BASE[:3], x[2:5], x[5], BASE[7:10], x[6],
                                   BASE[11:13], x[1]].tolist()
            (ROOT / "steerbot" / "morph_candidate.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} speed={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, BOUNDS, maxiter=60, seed=41,
                                    init=init, polish=False, callback=progress)
    progress(result.x, 0.)
