"""Tune leading-wheel LQR against nominal and randomized lateral kicks."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import optimize_lqr
from linearize_steer import identify


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "steerbot/lqr_negative_015_robust.json"
ANCHOR = np.asarray(json.loads((ROOT / "steerbot/lqr_negative_015.json").read_text())["params"])

spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
spec.body("fork").pos[0] = -.15
spec.body("drive_wheel").pos[0] = .15
spec.geom("body").mass = .645
spec.geom("body").pos[2] -= .062
spec.geom("steering_head").mass = .019
spec.geom("tire").size[1] = .004
spec.actuator("steer_motor").gear[0] = 2.4
optimize_lqr.MODEL = spec.compile()
optimize_lqr.LOCAL = identify(trail=-.15, tire_halfwidth=.004, steer_gear=2.4,
                               body_mass=.645, body_offset_z=-.062, head_mass=.019)


def evaluate(params, metrics=False):
    cases = [optimize_lqr.evaluate(params, True, full_yaw=True),
             optimize_lqr.evaluate(params, True, dr_seed=0, disturbance=True, full_yaw=True),
             optimize_lqr.evaluate(params, True, dr_seed=1, disturbance=True, full_yaw=True)]
    score = float(np.mean([r["score"] for r in cases]) + .5 * max(r["score"] for r in cases))
    return dict(score=score, cases=cases) if metrics else score


if __name__ == "__main__":
    bounds = [(5., 400.), (1., 400.), (1., 300.), (.1, 25.), (.005, .3),
              (.05, 1.5), (.01, 40.), (.1, 3.)]
    lower, upper = np.array(bounds).T
    rng = np.random.default_rng(163)
    population = rng.uniform(lower, upper, size=(60, len(bounds)))
    population[0] = np.clip(ANCHOR, lower, upper)
    if PATH.exists():
        population[0] = np.clip(json.loads(PATH.read_text())["params"], lower, upper)
    for i in range(1, 35):
        population[i] = np.clip(population[0] + rng.normal(0, .1, len(bounds)) *
                                (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(params, convergence):
        result = evaluate(params, True)
        if result["score"] < best["score"]:
            best.update(result)
            best["params"] = params.tolist()
            best.update(trail=-.15, body_mass=.645, body_offset_z=-.062,
                        head_mass=.019, tire_halfwidth=.004, steer_gear=2.4)
            PATH.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} "
              f"yaw={[round(r['yaw_rmse'], 3) for r in best['cases']]} "
              f"recovery={[round(r['recovery_yaw_rmse'], 3) for r in best['cases']]}",
              flush=True)

    result = differential_evolution(evaluate, bounds, init=population,
                                    maxiter=80, polish=False, seed=163,
                                    callback=progress)
    progress(result.x, 0.)
