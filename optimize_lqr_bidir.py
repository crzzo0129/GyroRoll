"""Tune one steering LQR for both turn orders under lateral kicks."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

import optimize_lqr
from linearize_steer import identify


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "steerbot/negative_bidir.json"
GEOMETRY = json.loads((ROOT / "steerbot/negative_candidate.json").read_text())
ANCHOR = np.asarray(GEOMETRY["params"])

spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
trail = GEOMETRY["trail"]
spec.body("fork").pos[0] = trail
spec.body("drive_wheel").pos[0] = -trail
spec.geom("body").mass = GEOMETRY["body_mass"]
spec.geom("body").pos[2] += GEOMETRY["body_offset_z"]
spec.geom("steering_head").mass = GEOMETRY["head_mass"]
spec.geom("tire").size[1] = GEOMETRY["tire_halfwidth"]
spec.actuator("steer_motor").gear[0] = GEOMETRY["steer_gear"]
optimize_lqr.MODEL = spec.compile()
optimize_lqr.LOCAL = identify(trail=trail, tire_halfwidth=GEOMETRY["tire_halfwidth"],
                               steer_gear=GEOMETRY["steer_gear"],
                               body_mass=GEOMETRY["body_mass"],
                               body_offset_z=GEOMETRY["body_offset_z"],
                               head_mass=GEOMETRY["head_mass"])


def evaluate(params, metrics=False):
    cases = [optimize_lqr.evaluate(params, True, dr_seed=0, disturbance=True,
                                   full_yaw=True, mirror=mirror) for mirror in (False, True)]
    score = float(20. * max(r["yaw_rmse"] for r in cases)
                  + .2 * sum(r["score"] for r in cases)
                  + max(0., max(r["recovery_yaw_rmse"] for r in cases) - .12) * 5.)
    return dict(score=score, cases=cases) if metrics else score


if __name__ == "__main__":
    bounds = [(30., 500.), (.1, 300.), (.1, 150.), (.1, 20.), (.005, .25),
              (.05, .8), (.1, 80.), (.6, 2.)]
    lower, upper = np.array(bounds).T
    rng = np.random.default_rng(181)
    population = rng.uniform(lower, upper, size=(60, len(bounds)))
    population[0] = np.clip(ANCHOR, lower, upper)
    if PATH.exists():
        population[0] = np.clip(json.loads(PATH.read_text())["params"], lower, upper)
    for i in range(1, 30):
        population[i] = np.clip(population[0] + rng.normal(0, .075, len(bounds)) *
                                (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(params, convergence):
        measured = evaluate(params, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best.update({key: GEOMETRY[key] for key in
                         ("trail", "body_mass", "body_offset_z", "head_mass",
                          "tire_halfwidth", "steer_gear")})
            best["params"] = params.tolist()
            PATH.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} "
              f"yaw={[round(r['yaw_rmse'], 3) for r in best['cases']]} "
              f"recovery={[round(r['recovery_yaw_rmse'], 3) for r in best['cases']]}",
              flush=True)

    result = differential_evolution(evaluate, bounds, init=population,
                                    maxiter=80, polish=False, seed=181,
                                    callback=progress)
    progress(result.x, 0.)
