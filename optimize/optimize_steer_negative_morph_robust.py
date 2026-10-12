"""Screen leading-wheel geometry under lateral velocity kicks and DR."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from optimize import optimize_lqr
from linearize_steer import identify


TUNING = np.asarray(json.loads((ROOT / "steerbot/lqr_negative_015_robust.json").read_text())["params"])
OUTPUT = ROOT / "steerbot/negative_morph_robust_screen.json"


if __name__ == "__main__":
    rng = np.random.default_rng(173)
    candidates = [(-.15, .645, -.062, .019)]
    candidates += [(float(rng.uniform(-.27, -.05)), float(rng.uniform(.3, .85)),
                    float(rng.uniform(-.11, .02)), float(rng.uniform(.005, .045)))
                   for _ in range(160)]
    results = []
    for i, (trail, mass, z, head) in enumerate(candidates):
        spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
        spec.body("fork").pos[0] = trail
        spec.body("drive_wheel").pos[0] = -trail
        spec.geom("body").mass = mass
        spec.geom("body").pos[2] += z
        spec.geom("steering_head").mass = head
        spec.geom("tire").size[1] = .004
        spec.actuator("steer_motor").gear[0] = 2.4
        try:
            optimize_lqr.MODEL = spec.compile()
            optimize_lqr.LOCAL = identify(trail=trail, tire_halfwidth=.004,
                                           steer_gear=2.4, body_mass=mass,
                                           body_offset_z=z, head_mass=head)
            nominal = optimize_lqr.evaluate(TUNING, True, full_yaw=True)
            pushed = optimize_lqr.evaluate(TUNING, True, dr_seed=0,
                                             disturbance=True, full_yaw=True)
            score = (nominal["score"] + 2 * pushed["score"]) / 3
            metrics = dict(score=score, nominal=nominal, pushed=pushed)
        except (ValueError, np.linalg.LinAlgError, mujoco.FatalError) as exc:
            metrics = {"score": 100., "error": str(exc)}
        results.append(dict(trail=trail, body_mass=mass, body_offset_z=z,
                            head_mass=head, **metrics))
        results.sort(key=lambda r: r["score"])
        OUTPUT.write_text(json.dumps(results, indent=2))
        best = results[0]
        print(i, round(best["score"], 3),
              round(best.get("pushed", {}).get("yaw_rmse", 3.), 3),
              [round(best[key], 3) for key in
               ("trail", "body_mass", "body_offset_z", "head_mass")], flush=True)
