"""Rank sampled leading-wheel morphologies on both turn orders."""

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


GEOMETRIES = json.loads((ROOT / "steerbot/negative_morph_robust_screen.json").read_text())
TUNING = json.loads((ROOT / "steerbot/negative_candidate.json").read_text())["params"]
OUTPUT = ROOT / "steerbot/negative_morph_bidir_screen.json"


if __name__ == "__main__":
    results = []
    for i, candidate in enumerate(GEOMETRIES):
        trail = candidate["trail"]
        mass = candidate["body_mass"]
        z = candidate["body_offset_z"]
        head = candidate["head_mass"]
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
            cases = [optimize_lqr.evaluate(TUNING, True, dr_seed=0,
                                            disturbance=True, full_yaw=True, mirror=m)
                     for m in (False, True)]
            score = (8 * max(r["yaw_rmse"] for r in cases)
                     + .5 * sum(r["score"] for r in cases))
            metrics = dict(score=score, cases=cases)
        except (ValueError, np.linalg.LinAlgError, mujoco.FatalError) as exc:
            metrics = dict(score=100., error=str(exc))
        results.append(dict(trail=trail, body_mass=mass, body_offset_z=z,
                            head_mass=head, **metrics))
        results.sort(key=lambda r: r["score"])
        OUTPUT.write_text(json.dumps(results, indent=2))
        best = results[0]
        print(i, round(best["score"], 3),
              [round(r["yaw_rmse"], 3) for r in best.get("cases", [])],
              [round(best[key], 3) for key in
               ("trail", "body_mass", "body_offset_z", "head_mass")], flush=True)
