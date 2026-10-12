"""Screen two-actuator steering morphologies using local LQR and full physics."""

import json
from pathlib import Path

import mujoco
import numpy as np

import optimize_lqr
from linearize_steer import identify


ROOT = Path(__file__).resolve().parent
TUNING = np.asarray(json.loads((ROOT / "steerbot/lqr_thin_004_g24.json").read_text())["params"])
OUTPUT = ROOT / "steerbot/morph_lqr_screen.json"


def evaluate(trail, body_mass, body_offset_z, head_mass):
    spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
    spec.body("fork").pos[0] = trail
    spec.body("drive_wheel").pos[0] = -trail
    spec.geom("body").mass = body_mass
    spec.geom("body").pos[2] += body_offset_z
    spec.geom("steering_head").mass = head_mass
    spec.geom("tire").size[1] = .004
    spec.actuator("steer_motor").gear[0] = 2.4
    optimize_lqr.MODEL = spec.compile()
    optimize_lqr.LOCAL = identify(trail=trail, tire_halfwidth=.004, steer_gear=2.4,
                                   body_mass=body_mass, body_offset_z=body_offset_z,
                                   head_mass=head_mass)
    return optimize_lqr.evaluate(TUNING, True)


if __name__ == "__main__":
    rng = np.random.default_rng(143)
    candidates = [(float(rng.uniform(.08, .50)), float(rng.uniform(.25, .87)),
                   float(rng.uniform(-.10, .02)), float(rng.uniform(.005, .045)))
                  for _ in range(100)]
    candidates.insert(0, (.12, .87, 0., .045))
    results = []
    for i, params in enumerate(candidates):
        try:
            metrics = evaluate(*params)
        except (ValueError, np.linalg.LinAlgError, mujoco.FatalError) as exc:
            metrics = {"score": 100., "error": str(exc)}
        results.append(dict(trail=params[0], body_mass=params[1],
                            body_offset_z=params[2], head_mass=params[3], **metrics))
        results.sort(key=lambda item: item["score"])
        OUTPUT.write_text(json.dumps(results, indent=2))
        print(i, "best", results[0]["score"], "yaw", results[0].get("yaw_rmse"),
              "geometry", [round(results[0][key], 3) for key in
                           ("trail", "body_mass", "body_offset_z", "head_mass")], flush=True)
