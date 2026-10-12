"""Screen wheel/chassis inertia ratio for steering authority at 2 m/s."""

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


TUNING = np.asarray(json.loads((ROOT / "steerbot/lqr_thin_004_g24.json").read_text())["params"])
OUTPUT = ROOT / "steerbot/morph_inertia_screen.json"


if __name__ == "__main__":
    rng = np.random.default_rng(149)
    results = []
    for i in range(200):
        trail = float(rng.uniform(.08, .50))
        body_mass = float(rng.uniform(.20, .90))
        wheel_mass = float(rng.uniform(.25, 1.0))
        body_offset_z = float(rng.uniform(-.11, .02))
        head_mass = float(rng.uniform(.005, .045))
        spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
        spec.body("fork").pos[0] = trail
        spec.body("drive_wheel").pos[0] = -trail
        spec.geom("body").mass = body_mass
        spec.geom("body").pos[2] += body_offset_z
        spec.geom("steering_head").mass = head_mass
        spec.geom("tire").size[1] = .004
        spec.geom("tire").mass = wheel_mass
        spec.actuator("steer_motor").gear[0] = 2.4
        try:
            optimize_lqr.MODEL = spec.compile()
            optimize_lqr.LOCAL = identify(trail=trail, tire_halfwidth=.004, steer_gear=2.4,
                                           body_mass=body_mass, body_offset_z=body_offset_z,
                                           head_mass=head_mass, wheel_mass=wheel_mass)
            metrics = optimize_lqr.evaluate(TUNING, True)
        except (ValueError, np.linalg.LinAlgError, mujoco.FatalError) as exc:
            metrics = {"score": 100., "error": str(exc)}
        results.append(dict(trail=trail, body_mass=body_mass, wheel_mass=wheel_mass,
                            body_offset_z=body_offset_z, head_mass=head_mass, **metrics))
        results.sort(key=lambda item: item["score"])
        OUTPUT.write_text(json.dumps(results, indent=2))
        print(i, "best", round(results[0]["score"], 3), "yaw",
              round(results[0].get("yaw_rmse", 3.), 3), "geometry",
              [round(results[0][key], 3) for key in ("trail", "body_mass",
               "wheel_mass", "body_offset_z", "head_mass")], flush=True)
