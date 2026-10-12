"""Test whether a rounded tire enables camber steering with two actuators."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from control import observe


def model_at(width, cylinder=False, reaction_gear=.48):
    spec = mujoco.MjSpec.from_file(str(ROOT / "wheelbot/model.xml"))
    tire = spec.geom("tire")
    if cylinder:
        tire.size[:] = (.12, width, 0.)
    else:
        tire.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        tire.size[:] = (.12, .12, width)
    spec.actuator("reaction_motor").gear[0] = reaction_gear
    return spec.compile()


MODEL = model_at(.016)


def evaluate(params, turning=False, metrics=False):
    speed_gain, pitch_kp, pitch_kd, roll_kp, roll_kd, roll_ff, yaw_kp = params
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[MODEL.joint("reaction").dofadr] = 110.
    mujoco.mj_forward(MODEL, data)
    speed_target = 0.
    speed_errors, yaw_errors, lateral_errors = [], [], []
    fallen = False
    for step in range(7500):
        s = observe(MODEL, data, "wheelbot")
        t = data.time
        speed = 1. if t < 2. else 2. if t < 11. else 1.5
        turn = .35 if turning and 5. <= t < 8. else -.35 if turning and 8. <= t < 11. else 0.
        speed_target += np.clip(speed - speed_target, -.004, .004)
        pitch_ref = np.clip(speed_gain * (speed_target - s.velocity[0]), -.3, .3)
        drive = np.clip(pitch_kp * (s.pitch - pitch_ref) + pitch_kd * s.omega[1], -1., 1.)
        roll_ref = np.clip(roll_ff * turn + yaw_kp * (turn - s.omega[2]), -.4, .4)
        rotor_friction = .0012 / MODEL.actuator("reaction_motor").gear[0]
        reaction = np.clip(roll_kp * (s.roll - roll_ref) + roll_kd * s.omega[0]
                           + rotor_friction * s.second_speed, -1., 1.)
        data.ctrl[:] = (drive, reaction)
        if turning and step in (4250, 5750):
            sign = 1 if step == 4250 else -1
            data.qvel[1] += sign * .25
            data.qvel[3] += sign * .25
        mujoco.mj_step(MODEL, data)
        if t > 3.:
            speed_errors.append((s.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_errors.append((s.omega[2] - turn) ** 2)
            lateral_errors.append(s.velocity[1] ** 2)
        if max(abs(s.roll), abs(s.pitch)) > .8 or s.height < .16:
            fallen = True
            break
    v = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 3.
    yaw = float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3.
    lateral = float(np.sqrt(np.mean(lateral_errors))) if lateral_errors else 3.
    score = (v + (4 * yaw + 2 * lateral if turning else 0.)
             + 3 * (15. - data.time) + (3 if fallen else 0))
    if metrics:
        return dict(score=score, speed_rmse=v, yaw_rmse=yaw,
                    lateral_rmse=lateral, duration=float(data.time), fallen=fallen)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turning", action="store_true")
    parser.add_argument("--width", type=float, default=.016,
                        help="Rounded tire half-width in meters")
    parser.add_argument("--cylinder", action="store_true",
                        help="Use a narrow cylindrical tire instead of an ellipsoid")
    parser.add_argument("--reaction-gear", type=float, default=.48)
    parser.add_argument("--iterations", type=int, default=70)
    args = parser.parse_args()
    MODEL = model_at(args.width, args.cylinder, args.reaction_gear)
    bounds = [(.05, .8), (2., 14.), (.1, 3.), (-15., 15.), (-4., 4.),
              (-2., 2.) if args.turning else (0., 0.),
              (-2., 2.) if args.turning else (0., 0.)]
    suffix = (f"{round(args.width * 1000):03d}_g{round(args.reaction_gear * 10):02d}"
              if args.cylinder else f"{round(args.width * 1000):03d}")
    label = "thin" if args.cylinder else "rounded"
    baseline_path = ROOT / "wheelbot" / f"{label}_straight_{suffix}.json"
    best = {"score": float("inf")}
    init = "latinhypercube"
    if baseline_path.exists():
        previous = np.array(json.loads(baseline_path.read_text())["params"])
        rng = np.random.default_rng(73)
        lower, upper = np.array(bounds).T
        init = rng.uniform(lower, upper, size=(56, len(bounds)))
        init[0] = np.clip(previous, lower, upper)
        for i in range(1, 25):
            init[i] = np.clip(previous + rng.normal(0, .08, len(bounds)) * (upper - lower), lower, upper)

    def progress(x, convergence):
        measured = evaluate(x, args.turning, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = x.tolist()
            path = ROOT / "wheelbot" / f"{label}_{'turning' if args.turning else 'straight'}_{suffix}.json"
            path.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} ", flush=True)

    result = differential_evolution(lambda x: evaluate(x, args.turning), bounds,
                                    seed=73, popsize=7, maxiter=args.iterations, init=init,
                                    polish=False, callback=progress)
    progress(result.x, 0.)
