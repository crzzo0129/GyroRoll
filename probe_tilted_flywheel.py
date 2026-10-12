"""Exploratory tilted-axis reaction wheel; it is not the published Wheelbot layout."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from control import observe


ROOT = Path(__file__).resolve().parent


def model_at(tilt):
    spec = mujoco.MjSpec.from_file(str(ROOT / "wheelbot/model.xml"))
    spec.body("reaction_wheel").quat[:] = (np.cos(tilt / 2), 0,
                                            -np.sin(tilt / 2), 0)
    return spec.compile()


def evaluate(model, gains, metrics=False, turn_amplitude=.35, mirror=False):
    roll_kp, roll_kd, yaw_ff, yaw_kp, yaw_kd, pitch_gain, speed_gain = gains
    data = mujoco.MjData(model)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[model.joint("reaction").dofadr] = 110.
    mujoco.mj_forward(model, data)
    speed_target = 0.
    speed_errors, yaw_errors, lateral_errors = [], [], []
    fallen = False
    for _ in range(7500):
        s = observe(model, data, "wheelbot")
        t = data.time
        speed = 1. if t < 2. else 2. if t < 11. else 1.5
        turn = turn_amplitude if 5. <= t < 8. else -turn_amplitude if 8. <= t < 11. else 0.
        if mirror:
            turn = -turn
        speed_target += np.clip(speed - speed_target, -.004, .004)
        pitch_ref = np.clip(speed_gain * (speed_target - s.velocity[0])
                            + pitch_gain * turn, -.35, .35)
        drive = np.clip(5. * (s.pitch - pitch_ref) + .75 * s.omega[1], -1., 1.)
        reaction = np.clip(roll_kp * s.roll + roll_kd * s.omega[0]
                           + .0025 * s.second_speed + yaw_ff * turn
                           + yaw_kp * (turn - s.omega[2]) - yaw_kd * s.velocity[1],
                           -1., 1.)
        data.ctrl[:] = (drive, reaction)
        if _ in (4250, 5750):
            sign = 1 if _ == 4250 else -1
            data.qvel[1] += sign * .25
            data.qvel[3] += sign * .25
        mujoco.mj_step(model, data)
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
    score = v + 4 * yaw + 2 * lateral + 3 * (15 - data.time) + (3 if fallen else 0)
    if metrics:
        return dict(score=score, speed_rmse=v, yaw_rmse=yaw, lateral_rmse=lateral,
                    duration=float(data.time), fallen=fallen)
    return score


def run(args):
    model = model_at(args.tilt)
    output = (ROOT / "wheelbot/tilted_axis_probe.json" if np.isclose(args.tilt, .4)
              else ROOT / "wheelbot" / f"tilted_axis_probe_{round(args.tilt * 100):03d}.json")
    bounds = [(1., 14.), (.15, 2.5), (-4., 4.), (-5., 5.),
              (-3., 3.), (-1., 1.), (.05, .6)]
    best = {"score": float("inf")}

    def progress(x, convergence):
        result = evaluate(model, x, True)
        if result["score"] < best["score"]:
            best.update(result)
            best["params"] = x.tolist()
            best["tilt"] = args.tilt
            output.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    previous = output
    init = "latinhypercube"
    if previous.exists():
        prior = json.loads(previous.read_text())
        if prior["tilt"] == args.tilt:
            lo, hi = np.array(bounds).T
            rng = np.random.default_rng(83)
            init = rng.uniform(lo, hi, size=(70, len(bounds)))
            init[0] = prior["params"]
            for i in range(1, 25):
                init[i] = np.clip(init[0] + rng.normal(0, .06, len(bounds)) * (hi - lo), lo, hi)
    result = differential_evolution(lambda x: evaluate(model, x), bounds,
                                    maxiter=args.iterations, popsize=10,
                                    init=init, seed=83, polish=False, callback=progress)
    progress(result.x, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilt", type=float, default=.4)
    parser.add_argument("--iterations", type=int, default=45)
    run(parser.parse_args())
