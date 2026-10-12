"""Search a gyro-steering gait while retaining roll and speed feedback."""

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


MODEL = mujoco.MjModel.from_xml_path(str(ROOT / "wheelbot/model.xml"))


def evaluate(params, return_metrics=False):
    pitch_amp, rotor_amp, phase, hz, rotor_kp, roll_kp, roll_kd = params
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[7] = 110.
    mujoco.mj_forward(MODEL, data)
    speed_target = 0.
    speed_errors = []
    yaw_errors = []
    max_roll = 0.
    fell = False

    for step in range(7500):
        state = observe(MODEL, data, "wheelbot")
        t = data.time
        speed = 1. if t < 2 else 2. if t < 11 else 1.5
        turn = .35 if 5 <= t < 8 else -.35 if 8 <= t < 11 else 0.
        speed_target += np.clip(speed - speed_target, -.004, .004)
        cycle = 2 * np.pi * hz * t
        strength = turn / .35
        pitch_offset = pitch_amp * strength * np.sin(cycle + phase)
        pitch_ref = np.clip(.18 * (speed_target - state.velocity[0]) + pitch_offset, -.3, .3)
        drive = np.clip(5. * (state.pitch - pitch_ref) + .75 * state.omega[1], -1., 1.)
        rotor_target = 110. + rotor_amp * np.sin(cycle)
        reaction = np.clip(roll_kp * state.roll + roll_kd * state.omega[0]
                           + .0025 * state.second_speed
                           + rotor_kp * (rotor_target - state.second_speed), -1., 1.)
        data.ctrl[:] = (drive, reaction)
        mujoco.mj_step(MODEL, data)
        max_roll = max(max_roll, abs(state.roll))
        if t > 3.:
            speed_errors.append((state.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_errors.append((state.omega[2] - turn) ** 2)
        if max(abs(state.roll), abs(state.pitch)) > .8 or state.height < .16:
            fell = True
            break

    speed_rmse = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 3.
    yaw_rmse = float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3.
    score = speed_rmse + 2. * yaw_rmse + 3. * (15. - data.time) + (3. if fell else 0.)
    if return_metrics:
        return dict(score=score, speed_rmse=speed_rmse, yaw_rmse=yaw_rmse,
                    max_roll=max_roll, duration=float(data.time), fallen=fell)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--popsize", type=int, default=6)
    args = parser.parse_args()
    bounds = [(0., .2), (0., 170.), (-np.pi, np.pi), (.3, 3.),
              (0., .01), (2., 7.), (.3, 1.5)]
    best = {"score": float("inf")}

    def progress(result, convergence):
        measured = evaluate(result, return_metrics=True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = result.tolist()
            (ROOT / "wheelbot" / "gyro_gait.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} speed={best['speed_rmse']:.3f} yaw={best['yaw_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, bounds, maxiter=args.maxiter,
                                    popsize=args.popsize, seed=31, polish=False,
                                    callback=progress)
    progress(result.x, 0.)
