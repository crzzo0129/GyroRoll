"""Probe gyroscopic yaw from sustained, bounded pitch-reference motion."""

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


def evaluate(params, metrics=False):
    ramp_gain, yaw_feedback, ramp_limit, rotor_target, rotor_kp, roll_kp, roll_kd = params
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[MODEL.joint("reaction").dofadr] = rotor_target
    mujoco.mj_forward(MODEL, data)
    speed_target = 0.
    pitch_offset = 0.
    speed_errors, yaw_errors = [], []
    fallen = False
    for _ in range(7500):
        s = observe(MODEL, data, "wheelbot")
        t = data.time
        speed = 1. if t < 2 else 2. if t < 11 else 1.5
        turn = .35 if 5 <= t < 8 else -.35 if 8 <= t < 11 else 0.
        speed_target += np.clip(speed - speed_target, -.004, .004)
        pitch_offset = np.clip(pitch_offset + .002 * ramp_gain
                               * (turn + yaw_feedback * (turn - s.omega[2])),
                               -ramp_limit, ramp_limit)
        if turn == 0.:
            pitch_offset *= .9997
        pitch_ref = np.clip(.18 * (speed_target - s.velocity[0]) + pitch_offset, -.3, .3)
        drive = np.clip(5. * (s.pitch - pitch_ref) + .75 * s.omega[1], -1., 1.)
        reaction = np.clip(roll_kp * s.roll + roll_kd * s.omega[0]
                           + .0025 * s.second_speed
                           + rotor_kp * (rotor_target - s.second_speed), -1., 1.)
        data.ctrl[:] = (drive, reaction)
        mujoco.mj_step(MODEL, data)
        if t > 3.:
            speed_errors.append((s.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_errors.append((s.omega[2] - turn) ** 2)
        if max(abs(s.roll), abs(s.pitch)) > .8 or s.height < .16:
            fallen = True
            break
    speed_rmse = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 3.
    yaw_rmse = float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3.
    score = speed_rmse + 2 * yaw_rmse + 3 * (15. - data.time) + (3. if fallen else 0.)
    if metrics:
        return dict(score=score, speed_rmse=speed_rmse, yaw_rmse=yaw_rmse,
                    duration=float(data.time), fallen=fallen)
    return score


if __name__ == "__main__":
    baseline = [0., 0., .2, 110., .005, 3.7, .65]
    print("baseline", evaluate(baseline, True), flush=True)
    for ramp in (-.4, -.15, -.05, .05, .15, .4):
        trial = baseline.copy()
        trial[0] = ramp
        print("ramp", ramp, evaluate(trial, True), flush=True)
    bounds = [(-.6, .6), (-2., 2.), (.04, .28), (50., 220.),
              (0., .02), (2., 8.), (.2, 1.5)]
    best = {"score": float("inf")}

    def progress(params, convergence):
        measured = evaluate(params, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = params.tolist()
            (ROOT / "wheelbot" / "ramp_controller.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} speed={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, bounds, maxiter=55, popsize=6,
                                    seed=17, polish=False, callback=progress)
    progress(result.x, 0.)
