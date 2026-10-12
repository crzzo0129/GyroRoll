"""Test noncommutative roll/pitch loops as a two-motor yaw mechanism."""

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
    roll_amp, pitch_amp, frequency, phase, roll_kp, roll_kd, command_tau = params[:7]
    rotor_init = params[7] if len(params) > 7 else 110.
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[MODEL.joint("reaction").dofadr] = rotor_init
    mujoco.mj_forward(MODEL, data)
    speed_target = 0.
    turn_filtered = 0.
    speed_error, yaw_error, lateral_error = [], [], []
    fallen = False
    for _ in range(7500):
        s = observe(MODEL, data, "wheelbot")
        t = data.time
        speed = 1. if t < 2. else 2. if t < 11. else 1.5
        turn = .35 if 5. <= t < 8. else -.35 if 8. <= t < 11. else 0.
        speed_target += np.clip(speed - speed_target, -.004, .004)
        turn_filtered += (turn - turn_filtered) * .002 / (command_tau + .002)
        cycle = 2 * np.pi * frequency * t
        strength = turn_filtered / .35
        roll_ref = roll_amp * abs(strength) * np.sin(cycle)
        pitch_offset = pitch_amp * strength * np.sin(cycle + phase)
        pitch_ref = np.clip(.18 * (speed_target - s.velocity[0]) + pitch_offset, -.52, .52)
        drive = np.clip(5. * (s.pitch - pitch_ref) + .75 * s.omega[1], -1., 1.)
        rotor_friction = .0012 / MODEL.actuator("reaction_motor").gear[0]
        reaction = np.clip(roll_kp * (s.roll - roll_ref) + roll_kd * s.omega[0]
                           + rotor_friction * s.second_speed, -1., 1.)
        data.ctrl[:] = (drive, reaction)
        mujoco.mj_step(MODEL, data)
        if t > 3.:
            speed_error.append((s.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_error.append((s.omega[2] - turn) ** 2)
            lateral_error.append(s.velocity[1] ** 2)
        if max(abs(s.roll), abs(s.pitch)) > .8 or s.height < .16:
            fallen = True
            break
    v = float(np.sqrt(np.mean(speed_error))) if speed_error else 3.
    yaw = float(np.sqrt(np.mean(yaw_error))) if yaw_error else 3.
    lateral = float(np.sqrt(np.mean(lateral_error))) if lateral_error else 3.
    score = v + 4 * yaw + 2 * lateral + 3 * (15. - data.time) + (3 if fallen else 0)
    if metrics:
        return dict(score=score, speed_rmse=v, yaw_rmse=yaw,
                    lateral_rmse=lateral, duration=float(data.time), fallen=fallen)
    return score


if __name__ == "__main__":
    print("baseline", evaluate([0, 0, 1, 0, 3.7, .65, .2], True), flush=True)
    bounds = [(0, .45), (0, .45), (.25, 3.5), (-np.pi, np.pi),
              (2., 9.), (.2, 2.), (.02, .7)]
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = evaluate(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = x.tolist()
            (ROOT / "wheelbot" / "roll_pitch_loop.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, bounds, seed=71, popsize=7,
                                    maxiter=75, polish=False, callback=progress)
    progress(result.x, 0.)
