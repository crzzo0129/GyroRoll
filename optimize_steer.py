"""Search for a reproducible steerbot feedback controller on actual CPU physics.

The search objective is the same 15-second command schedule as accept.py. The
controller receives the current command only; it cannot see future switches.
"""

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from control import observe


ROOT = Path(__file__).resolve().parent
MODEL = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot/model.xml"))


def evaluate(params, return_metrics=False, dr_seed=None, disturbance=False,
             full_yaw=False):
    (speed_gain, pitch_kp, pitch_kd, yaw_ff, yaw_kp, yaw_ki,
     roll_kp, roll_kd, steer_kp, steer_kd, command_tau,
     lean_gain, lateral_gain) = params[:13]
    pitch_trim = params[13] if len(params) > 13 else 0.
    model = copy.deepcopy(MODEL) if dr_seed is not None else MODEL
    data = mujoco.MjData(model)
    if dr_seed is not None:
        rng = np.random.default_rng(dr_seed)
        body = model.body("base").id
        tire = model.geom("tire").id
        scale = rng.uniform(.85, 1.15)
        model.body_mass[body] *= scale
        model.body_inertia[body] *= scale
        model.geom_friction[tire, 0] = rng.uniform(1.1, 1.9)
        mujoco.mj_setConst(model, data)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[0] = .8
    data.qvel[7] = .8 / .12
    mujoco.mj_forward(model, data)
    speed_target = 0.
    turn_filtered = 0.
    yaw_integral = 0.
    speed_error = []
    yaw_error = []
    straight_yaw_error = []
    recovery_yaw_error = []
    lateral_error = []
    max_roll = 0.
    fell = False

    for step in range(7500):
        state = observe(model, data, "steerbot")
        t = data.time
        speed = 1. if t < 2 else 2. if t < 11 else 1.5
        turn = .35 if 5 <= t < 8 else -.35 if 8 <= t < 11 else 0.
        speed_target += np.clip(speed - speed_target, -.004, .004)
        turn_filtered += (turn - turn_filtered) * .002 / (command_tau + .002)
        yaw_integral = np.clip(yaw_integral + .002 * (turn_filtered - state.omega[2]), -.6, .6)
        pitch_ref = np.clip(speed_gain * (speed_target - state.velocity[0]) + pitch_trim, -.6, .6)
        drive = np.clip(pitch_kp * (state.pitch - pitch_ref) + pitch_kd * state.omega[1], -1., 1.)
        roll_ref = lean_gain * state.velocity[0] * turn_filtered / 9.81
        desired = np.clip(yaw_ff * turn_filtered
                          + yaw_kp * (turn_filtered - state.omega[2])
                          + yaw_ki * yaw_integral
                          - roll_kp * (state.roll - roll_ref) - roll_kd * state.omega[0]
                          + lateral_gain * state.velocity[1], -.72, .72)
        steer = np.clip(steer_kp * (desired - state.second_pos)
                        - steer_kd * state.second_speed, -1., 1.)
        data.ctrl[:] = (drive, steer)
        if disturbance:
            if step == 4250:
                data.qvel[1] += .25
                data.qvel[3] += .25
            elif step == 5750:
                data.qvel[1] -= .25
                data.qvel[3] -= .25
        mujoco.mj_step(model, data)
        max_roll = max(max_roll, abs(state.roll))
        if t > 3.:
            speed_error.append((state.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_error.append((state.omega[2] - turn) ** 2)
            lateral_error.append(state.velocity[1] ** 2)
        if 3. < t < 4.8:
            straight_yaw_error.append(state.omega[2] ** 2)
        if 12. < t < 15.:
            recovery_yaw_error.append(state.omega[2] ** 2)
        if max(abs(state.roll), abs(state.pitch)) > .8 or state.height < .16:
            fell = True
            break

    speed_rmse = float(np.sqrt(np.mean(speed_error))) if speed_error else 3.
    yaw_rmse = float(np.sqrt(np.mean(yaw_error))) if yaw_error else 3.
    lateral_rmse = float(np.sqrt(np.mean(lateral_error))) if lateral_error else 3.
    straight_yaw_rmse = float(np.sqrt(np.mean(straight_yaw_error))) if straight_yaw_error else 3.
    recovery_yaw_rmse = float(np.sqrt(np.mean(recovery_yaw_error))) if recovery_yaw_error else 3.
    score = (speed_rmse + 4. * yaw_rmse + 2. * lateral_rmse
             + 3. * (15. - data.time) + (3. if fell else 0.))
    if full_yaw:
        score += 4. * straight_yaw_rmse + 5. * recovery_yaw_rmse
    if return_metrics:
        return dict(score=score, speed_rmse=speed_rmse, yaw_rmse=yaw_rmse,
                    straight_yaw_rmse=straight_yaw_rmse,
                    recovery_yaw_rmse=recovery_yaw_rmse, lateral_rmse=lateral_rmse,
                    max_roll=max_roll, duration=float(data.time), fallen=fell)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--popsize", type=int, default=7)
    parser.add_argument("--warm-start", action="store_true")
    args = parser.parse_args()
    bounds = [(.1, .8), (3., 12.), (.2, 2.), (-3., 3.), (-3., 3.),
              (-3., 3.), (-6., 6.), (-2., 2.), (2., 12.), (.1, 1.5), (.02, .9),
              (-3., 3.), (-2., 2.)]
    init = "latinhypercube"
    if args.warm_start:
        previous = np.array(json.loads((ROOT / "steerbot" / "optimized_controller.json").read_text())["params"])
        if len(previous) == 11:
            previous = np.r_[previous, 0., 0.]
        rng = np.random.default_rng(19)
        lower, upper = np.array(bounds).T
        init = rng.uniform(lower, upper, size=(args.popsize * len(bounds), len(bounds)))
        init[0] = previous
        for i in range(1, min(15, len(init))):
            init[i] = np.clip(previous + rng.normal(0, .08, len(bounds)) * (upper - lower), lower, upper)
    best = {"score": float("inf")}
    if args.warm_start:
        best.update(evaluate(previous, return_metrics=True))
        best["params"] = previous.tolist()

    def progress(result, convergence):
        nonlocal_best = evaluate(result, return_metrics=True)
        if nonlocal_best["score"] < best["score"]:
            best.update(nonlocal_best)
            best["params"] = result.tolist()
            (ROOT / "steerbot" / "optimized_controller.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} speed={best['speed_rmse']:.3f} yaw={best['yaw_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, bounds, maxiter=args.maxiter,
                                    popsize=args.popsize, seed=19, polish=False,
                                    callback=progress, updating="immediate", init=init)
    progress(result.x, 0.)
