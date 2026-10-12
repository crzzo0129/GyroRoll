"""Tune local steering LQR on the full nonlinear MuJoCo 15-second test."""

import json
import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from control import OptimizedSteerController, observe
from linearize_steer import identify, lqr


MODEL = mujoco.MjModel.from_xml_path(str(ROOT / "steerbot/model.xml"))
LOCAL = identify()
LONGITUDINAL = ROOT / "steerbot" / "optimized_controller.json"


def evaluate(params, metrics=False, dr_seed=None, disturbance=False,
             full_yaw=False, mirror=False):
    roll_q, yaw_q, integral_q, effort_r, tau, limit, lateral_q = params[:7]
    ff_scale = params[7] if len(params) > 7 else 1.
    try:
        gain = lqr(LOCAL, q=(roll_q, 2., lateral_q, yaw_q, 2., .2, integral_q), r=effort_r)
    except (ValueError, np.linalg.LinAlgError):
        return {"score": 100.} if metrics else 100.
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
    data.qvel[model.joint("drive").dofadr] = .8 / .12
    mujoco.mj_forward(model, data)
    drive_controller = OptimizedSteerController(LONGITUDINAL)
    steady_response = np.linalg.solve(np.eye(6) - LOCAL["A"], LOCAL["B"]).ravel()
    integral = 0.
    filtered = 0.
    speed_errors, yaw_errors, lateral_errors = [], [], []
    straight_yaw_errors, recovery_yaw_errors = [], []
    fallen = False
    for step in range(7500):
        state = observe(model, data, "steerbot")
        t = data.time
        speed = 1. if t < 2 else 2. if t < 11 else 1.5
        turn = (-.35 if mirror else .35) if 5 <= t < 8 else (
               .35 if mirror else -.35) if 8 <= t < 11 else 0.
        drive = drive_controller.act(state, speed, 0., .002)[0]
        filtered += (turn - filtered) * .002 / (tau + .002)
        integral = np.clip(integral + (filtered - state.omega[2]) * .002, -limit, limit)
        lateral = np.array((state.roll, state.omega[0], state.velocity[1],
                            state.omega[2], state.second_pos, state.second_speed))
        feedforward = ff_scale * filtered / steady_response[3]
        reference = LOCAL["trim"] + steady_response * feedforward
        steer = np.clip(feedforward - gain[:6] @ (lateral - reference) - gain[6] * integral, -1., 1.)
        data.ctrl[:] = (drive, steer)
        if disturbance:
            if step == 4250:
                data.qvel[1] += .25
                data.qvel[3] += .25
            elif step == 5750:
                data.qvel[1] -= .25
                data.qvel[3] -= .25
        mujoco.mj_step(model, data)
        if t > 3.:
            speed_errors.append((state.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_errors.append((state.omega[2] - turn) ** 2)
            lateral_errors.append(state.velocity[1] ** 2)
        if 3. < t < 4.8:
            straight_yaw_errors.append(state.omega[2] ** 2)
        if 12. < t < 15.:
            recovery_yaw_errors.append(state.omega[2] ** 2)
        if max(abs(state.roll), abs(state.pitch)) > .8 or state.height < .16:
            fallen = True
            break
    speed_rmse = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 3.
    yaw_rmse = float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3.
    lateral_rmse = float(np.sqrt(np.mean(lateral_errors))) if lateral_errors else 3.
    straight_yaw_rmse = float(np.sqrt(np.mean(straight_yaw_errors))) if straight_yaw_errors else 3.
    recovery_yaw_rmse = float(np.sqrt(np.mean(recovery_yaw_errors))) if recovery_yaw_errors else 3.
    score = (speed_rmse + 4 * yaw_rmse + 2 * lateral_rmse
             + 3 * (15. - data.time) + (3. if fallen else 0.))
    if full_yaw:
        score += 4 * straight_yaw_rmse + 5 * recovery_yaw_rmse
    if metrics:
        return dict(score=score, speed_rmse=speed_rmse, yaw_rmse=yaw_rmse,
                    straight_yaw_rmse=straight_yaw_rmse,
                    recovery_yaw_rmse=recovery_yaw_rmse, lateral_rmse=lateral_rmse,
                    duration=float(data.time), fallen=fallen)
    return score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trail", type=float, default=.5)
    parser.add_argument("--tire-halfwidth", type=float, default=None)
    parser.add_argument("--steer-gear", type=float, default=None)
    parser.add_argument("--body-mass", type=float, default=None)
    parser.add_argument("--body-offset-z", type=float, default=None)
    parser.add_argument("--head-mass", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--maxiter", type=int, default=50)
    args = parser.parse_args()
    spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
    spec.body("fork").pos[0] = args.trail
    spec.body("drive_wheel").pos[0] = -args.trail
    if args.tire_halfwidth is not None:
        spec.geom("tire").size[1] = args.tire_halfwidth
    if args.steer_gear is not None:
        spec.actuator("steer_motor").gear[0] = args.steer_gear
    if args.body_mass is not None:
        spec.geom("body").mass = args.body_mass
    if args.body_offset_z is not None:
        spec.geom("body").pos[2] += args.body_offset_z
    if args.head_mass is not None:
        spec.geom("steering_head").mass = args.head_mass
    MODEL = spec.compile()
    LOCAL = identify(trail=args.trail, tire_halfwidth=args.tire_halfwidth,
                     steer_gear=args.steer_gear, body_mass=args.body_mass,
                     body_offset_z=args.body_offset_z, head_mass=args.head_mass)
    bounds = [(3., 300.), (.1, 200.), (.02, 200.), (.1, 20.), (.01, .8),
              (.05, 2.), (.001, 8.), (.1, 2.)]
    rng = np.random.default_rng(29)
    lower, upper = np.array(bounds).T
    population = rng.uniform(lower, upper, size=(64, len(bounds)))
    previous = args.output or (ROOT / "steerbot/lqr_morph_028.json" if args.body_mass is not None else
                ROOT / "steerbot/lqr_thin_004_g24.json" if args.tire_halfwidth is not None else
                ROOT / "steerbot/lqr_longtrail.json")
    if previous.exists():
        old = json.loads(previous.read_text())["params"]
        population[0] = old if len(old) == 8 else np.r_[old[:6], .05, 1.]
        for i in range(1, 20):
            population[i] = np.clip(population[0] + rng.normal(0, .12, len(bounds)) * (upper - lower), lower, upper)
    elif args.body_mass is not None:
        population[0] = json.loads((ROOT / "steerbot/lqr_thin_004_g24.json").read_text())["params"]
        for i in range(1, 20):
            population[i] = np.clip(population[0] + rng.normal(0, .12, len(bounds)) *
                                    (upper - lower), lower, upper)
    best = {"score": float("inf")}

    def progress(params, convergence):
        measured = evaluate(params, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["params"] = params.tolist()
            best["trail"] = args.trail
            if args.tire_halfwidth is not None:
                best["tire_halfwidth"] = args.tire_halfwidth
            if args.steer_gear is not None:
                best["steer_gear"] = args.steer_gear
            if args.body_mass is not None:
                best["body_mass"] = args.body_mass
                best["body_offset_z"] = args.body_offset_z
                best["head_mass"] = args.head_mass
            previous.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} speed={best.get('speed_rmse', 0):.3f} "
              f"yaw={best.get('yaw_rmse', 0):.3f} lateral={best.get('lateral_rmse', 0):.3f}", flush=True)

    result = differential_evolution(evaluate, bounds, maxiter=args.maxiter,
                                    init=population, seed=29, polish=False, callback=progress)
    progress(result.x, 0.)
