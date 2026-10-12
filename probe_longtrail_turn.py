"""Find a steady, low-crab single turn before attempting command reversal."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from control import OptimizedSteerController, observe


ROOT = Path(__file__).resolve().parent
SPEC = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
SPEC.body("fork").pos[0] = .5
SPEC.body("drive_wheel").pos[0] = -.5
MODEL = SPEC.compile()


def export_controller():
    result = json.loads((ROOT / "steerbot" / "longtrail_turn.json").read_text())
    ff, yaw_kp, roll_kp, roll_kd, steer_kp, steer_kd, tau = result["params"]
    params = json.loads((ROOT / "steerbot" / "optimized_controller.json").read_text())["params"][:11]
    params[3:11] = [ff, yaw_kp, 0., roll_kp, roll_kd, steer_kp, steer_kd, tau]
    params += [0., 0.]
    target = ROOT / "steerbot" / "longtrail_controller.json"
    target.write_text(json.dumps({"params": params, "single_turn": result,
                                  "trail": .5}, indent=2))
    return target


def evaluate(params, metrics=False, duration=8., turn_end=None):
    ff, yaw_kp, roll_kp, roll_kd, steer_kp, steer_kd, tau = params
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[0] = .8
    data.qvel[MODEL.joint("drive").dofadr] = .8 / .12
    mujoco.mj_forward(MODEL, data)
    drive_controller = OptimizedSteerController(ROOT / "steerbot/optimized_controller.json")
    filtered = 0.
    speed_errors, yaw_errors, lateral_errors = [], [], []
    fallen = False
    for _ in range(round(duration / MODEL.opt.timestep)):
        s = observe(MODEL, data, "steerbot")
        t = data.time
        speed = 1. if t < 2. else 2.
        target = .35 if 5. <= t and (turn_end is None or t < turn_end) else 0.
        filtered += (target - filtered) * .002 / (tau + .002)
        drive = drive_controller.act(s, speed, 0., .002)[0]
        desired = np.clip(ff * filtered + yaw_kp * (filtered - s.omega[2])
                          - roll_kp * s.roll - roll_kd * s.omega[0], -.72, .72)
        steer = np.clip(steer_kp * (desired - s.second_pos)
                        - steer_kd * s.second_speed, -1., 1.)
        data.ctrl[:] = (drive, steer)
        mujoco.mj_step(MODEL, data)
        if 6. < t < (8. if duration <= 8. else duration - 1.):
            speed_errors.append((s.velocity[0] - speed) ** 2)
            yaw_errors.append((s.omega[2] - target) ** 2)
            lateral_errors.append(s.velocity[1] ** 2)
        if max(abs(s.roll), abs(s.pitch)) > .8 or s.height < .16:
            fallen = True
            break
    v = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 3.
    yaw = float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3.
    lateral = float(np.sqrt(np.mean(lateral_errors))) if lateral_errors else 3.
    score = v + 4 * yaw + 2 * lateral + 3 * (duration - data.time) + (3 if fallen else 0)
    if metrics:
        return dict(score=score, speed_rmse=v, yaw_rmse=yaw,
                    lateral_rmse=lateral, duration=float(data.time), fallen=fallen)
    return score


if __name__ == "__main__":
    bounds = [(-3, 3), (-4, 4), (-24, 12), (-5, 5), (.2, 10), (.05, 3), (.01, 1)]
    path = ROOT / "steerbot" / "longtrail_turn.json"
    best = json.loads(path.read_text())
    rng = np.random.default_rng(59)
    lower, upper = np.array(bounds).T
    population = rng.uniform(lower, upper, size=(70, len(bounds)))
    population[0] = np.clip(best["params"], lower, upper)
    for i in range(1, 25):
        population[i] = np.clip(population[0] + rng.normal(0, .08, len(bounds)) * (upper - lower), lower, upper)

    def progress(x, convergence):
        result = evaluate(x, True)
        if result["score"] < best["score"]:
            best.update(result)
            best["params"] = x.tolist()
            path.write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f}", flush=True)

    result = differential_evolution(evaluate, bounds, seed=59, init=population,
                                    maxiter=75, polish=False, callback=progress)
    progress(result.x, 0.)
    print("controller", export_controller(), flush=True)
