"""Find the reversal controller after a validated low-slip first turn."""

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
POS = np.array(json.loads((ROOT / "steerbot" / "longtrail_turn.json").read_text())["params"])


def evaluate(negative, metrics=False):
    brake_bias, brake_duration = negative[8:] if len(negative) > 8 else (0., 0.)
    negative = negative[:8]
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[0] = .8
    data.qvel[MODEL.joint("drive").dofadr] = .8 / .12
    mujoco.mj_forward(MODEL, data)
    drive_controller = OptimizedSteerController(ROOT / "steerbot/optimized_controller.json")
    filtered = 0.
    integral = 0.
    speed_errors, yaw_errors, lateral_errors = [], [], []
    fallen = False
    for _ in range(7500):
        s = observe(MODEL, data, "steerbot")
        t = data.time
        speed = 1. if t < 2. else 2. if t < 11. else 1.5
        target = .35 if 5. <= t < 8. else -.35 if 8. <= t < 11. else 0.
        if t < 8.:
            ff, yaw_kp, roll_kp, roll_kd, steer_kp, steer_kd, tau = POS
            yaw_ki = 0.
        else:
            ff, yaw_kp, yaw_ki, roll_kp, roll_kd, steer_kp, steer_kd, tau = negative
        filtered += (target - filtered) * .002 / (tau + .002)
        if 8. <= t < 8.002:
            integral = 0.
        integral = np.clip(integral + .002 * (filtered - s.omega[2]), -.7, .7)
        drive_target = speed + brake_bias if 8. <= t < 8. + brake_duration else speed
        drive = drive_controller.act(s, drive_target, 0., .002)[0]
        desired = np.clip(ff * filtered + yaw_kp * (filtered - s.omega[2])
                          + yaw_ki * integral - roll_kp * s.roll
                          - roll_kd * s.omega[0], -.72, .72)
        steer = np.clip(steer_kp * (desired - s.second_pos)
                        - steer_kd * s.second_speed, -1., 1.)
        data.ctrl[:] = (drive, steer)
        mujoco.mj_step(MODEL, data)
        if t > 3.:
            speed_errors.append((s.velocity[0] - speed) ** 2)
        if 5.5 < t < 10.8:
            yaw_errors.append((s.omega[2] - target) ** 2)
            lateral_errors.append(s.velocity[1] ** 2)
        if max(abs(s.roll), abs(s.pitch)) > .8 or s.height < .16:
            fallen = True
            break
    v = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 3.
    yaw = float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3.
    lateral = float(np.sqrt(np.mean(lateral_errors))) if lateral_errors else 3.
    score = v + 4 * yaw + 2 * lateral + 3 * (15. - data.time) + (3 if fallen else 0)
    if metrics:
        return dict(score=score, speed_rmse=v, yaw_rmse=yaw, lateral_rmse=lateral,
                    duration=float(data.time), fallen=fallen)
    return score


if __name__ == "__main__":
    bounds = [(-3, 3), (-4, 4), (-3, 3), (-24, 12), (-5, 5),
              (.2, 10), (.05, 3), (.01, 1)]
    base = np.r_[POS[:2], 0., POS[2:]]
    rng = np.random.default_rng(67)
    low, high = np.array(bounds).T
    population = rng.uniform(low, high, size=(80, len(bounds)))
    population[0] = np.clip(base, low, high)
    for i in range(1, 35):
        population[i] = np.clip(base + rng.normal(0, .08, len(bounds)) * (high - low), low, high)
    best = {"score": float("inf")}

    def progress(x, convergence):
        measured = evaluate(x, True)
        if measured["score"] < best["score"]:
            best.update(measured)
            best["positive"] = POS.tolist()
            best["negative"] = x.tolist()
            best["trail"] = .5
            (ROOT / "steerbot" / "longtrail_switch.json").write_text(json.dumps(best, indent=2))
        print(f"best={best['score']:.3f} v={best['speed_rmse']:.3f} "
              f"yaw={best['yaw_rmse']:.3f} lat={best['lateral_rmse']:.3f} "
              f"duration={best['duration']:.1f}", flush=True)

    result = differential_evolution(evaluate, bounds, seed=67, init=population,
                                    maxiter=70, polish=False, callback=progress)
    progress(result.x, 0.)
