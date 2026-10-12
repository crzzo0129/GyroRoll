"""Experimental MuJoCo-in-the-loop shooting MPC for the long-trail unicycle.

This is an offline feasibility probe, not a real-time joystick controller.
It optimizes steering motor commands while a proven pitch loop drives the wheel.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from control import OptimizedSteerController, observe
from sim import sample_commands


ROOT = Path(__file__).resolve().parent
SPEC = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
SPEC.body("fork").pos[0] = .5
SPEC.body("drive_wheel").pos[0] = -.5
MODEL = SPEC.compile()
PARAMS = json.loads((ROOT / "steerbot" / "optimized_controller.json").read_text())["params"]
DT = MODEL.opt.timestep


def drive(s, target, filtered):
    gain, kp, kd = PARAMS[:3]
    filtered += np.clip(target - filtered, -2 * DT, 2 * DT)
    pitch_ref = np.clip(gain * (filtered - s.velocity[0]), -.3, .3)
    return np.clip(kp * (s.pitch - pitch_ref) + kd * s.omega[1], -1, 1), filtered


def initial():
    data = mujoco.MjData(MODEL)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[0] = .8
    data.qvel[MODEL.joint("drive").dofadr] = .8 / .12
    mujoco.mj_forward(MODEL, data)
    return data


def predict(work, actual, controls, speed, turn, filtered, horizon, knot_steps):
    work.qpos[:] = actual.qpos
    work.qvel[:] = actual.qvel
    work.time = actual.time
    mujoco.mj_forward(MODEL, work)
    cost = 0.
    steps = round(horizon / DT)
    for k in range(steps):
        s = observe(MODEL, work, "steerbot")
        knot = k / knot_steps
        u = np.interp(knot, np.arange(len(controls)), controls)
        motor, filtered = drive(s, speed, filtered)
        work.ctrl[:] = (motor, u)
        mujoco.mj_step(MODEL, work)
        if k % 10 == 0:
            roll_ref = -s.velocity[0] * turn / 9.81
            cost += (.12 * (s.velocity[0] - speed) ** 2
                     + 12. * (s.omega[2] - turn) ** 2
                     + 3. * s.velocity[1] ** 2
                     + 6. * (s.roll - roll_ref) ** 2
                     + .6 * s.pitch ** 2 + .003 * u ** 2)
        if max(abs(s.roll), abs(s.pitch)) > .65 or s.height < .17:
            return cost + 200 + 2 * (steps - k)
    s = observe(MODEL, work, "steerbot")
    return cost + (70 * (s.omega[2] - turn) ** 2 + 10 * s.velocity[1] ** 2
                   + 50 * (s.roll + s.velocity[0] * turn / 9.81) ** 2
                   + 2 * (s.velocity[0] - speed) ** 2)


def optimize(work, actual, previous, speed, turn, filtered, args, rng):
    knot_steps = round(args.horizon / len(previous) / DT)
    # Receding-horizon plans are time-indexed; reuse the remaining future rather
    # than replaying the previous first knot at every optimization call.
    shifted = args.control_period / args.horizon * len(previous)
    mean = np.interp(np.arange(len(previous)) + shifted,
                     np.arange(len(previous)), previous)
    spread = .45
    best = mean.copy()
    best_cost = predict(work, actual, best, speed, turn, filtered, args.horizon, knot_steps)
    for _ in range(args.cem_iters):
        candidates = np.clip(rng.normal(mean, spread, size=(args.samples, len(mean))), -1, 1)
        candidates[0] = best
        scores = np.array([predict(work, actual, c, speed, turn, filtered,
                                   args.horizon, knot_steps) for c in candidates])
        elite = candidates[np.argsort(scores)[:max(4, args.samples // 6)]]
        mean = elite.mean(axis=0)
        spread = max(.10, float(np.mean(elite.std(axis=0))) * 1.5)
        if scores.min() < best_cost:
            best_cost = float(scores.min())
            best = candidates[scores.argmin()].copy()
    return best, best_cost


def run(args):
    data = initial()
    work = mujoco.MjData(MODEL)
    baseline = OptimizedSteerController(ROOT / "steerbot" / "optimized_controller.json")
    rng = np.random.default_rng(83)
    sequence = np.zeros(args.knots)
    action = 0.
    filtered = 0.
    trajectory = []
    errors = [[], [], []]
    fallen = False
    control_steps = round(args.control_period / DT)
    for step in range(round(args.duration / DT)):
        t = data.time
        speed, turn = sample_commands(t, 2., 0., True)
        s = observe(MODEL, data, "steerbot")
        if step % control_steps == 0 and t >= 4.9:
            sequence, cost = optimize(work, data, sequence, speed, turn,
                                      filtered, args, rng)
            action = sequence[0]
            if args.verbose:
                print(f"t={t:.2f} u={action:.3f} cost={cost:.2f} yaw={s.omega[2]:.3f} roll={s.roll:.3f}", flush=True)
        if t < 4.9:
            baseline_action = baseline.act(s, speed, 0., DT)
            action = baseline_action[1]
            filtered = baseline.speed_target
        motor, filtered = drive(s, speed, filtered)
        data.ctrl[:] = (motor, action)
        mujoco.mj_step(MODEL, data)
        if t > 3.:
            errors[0].append((s.velocity[0] - speed) ** 2)
        if 5.5 < t < min(10.8, args.duration):
            errors[1].append((s.omega[2] - turn) ** 2)
            errors[2].append(s.velocity[1] ** 2)
        if step % 10 == 0:
            trajectory.append(dict(t=round(t, 3), v=round(s.velocity[0], 3),
                                   vy=round(s.velocity[1], 3), yaw=round(s.omega[2], 3),
                                   turn=turn, roll=round(s.roll, 3), steer=round(s.second_pos, 3),
                                   action=round(float(action), 3)))
        if max(abs(s.roll), abs(s.pitch)) > .8 or s.height < .16:
            fallen = True
            break
    summary = dict(duration=float(data.time), fallen=fallen,
                   speed_rmse=float(np.sqrt(np.mean(errors[0]))) if errors[0] else None,
                   yaw_rmse=float(np.sqrt(np.mean(errors[1]))) if errors[1] else None,
                   lateral_rmse=float(np.sqrt(np.mean(errors[2]))) if errors[2] else None)
    output = ROOT / "results" / "steerbot_mpc_probe.json"
    output.write_text(json.dumps(dict(summary=summary, trajectory=trajectory), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--horizon", type=float, default=.8)
    parser.add_argument("--knots", type=int, default=4)
    parser.add_argument("--control-period", type=float, default=.1)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--cem-iters", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    run(parser.parse_args())
