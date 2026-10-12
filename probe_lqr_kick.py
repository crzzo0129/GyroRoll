"""Probe state-triggered steer corrections on both kick/turn directions."""

import json
from pathlib import Path

import mujoco
import numpy as np

from control import LqrSteerController, observe
from sim import sample_commands


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "steerbot/negative_candidate.json").read_text())


def run(gain, threshold, mirror=False, seed=0):
    spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
    trail = CFG["trail"]
    spec.body("fork").pos[0] = trail
    spec.body("drive_wheel").pos[0] = -trail
    spec.geom("body").mass = CFG["body_mass"]
    spec.geom("body").pos[2] += CFG["body_offset_z"]
    spec.geom("steering_head").mass = CFG["head_mass"]
    spec.geom("tire").size[1] = CFG["tire_halfwidth"]
    spec.actuator("steer_motor").gear[0] = CFG["steer_gear"]
    model = spec.compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    base = model.body("base").id
    tire = model.geom("tire").id
    scale = rng.uniform(.85, 1.15)
    model.body_mass[base] *= scale
    model.body_inertia[base] *= scale
    model.geom_friction[tire, 0] = rng.uniform(1.1, 1.9)
    mujoco.mj_setConst(model, data)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[0] = .8
    data.qvel[model.joint("drive").dofadr] = .8 / .12
    mujoco.mj_forward(model, data)
    controller = LqrSteerController(ROOT / "steerbot/optimized_controller.json",
                                    trail=trail, tire_halfwidth=CFG["tire_halfwidth"],
                                    steer_gear=CFG["steer_gear"],
                                    body_mass=CFG["body_mass"],
                                    body_offset_z=CFG["body_offset_z"],
                                    head_mass=CFG["head_mass"],
                                    tuning_path=ROOT / "steerbot/negative_candidate.json")
    yaw_errors = []
    recovery_errors = []
    for step in range(7500):
        state = observe(model, data, "steerbot")
        speed, turn = sample_commands(data.time, 0., 0., True, mirror)
        if 5.5 < data.time < 10.8:
            yaw_errors.append((state.omega[2] - turn) ** 2)
        if 12. < data.time < 15.:
            recovery_errors.append(state.omega[2] ** 2)
        if abs(state.roll) > .8 or abs(state.pitch) > .8 or state.height < .16:
            break
        action = controller.act(state, speed, turn, .002)
        if turn > .02:
            action[1] = np.clip(action[1] + gain * max(state.omega[2] - turn - threshold, 0.),
                                -1., 1.)
        data.ctrl[:] = action
        if step == 4250:
            data.qvel[1] += .25
            data.qvel[3] += .25
        elif step == 5750:
            data.qvel[1] -= .25
            data.qvel[3] -= .25
        mujoco.mj_step(model, data)
    return (round(float(np.sqrt(np.mean(yaw_errors))) if yaw_errors else 3., 4),
            round(float(np.sqrt(np.mean(recovery_errors))) if recovery_errors else 3., 4),
            round(float(data.time), 3))


if __name__ == "__main__":
    for threshold in (.05, .1, .2, .3, .4):
        for gain in (-.4, -.2, -.1, -.05, 0., .025, .05, .1, .2, .4):
            print(threshold, gain, run(gain, threshold), run(gain, threshold, True), flush=True)
