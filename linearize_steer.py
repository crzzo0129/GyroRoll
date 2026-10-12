"""Finite-difference lateral model of the moving, single-wheel MuJoCo plant.

The state is [roll, roll rate, lateral speed, yaw rate, steer angle,
steer rate]. The longitudinal drive loop is closed; steering input is motor
command. This model is only local to the straight ~2 m/s trim trajectory.
"""

from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from control import OptimizedSteerController, observe


ROOT = Path(__file__).resolve().parent
DT = .02


def state(model, data):
    s = observe(model, data, "steerbot")
    return np.array((s.roll, s.omega[0], s.velocity[1], s.omega[2],
                     s.second_pos, s.second_speed))


def perturb(model, data, index, epsilon):
    if index == 0:
        q = np.array((np.cos(epsilon / 2), np.sin(epsilon / 2), 0., 0.))
        original = data.qpos[3:7].copy()
        mujoco.mju_mulQuat(data.qpos[3:7], original, q)
    elif index == 1:
        data.qvel[3] += epsilon
    elif index == 2:
        data.qvel[1] += epsilon
    elif index == 3:
        data.qvel[5] += epsilon
    elif index == 4:
        data.qpos[model.joint("steer").qposadr] += epsilon
    elif index == 5:
        data.qvel[model.joint("steer").dofadr] += epsilon
    mujoco.mj_forward(model, data)


def identify(com_offset=None, trail=None, body_width=None,
             tire_halfwidth=None, steer_gear=None,
             body_mass=None, body_offset_z=None, head_mass=None, wheel_mass=None):
    spec = mujoco.MjSpec.from_file(str(ROOT / "steerbot/model.xml"))
    if com_offset is not None:
        spec.geom("body").pos[0] = com_offset
    if trail is not None:
        spec.body("fork").pos[0] = trail
        spec.body("drive_wheel").pos[0] = -trail
    if body_width is not None:
        spec.geom("body").size[:2] = body_width
    if tire_halfwidth is not None:
        spec.geom("tire").size[1] = tire_halfwidth
    if steer_gear is not None:
        spec.actuator("steer_motor").gear[0] = steer_gear
    if body_mass is not None:
        spec.geom("body").mass = body_mass
    if body_offset_z is not None:
        spec.geom("body").pos[2] += body_offset_z
    if head_mass is not None:
        spec.geom("steering_head").mass = head_mass
    if wheel_mass is not None:
        spec.geom("tire").mass = wheel_mass
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:7] = (0, 0, .28, 1, 0, 0, 0)
    data.qvel[0] = .8
    data.qvel[model.joint("drive").dofadr] = .8 / .12
    mujoco.mj_forward(model, data)
    controller = OptimizedSteerController(ROOT / "steerbot/optimized_controller.json")
    for _ in range(2400):
        s = observe(model, data, "steerbot")
        drive = controller.act(s, 2., 0., model.opt.timestep)[0]
        data.ctrl[:] = (drive, 0.)
        mujoco.mj_step(model, data)
    trim = data.qpos.copy(), data.qvel.copy()
    speed_target = controller.speed_target
    gain, kp, kd = controller.params[:3]

    def advance(delta=None, control=0.):
        temp = mujoco.MjData(model)
        temp.qpos[:] = trim[0]
        temp.qvel[:] = trim[1]
        mujoco.mj_forward(model, temp)
        if delta is not None:
            perturb(model, temp, *delta)
        for _ in range(round(DT / model.opt.timestep)):
            s = observe(model, temp, "steerbot")
            pitch_ref = np.clip(gain * (speed_target - s.velocity[0]), -.3, .3)
            temp.ctrl[:] = (np.clip(kp * (s.pitch - pitch_ref) + kd * s.omega[1], -1, 1), control)
            mujoco.mj_step(model, temp)
        return state(model, temp)

    eps = 1.e-4
    a = np.column_stack([(advance((i, eps)) - advance((i, -eps))) / (2 * eps)
                         for i in range(6)])
    b = ((advance(control=eps) - advance(control=-eps)) / (2 * eps))[:, None]
    x = state(model, data)
    ctrb = np.column_stack([np.linalg.matrix_power(a, i) @ b for i in range(6)])
    return dict(A=a, B=b, trim=x, speed=observe(model, data, "steerbot").velocity[0],
                controllability_rank=np.linalg.matrix_rank(ctrb))


def lqr(model, q=(80., 2., 2., 8., 2., .2, 80.), r=1.):
    a, b = model["A"], model["B"]
    extended_a = np.eye(7)
    extended_a[:6, :6] = a
    extended_a[6, 3] = -DT
    extended_b = np.r_[b, [[0.]]]
    p = solve_discrete_are(extended_a, extended_b, np.diag(q), np.array([[r]]))
    gain = np.linalg.solve(extended_b.T @ p @ extended_b + r,
                           extended_b.T @ p @ extended_a)
    return gain.ravel()


if __name__ == "__main__":
    result = identify()
    for key, value in result.items():
        print(key, value)
    print("LQR gain", lqr(result))
