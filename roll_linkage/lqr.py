"""Free-body LQR on full MuJoCo tangent states with compliant loop/contact modes.

The manifold lift is used only for initial conditions; finite differences retain
all internal joint and contact degrees of freedom. Rollouts use mj_step without
state resets or torso orientation locks.
"""
from pathlib import Path
import math
import sys

import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roll_linkage.build import make_xml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/roll_linkage/lqr"
DT = .01
STATE_NAMES = ["torso_roll", "torso_pitch", "roll_rate", "pitch_rate",
               "forward_speed", "lateral_speed", "heading_rate",
               "wheel_roll_relative", "wheel_roll_rate", "drive_speed"]


def dare(a, b, q, r, discount=.999):
    """Discrete Riccati iteration; optional small discount for neutral yaw modes."""
    p = q.copy()
    for iteration in range(50000):
        k = np.linalg.solve(r + discount*b.T@p@b, discount*b.T@p@a)
        pn = q + discount*a.T@p@a - discount*a.T@p@b@k
        pn = (pn + pn.T)/2
        if np.max(np.abs(pn-p)) < 1e-10 * max(1., np.max(np.abs(pn))):
            break
        p = pn
    else:
        raise RuntimeError("Riccati iteration did not converge")
    return np.linalg.solve(r + discount*b.T@pn@b, discount*b.T@pn@a)


class Plant:
    def __init__(self, timestep=.001, original=False):
        self.xml = make_xml(level_torso=not original)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.m.opt.timestep = timestep
        self.base = self.m.body("base").id
        self.carrier = self.m.body("wheel_carrier").id
        self.q = {name: int(self.m.joint(name).qposadr[0])
                  for name in ("linkage_motor", "passive_rod", "wheel_roll", "drive")}
        self.v = {name: int(self.m.joint(name).dofadr[0]) for name in self.q}
        self.tires = np.array([self.m.geom(f"tire_{i}").id for i in range(128)])
        self.sites = [self.m.site(n).id for n in ("rod_end", "rocker_end")]
        assert self.m.nu == 2 and self.m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.penetration = .0001
        self.trim_state = np.zeros(10)
        self.trim_control = np.zeros(2)
        # Select nominal tire compression to support weight without fake supports.
        lo, hi = 0., .003
        for _ in range(45):
            self.penetration = (lo+hi)/2
            d = self.lift(self.trim_state)
            if d.qacc[2] > 0:
                hi = self.penetration
            else:
                lo = self.penetration
        self.penetration = (lo+hi)/2
        d = self.lift(self.trim_state)
        tangent = np.zeros(self.m.nv)
        tangent[[self.v["linkage_motor"], self.v["passive_rod"], self.v["wheel_roll"]]] = [1, -1, 1]
        self.trim_control[1] = (tangent @ d.qfrc_bias) / 8.

    def state(self, d):
        r = d.xmat[self.base].reshape(3, 3)
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], np.linalg.norm(r[2, 1:]))
        omega = d.qvel[3:6]
        velocity = r.T @ d.qvel[:3]
        forward = r[:, 0]
        fdot = np.cross(r @ omega, forward)
        heading_rate = (forward[0]*fdot[1]-forward[1]*fdot[0]) / max(1e-8, forward[:2]@forward[:2])
        return np.array([roll, pitch, omega[0], omega[1], velocity[0], velocity[1],
                         heading_rate, d.qpos[self.q["wheel_roll"]],
                         d.qvel[self.v["wheel_roll"]], d.qvel[self.v["drive"]]])

    def lift(self, x):
        d = mujoco.MjData(self.m)
        qx = np.array([math.cos(x[0]/2), math.sin(x[0]/2), 0, 0])
        qy = np.array([math.cos(x[1]/2), 0, math.sin(x[1]/2), 0])
        mujoco.mju_mulQuat(d.qpos[3:7], qy, qx)
        for name, sign in (("linkage_motor", 1), ("passive_rod", -1), ("wheel_roll", 1)):
            d.qpos[self.q[name]] = sign*x[7]
            d.qvel[self.v[name]] = sign*x[8]
        d.qvel[self.v["drive"]] = x[9]
        mujoco.mj_forward(self.m, d)
        r = d.xmat[self.base].reshape(3, 3)
        d.qvel[:3] = r @ [x[4], x[5], 0.]
        # Convert desired Euler yaw rate to body wz at this roll/pitch.
        d.qvel[3:6] = [x[2], x[3], (x[6]*math.cos(x[1])-x[3]*math.sin(x[0])) / math.cos(x[0])]
        tire_axes = d.geom_xmat[self.tires].reshape(-1, 3, 3)[:, 2, 2]
        low = np.min(d.geom_xpos[self.tires, 2] - np.abs(tire_axes)*self.m.geom_size[self.tires, 1]
                     - self.m.geom_size[self.tires, 0])
        d.qpos[2] -= low + self.penetration
        mujoco.mj_forward(self.m, d)
        return d

    def input(self, d, u):
        d.ctrl[:] = np.clip(u, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])






class FullController:
    """LQR on MuJoCo tangent coordinates, retaining compliant loop/contact modes."""
    def __init__(self, plant, speed, discount=.999, effort=1., yaw_weight=.1, phases=1, decoupled=False, epsilon=1e-5):
        self.plant = plant
        x = np.zeros(10)
        x[4], x[9] = speed, speed/.12
        self.trim = plant.lift(x)
        self.utrim = plant.trim_control.copy()
        self.utrim[0] = speed/.12
        self.indices = np.array([2, 3, 4, 5, 6, 7, 8, *range(10, 20)])
        n = len(self.indices)
        eps = epsilon
        def advance(column=None, sign=0., u=None):
            d = mujoco.MjData(plant.m)
            d.qpos[:] = self.trim.qpos
            d.qvel[:] = self.trim.qvel
            if column is not None:
                idx = self.indices[column]
                if idx < 10:
                    dq = np.zeros(10); dq[idx] = sign*eps
                    mujoco.mj_integratePos(plant.m, d.qpos, dq, 1.)
                else:
                    d.qvel[idx-10] += sign*eps
            plant.input(d, self.utrim if u is None else u)
            mujoco.mj_step(plant.m, d, nstep=round(DT/plant.m.opt.timestep))
            return self.error(d)
        matrices = []
        for phase in np.linspace(0., 2*math.pi/128, phases, endpoint=False):
            self.trim.qpos[plant.q["drive"]] = phase
            matrices.append((
                np.column_stack([(advance(j, 1)-advance(j, -1))/(2*eps) for j in range(n)]),
                np.column_stack([(advance(u=self.utrim+np.eye(2)[j]*eps)-
                                  advance(u=self.utrim-np.eye(2)[j]*eps))/(2*eps) for j in range(2)]),
                advance()/DT))
        self.trim.qpos[plant.q["drive"]] = 0.
        self.a, self.b, self.drift = [np.mean([item[j] for item in matrices], axis=0) for j in range(3)]
        q = np.array([100., 500., 250., yaw_weight, 1., 0., 4.,
                      20., 1., .1, 5., 4., .1, .01, .001, .1, .001])
        self.k = dare(self.a, self.b, np.diag(q), np.diag([.02, 1.])*effort, discount)
        if decoupled:
            self.k[:] = 0.
            # Independent sagittal/coronal LQR ignores weak yaw cross-coupling;
            # yaw is observed and reported but is not a controlled objective.
            for motor, dofs in ((0, [4, 10, 14, 19]), (1, [3, 6, 7, 8, 11, 13, 16, 17, 18])):
                ix = np.array([list(self.indices).index(j) for j in dofs])
                gain = dare(self.a[np.ix_(ix, ix)], self.b[ix, motor: motor+1],
                            np.diag(q[ix]), np.array([[.02 if motor == 0 else 1.]])*effort, discount)
                self.k[motor, ix] = gain[0]
        self.radius = float(max(abs(np.linalg.eigvals(self.a-self.b@self.k))))

    def error(self, d, speed=None):
        dq = np.empty(self.plant.m.nv)
        mujoco.mj_differentiatePos(self.plant.m, dq, 1., self.trim.qpos, d.qpos)
        dv = d.qvel-self.trim.qvel
        if speed is not None:
            dv[0] += self.trim.qvel[0]-speed
            dv[self.plant.v["drive"]] += self.trim.qvel[self.plant.v["drive"]]-speed/.12
        return np.r_[dq, dv][self.indices]

    def act_data(self, d, speed):
        u = self.utrim.copy()
        u[0] = speed/.12
        return u-self.k@self.error(d, speed)


def trial(plant, controller, name, speed=0., initial_speed=0., initial_roll=1., duration=15., push=0.):
    x = np.zeros(10)
    x[0] = math.radians(initial_roll)
    x[4], x[9] = initial_speed, initial_speed/.12
    d = plant.lift(x)
    rows, poses = [], []
    target = initial_speed
    failed = False
    peak_closure = peak_force = peak_limit = 0.
    saturation = 0
    for step in range(round(duration/DT)):
        t = step*DT
        x = plant.state(d)
        target += np.clip(speed-target, -.3*DT, .3*DT)
        u = controller.act_data(d, target) if hasattr(controller, "act_data") else controller.act(x, target)
        plant.input(d, u)
        d.xfrc_applied[:] = 0
        if push and 5 <= t < 5.2:
            d.xfrc_applied[plant.base, 1] = push
        if abs(x[0]) > .7 or abs(x[1]) > .7 or d.qpos[2] < .24 or not np.isfinite(x).all():
            failed = True
            break
        for _ in range(round(DT/plant.m.opt.timestep)):
            mujoco.mj_step(plant.m, d)
            peak_force = max(peak_force, float(np.abs(d.actuator_force).max()))
            peak_limit = max(peak_limit, float(abs(d.qpos[plant.q["wheel_roll"]])))
        mujoco.mj_forward(plant.m, d)
        x = plant.state(d)
        closure = np.linalg.norm(d.site_xpos[plant.sites[0]] - d.site_xpos[plant.sites[1]])
        peak_closure = max(peak_closure, float(closure))
        peak_force = max(peak_force, float(np.abs(d.actuator_force).max()))
        peak_limit = max(peak_limit, float(abs(x[7])))
        saturation += int(np.any(np.abs(u) >= plant.m.actuator_ctrlrange[:, 1]))
        r = d.xmat[plant.base].reshape(3, 3)
        yaw = math.atan2(r[1, 0], r[0, 0])
        rows.append([d.time, *x, target, yaw, *d.qpos[:3], *d.ctrl, *d.actuator_force, closure])
        end_failed = abs(x[0])>.7 or abs(x[1])>.7 or d.qpos[2]<.24 or not np.isfinite(x).all()
        if step % 4 == 0 or end_failed:
            poses.append(d.qpos.copy())
        if end_failed:
            failed = True
            break
    z = np.asarray(rows)
    mask = z[:, 0] > min(3., duration/2)
    def rms(col):
        return float(np.sqrt(np.mean(col**2))) if len(col) else None
    roll_rms = rms(np.degrees(z[mask, 1]))
    pitch_rms = rms(np.degrees(z[mask, 2]))
    vx_rms = rms(z[mask, 5]-z[mask, 11])
    result = dict(case=name, survived=not failed, duration_s=float(d.time),
                  initial_roll_deg=initial_roll, initial_speed=initial_speed, target_speed=speed,
                  lateral_push_N=push, timestep=plant.m.opt.timestep,
                  torso_roll_rms_deg=roll_rms, torso_pitch_rms_deg=pitch_rms,
                  max_torso_roll_deg=float(abs(np.degrees(z[:, 1])).max()),
                  max_torso_pitch_deg=float(abs(np.degrees(z[:, 2])).max()),
                  speed_rmse=vx_rms, heading_final_deg=float(math.degrees(z[-1, 12])),
                  lateral_displacement_m=float(z[-1, 14]),
                  max_relative_wheel_roll_deg=math.degrees(peak_limit),
                  peak_torque_Nm=peak_force, peak_closure_error_mm=peak_closure*1000,
                  command_saturation_fraction=saturation/len(rows),
                  level_pass=bool(not failed and roll_rms is not None and roll_rms<3
                                  and pitch_rms<3 and vx_rms<.15
                                  and abs(np.degrees(z[:, 1:3])).max()<5
                                  and math.degrees(peak_limit)<31 and peak_closure<.001))
    return result, z, np.asarray(poses)
