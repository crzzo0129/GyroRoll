"""CPU side-lying recovery experiments using the current RL velocity actuators.

This is a separate deterministic control experiment, not a trained RL policy.
No forces, state resets, wheel pre-spin or altered motor limits during rollout.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from miniwheelbot_rl.model import MODEL, velocity_model_xml
from miniwheelbot_rl.settings import Settings
from miniwheelbot_rl.spawn import CollisionSupport


class Experiment:
    def __init__(self, torque_limit=2.4, flywheel_limit=250., timestep=.002, model_path=MODEL):
        self.model_path = Path(model_path)
        self.settings = Settings(torque_limit=torque_limit, flywheel_speed_limit=flywheel_limit,
                                 timestep=timestep, decimation=round(.01/timestep))
        self.model = mujoco.MjModel.from_xml_string(velocity_model_xml(self.settings, ground=True, model_path=self.model_path))
        self.model.opt.timestep = timestep
        self.model.opt.iterations = 30
        self.model.opt.ls_iterations = 10
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.support = CollisionSupport(self.settings, self.model_path, "cpu")
        self.base = self.model.body("base").id
        self.dofs = [int(self.model.joint(n).dofadr[0]) for n in ("reaction", "drive")]
        self.dt = self.settings.timestep * self.settings.decimation

    def reset(self, roll=90., pitch=0., yaw=0., settle=.6):
        d = mujoco.MjData(self.model)
        r, p, y = np.radians([roll, pitch, yaw]) / 2
        cr, cp, cy = np.cos([r, p, y])
        sr, sp, sy = np.sin([r, p, y])
        d.qpos[3:7] = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, d.qpos[3:7])
        normal = rotation.reshape(3, 3)[2]
        low = float(self.support.minimum(torch.tensor(normal[None], dtype=torch.float32))[0])
        d.qpos[:3] = (0, 0, .002 - low)
        d.qpos[7:] = 0
        d.qvel[:] = 0
        mujoco.mj_forward(self.model, d)
        # Let the robot settle on the actual ground before enabling recovery.
        for _ in range(round(settle / self.settings.timestep)):
            mujoco.mj_step(self.model, d)
        return d

    def state(self, d):
        r = d.xmat[self.base].reshape(3, 3)
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
        omega = r.T @ d.cvel[self.base, :3]
        velocity = r.T @ d.qvel[:3]
        tilt = math.acos(np.clip(r[2, 2], -1, 1))
        return roll, pitch, omega, velocity, tilt

    def run(self, params, roll=90., pitch=0., yaw=0., duration=8., record=False, precharge=0.,
            schedule=None, settle=.6, capture_angle=25., yaw_gain=0., gyro_gain=0.,
            roll_gyro_gain=0., trajectory_controller=None):
        d = self.reset(roll, pitch, yaw, settle)
        start = d.time
        initial = {"qpos": d.qpos.tolist(), "qvel": d.qvel.tolist()}
        # Torque feedback is mapped to bounded target speeds, then executed by
        # the same finite-torque velocity servos used in reinforcement learning.
        kr, dr, hr, kp, dp, vp = params
        stable_for = 0.
        recovered_at = None
        min_tilt = math.pi
        peak_speed = np.zeros(2)
        peak_torque = np.zeros(2)
        peak_target = np.zeros(2)
        minimum_contact_distance = 0.
        fell_after_recovery = False
        rows, frames = [], []
        late_cost = 0.
        total_cost = 0.
        captured = False
        best_capture_cost = 1e9
        max_stable = 0.
        for step in range(round(duration / self.dt)):
            phi, theta, omega, v, tilt = self.state(d)
            speeds = d.qvel[self.dofs].copy()
            torque = np.array([kr * phi + dr * omega[0] + hr * speeds[0],
                               kp * theta + dp * omega[1] + vp * v[0]])
            torque[0] += yaw_gain * omega[2]
            torque[0] += roll_gyro_gain * float(self.model.body('drive_wheel').inertia.max()) * speeds[1] * omega[2]
            torque[1] -= gyro_gain * float(self.model.body('reaction_wheel').inertia.max()) * speeds[0] * omega[2]
            limits = np.array([self.settings.flywheel_speed_limit, self.settings.rolling_speed_limit])
            d.ctrl[:] = np.clip(speeds + torque / [.12, .25], -limits, limits)
            if step * self.dt < precharge:
                d.ctrl[:] = [-math.copysign(limits[0], roll), 0.]
            if schedule:
                captured |= tilt < math.radians(capture_angle)
                phase_end = 0.
                for phase_duration, target_f, target_d in schedule:
                    phase_end += phase_duration
                    # Compare integer control ticks: floating sums such as
                    # 0.2+0.2+0.2 must not extend a pulse by one extra tick.
                    if step < round(phase_end/self.dt):
                        if not captured:
                            d.ctrl[:] = np.clip([math.copysign(1., roll)*target_f, target_d], -limits, limits)
                        break
            if trajectory_controller is not None:
                target = trajectory_controller.control(d, step*self.dt)
                if target is not None:
                    d.ctrl[:] = target
            stable = tilt < math.radians(20) and d.qpos[2] > .30 and np.linalg.norm(omega[:2]) < 1.5
            stable_for = stable_for + self.dt if stable else 0.
            max_stable = max(max_stable, stable_for)
            best_capture_cost = min(best_capture_cost, tilt**2 + .03*np.dot(omega[:2],omega[:2])
                                    + .002*omega[2]**2 + .000005*speeds[0]**2)
            if recovered_at is None and stable_for >= 1. - 1e-8:
                recovered_at = step * self.dt
            if recovered_at is not None and (tilt > .7 or d.qpos[2] < .185):
                fell_after_recovery = True
            min_tilt = min(min_tilt, tilt)
            total_cost += tilt**2 * self.dt
            if step * self.dt > duration - 2:
                late_cost += (tilt**2 + .02 * np.dot(omega[:2], omega[:2])) * self.dt
            if record:
                rows.append([step*self.dt, phi, theta, tilt, d.qpos[2], *omega, *speeds,
                             *d.ctrl, *d.actuator_force, float(stable)])
                if step % 3 == 0:
                    frames.append(d.qpos.copy())
            for _ in range(self.settings.decimation):
                previous_time = d.time
                mujoco.mj_step(self.model, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time <= previous_time:
                    raise RuntimeError("Invalid simulation state or automatic reset; reject this trial")
                peak_speed = np.maximum(peak_speed, np.abs(d.qvel[self.dofs]))
                peak_torque = np.maximum(peak_torque, np.abs(d.actuator_force))
                peak_target = np.maximum(peak_target, np.abs(d.ctrl))
                if d.ncon:
                    minimum_contact_distance = min(minimum_contact_distance, float(d.contact.dist.min()))
        final = self.state(d)
        report = dict(params=list(params), precharge_s=precharge,
                      physics_timestep=self.settings.timestep, control_timestep=self.dt,
                      torque_limit=self.settings.torque_limit, flywheel_target_limit=self.settings.flywheel_speed_limit,
                      schedule=schedule, capture_angle=capture_angle,
                      yaw_gain=yaw_gain, gyro_gain=gyro_gain,
                      roll_gyro_gain=roll_gyro_gain,
                      initial_roll_deg=roll, initial_pitch_deg=pitch,
                      initial_yaw_deg=yaw, duration_s=duration, settle_s=start,
                      recovered_at_s=recovered_at, within_5s=recovered_at is not None and recovered_at <= 5.,
                      stable_at_end=stable_for >= 1.-1e-8, final_stable_seconds=stable_for,
                      max_stable_seconds=max_stable, best_capture_cost=best_capture_cost,
                      minimum_tilt_deg=math.degrees(min_tilt), final_tilt_deg=math.degrees(final[4]),
                      max_wheel_speeds=peak_speed.tolist(), max_motor_torques=peak_torque.tolist(),
                      max_target_speeds=peak_target.tolist(), minimum_contact_distance_m=minimum_contact_distance,
                      fell_after_recovery=fell_after_recovery,
                      objective=late_cost + .15 * total_cost, initial=initial)
        return report, rows, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", nargs=6, type=float, default=[8., 1.2, .002, 10., 1.5, 1.5])
    parser.add_argument("--roll", type=float, default=90.)
    parser.add_argument("--pitch", type=float, default=0.)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--precharge", type=float, default=0.)
    parser.add_argument("--settle", type=float, default=None, help="Unpowered settling seconds before control")
    parser.add_argument("--torque-limit", type=float, default=2.4)
    parser.add_argument("--flywheel-limit", type=float, default=250.)
    parser.add_argument("--timestep", type=float, default=.002)
    parser.add_argument("--controller", type=Path, help="Saved experiment JSON with params/schedule")
    parser.add_argument("--reference", type=Path, help="Optional planned trajectory NPZ")
    parser.add_argument("--feedback", choices=['lqr','mpc'], default='lqr')
    parser.add_argument("--feedback-scale", type=float, default=1.)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/wheelbot_cargo/local_rise/baseline.json"))
    args = parser.parse_args()
    schedule, capture_angle, yaw_gain, gyro_gain, roll_gyro_gain = None, 25., 0., 0., 0.
    if args.controller:
        config = json.loads(args.controller.read_text())
        args.params, schedule = config['params'], config.get('schedule')
        args.precharge = config.get('precharge_s', 0.)
        capture_angle = config.get('capture_angle', 25.)
        yaw_gain, gyro_gain = config.get('yaw_gain',0.), config.get('gyro_gain',0.)
        roll_gyro_gain = config.get('roll_gyro_gain', 0.)
        if args.settle is None:
            args.settle = config.get('settle_s', .6)
    if args.settle is None:
        args.settle = .6
    experiment = Experiment(args.torque_limit, args.flywheel_limit, args.timestep, args.model)
    trajectory = None
    if args.reference:
        from wheelbot.recovery_control import RecoveryTrajectory, PredictiveRecovery
        controller_type = PredictiveRecovery if args.feedback == 'mpc' else RecoveryTrajectory
        trajectory = controller_type(args.reference, experiment.model, args.model,
                                     scale=args.feedback_scale, side=args.roll)
    try:
        report, rows, frames = experiment.run(args.params, roll=args.roll, pitch=args.pitch, yaw=args.yaw,
                                         duration=args.duration, record=True, precharge=args.precharge,
                                         schedule=schedule, capture_angle=capture_angle,
                                         settle=args.settle,
                                         yaw_gain=yaw_gain, gyro_gain=gyro_gain,
                                         roll_gyro_gain=roll_gyro_gain, trajectory_controller=trajectory)
    finally:
        if trajectory is not None and hasattr(trajectory, 'close'):
            trajectory.close()
    if trajectory is not None:
        report['trajectory_feedback'] = args.feedback
        report['feedback_scale'] = args.feedback_scale
        report['reference'] = str(args.reference)
        if hasattr(trajectory, 'records'):
            report['replanning'] = trajectory.records
    report["geometry_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), trace=rows, qpos=frames)
    if args.video:
        render(experiment, report, rows, frames, args.video)
    print(json.dumps({k:v for k,v in report.items() if k != "initial"}, indent=2))


def render(experiment, report, rows, frames, path):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    m = experiment.model
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 640
    m.vis.quality.offsamples = 0
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.35, 145, -18
    path.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    montage = Image.new('RGB', (1440, 640))
    sample_ids = np.linspace(0, len(frames)-1, 6).astype(int).tolist()
    with mujoco.Renderer(m, height=640, width=960) as renderer, imageio.get_writer(
            str(path), fps=1/(3*experiment.dt), codec='libx264', quality=8) as writer:
        for i, qpos in enumerate(frames):
            d.qpos[:] = qpos
            mujoco.mj_forward(m, d)
            camera.lookat[:] = [d.qpos[0], d.qpos[1], .24]
            renderer.update_scene(d, camera=camera)
            picture = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0,0,960,72),fill='#15252d')
            draw.text((18,10),f"LOCAL CPU / DIRECT CONTROL / {report['torque_limit']:g} Nm / {report['flywheel_target_limit']:g} rad/s target limit",font=font,fill='white')
            row = rows[min(3*i,len(rows)-1)]
            draw.text((18,40),f"t = {row[0]:.2f} s    tilt = {math.degrees(row[3]):.1f} deg    rotor = {row[8]:.1f} rad/s",font=font,fill='white')
            writer.append_data(np.asarray(picture))
            if i in sample_ids:
                j=sample_ids.index(i)
                montage.paste(picture.resize((480,320)),((j%3)*480,(j//3)*320))
    montage.save(path.with_suffix('.png'))


if __name__ == "__main__":
    main()
