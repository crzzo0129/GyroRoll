"""Small deterministic baselines and state extraction for both physical models."""

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class State:
    roll: float
    pitch: float
    omega: np.ndarray
    velocity: np.ndarray
    drive_speed: float
    second_pos: float
    second_speed: float
    height: float
    position: np.ndarray


def observe(model: mujoco.MjModel, data: mujoco.MjData, kind: str) -> State:
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    transform = data.xmat[base].reshape(3, 3)
    # The free-joint linear velocity is world-frame; cvel's angular part is
    # world-frame too. Express both in the chassis frame for the controller.
    velocity = transform.T @ data.qvel[:3]
    omega = transform.T @ data.cvel[base, :3]
    drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drive")
    second = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                "reaction" if kind == "wheelbot" else "steer")
    return State(
        roll=float(np.arctan2(transform[2, 1], transform[2, 2])),
        pitch=float(np.arctan2(-transform[2, 0], transform[2, 2])),
        omega=omega.copy(), velocity=velocity.copy(),
        drive_speed=float(data.qvel[model.jnt_dofadr[drive]]),
        second_pos=float(data.qpos[model.jnt_qposadr[second]]),
        second_speed=float(data.qvel[model.jnt_dofadr[second]]),
        height=float(data.xpos[base, 2]), position=data.xpos[base].copy(),
    )


class Baseline:
    def __init__(self, kind: str):
        if kind not in ("wheelbot", "steerbot"):
            raise ValueError(kind)
        self.kind = kind
        self.filtered_target = 0.0

    def reset(self):
        self.filtered_target = 0.0

    def act(self, state: State, speed: float, turn: float, dt: float) -> np.ndarray:
        # Do not request instantaneous 2 m/s at reset; wheel traction has a limit.
        self.filtered_target += np.clip(speed - self.filtered_target, -2.0 * dt, 2.0 * dt)
        v = float(state.velocity[0])
        pitch_ref = np.clip(0.14 * (self.filtered_target - v), -0.22, 0.22)
        drive = np.clip(5.0 * (state.pitch - pitch_ref) + 0.75 * state.omega[1], -1, 1)

        if self.kind == "wheelbot":
            # Reaction torque acts opposite to motor torque on the rotor.
            reaction = np.clip(3.7 * state.roll + 0.65 * state.omega[0]
                               + 0.0025 * state.second_speed, -1, 1)
            # This baseline only stabilizes roll/pitch. Yaw is handled by the learned policy.
            return np.array([drive, reaction])

        # Steering is a velocity-dependent roll actuator, not a second roll motor.
        # The commanded turn is subordinate to recovering the lean angle.
        desired = np.clip(turn * 0.32 - 1.5 * state.roll
                          - 0.27 * state.omega[0], -0.62, 0.62)
        steer = np.clip(5.0 * (desired - state.second_pos)
                        - 0.48 * state.second_speed, -1, 1)
        return np.array([drive, steer])


class OptimizedSteerController:
    """Nonlinear outer-loop gains optimized on the 15-second command sequence."""

    def __init__(self, path: Path):
        self.params = json.loads(path.read_text())["params"]
        self.rotor_speed = 0.
        self.reset()

    def reset(self):
        self.speed_target = 0.
        self.turn_filtered = 0.
        self.yaw_integral = 0.

    def act(self, state: State, speed: float, turn: float, dt: float) -> np.ndarray:
        (speed_gain, pitch_kp, pitch_kd, yaw_ff, yaw_kp, yaw_ki,
         roll_kp, roll_kd, steer_kp, steer_kd, command_tau) = self.params[:11]
        lean_gain, lateral_gain = self.params[11:13] if len(self.params) >= 13 else (0., 0.)
        pitch_trim = self.params[13] if len(self.params) >= 14 else 0.
        self.speed_target += np.clip(speed - self.speed_target, -2. * dt, 2. * dt)
        self.turn_filtered += (turn - self.turn_filtered) * dt / (command_tau + dt)
        self.yaw_integral = np.clip(self.yaw_integral + dt * (self.turn_filtered - state.omega[2]), -.6, .6)
        pitch_ref = np.clip(speed_gain * (self.speed_target - state.velocity[0]) + pitch_trim, -.6, .6)
        drive = np.clip(pitch_kp * (state.pitch - pitch_ref) + pitch_kd * state.omega[1], -1., 1.)
        roll_ref = lean_gain * state.velocity[0] * self.turn_filtered / 9.81
        desired = np.clip(yaw_ff * self.turn_filtered
                          + yaw_kp * (self.turn_filtered - state.omega[2])
                          + yaw_ki * self.yaw_integral
                          - roll_kp * (state.roll - roll_ref) - roll_kd * state.omega[0]
                          + lateral_gain * state.velocity[1], -.72, .72)
        steer = np.clip(steer_kp * (desired - state.second_pos)
                        - steer_kd * state.second_speed, -1., 1.)
        return np.array([drive, steer])


class TiltedWheelController:
    """Experimental tilted-flywheel feedback, not the published Wheelbot geometry."""

    def __init__(self, path: Path):
        self.configuration = json.loads(path.read_text())
        self.params = self.configuration["params"]
        self.rotor_speed = 110.
        self.reset()

    def reset(self):
        self.speed_target = 0.

    def act(self, state: State, speed: float, turn: float, dt: float) -> np.ndarray:
        roll_kp, roll_kd, yaw_ff, yaw_kp, lateral_kp, pitch_ff, speed_gain = self.params
        self.speed_target += np.clip(speed - self.speed_target, -2 * dt, 2 * dt)
        pitch_ref = np.clip(speed_gain * (self.speed_target - state.velocity[0])
                            + pitch_ff * turn, -.35, .35)
        drive = np.clip(5. * (state.pitch - pitch_ref) + .75 * state.omega[1], -1., 1.)
        reaction = np.clip(roll_kp * state.roll + roll_kd * state.omega[0]
                           + .0025 * state.second_speed + yaw_ff * turn
                           + yaw_kp * (turn - state.omega[2])
                           - lateral_kp * state.velocity[1], -1., 1.)
        return np.array([drive, reaction])


class ThinWheelController:
    """Feedback for a narrow, strong-motor flywheel-wheel candidate."""

    def __init__(self, path: Path, reaction_gear: float,
                 rotor_friction_scale: float | None = None,
                 recovery_rotor_gain: float | None = None,
                 recovery_yaw_gain: float | None = None,
                 recovery_yaw_integral_gain: float | None = None,
                 pre_turn_yaw_gain: float | None = None,
                 pre_turn_rotor_gain: float | None = None,
                 speed_slew: float | None = None,
                 roll_ff_scale: float | None = None, yaw_kp_scale: float = 1.):
        configuration = json.loads(path.read_text())
        self.params = configuration["params"]
        scale = (configuration.get("rotor_friction_scale", 1.)
                 if rotor_friction_scale is None else rotor_friction_scale)
        self.rotor_friction = .0012 / reaction_gear * scale
        self.recovery_rotor_gain = (configuration.get("recovery_rotor_gain", 0.)
                                    if recovery_rotor_gain is None else recovery_rotor_gain)
        self.recovery_yaw_gain = (configuration.get("recovery_yaw_gain", 0.)
                                  if recovery_yaw_gain is None else recovery_yaw_gain)
        self.recovery_yaw_integral_gain = (
            configuration.get("recovery_yaw_integral_gain", 0.)
            if recovery_yaw_integral_gain is None else recovery_yaw_integral_gain)
        self.pre_turn_yaw_gain = (configuration.get("pre_turn_yaw_gain", 0.)
                                  if pre_turn_yaw_gain is None else pre_turn_yaw_gain)
        self.pre_turn_rotor_gain = (configuration.get("pre_turn_rotor_gain", 0.)
                                    if pre_turn_rotor_gain is None else pre_turn_rotor_gain)
        self.speed_slew = (configuration.get("speed_slew", 2.)
                           if speed_slew is None else speed_slew)
        self.turn_min_speed = configuration.get("turn_min_speed", 0.)
        self.turn_full_speed = configuration.get("turn_full_speed", 0.)
        self.startup_seconds = configuration.get("startup_seconds", 0.)
        self.startup_speed_limit = configuration.get("startup_speed_limit", 1.)
        self.roll_ff_scale = (configuration.get("roll_ff_scale", 1.)
                              if roll_ff_scale is None else roll_ff_scale)
        self.yaw_kp_scale = yaw_kp_scale
        self.rotor_speed = configuration.get("initial_rotor_speed", 110.)
        self.reset()

    def reset(self):
        self.speed_target = 0.
        self.elapsed = 0.
        self.has_turned = False
        self.yaw_integral = 0.

    def act(self, state: State, speed: float, turn: float, dt: float) -> np.ndarray:
        speed_gain, pitch_kp, pitch_kd, roll_kp, roll_kd, roll_ff, yaw_kp = self.params
        self.elapsed += dt
        if self.elapsed < self.startup_seconds:
            speed = min(speed, self.startup_speed_limit)
        if self.turn_full_speed > self.turn_min_speed:
            turn *= np.clip((state.velocity[0] - self.turn_min_speed) /
                            (self.turn_full_speed - self.turn_min_speed), 0., 1.)
        self.speed_target += np.clip(speed - self.speed_target,
                                     -self.speed_slew * dt, self.speed_slew * dt)
        pitch_ref = np.clip(speed_gain * (self.speed_target - state.velocity[0]), -.3, .3)
        drive = np.clip(pitch_kp * (state.pitch - pitch_ref) + pitch_kd * state.omega[1], -1., 1.)
        roll_ref = np.clip(self.roll_ff_scale * roll_ff * turn
                           + self.yaw_kp_scale * yaw_kp * (turn - state.omega[2]), -.4, .4)
        reaction = np.clip(roll_kp * (state.roll - roll_ref) + roll_kd * state.omega[0]
                           + self.rotor_friction * state.second_speed, -1., 1.)
        if (not self.has_turned and abs(turn) <= .02
                and self.elapsed >= self.startup_seconds
                and state.velocity[0] > 1.7):
            reaction = np.clip(reaction - self.pre_turn_yaw_gain * state.omega[2]
                               + self.pre_turn_rotor_gain * (self.rotor_speed - state.second_speed),
                               -1., 1.)
        if abs(turn) > .02:
            self.has_turned = True
            self.yaw_integral = 0.
        elif self.has_turned:
            self.yaw_integral = np.clip(self.yaw_integral + dt * state.omega[2], -1., 1.)
            reaction = np.clip(reaction
                               + self.recovery_rotor_gain * (self.rotor_speed - state.second_speed)
                               - self.recovery_yaw_gain * state.omega[2]
                               - self.recovery_yaw_integral_gain * self.yaw_integral,
                               -1., 1.)
        return np.array([drive, reaction])


class LqrSteerController:
    """Local lateral LQR at 2 m/s plus integral yaw-rate tracking."""

    def __init__(self, path: Path, *, com_offset=None, trail=None,
                 tire_halfwidth=None, steer_gear=None, head_mass=None,
                 body_mass=None, body_offset_z=None, tuning_path=None):
        from linearize_steer import identify, lqr

        self.longitudinal = OptimizedSteerController(path)
        self.model = identify(com_offset=com_offset, trail=trail,
                              tire_halfwidth=tire_halfwidth, steer_gear=steer_gear,
                              head_mass=head_mass, body_mass=body_mass,
                              body_offset_z=body_offset_z)
        tuning_path = Path(tuning_path) if tuning_path else path.with_name("lqr_tuning.json")
        self.tuning = (json.loads(tuning_path.read_text())["params"]
                       if tuning_path.exists() else [80., 8., 80., 1., 0., 2.])
        roll_q, yaw_q, integral_q, effort_r, self.command_tau, self.integral_limit = self.tuning[:6]
        lateral_q = self.tuning[6] if len(self.tuning) >= 7 else 2.
        self.ff_scale = self.tuning[7] if len(self.tuning) >= 8 else 1.
        self.gain = lqr(self.model, q=(roll_q, 2., lateral_q, yaw_q, 2., .2, integral_q), r=effort_r)
        self.steady_response = np.linalg.solve(np.eye(6) - self.model["A"], self.model["B"]).ravel()
        self.rotor_speed = 0.
        self.reset()

    def reset(self):
        self.longitudinal.reset()
        self.yaw_integral = 0.
        self.turn_filtered = 0.

    def act(self, state: State, speed: float, turn: float, dt: float) -> np.ndarray:
        drive = self.longitudinal.act(state, speed, 0., dt)[0]
        lateral = np.array((state.roll, state.omega[0], state.velocity[1],
                            state.omega[2], state.second_pos, state.second_speed))
        self.turn_filtered += (turn - self.turn_filtered) * dt / (self.command_tau + dt)
        self.yaw_integral = np.clip(self.yaw_integral + (self.turn_filtered - state.omega[2]) * dt,
                                    -self.integral_limit, self.integral_limit)
        feedforward = self.ff_scale * self.turn_filtered / self.steady_response[3]
        reference = self.model["trim"] + self.steady_response * feedforward
        steer = np.clip(feedforward - self.gain[:6] @ (lateral - reference)
                        - self.gain[6] * self.yaw_integral, -1., 1.)
        return np.array((drive, steer))
