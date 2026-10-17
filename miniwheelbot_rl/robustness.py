"""Standing commands and finite-duration horizontal pushes for flat tracking."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class RobustTracking:
    standing_fraction: float = 0.25
    force_newtons: tuple[float, float] = (0.5, 1.5)
    duration_seconds: tuple[float, float] = (0.10, 0.25)
    cooldown_seconds: tuple[float, float] = (4.0, 8.0)


ROBUST_TRACKING = RobustTracking()


def training_manifest(stage):
    return asdict(ROBUST_TRACKING) if stage == "full_robust" else None


def standing_mask(env):
    return env.command_manager.get_command("twist").abs().amax(dim=-1) < 1e-6


def standing_fraction(env):
    return standing_mask(env).float()


def standing_reward(env):
    """Reward low body velocity, while allowing wheel motion needed for balance."""
    from .task import heading_rate
    robot = env.scene["robot"].data
    return standing_score(standing_mask(env), robot.root_link_lin_vel_b[:, :2],
                          heading_rate(env), -robot.projected_gravity_b[:, 2])


def standing_score(mask, planar_velocity, yaw_rate, up):
    speed_squared = planar_velocity.square().sum(dim=-1)
    return mask * up.clamp(0, 1).square() * torch.exp(
        -speed_squared / 0.10**2 - (yaw_rate / 0.15).square())


class HorizontalPushes:
    """MJLab step event: world-horizontal force at the selected body's CoM.

    Owns the external wrench on this body. Each environment has independent
    pulse/cooldown timers; reset clears only its force and starts a fresh delay.
    No root velocity or pose is changed. Forces persist through physics substeps.
    """

    def __init__(self, cfg, env):
        self.robot = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids = cfg.params["asset_cfg"].body_ids
        if not isinstance(self.body_ids, list) or len(self.body_ids) != 1:
            raise ValueError("HorizontalPushes requires exactly one resolved body")
        self.device, self.dt = env.device, env.step_dt
        self.cooldown_seconds = cfg.params["cooldown_seconds"]
        self.forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
        self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.wait = torch.zeros_like(self.remaining)
        self.reset()

    def _ticks(self, count, bounds):
        return (torch.empty(count, device=self.device).uniform_(*bounds)
                / self.dt).round().long().clamp_min(1)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.forces[ids] = 0
        self.remaining[ids] = 0
        self.wait[ids] = self._ticks(self.remaining[ids].numel(), self.cooldown_seconds)
        self.robot.write_external_wrench_to_sim(
            self.forces[ids], torch.zeros_like(self.forces[ids]),
            env_ids=env_ids, body_ids=self.body_ids)

    def __call__(self, env, env_ids, force_newtons, duration_seconds,
                 cooldown_seconds, asset_cfg):
        del env, env_ids, asset_cfg
        was_active = self.remaining > 0
        self.remaining = (self.remaining - 1).clamp_min(0)
        expired = was_active & (self.remaining == 0)
        self.forces[expired] = 0
        self.wait[expired] = self._ticks(int(expired.sum()), cooldown_seconds)
        waiting = (self.remaining == 0) & ~expired
        self.wait[waiting] -= 1
        ready = waiting & (self.wait <= 0)
        count = int(ready.sum())
        if count:
            angle = torch.rand(count, device=self.device) * (2 * math.pi)
            magnitude = torch.empty(count, device=self.device).uniform_(*force_newtons)
            self.forces[ready, 0, 0] = magnitude * angle.cos()
            self.forces[ready, 0, 1] = magnitude * angle.sin()
            self.remaining[ready] = self._ticks(count, duration_seconds)
        self.robot.write_external_wrench_to_sim(
            self.forces, torch.zeros_like(self.forces), body_ids=self.body_ids)
