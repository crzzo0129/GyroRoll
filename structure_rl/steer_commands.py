"""Balanced start/stop/speed-change commands; no imposed velocity at changes."""
from dataclasses import dataclass
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from .steer_training import sample_commands


class BalancedSteeringCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        command, category = sample_commands(self.robot.data.root_link_lin_vel_b[env_ids, 0])
        self.vel_command_b[env_ids] = command
        self.vel_command_w[env_ids] = command
        self.is_standing_env[env_ids] = category == 0
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class BalancedSteeringCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return BalancedSteeringCommand(self, env)
