"""Steering v2 sampling and reflection utilities, independent of MJLab imports."""
import torch

PROFILE = 'steer_v2'
TRAINING_REVISION = 'mixed_maneuvers_pitch_20261020'
OBS_REFLECTION = (1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1)
ACTION_REFLECTION = (1, -1)


def sample_commands(actual_forward):
    """20% stop, 20% each straight direction, 15% each turn, 10% mixed maneuvers.

    actual_forward supplies batch size/device only; commands are independent of
    actual velocity, allowing same-direction speed changes as well as reversals.
    """
    n = len(actual_forward)
    device = actual_forward.device
    category = torch.bucketize(torch.rand(n, device=device),
                              torch.tensor([.2, .4, .6, .75, .9], device=device))
    # Mixed group chooses forward/reverse/left/right equally and independently.
    maneuver = torch.where(category == 5, torch.randint(1, 5, (n,), device=device), category)
    magnitude = torch.empty(n, device=device).uniform_(.3, 1.)
    sign = torch.where(torch.rand(n, device=device) < .5, -1., 1.)
    speed = magnitude * sign
    speed = torch.where(maneuver == 1, magnitude, speed)
    speed = torch.where(maneuver == 2, -magnitude, speed)
    yaw = torch.empty(n, device=device).uniform_(.1, .3)
    yaw = torch.where(maneuver == 3, yaw, torch.where(maneuver == 4, -yaw, 0.))
    command = torch.stack((speed, torch.zeros_like(speed), yaw), dim=1)
    command[category == 0] = 0.
    return command, category


def pitch_angle_squared(gravity):
    """Squared ZYX pitch (rad^2), independent of roll in upright poses."""
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=1))
    return pitch.square()


def sample_initial_speed(n, device):
    """Half from rest, one quarter forward, one quarter reverse."""
    choice = torch.rand(n, device=device)
    magnitude = torch.empty(n, device=device).uniform_(.3, .9)
    return torch.where(choice < .5, 0., torch.where(choice < .75, magnitude, -magnitude))


def mirror(obs=None, actions=None, env=None):
    """Original + sagittal reflection; consumes raw observations before normalization."""
    if obs is not None:
        reflected = obs.clone()
        for key in ('actor', 'critic'):
            if obs[key].shape[-1] != len(OBS_REFLECTION):
                raise ValueError('Reflection requires the 16D raked-steering observation contract')
            reflected[key] = obs[key] * obs[key].new_tensor(OBS_REFLECTION)
        obs = torch.cat((obs, reflected), dim=0)
    if actions is not None:
        actions = torch.cat((actions, actions * actions.new_tensor(ACTION_REFLECTION)), dim=0)
    return obs, actions


def reset_exploration(actor, std=.08):
    distribution = actor.distribution
    fraction = (std - distribution.min_std) / (distribution.max_std - distribution.min_std)
    if not 0 < fraction.item() < 1:
        raise ValueError('Exploration reset must lie inside checkpoint std bounds')
    with torch.no_grad():
        distribution.std_logit.fill_(torch.logit(fraction).item())
