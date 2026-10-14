"""Gaussian exploration around a bounded mean, for RSL-RL 5.0.1.

Only the mean is passed through tanh. Samples are still Gaussian; the environment
clips them to [-1, 1]. Log probabilities, entropy and KL therefore remain those of
the ordinary Gaussian over the pre-clip samples stored by PPO.
"""

import math

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import Distribution, GaussianDistribution


class BoundedMeanGaussianDistribution(GaussianDistribution):
    """Per-action learned std in [min_std, max_std], with identical train/play means."""

    def __init__(self, output_dim: int, init_std: float = 0.1,
                 min_std: float = 0.02, max_std: float = 0.2):
        if not all(math.isfinite(v) for v in (init_std, min_std, max_std)):
            raise ValueError("Standard deviations must be finite")
        if not 0 < min_std < init_std < max_std:
            raise ValueError("Require 0 < min_std < init_std < max_std")
        # Replace GaussianDistribution's unconstrained std with a sigmoid
        # parameterization; retain its sampling/log-prob/entropy/KL interface.
        Distribution.__init__(self, output_dim)
        self.register_buffer("min_std", torch.tensor(min_std))
        self.register_buffer("max_std", torch.tensor(max_std))
        fraction = (init_std - min_std) / (max_std - min_std)
        self.std_logit = nn.Parameter(torch.full((output_dim,), math.log(fraction / (1 - fraction))))
        self._distribution = None

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = self.deterministic_output(mlp_output)
        std = self.min_std + (self.max_std - self.min_std) * self.std_logit.sigmoid()
        self._distribution = Normal(mean, std.expand_as(mean), validate_args=False)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output.tanh()

    def as_deterministic_output_module(self) -> nn.Module:
        # RSL-RL's JIT and ONNX export paths call this separately from inference.
        return nn.Tanh()
