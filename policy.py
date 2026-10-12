"""CPU inference for RSL-RL's two-layer MLP checkpoints, with fixed observation order."""

import numpy as np


class Policy:
    def __init__(self, path):
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mean = state["obs_normalizer._mean"].numpy().reshape(-1)
        self.std = state["obs_normalizer._std"].numpy().reshape(-1)
        self.layers = [(state[f"mlp.{i}.weight"].numpy(),
                        state[f"mlp.{i}.bias"].numpy()) for i in (0, 2, 4)]

    def act(self, observation):
        x = (np.asarray(observation, dtype=np.float32) - self.mean) / (self.std + 1e-8)
        for weight, bias in self.layers[:-1]:
            x = weight @ x + bias
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        weight, bias = self.layers[-1]
        return np.clip(weight @ x + bias, -1., 1.)
