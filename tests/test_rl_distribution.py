"""Exercise the real RSL-RL distribution, PPO, inference and export contracts."""

from copy import deepcopy
from importlib.util import find_spec
import io
import unittest

import torch

from miniwheelbot_rl.policy_config import (
    LEGACY_POLICY_VERSION, POLICY_VERSION, checkpoint_policy_version, distribution_config,
)


class PolicyVersionTests(unittest.TestCase):
    def test_legacy_replay_is_preserved_and_legacy_resume_rejected(self):
        self.assertEqual(checkpoint_policy_version({}), LEGACY_POLICY_VERSION)
        self.assertEqual(distribution_config(checkpoint_policy_version({}))["class_name"],
                         "GaussianDistribution")
        with self.assertRaisesRegex(ValueError, "without --checkpoint"):
            checkpoint_policy_version({}, training=True)
        self.assertEqual(checkpoint_policy_version({"policy_version": POLICY_VERSION}, training=True),
                         POLICY_VERSION)
        with self.assertRaisesRegex(ValueError, "Unknown policy"):
            checkpoint_policy_version({"policy_version": "unknown"})


@unittest.skipIf(find_spec("rsl_rl") is None, "requires rsl-rl-lib==5.0.1")
class DistributionTests(unittest.TestCase):
    def make_distribution(self, **kwargs):
        from miniwheelbot_rl.distributions import BoundedMeanGaussianDistribution
        return BoundedMeanGaussianDistribution(2, **kwargs)

    def test_bounds_probabilities_and_gradients(self):
        dist = self.make_distribution()
        logits = torch.tensor([[0.3, -0.4], [5000., -5000.]], requires_grad=True)
        dist.update(logits)
        torch.testing.assert_close(dist.mean, logits.tanh())
        torch.testing.assert_close(dist.std, torch.full_like(logits, 0.1))
        # Gaussian log probability must use the bounded mean, without applying
        # a tanh change-of-variables correction to these unsquashed samples.
        actions = dist.mean.detach() + 0.05
        expected = torch.distributions.Normal(logits.tanh(), torch.full_like(logits, 0.1))
        torch.testing.assert_close(dist.log_prob(actions), expected.log_prob(actions).sum(-1))
        torch.testing.assert_close(dist.entropy, expected.entropy().sum(-1))
        (-dist.log_prob(actions).mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(dist.std_logit.grad).all())
        self.assertGreater(float(dist.std_logit.grad.abs().sum()), 0)
        with torch.no_grad():
            dist.std_logit.copy_(torch.tensor([-10000., 10000.]))
        dist.update(logits)
        torch.testing.assert_close(dist.std[0], torch.tensor([0.02, 0.2]))
        self.assertTrue(torch.isfinite(dist.log_prob(dist.sample())).all())
        torch.testing.assert_close(dist.kl_divergence(dist.params, dist.params), torch.zeros(2))

    def test_std_remains_bounded_under_optimizer_updates(self):
        for direction in (-1, 1):
            dist = self.make_distribution()
            optimizer = torch.optim.Adam(dist.parameters(), lr=1.0)
            before = dist.std_logit.detach().clone()
            for _ in range(30):
                dist.update(torch.zeros(16, 2))
                loss = direction * dist.entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            dist.update(torch.zeros(16, 2))
            self.assertFalse(torch.equal(before, dist.std_logit))
            self.assertTrue((dist.std >= dist.min_std).all())
            self.assertTrue((dist.std <= dist.max_std).all())

    def test_invalid_bounds(self):
        for kwargs in ({"init_std": 0.5}, {"min_std": 0}, {"max_std": float("inf")}):
            with self.assertRaises(ValueError):
                self.make_distribution(**kwargs)

    def test_legacy_model_replay_keeps_unbounded_mean(self):
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict
        obs = TensorDict({"actor": torch.zeros(1, 15)}, batch_size=[1])
        model = MLPModel(obs, {"actor": ["actor"]}, "actor", 2, hidden_dims=(8,),
                         distribution_cfg=distribution_config(checkpoint_policy_version({})))
        with torch.no_grad():
            model.mlp[-1].weight.zero_()
            model.mlp[-1].bias.copy_(torch.tensor([2., -3.]))
        torch.testing.assert_close(model(obs), torch.tensor([[2., -3.]]))
        self.assertIn("distribution.std_param", model.state_dict())

    def test_real_ppo_update_checkpoint_and_export(self):
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict

        torch.manual_seed(42)
        obs = TensorDict({"actor": torch.randn(8, 15)}, batch_size=[8])
        groups = {"actor": ["actor"], "critic": ["actor"]}

        def make_ppo():
            actor = MLPModel(obs, groups, "actor", 2, hidden_dims=(16, 16),
                             distribution_cfg=distribution_config())
            critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(16, 16))
            storage = RolloutStorage("rl", 8, 4, obs, (2,))
            return PPO(actor, critic, storage, entropy_coef=0.0, num_learning_epochs=2,
                       num_mini_batches=2, learning_rate=3e-4)

        alg = make_ppo()
        original = deepcopy(alg.actor.state_dict())
        for _ in range(2):
            with torch.no_grad():
                for _ in range(4):
                    actions = alg.act(obs)
                    rewards = -(actions.clamp(-1, 1) - obs["actor"][:, :2].tanh()).square().sum(-1)
                    alg.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.bool), {})
                alg.compute_returns(obs)
            losses = alg.update()
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
        self.assertTrue(any(not torch.equal(original[k], v) for k, v in alg.actor.state_dict().items()))
        alg.actor(obs, stochastic_output=True)
        self.assertTrue((alg.actor.output_mean.abs() <= 1).all())
        self.assertTrue((alg.actor.output_std >= 0.02 - 1e-7).all())
        self.assertTrue((alg.actor.output_std <= 0.2 + 1e-7).all())
        expected = alg.actor(obs).detach()
        torch.testing.assert_close(expected, alg.actor.output_mean)

        checkpoint = io.BytesIO()
        torch.save(alg.save(), checkpoint)
        checkpoint.seek(0)
        restored = make_ppo()
        restored.load(torch.load(checkpoint, weights_only=True), load_cfg=None, strict=True)
        torch.testing.assert_close(restored.actor(obs), expected)
        restored.actor(obs, stochastic_output=True)
        torch.testing.assert_close(restored.actor.output_std, alg.actor.output_std)
        # Use the real RSL-RL export adapters; they must retain tanh as well.
        torch.testing.assert_close(torch.jit.script(restored.actor.as_jit())(obs["actor"]), expected)
        torch.testing.assert_close(restored.actor.as_onnx(verbose=False)(obs["actor"]), expected)


if __name__ == "__main__":
    unittest.main()
