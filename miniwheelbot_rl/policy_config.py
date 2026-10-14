"""Version the policy's action distribution independently of motor geometry."""

POLICY_VERSION = "bounded_mean_gaussian_v1"
LEGACY_POLICY_VERSION = "unbounded_gaussian_v0"


def distribution_config(policy_version=POLICY_VERSION):
    if policy_version == POLICY_VERSION:
        return {"class_name": "miniwheelbot_rl.distributions:BoundedMeanGaussianDistribution",
                "init_std": 0.1, "min_std": 0.02, "max_std": 0.2}
    if policy_version == LEGACY_POLICY_VERSION:
        return {"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"}
    raise ValueError(f"Unknown policy version: {policy_version}")


def checkpoint_policy_version(manifest, *, training=False):
    version = manifest.get("policy_version", LEGACY_POLICY_VERSION)
    distribution_config(version)  # Reject unknown contracts before constructing an environment.
    if training and version != POLICY_VERSION:
        raise ValueError("This checkpoint uses the old unbounded policy. Start the bounded-policy "
                         "training without --checkpoint; old models remain usable for "
                         "evaluate/play/video/diagnose without changing their actions.")
    return version
