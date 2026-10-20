"""Result destinations for the legacy wheelbot/steerbot experiment entry points."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def legacy_result_dir(experiment):
    if experiment.startswith(("wheelbot_nested", "nested_", "chamfer_", "local_rise")):
        family = "wheelbot_cargo"
    elif experiment.startswith("wheelbot"):
        family = "wheelbot_legacy"
    elif experiment.startswith("steerbot"):
        family = "steerbot_legacy"
    else:
        raise ValueError(f"Unknown legacy result family: {experiment}")
    return ROOT / "results" / family / experiment
