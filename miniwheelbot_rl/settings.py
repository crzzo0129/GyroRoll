from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Settings:
    # Action order is always [reaction/flywheel, drive/rolling wheel].
    flywheel_speed_limit: float = 250.0  # signed rad/s
    rolling_speed_limit: float = 35.0  # signed rad/s
    flywheel_kv: float = 0.12  # N m per rad/s of speed error
    rolling_kv: float = 0.25
    torque_limit: float = 2.4  # N m, both joints, gear=1
    timestep: float = 0.002
    decimation: int = 5  # 100 Hz policy; 500 Hz physics/velocity servo
    episode_seconds: float = 20.0
    initial_height: float = 0.370
    minimum_height: float = 0.185
    maximum_tilt: float = 0.70

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.decimation, int):
            raise ValueError("decimation must be an integer")


STAGES = {
    "warmup": (0.5, 0.20),
    "tracking": (1.2, 0.40),
    "full": (2.0, 0.60),
    "rise_tilt": (0.5, 0.20),
    "rise": (0.5, 0.20),
    "recover": (0.5, 0.20),
    "slope3": (0.5, 0.20),
    "slope6": (0.8, 0.30),
    "slope10": (1.2, 0.40),
    "mixed": (1.2, 0.40),
}
