"""Explicit curriculum stages; every stage trains the same 15-input actor."""

from dataclasses import dataclass

TASK_VERSION = "recovery_slopes_v1"


@dataclass(frozen=True)
class Lesson:
    fallen_fraction: float
    roll_degrees: tuple[float, float]
    pitch_fraction: float = 0.0
    slope_degrees: float = 0.0
    push: float = 0.0  # angular velocity impulse, rad/s
    recovery_seconds: float = 5.0
    stable_seconds: float = 1.0


# Advance explicitly after evaluation, not after an arbitrary iteration count.
LESSONS = {
    "rise_tilt": Lesson(0.5, (20, 60)),
    "rise": Lesson(0.6, (60, 100)),
    "recover": Lesson(0.4, (45, 100), pitch_fraction=0.25, push=1.5),
    "slope3": Lesson(0.25, (45, 100), slope_degrees=3, push=0.5),
    "slope6": Lesson(0.3, (45, 100), pitch_fraction=0.15, slope_degrees=6, push=1.0),
    "slope10": Lesson(0.3, (45, 100), pitch_fraction=0.25, slope_degrees=10, push=1.5),
    "mixed": Lesson(0.4, (20, 100), pitch_fraction=0.25, slope_degrees=10, push=1.5),
}


def slope_angles(lesson):
    """Columns have fixed angles; random spawn yaw supplies up/down/cross slope."""
    if not lesson.slope_degrees:
        return (0.0,)
    limit = lesson.slope_degrees
    return (0.0, limit / 2, -limit / 2, limit, -limit)
