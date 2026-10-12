"""Generate the deterministic, gentle MuJoCo height field used by hill_model.xml."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "hill_height.png"
HALF_LENGTH = 200.
HILL_START = 2.
HILL_END = 6.


def main():
    x = np.linspace(-HALF_LENGTH, HALF_LENGTH, 4096)
    phase = (x - HILL_START) / (HILL_END - HILL_START)
    height = np.where((phase >= 0.) & (phase <= 1.),
                      np.sin(np.pi * np.clip(phase, 0., 1.)) ** 2, 0.)
    pixels = np.repeat(np.rint(height[None, :] * 255).astype(np.uint8), 128, axis=0)
    Image.fromarray(pixels, mode="L").save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
