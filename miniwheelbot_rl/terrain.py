"""Large planar ramp patches with analytic surface normals and safe boundaries."""

from dataclasses import dataclass
import math
import mujoco
import numpy as np
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainOutput, TerrainGeometry, TerrainGeneratorCfg,
)
from .curriculum import slope_angles

PATCH_SIZE = 64.0
SAFE_RADIUS = 28.0


@dataclass(kw_only=True)
class RampCfg(SubTerrainCfg):
    degrees: float = 0.0

    def function(self, difficulty, spec, rng):
        angle = math.radians(self.degrees)
        normal = np.array([-math.sin(angle), 0, math.cos(angle)])
        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=origin - 0.1 * normal,
            quat=(math.cos(angle / 2), 0, -math.sin(angle / 2), 0),
            size=(self.size[0] / 2, self.size[1] / 2, 0.1),
            friction=(1.6, 0.03, 0.003),
        )
        return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=geom)])


def terrain_cfg(lesson):
    if not lesson.slope_degrees:
        return TerrainEntityCfg(terrain_type="plane")
    return TerrainEntityCfg(terrain_type="generator", terrain_generator=TerrainGeneratorCfg(
        size=(PATCH_SIZE, PATCH_SIZE), num_rows=1, curriculum=True, seed=0,
        sub_terrains={f"ramp_{i}": RampCfg(degrees=angle,
            proportion=0.4 if i == 0 else 0.15)
            for i, angle in enumerate(slope_angles(lesson))},
    ))
