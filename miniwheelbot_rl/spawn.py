"""Collision support of the actual XML, for non-penetrating tilted resets.

Meshes/boxes use their vertices; cylinders and capsules use analytic support.
The reset sets joint angles to zero, matching this cached collision geometry.
"""

from itertools import product
import mujoco
import numpy as np
import torch
from .model import make_spec


class CollisionSupport:
    def __init__(self, settings, model_path, device):
        model = make_spec(settings, model_path).compile()
        data = mujoco.MjData(model)
        data.qpos[:3] = 0
        data.qpos[3:7] = (1, 0, 0, 0)
        data.qpos[7:] = 0
        mujoco.mj_forward(model, data)
        vertices, centers, axes, radii, lengths, capsules = [], [], [], [], [], []
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            kind, size = model.geom_type[i], model.geom_size[i]
            rot, pos = data.geom_xmat[i].reshape(3, 3), data.geom_xpos[i]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[i]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                local = model.mesh_vert[start:start + count]
                vertices.extend(local @ rot.T + pos)
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                local = np.array(list(product((-1, 1), repeat=3))) * size
                vertices.extend(local @ rot.T + pos)
            elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE,
                          mujoco.mjtGeom.mjGEOM_SPHERE):
                centers.append(pos.copy())
                axes.append(rot[:, 2].copy())
                radii.append(size[0])
                lengths.append(0 if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size[1])
                capsules.append(kind != mujoco.mjtGeom.mjGEOM_CYLINDER)
            else:
                raise ValueError(f"Unsupported collision geom type: {kind}")
        def tensor(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.vertices = tensor(vertices).reshape(-1, 3)
        self.centers = tensor(centers).reshape(-1, 3)
        self.axes = tensor(axes).reshape(-1, 3)
        self.radii, self.lengths, self.capsules = map(tensor, (radii, lengths, capsules))

    def minimum(self, normal_body):
        lowest = (normal_body @ self.vertices.T).amin(dim=1)
        if len(self.centers):
            along = (normal_body @ self.axes.T).abs().clamp(max=1)
            radial = torch.where(self.capsules.bool(), torch.ones_like(along),
                                 (1 - along.square()).clamp_min(0).sqrt())
            support = normal_body @ self.centers.T - self.lengths * along - self.radii * radial
            lowest = torch.minimum(lowest, support.amin(dim=1))
        return lowest
