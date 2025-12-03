# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Voxel grid utilities for differentiable volumetric scene representations.

This module defines helpers for constructing *voxel-level* variables and
their associated factors on top of the DSG-JIT world model.

Key responsibilities
--------------------
- Create voxel chains or grids:
    • 1D voxel chains (for smooth curves or “lines” in space).
    • Higher-dimensional voxel layouts (as needed by experiments).
- Register and attach voxel-related factors:
    • Smoothness factors between neighboring voxels
      (using `voxel_smoothness_residual`).
    • Point-observation factors tying voxels to measurements in world
      coordinates (using `voxel_point_observation_residual`).
    • Optional voxel priors for regularization or supervision.

- Provide convenience routines for:
    • Initializing voxel positions (e.g. along an axis).
    • Accessing the optimized voxel centers from the packed state.

Role in the DSG-JIT stack
-------------------------
Voxel grids are a key piece of the *volumetric* side of the engine.
They allow us to:

    • Represent surfaces or occupancy with a differentiable structure.
    • Run Gauss–Newton over large chains / grids of voxels.
    • Jointly optimize voxels with SE3 poses and other scene graph nodes
      (hybrid SE3 + voxel experiments and benchmarks).

Integration points
------------------
- Uses `world.model.WorldModel` to create voxel variables and factors.
- Relies on residuals defined in `slam.measurements` for:
    - smoothness,
    - point observations,
    - and priors.
- Works seamlessly with `optimization.solvers.gauss_newton_manifold`
  and related JIT-compiled solvers.

Design goals
------------
- **Scalable**:
    Able to create hundreds or thousands of voxel nodes and factors that
    still admit fast, JIT-compiled optimization.
- **Composable**:
    Plays nicely with SE3 poses, places, and other world entities in a
    single factor graph.
- **Experiment-oriented**:
    Keeps the voxel construction boilerplate out of experiment scripts,
    making it easier to design new voxel-based learning tasks.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp

from dsg_jit.world.scene_graph import SceneGraphWorld

GridIndex = Tuple[int, int, int]


@dataclass
class VoxelGridSpec:
    """
    Specification for constructing a regular voxel grid.

    This lightweight container defines the spatial layout of a voxel grid,
    including its world-space origin, discrete grid dimensions, and the
    physical resolution of each voxel cell.

    :param origin: A 3-element array giving the world-space center of voxel
        coordinate (0, 0, 0). This is the reference point from which all voxel
        centers are computed.
    :param dims: A tuple ``(nx, ny, nz)`` representing the number of voxels
        along the x-, y-, and z-axes respectively.
    :param resolution: The edge length of each voxel cell in world units.
        The spacing between voxel centers is equal to this resolution.
    """
    origin: jnp.ndarray
    dims: Tuple[int, int, int]
    resolution: float


def build_voxel_grid(
    sg: SceneGraphWorld,
    spec: VoxelGridSpec,
) -> Dict[GridIndex, int]:
    """
    Construct a regular voxel grid inside the SceneGraphWorld.

    This allocates one `voxel_cell` variable per grid coordinate `(ix, iy, iz)`
    using the voxel resolution and origin defined in `spec`. Each voxel is
    positioned at:

        center = origin + [ix * res, iy * res, iz * res]

    The resulting mapping enables downstream creation of voxel smoothness
    constraints and scene-graph integration.

    :param sg: The active `SceneGraphWorld` instance where voxel nodes will be
        created. Must expose `add_voxel_cell(center)` which returns a node ID.
    :param spec: Voxel grid specification containing:
        - `spec.origin`: 3D world origin of the grid.
        - `spec.dims`: Tuple `(nx, ny, nz)` specifying grid dimensions.
        - `spec.resolution`: Edge length of each voxel cell.
    :return: A dictionary mapping each grid index `(ix, iy, iz)` to the
        corresponding voxel node ID allocated within the scene graph.
    """
    origin = jnp.array(spec.origin, dtype=jnp.float32).reshape(3,)
    nx, ny, nz = spec.dims
    res = float(spec.resolution)

    index_to_id: Dict[GridIndex, int] = {}

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = jnp.array([ix * res, iy * res, iz * res], dtype=jnp.float32)
                center = origin + offset
                nid = sg.add_voxel_cell(center)
                index_to_id[(ix, iy, iz)] = nid

    return index_to_id


def connect_grid_neighbors_1d_x(
    sg: SceneGraphWorld,
    index_to_id: Dict[GridIndex, int],
    spec: VoxelGridSpec,
    sigma: float | None = None,
) -> None:
    """
    Connect 3D voxel grid nodes along the +x direction using smoothness factors.

    This function iterates over all voxel indices `(ix, iy, iz)` such that
    `ix + 1 < nx`, and adds a voxel smoothness constraint between each voxel
    and its +x neighbor. The enforced residual encourages:

        voxel(ix+1, iy, iz) - voxel(ix, iy, iz) ≈ [resolution, 0, 0]

    This is sufficient to enforce a 1D chain structure along the x-axis
    and is used when constructing structured voxel grids for optimization.

    :param sg: The active `SceneGraphWorld` instance to which smoothness
        factors will be added. Must expose `add_voxel_smoothness(i, j, offset, sigma)`.
    :param index_to_id: Mapping from grid index `(ix, iy, iz)` to the corresponding
        node ID in the scene graph or factor graph.
    :param spec: Voxel grid specification containing dimensions and voxel resolution.
        Expected to provide:
            - `spec.dims`: Tuple `(nx, ny, nz)` with number of voxels.
            - `spec.resolution`: Voxel edge length in world units.
    :param sigma: Optional noise standard deviation for the smoothness factor.
        If `None`, the default sigma inside `sg.add_voxel_smoothness` is used.
    :return: None. This function mutates the scene graph world in-place by
        adding smoothness edges between neighboring x-axis voxels.
    """
    nx, ny, nz = spec.dims
    res = float(spec.resolution)

    offset = jnp.array([res, 0.0, 0.0], dtype=jnp.float32)

    for ix in range(nx - 1):
        for iy in range(ny):
            for iz in range(nz):
                vid_i = index_to_id[(ix, iy, iz)]
                vid_j = index_to_id[(ix + 1, iy, iz)]
                sg.add_voxel_smoothness(vid_i, vid_j, offset, sigma=sigma)