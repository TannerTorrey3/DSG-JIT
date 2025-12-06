import jax.numpy as jnp
import pytest

from dsg_jit.world.scene_graph import SceneGraphWorld
from dsg_jit.world.voxel_grid import VoxelGridSpec, build_voxel_grid, connect_grid_neighbors_1d_x
from dsg_jit.core.types import NodeId
from dsg_jit.slam.measurements import prior_residual, voxel_smoothness_residual, voxel_point_observation_residual
from dsg_jit.optimization.solvers import GNConfig


def test_voxel_point_observation_on_grid_chain():
    """
    1D voxel chain + one world point observation on the middle voxel.

    Grid:
        v0, v1, v2 along x with spacing 1.0 (nominally [0,1,2]).

    Factors:
        - priors:
            v0 ~ [0, 0, 0]
            v2 ~ [2, 0, 0]
        - voxel_smoothness between neighbors:
            v1 - v0 ~ [1, 0, 0]
            v2 - v1 ~ [1, 0, 0]
        - voxel_point_obs on v1 at [1, 0, 0]

    We then perturb initial voxel positions and ask the manifold GN to
    recover the structure.

    Expected:
        - v0 ~ [0, 0, 0]
        - v1 ~ [1, 0, 0]
        - v2 ~ [2, 0, 0]
    """
    sg = SceneGraphWorld()

    # Just in case __init__ changes in future, make sure residual registration is explicit
    sg.wm.register_residual("prior", prior_residual)
    sg.wm.register_residual("voxel_smoothness", voxel_smoothness_residual)
    sg.wm.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # Build a 1D voxel grid with 3 voxels: [0,1,2] on x-axis
    spec = VoxelGridSpec(
        origin=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        dims=(3, 1, 1),   # nx=3
        resolution=1.0,
    )
    index_to_id = build_voxel_grid(sg, spec)

    # Smoothness along x
    connect_grid_neighbors_1d_x(sg, index_to_id, spec, sigma=None)

    v0_id = index_to_id[(0, 0, 0)]
    v1_id = index_to_id[(1, 0, 0)]
    v2_id = index_to_id[(2, 0, 0)]

    # Priors on v0 and v2: fix gauge + target endpoints
    sg.wm.add_factor(
        "prior",
        (v0_id,),
        {"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)},
    )
    sg.wm.add_factor(
        "prior",
        (v2_id,),
        {"target": jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)},
    )

    # Observation on v1 at [1, 0, 0] in world coordinates
    sg.add_voxel_point_observation(v1_id, jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))

    # Perturb initial voxel positions so solver has work to do
    for idx, nid in index_to_id.items():
        var = sg.wm.fg.variables[NodeId(nid)]
        noise = jnp.array(
            [0.05 * (nid + 1), -0.02 * (nid + 1), 0.01 * (nid + 1)],
            dtype=jnp.float32,
        )
        sg.wm.fg.variables[NodeId(nid)] = type(var)(
            id=var.id,
            type=var.type,
            value=var.value + noise,
        )

    # Optimize with manifold GN
    cfg = GNConfig(max_iters=40, damping=1e-2, max_step_norm=0.1)
    sg.wm.optimize(iters=cfg.max_iters, method="gn")

    v0 = sg.wm.fg.variables[NodeId(v0_id)].value
    v1 = sg.wm.fg.variables[NodeId(v1_id)].value
    v2 = sg.wm.fg.variables[NodeId(v2_id)].value

    # Check v0 and v2 hit their priors
    assert float(v0[0]) == pytest.approx(0.0, abs=1e-2)
    assert float(v0[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(v0[2]) == pytest.approx(0.0, abs=1e-2)

    assert float(v2[0]) == pytest.approx(2.0, abs=1e-2)
    assert float(v2[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(v2[2]) == pytest.approx(0.0, abs=1e-2)

    # Middle voxel should be pulled near [1, 0, 0] due to obs + smoothness
    assert float(v1[0]) == pytest.approx(1.0, abs=2e-2)
    assert float(v1[1]) == pytest.approx(0.0, abs=2e-2)
    assert float(v1[2]) == pytest.approx(0.0, abs=2e-2)