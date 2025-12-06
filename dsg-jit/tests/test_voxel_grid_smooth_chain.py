import jax.numpy as jnp
import pytest

from dsg_jit.world.scene_graph import SceneGraphWorld
from dsg_jit.world.voxel_grid import VoxelGridSpec, build_voxel_grid, connect_grid_neighbors_1d_x
from dsg_jit.core.types import NodeId
from dsg_jit.slam.measurements import prior_residual, voxel_smoothness_residual
from dsg_jit.optimization.solvers import GNConfig


def test_voxel_grid_1d_chain_smoothing():
    """
    Build a 1D voxel chain along x, perturb the positions, then use
    voxel_smoothness + a couple of priors to pull it into a regular grid.

    Setup:
      - 4 voxels with nominal centers: [0, 1, 2, 3] x-axis
      - We add small perturbations to their initial positions.
      - Smoothness factors enforce:
            v_{i+1} - v_i ≈ [1, 0, 0]
      - A prior on v0: [0, 0, 0]
      - A prior on v3: [3, 0, 0]

    Expect:
      - v0 ≈ [0, 0, 0]
      - v3 ≈ [3, 0, 0]
      - v1, v2 evenly spaced in between.
    """
    sg = SceneGraphWorld()

    # Manually ensure residuals are registered (SceneGraphWorld.__init__ should
    # already do this, but this makes the test self-contained if that changes).
    sg.wm.register_residual("prior", prior_residual)
    sg.wm.register_residual("voxel_smoothness", voxel_smoothness_residual)

    # Build a 1D grid: 4 voxels along x, spacing 1.0
    spec = VoxelGridSpec(
        origin=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        dims=(4, 1, 1),   # nx=4
        resolution=1.0,
    )
    index_to_id = build_voxel_grid(sg, spec)

    # Add smoothness constraints along x
    connect_grid_neighbors_1d_x(sg, index_to_id, spec, sigma=None)

    # Add priors to fix gauge: v0 ~ [0, 0, 0], v3 ~ [3, 0, 0]
    v0_id = index_to_id[(0, 0, 0)]
    v3_id = index_to_id[(3, 0, 0)]

    sg.wm.add_factor(
        "prior",
        (v0_id,),
        {"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)},
    )
    sg.wm.add_factor(
        "prior",
        (v3_id,),
        {"target": jnp.array([3.0, 0.0, 0.0], dtype=jnp.float32)},
    )

    # Perturb initial voxel positions a bit so the optimizer has work to do
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

    # Optimize the whole world model using manifold GN
    cfg = GNConfig(max_iters=40, damping=1e-2, max_step_norm=0.1)
    sg.wm.optimize(iters=cfg.max_iters, method="gn")

    # Fetch optimized voxel positions
    v0 = sg.wm.fg.variables[NodeId(v0_id)].value
    v1 = sg.wm.fg.variables[NodeId(index_to_id[(1, 0, 0)])].value
    v2 = sg.wm.fg.variables[NodeId(index_to_id[(2, 0, 0)])].value
    v3 = sg.wm.fg.variables[NodeId(v3_id)].value

    # v0 and v3 should be close to the priors
    assert float(v0[0]) == pytest.approx(0.0, abs=1e-2)
    assert float(v0[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(v0[2]) == pytest.approx(0.0, abs=1e-2)

    assert float(v3[0]) == pytest.approx(3.0, abs=1e-2)
    assert float(v3[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(v3[2]) == pytest.approx(0.0, abs=1e-2)

    # Check spacing: v1 and v2 should be approximately [1,0,0] and [2,0,0]
    assert float(v1[0]) == pytest.approx(1.0, abs=2e-2)
    assert float(v2[0]) == pytest.approx(2.0, abs=2e-2)

    # y,z should remain near zero for all voxels
    for v in (v0, v1, v2, v3):
        assert float(v[1]) == pytest.approx(0.0, abs=2e-2)
        assert float(v[2]) == pytest.approx(0.0, abs=2e-2)