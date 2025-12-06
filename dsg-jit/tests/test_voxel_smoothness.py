import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import voxel_smoothness_residual
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def test_voxel_smoothness_grid_spacing():
    """
    Two voxels, smoothness factor enforces grid offset [1, 0, 0].
    """
    wm = WorldModel()

    v0_id = wm.add_variable(
        var_type="voxel_cell",
        value=jnp.array([0.1, 0.0, 0.0]),
    )
    v1_id = wm.add_variable(
        var_type="voxel_cell",
        value=jnp.array([0.8, 0.2, 0.0]),
    )

    offset = jnp.array([1.0, 0.0, 0.0])
    wm.add_factor(
        f_type="voxel_smoothness",
        var_ids=(v0_id, v1_id),
        params={"offset": offset},
    )

    wm.register_residual("voxel_smoothness", voxel_smoothness_residual)

    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = wm.unpack_state(x_opt, index)
    v0_opt = values[v0_id]
    v1_opt = values[v1_id]

    # We expect v1 - v0 ≈ [1, 0, 0]. The absolute position is underdetermined,
    # but we can at least enforce this relative structure.
    diff = v1_opt - v0_opt
    assert float(diff[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(diff[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(diff[2]) == pytest.approx(0.0, abs=1e-2)