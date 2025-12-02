import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import voxel_smoothness_residual
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def test_voxel_smoothness_grid_spacing():
    """
    Two voxels, smoothness factor enforces grid offset [1, 0, 0].
    """
    fg = FactorGraph()

    v0 = Variable(
        id=NodeId(0),
        type="voxel_cell",
        value=jnp.array([0.1, 0.0, 0.0]),
    )
    v1 = Variable(
        id=NodeId(1),
        type="voxel_cell",
        value=jnp.array([0.8, 0.2, 0.0]),
    )

    fg.add_variable(v0)
    fg.add_variable(v1)

    offset = jnp.array([1.0, 0.0, 0.0])
    f_smooth = Factor(
        id=FactorId(0),
        type="voxel_smoothness",
        var_ids=(NodeId(0), NodeId(1)),
        params={"offset": offset},
    )

    fg.add_factor(f_smooth)
    fg.register_residual("voxel_smoothness", voxel_smoothness_residual)

    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = fg.unpack_state(x_opt, index)
    v0_opt = values[NodeId(0)]
    v1_opt = values[NodeId(1)]

    # We expect v1 - v0 ≈ [1, 0, 0]. The absolute position is underdetermined,
    # but we can at least enforce this relative structure.
    diff = v1_opt - v0_opt
    assert float(diff[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(diff[1]) == pytest.approx(0.0, abs=1e-2)
    assert float(diff[2]) == pytest.approx(0.0, abs=1e-2)