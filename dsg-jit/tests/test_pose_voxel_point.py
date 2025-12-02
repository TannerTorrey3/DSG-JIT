import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import (
    prior_residual,
    pose_voxel_point_residual,
)
from dsg_jit.optimization.solvers import GNConfig, gauss_newton


def test_pose_voxel_point_alignment():
    """
    One pose, one voxel, one 3D point measurement.

    Ground truth:
      - pose0: identity (world = pose frame)
      - point_meas (pose frame): [1, 0.5, 0.0]
      - voxel_world should converge to [1, 0.5, 0.0]

    Setup:
      Variables:
        - pose0: pose_se3 (R^6)
        - voxel0: voxel_cell (R^3)

      Factors:
        - prior on pose0 -> identity
        - weak prior on voxel0 around [1, 0.5, 0]
        - pose_voxel_point: enforces voxel_center ≈ T(pose0) * point_meas
    """
    fg = FactorGraph()

    # Initial guesses (intentionally a bit off)
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.0]),
    )
    voxel0 = Variable(
        id=NodeId(1),
        type="voxel_cell",
        value=jnp.array([0.5, 0.0, 0.1]),
    )

    fg.add_variable(pose0)
    fg.add_variable(voxel0)

    # Prior on pose0: identity
    f_prior_pose = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6)},
    )

    # Weak prior on voxel0 near the expected solution
    f_prior_voxel = Factor(
        id=FactorId(2),
        type="prior",
        var_ids=(NodeId(1),),
        params={"target": jnp.array([1.0, 0.5, 0.0])},
    )

    # 3D point measurement in the pose frame
    point_meas = jnp.array([1.0, 0.5, 0.0])
    f_pv = Factor(
        id=FactorId(1),
        type="pose_voxel_point",
        var_ids=(NodeId(0), NodeId(1)),
        params={"point_meas": point_meas},
    )

    fg.add_factor(f_prior_pose)
    fg.add_factor(f_prior_voxel)
    fg.add_factor(f_pv)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("pose_voxel_point", pose_voxel_point_residual)

    # Optimize with conservative Gauss-Newton (to avoid SE(3) blowups)
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(
        max_iters=40,
        damping=1e-2,       # stronger damping for stability
        max_step_norm=0.05  # smaller steps in SE(3)
    )
    x_opt = gauss_newton(residual_fn, x_init, cfg)

    values = fg.unpack_state(x_opt, index)
    p_opt = values[NodeId(0)]
    v_opt = values[NodeId(1)]

    # pose0 should be near identity
    for i in range(6):
        assert float(p_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # voxel0 should be near the world point [1, 0.5, 0.0]
    assert float(v_opt[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(v_opt[1]) == pytest.approx(0.5, abs=1e-2)
    assert float(v_opt[2]) == pytest.approx(0.0, abs=1e-2)