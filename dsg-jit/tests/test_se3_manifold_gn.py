
from __future__ import annotations

import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import prior_residual, odom_se3_geodesic_residual, odom_se3_residual
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig


def test_se3_manifold_gn_two_poses_translation():
    """
    Manifold-aware GN on a simple SE(3) problem:

      - pose0, pose1 in R^6: [tx, ty, tz, wx, wy, wz]

    Factors:
      - prior on pose0: identity
      - odom_geodesic between pose0 and pose1: +1m in x, no rotation

    Expected optimum:
      - pose0 ~ [0, 0, 0, 0, 0, 0]
      - pose1 ~ [1, 0, 0, 0, 0, 0]
    """

    fg = FactorGraph()

    # Initial guesses (intentionally off)
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.005]),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([0.4, 0.2, -0.05, 0.05, 0.03, -0.01]),
    )

    fg.add_variable(pose0)
    fg.add_variable(pose1)

    # Prior on pose0: identity
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6)},
    )

    # Geodesic SE(3) odom: 1m along x, zero rotation
    meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    f_odom = Factor(
        id=FactorId(1),
        type="odom_se3",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": meas},
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_odom)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)

    # Build residual fn and manifold metadata
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # Manifold-aware Gauss-Newton
    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    values = fg.unpack_state(x_opt, index)
    p0_opt = values[NodeId(0)]
    p1_opt = values[NodeId(1)]

    # pose0 ~ identity
    for i in range(6):
        assert float(p0_opt[i]) == pytest.approx(0.0, abs=1e-2)

    # pose1 ~ [1, 0, 0, 0, 0, 0]
    assert float(p1_opt[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(p1_opt[1]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[2]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[5]) == pytest.approx(0.0, abs=5e-2)