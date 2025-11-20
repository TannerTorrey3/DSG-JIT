from __future__ import annotations

import jax.numpy as jnp
import pytest

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import prior_residual, odom_se3_geodesic_residual
from slam.manifold import build_manifold_metadata
from optimization.solvers import gauss_newton_manifold, GNConfig


def test_se3_manifold_geodesic_small_rotation():
    """
    Manifold GN with true SE(3) geodesic residual, small rotation.

    Variables:
      pose0, pose1 in R^6: [tx, ty, tz, wx, wy, wz]

    Factors:
      - prior on pose0: identity
      - odom_geodesic between pose0 and pose1:
            translation: [1, 0, 0]
            rotation:    small yaw (e.g. 0.1 rad about z)

    Expected optimum:
      pose0 ~ [0, 0, 0, 0, 0, 0]
      pose1 ~ [1, 0, 0, 0, 0, theta]
    """

    fg = FactorGraph()

    # Small rotation about z
    theta = 0.1

    # Initial guesses close to optimum
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.05, -0.02, 0.0, 0.01, -0.01, 0.0]),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([1.05, 0.03, 0.0, 0.0, 0.0, theta + 0.02]),
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

    # Geodesic SE(3) odom: 1m along x, small yaw on z
    measurement = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    f_odom = Factor(
        id=FactorId(1),
        type="odom_se3_geodesic",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": measurement},
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_odom)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    # Build residual fn and manifold metadata
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # Manifold-aware Gauss-Newton
    cfg = GNConfig(max_iters=15, damping=5e-3, max_step_norm=0.5)
    x_opt = gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    values = fg.unpack_state(x_opt, index)
    p0_opt = values[NodeId(0)]
    p1_opt = values[NodeId(1)]

    # pose0 ~ identity
    for i in range(6):
        assert float(p0_opt[i]) == pytest.approx(0.0, abs=5e-3)

    # pose1 translation ~ [1, 0, 0]
    assert float(p1_opt[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(p1_opt[1]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[2]) == pytest.approx(0.0, abs=5e-2)

    # pose1 rotation ~ [0, 0, theta] (yaw)
    assert float(p1_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[5]) == pytest.approx(theta, abs=5e-2)