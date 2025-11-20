from __future__ import annotations

import jax.numpy as jnp
import pytest

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import prior_residual, odom_se3_geodesic_residual
from slam.manifold import build_manifold_metadata
from optimization.solvers import gauss_newton_manifold, GNConfig


def test_se3_manifold_geodesic_three_pose_chain():
    """
    Manifold GN + true SE(3) geodesic residual on a 3-pose chain.

    Ground truth (in se(3)-param form):
      pose0: [0, 0, 0, 0, 0, 0]
      pose1: [1, 0, 0, 0, 0, theta]
      pose2: [2, 0, 0, 0, 0, 2*theta]

    Factors:
      - prior on pose0: identity
      - odom_geodesic between pose0 and pose1: [1, 0, 0, 0, 0, theta]
      - odom_geodesic between pose1 and pose2: [1, 0, 0, 0, 0, theta]

    We check that manifold GN recovers this configuration from a perturbed
    initial guess.
    """

    fg = FactorGraph()

    theta = 0.1  # small yaw

    # Initial guesses (intentionally a bit off)
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.10, -0.05, 0.02, 0.02, -0.01, -0.01]),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([0.9, 0.05, -0.02, 0.01, 0.01, theta + 0.03]),
    )
    pose2 = Variable(
        id=NodeId(2),
        type="pose_se3",
        value=jnp.array([2.1, -0.02, 0.01, -0.02, 0.02, 2 * theta - 0.04]),
    )

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(pose2)

    # Prior on pose0: identity
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6)},
    )

    # Geodesic SE(3) odom: 1m along x, yaw = theta
    meas01 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    meas12 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])

    f_odom01 = Factor(
        id=FactorId(1),
        type="odom_se3_geodesic",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": meas01},
    )
    f_odom12 = Factor(
        id=FactorId(2),
        type="odom_se3_geodesic",
        var_ids=(NodeId(1), NodeId(2)),
        params={"measurement": meas12},
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_odom01)
    fg.add_factor(f_odom12)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    # Build residual fn and manifold metadata
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # Manifold-aware Gauss-Newton (slightly conservative damping/step)
    cfg = GNConfig(max_iters=20, damping=5e-3, max_step_norm=0.5)
    x_opt = gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    values = fg.unpack_state(x_opt, index)
    p0_opt = values[NodeId(0)]
    p1_opt = values[NodeId(1)]
    p2_opt = values[NodeId(2)]

    # pose0 ~ identity
    for i in range(6):
        assert float(p0_opt[i]) == pytest.approx(0.0, abs=5e-3)

    # pose1 ~ [1, 0, 0, 0, 0, theta]
    assert float(p1_opt[0]) == pytest.approx(1.0, abs=5e-2)
    assert float(p1_opt[1]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[2]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p1_opt[5]) == pytest.approx(theta, abs=5e-2)

    # pose2 ~ [2, 0, 0, 0, 0, 2*theta]
    """
    Ground truth (world frame):
    pose0: t=[0, 0, 0], yaw=0
    pose1: t=[1, 0, 0], yaw=theta
    pose2: t=[1+cos(theta), sin(theta), 0], yaw=2*theta
    """
    expected_t2x = 1.0 + jnp.cos(theta)
    expected_t2y = jnp.sin(theta)

    assert float(p2_opt[0]) == pytest.approx(float(expected_t2x), abs=5e-2)
    assert float(p2_opt[1]) == pytest.approx(float(expected_t2y), abs=5e-2)
    assert float(p2_opt[2]) == pytest.approx(0.0, abs=5e-2)

    assert float(p2_opt[3]) == pytest.approx(0.0, abs=5e-2)
    assert float(p2_opt[4]) == pytest.approx(0.0, abs=5e-2)
    assert float(p2_opt[5]) == pytest.approx(2 * theta, abs=5e-2)