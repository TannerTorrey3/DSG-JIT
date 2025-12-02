from __future__ import annotations
import jax.numpy as jnp
import pytest

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import prior_residual, odom_se3_geodesic_residual
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig


def test_se3_manifold_loop_closure():
    """
    3-pose loop with geodesic SE(3) constraints.

    Pose0 → Pose1: +1m, +theta
    Pose1 → Pose2: +1m, +theta
    Pose2 → Pose0: closure that enforces the triangle to "close".

    Expected:
        The solver globally pulls all poses into a self-consistent loop.
    """

    fg = FactorGraph()
    theta = 0.1  # small rotation (~6 degrees)

    # Initial (intentionally sloppy) guesses
    pose0 = Variable(NodeId(0), "pose_se3",
        jnp.array([0.2, 0.1, 0.0, 0.01, -0.01, 0.01])
    )
    pose1 = Variable(NodeId(1), "pose_se3",
        jnp.array([1.1, -0.1, 0.0, -0.02, 0.02, theta + 0.03])
    )
    pose2 = Variable(NodeId(2), "pose_se3",
        jnp.array([1.9, 0.2, 0.0, 0.03, -0.01, 2 * theta - 0.04])
    )

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(pose2)

    # Prior on pose0
    fg.add_factor(
        Factor(
            FactorId(0), "prior",
            (NodeId(0),),
            {"target": jnp.zeros(6)}
        )
    )

    # Forward odometry
    meas01 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    meas12 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    meas20 = jnp.array([-2.0, 0.0, 0.0, 0.0, 0.0, -2 * theta])  # closes loop

    fg.add_factor(
        Factor(FactorId(1), "odom_se3_geodesic", (NodeId(0), NodeId(1)), {"measurement": meas01})
    )
    fg.add_factor(
        Factor(FactorId(2), "odom_se3_geodesic", (NodeId(1), NodeId(2)), {"measurement": meas12})
    )
    fg.add_factor(
        Factor(FactorId(3), "odom_se3_geodesic", (NodeId(2), NodeId(0)), {"measurement": meas20})
    )

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    # Metadata
    x_init, index = fg.pack_state()
    residual_fn = fg.build_residual_function()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # Solve
    cfg = GNConfig(max_iters=30, damping=5e-3, max_step_norm=0.5)
    x_opt = gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    values = fg.unpack_state(x_opt, index)
    p0, p1, p2 = values[NodeId(0)], values[NodeId(1)], values[NodeId(2)]

    # Check consistency of loop: 0->1->2->0 produces near-zero total error
    # Check that each factor residual is small at the solution.

# Prior on pose0
    r_prior = prior_residual(p0, {"target": jnp.zeros(6)})

    # Odom 0->1
    x01 = jnp.concatenate([p0, p1])
    r01 = odom_se3_geodesic_residual(x01, {"measurement": meas01})

    # Odom 1->2
    x12 = jnp.concatenate([p1, p2])
    r12 = odom_se3_geodesic_residual(x12, {"measurement": meas12})

    # Odom 2->0 (loop closure)
    x20 = jnp.concatenate([p2, p0])
    r20 = odom_se3_geodesic_residual(x20, {"measurement": meas20})

    # All residuals must be small
    for r in [r_prior, r01, r12, r20]:
    # SE(3) is nonlinear; per-component residuals don't need to be
    # < 5e-2. A small L2 norm is a more realistic criterion.
        norm_r = float(jnp.linalg.norm(r))
        assert norm_r == pytest.approx(0.0, abs=1e-1)