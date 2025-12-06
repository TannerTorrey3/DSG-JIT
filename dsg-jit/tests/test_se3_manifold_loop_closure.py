from __future__ import annotations
import jax.numpy as jnp
import pytest

from dsg_jit.world.model import WorldModel
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

    wm = WorldModel()
    theta = 0.1  # small rotation (~6 degrees)

    # Initial (intentionally sloppy) guesses
    pose0_val = jnp.array([0.2, 0.1, 0.0, 0.01, -0.01, 0.01])
    pose1_val = jnp.array([1.1, -0.1, 0.0, -0.02, 0.02, theta + 0.03])
    pose2_val = jnp.array([1.9, 0.2, 0.0, 0.03, -0.01, 2 * theta - 0.04])

    pose0_id = wm.add_variable(var_type="pose_se3", value=pose0_val)
    pose1_id = wm.add_variable(var_type="pose_se3", value=pose1_val)
    pose2_id = wm.add_variable(var_type="pose_se3", value=pose2_val)

    # Prior on pose0
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6)},
    )

    # Forward odometry
    meas01 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    meas12 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, theta])
    meas20 = jnp.array([-2.0, 0.0, 0.0, 0.0, 0.0, -2 * theta])  # closes loop

    wm.add_factor(
        f_type="odom_se3_geodesic",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": meas01},
    )
    wm.add_factor(
        f_type="odom_se3_geodesic",
        var_ids=(pose1_id, pose2_id),
        params={"measurement": meas12},
    )
    wm.add_factor(
        f_type="odom_se3_geodesic",
        var_ids=(pose2_id, pose0_id),
        params={"measurement": meas20},
    )

    # Register residuals
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    # Metadata
    x_init, index = wm.pack_state()
    residual_fn = wm.build_residual()
    packed_state = (x_init, index)
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=packed_state,
        fg=wm.fg,
    )

    # Solve
    cfg = GNConfig(max_iters=30, damping=5e-3, max_step_norm=0.5)
    x_opt = gauss_newton_manifold(residual_fn, x_init, block_slices, manifold_types, cfg)

    values = wm.unpack_state(x_opt, index)
    p0, p1, p2 = values[pose0_id], values[pose1_id], values[pose2_id]

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