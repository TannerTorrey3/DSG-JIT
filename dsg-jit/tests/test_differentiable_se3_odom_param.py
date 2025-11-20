# tests/test_differentiable_se3_odom_param.py

import jax
import jax.numpy as jnp
import pytest

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import prior_residual, odom_se3_residual
from optimization.solvers import GDConfig, gradient_descent


def _to_slice(idx):
    """Normalize FactorGraph index entry (slice or (start, length)) into a slice."""
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def _build_chain():
    """
    Build a small 3-pose SE(3) chain with additive odom, suitable
    for differentiable testing w.r.t. odometry measurements.

    State:
        pose0, pose1, pose2 in R^6: [tx, ty, tz, wx, wy, wz]

    Ground truth:
        pose0: [0, 0, 0, 0, 0, 0]
        pose1: [1, 0, 0, 0, 0, 0]
        pose2: [2, 0, 0, 0, 0, 0]
    """
    fg = FactorGraph()

    # Slightly perturbed initial guesses
    p0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.0], dtype=jnp.float32),
    )
    p1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([1.2, 0.2, 0.0, 0.02, 0.01, 0.0], dtype=jnp.float32),
    )
    p2 = Variable(
        id=NodeId(2),
        type="pose_se3",
        value=jnp.array([1.8, -0.2, 0.0, -0.01, 0.02, 0.0], dtype=jnp.float32),
    )

    fg.add_variable(p0)
    fg.add_variable(p1)
    fg.add_variable(p2)

    # Prior on pose0 at identity
    f_prior0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6, dtype=jnp.float32)},
    )

    # Base additive odometry measurements (a bit off from [1, 0, 0, 0, 0, 0])
    base_meas01 = jnp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    base_meas12 = jnp.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    f_odom01 = Factor(
        id=FactorId(1),
        type="odom_se3",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": base_meas01},
    )
    f_odom12 = Factor(
        id=FactorId(2),
        type="odom_se3",
        var_ids=(NodeId(1), NodeId(2)),
        params={"measurement": base_meas12},
    )

    for f in (f_prior0, f_odom01, f_odom12):
        fg.add_factor(f)

    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)

    x_init, index = fg.pack_state()

    # Parametric residual: r(x, theta) where theta has one row per odom factor
    residual_param_fn, _ = fg.build_residual_function_se3_odom_param_multi()

    # Ground-truth poses
    gt0 = jnp.zeros(6, dtype=jnp.float32)
    gt1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt2 = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt_stack = jnp.stack([gt0, gt1, gt2], axis=0)

    # Initial theta (stack the two measurements)
    theta0 = jnp.stack([base_meas01, base_meas12], axis=0)  # (2, 6)

    p0_slice = _to_slice(index[NodeId(0)])
    p1_slice = _to_slice(index[NodeId(1)])
    p2_slice = _to_slice(index[NodeId(2)])

    return (
        fg,
        x_init,
        residual_param_fn,
        (p0_slice, p1_slice, p2_slice),
        gt_stack,
        theta0,
    )


def test_differentiable_se3_odom_param_reduces_loss():
    """
    Check that:
      - loss(theta0) is finite
      - grad wrt theta contains no NaNs
      - one gradient step on theta reduces the loss

    This is a regression guard for the differentiable SE3 odom chain
    using additive odometry and gradient-descent inner optimization.
    """
    (
        fg,
        x_init,
        residual_param_fn,
        (p0_slice, p1_slice, p2_slice),
        gt_stack,
        theta0,
    ) = _build_chain()

    gd_cfg = GDConfig(learning_rate=0.1, max_iters=200)

    def solve_and_loss(theta: jnp.ndarray) -> jnp.ndarray:
        """Optimize x via GD for given theta, then compute pose loss."""
        def objective_for_x(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_param_fn(x, theta)
            return jnp.sum(r * r)

        x_opt = gradient_descent(objective_for_x, x_init, gd_cfg)

        p0_opt = x_opt[p0_slice]
        p1_opt = x_opt[p1_slice]
        p2_opt = x_opt[p2_slice]
        p_stack = jnp.stack([p0_opt, p1_opt, p2_opt], axis=0)

        return jnp.sum((p_stack - gt_stack) ** 2)

    # Compute initial loss and gradient
    loss0 = solve_and_loss(theta0)
    assert jnp.isfinite(loss0), "Initial loss should be finite"

    grad_theta0 = jax.grad(solve_and_loss)(theta0)
    assert not bool(jnp.any(jnp.isnan(grad_theta0))), "Gradient has NaNs"

    # One gradient step on theta
    alpha = 0.3
    theta1 = theta0 - alpha * grad_theta0
    loss1 = solve_and_loss(theta1)

    # Loss should decrease after the update
    assert float(loss1) < float(loss0), "Loss did not decrease after grad step"