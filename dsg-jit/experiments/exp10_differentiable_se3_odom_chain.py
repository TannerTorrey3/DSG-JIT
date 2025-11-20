# experiments/exp10_differentiable_se3_odom_chain.py

import jax
import jax.numpy as jnp

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import (
    prior_residual,
    odom_se3_residual,  # additive SE3 residual (no geodesic manifold here)
)
from optimization.solvers import GDConfig, gradient_descent  # first-order solver


def _to_slice(idx):
    """
    Normalize FactorGraph index entry (slice or (start, length)) into a slice.
    """
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def build_se3_chain_with_param_odom():
    """
    3-pose SE(3) chain with parameterized *additive* odometry measurements.

    State:
      - pose0, pose1, pose2 in R^6: [tx, ty, tz, wx, wy, wz]

    Ground truth (in se(3)-param form):
      pose0: [0, 0, 0, 0, 0, 0]
      pose1: [1, 0, 0, 0, 0, 0]
      pose2: [2, 0, 0, 0, 0, 0]

    Factors:
      - prior on pose0: identity (zero pose)
      - odom_se3 between pose0 and pose1: measurement ~ [1, 0, 0, 0, 0, 0]
      - odom_se3 between pose1 and pose2: measurement ~ [1, 0, 0, 0, 0, 0]

    We treat the two odometry measurements as a 2x6 parameter matrix theta:
      theta[0] -> measurement for factor (0 -> 1)
      theta[1] -> measurement for factor (1 -> 2)

    The experiment:
      - builds a parametric residual function r(x, theta)
      - runs gradient descent to get x*(theta)
      - defines loss(theta) = sum_i || pose_i(theta) - pose_i_gt ||^2
      - differentiates loss wrt theta using jax.grad

    Using first-order GD avoids the numerical fragility of backpropagating
    through repeated linear solves inside Gauss-Newton.
    """
    fg = FactorGraph()

    # Initial pose guesses, slightly perturbed from GT.
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

    # Strong prior on pose0 at identity.
    f_prior0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6, dtype=jnp.float32)},
    )

    # Base additive odometry measurements (intentionally a bit off).
    base_meas01 = jnp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    base_meas12 = jnp.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Odom factors using the additive SE3 residual.
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

    # Register residuals.
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)

    # Pack state and get index map.
    x_init, index = fg.pack_state()

    # Build parametric residual function:
    #   residual_param_fn(x, theta) -> stacked residuals,
    # with theta of shape (K, 6) where K is the number of odom_se3 factors.
    residual_param_fn, _ = fg.build_residual_function_se3_odom_param_multi()

    # Ground truth poses (stacked).
    gt0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt2 = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt_stack = jnp.stack([gt0, gt1, gt2], axis=0)

    # Initial theta: stack of the two odom measurements (2, 6).
    theta0 = jnp.stack([base_meas01, base_meas12], axis=0)  # shape (2, 6)

    # Pose slices in the flat state vector.
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


def main():
    (
        fg,
        x_init,
        residual_fn_param,
        (p0_slice, p1_slice, p2_slice),
        gt_stack,
        theta0,
    ) = build_se3_chain_with_param_odom()

    # Gradient-descent configuration (simple, robust).
    gd_cfg = GDConfig(learning_rate=0.1, max_iters=200)

    # Define solve-and-loss as a function of theta (2x6).
    def solve_and_loss(theta: jnp.ndarray) -> jnp.ndarray:
        """
        Run gradient descent with parametric odometry theta and compute pose loss:

            loss(theta) = sum_i || pose_i(theta) - gt_i ||^2
        """

        def objective_for_x(x: jnp.ndarray) -> jnp.ndarray:
            # Parametric residuals:
            r = residual_fn_param(x, theta)
            return jnp.sum(r * r)

        # Optimize x with GD; this is fully differentiable w.r.t. theta.
        x_opt = gradient_descent(objective_for_x, x_init, gd_cfg)

        p0_opt = x_opt[p0_slice]
        p1_opt = x_opt[p1_slice]
        p2_opt = x_opt[p2_slice]

        p_stack = jnp.stack([p0_opt, p1_opt, p2_opt], axis=0)
        return jnp.sum((p_stack - gt_stack) ** 2)

    # JIT the loss and its gradient wrt theta.
    loss_jit = jax.jit(solve_and_loss)
    grad_fn = jax.jit(jax.grad(solve_and_loss))

    # Evaluate initial loss and gradient.
    loss0 = float(loss_jit(theta0))
    g0 = grad_fn(theta0)

    print("=== Differentiable SE3 Odom Chain Experiment (Additive + GD) ===")
    print(f"Initial theta (odom meas):\n{theta0}")
    print(f"Initial loss: {loss0:.6f}")
    print(f"Grad wrt theta:\n{g0}")

    # One gradient step on theta.
    alpha = 0.3
    theta1 = theta0 - alpha * g0
    loss1 = float(loss_jit(theta1))

    print("\n=== After one grad step on theta ===")
    print(f"Updated theta:\n{theta1}")
    print(f"Loss after update: {loss1:.6f}")

    # Helper to solve for poses given a theta (using the same GD inner solver).
    def solve_for(theta: jnp.ndarray) -> jnp.ndarray:
        def objective_for_x(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_fn_param(x, theta)
            return jnp.sum(r * r)

        x_opt = gradient_descent(objective_for_x, x_init, gd_cfg)
        p0 = x_opt[p0_slice]
        p1 = x_opt[p1_slice]
        p2 = x_opt[p2_slice]
        return jnp.stack([p0, p1, p2], axis=0)

    p_opt0 = solve_for(theta0)
    p_opt1 = solve_for(theta1)

    print("\n=== Optimized poses ===")
    print(f"p_opt (theta0):\n{p_opt0}")
    print(f"p_opt (theta1):\n{p_opt1}")
    print(f"Ground truth:\n{gt_stack}")


if __name__ == "__main__":
    main()