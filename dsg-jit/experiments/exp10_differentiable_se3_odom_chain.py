# experiments/exp10_differentiable_se3_odom_chain.py

import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,  # additive SE3 residual (no geodesic manifold here)
)
from dsg_jit.optimization.solvers import GDConfig, gradient_descent  # first-order solver


def _to_slice(idx):
    """
    Normalize an index entry (slice or (start, length)) into a slice.
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
    wm = WorldModel()

    # Initial pose guesses, slightly perturbed from GT.
    p0_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.0], dtype=jnp.float32),
    )
    p1_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([1.2, 0.2, 0.0, 0.02, 0.01, 0.0], dtype=jnp.float32),
    )
    p2_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([1.8, -0.2, 0.0, -0.01, 0.02, 0.0], dtype=jnp.float32),
    )

    # Strong prior on pose0 at identity.
    wm.add_factor(
        f_type="prior",
        var_ids=(p0_id,),
        params={"target": jnp.zeros(6, dtype=jnp.float32)},
    )

    # Base additive odometry measurements (intentionally a bit off).
    base_meas01 = jnp.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    base_meas12 = jnp.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    # Odom factors using the additive SE3 residual. The 'measurement'
    # fields will be overridden by theta in the parametric residual.
    f_odom01_id = wm.add_factor(
        f_type="odom_se3",
        var_ids=(p0_id, p1_id),
        params={"measurement": base_meas01},
    )
    f_odom12_id = wm.add_factor(
        f_type="odom_se3",
        var_ids=(p1_id, p2_id),
        params={"measurement": base_meas12},
    )

    # Register residuals at the WorldModel level.
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)

    # Pack state and get index map.
    x_init, index = wm.pack_state()

    # Pose slices in the flat state vector.
    p0_slice = _to_slice(index[p0_id])
    p1_slice = _to_slice(index[p1_id])
    p2_slice = _to_slice(index[p2_id])

    # Build a parametric residual function:
    #   residual_param_fn(x, theta) -> stacked residuals,
    # where theta has shape (K, 6) with K equal to the number of odom_se3 factors.
    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry

    def residual_param_fn(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Parametric residual r(x, theta), where theta[k] is the SE(3) odometry
        measurement for the k-th odom_se3 factor in factor iteration order.
        """
        var_values = wm.unpack_state(x, index)
        res_list = []
        odom_idx = 0

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(
                    f"No residual fn registered for factor type '{f.type}'"
                )

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

            if f.type == "odom_se3":
                # Override the measurement parameter from theta[odom_idx]
                meas = theta[odom_idx]
                odom_idx += 1
                base_params = dict(f.params)
                base_params["measurement"] = meas
                params = base_params
            else:
                params = f.params

            r = res_fn(stacked, params)
            w = params.get("weight", 1.0)
            res_list.append(jnp.sqrt(w) * r)

        return jnp.concatenate(res_list)

    # Ground truth poses (stacked).
    gt0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt1 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt2 = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    gt_stack = jnp.stack([gt0, gt1, gt2], axis=0)

    # Initial theta: stack of the two odom measurements (2, 6).
    theta0 = jnp.stack([base_meas01, base_meas12], axis=0)  # shape (2, 6)

    return (
        wm,
        x_init,
        residual_param_fn,
        (p0_slice, p1_slice, p2_slice),
        gt_stack,
        theta0,
    )


def main():
    (
        wm,
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