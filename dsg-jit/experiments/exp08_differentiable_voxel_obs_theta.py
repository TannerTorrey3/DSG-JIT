# experiments/exp08_differentiable_voxel_obs_theta.py

import jax
import jax.numpy as jnp

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import (
    prior_residual,
    voxel_point_observation_residual,
    sigma_to_weight,
)
from slam.manifold import build_manifold_metadata
from optimization.solvers import GNConfig, gauss_newton_manifold


def build_single_voxel_graph_with_param_obs():
    """
    Build a tiny factor graph:

      - one voxel_cell variable v in R^3
      - prior: v ~ [0, 0, 0] (weak)
      - voxel_point_obs: v ~ point_world (we will treat point_world as a parameter)

    Returns:
      fg, x_init, residual_fn(x, point_world), block_slices, manifold_types, voxel_slice, gt
    """
    fg = FactorGraph()

    # Single voxel variable, intentionally off the ground truth
    voxel0 = Variable(
        id=NodeId(0),
        type="voxel_cell",
        value=jnp.array([-0.5, 0.2, 0.0], dtype=jnp.float32),
    )
    fg.add_variable(voxel0)

    # Prior on voxel at [0,0,0] (weak-ish)
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)},
    )

    # Observation factor: placeholder point_world (will be overridden by param)
    # Use a dummy base, e.g. [0.5, 0, 0]
    base_point = jnp.array([0.5, 0.0, 0.0], dtype=jnp.float32)
    weight_obs = sigma_to_weight(0.05)  # strong weight

    f_obs = Factor(
        id=FactorId(1),
        type="voxel_point_obs",
        var_ids=(NodeId(0),),
        params={
            "point_world": base_point,  # will be overridden in residual_fn
            "weight": weight_obs,
        },
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_obs)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # Pack state, manifold metadata
    x_init, index = fg.pack_state()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # For this tiny graph, the voxel is the only variable; should occupy first 3 entries.
    voxel_slice = index[NodeId(0)]
    # Normalize to slice if index is a tuple (start, length)
    if not isinstance(voxel_slice, slice):
        start, length = voxel_slice
        voxel_slice = slice(start, start + length)

    # Build parametric residual function: residual(x, point_world)
    residual_param_fn, _ = fg.build_residual_function_voxel_point_param()

    # Ground-truth voxel position we want the solver to reach
    gt = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)

    return (
        fg,
        x_init,
        residual_param_fn,
        block_slices,
        manifold_types,
        voxel_slice,
        gt,
        base_point,
    )


def main():
    (
        fg,
        x_init,
        residual_fn_param,
        block_slices,
        manifold_types,
        voxel_slice,
        gt,
        base_point,
    ) = build_single_voxel_graph_with_param_obs()

    cfg = GNConfig(
        max_iters=20,
        damping=1e-2,
        max_step_norm=0.1,
    )

    # Define solve-and-loss as a function of the observation point
    def solve_and_loss(point_world: jnp.ndarray) -> jnp.ndarray:
        """
        Run manifold Gauss-Newton with observation point_world and compute:

            loss = || v_opt - gt ||^2
        """
        def residual_x(x):
            return residual_fn_param(x, point_world)

        x_opt = gauss_newton_manifold(
            residual_x,
            x_init,
            block_slices,
            manifold_types,
            cfg,
        )
        v_opt = x_opt[voxel_slice]  # shape (3,)
        return jnp.sum((v_opt - gt) ** 2)

    loss_jit = jax.jit(solve_and_loss)
    grad_fn = jax.jit(jax.grad(solve_and_loss))

    # Start from an observation slightly wrong: base_point = [0.5, 0, 0]
    theta0 = base_point

    loss0 = loss_jit(theta0)
    g0 = grad_fn(theta0)

    print("=== Differentiable Voxel Obs wrt point_world ===")
    print(f"Initial point_world (theta0): {theta0}")
    print(f"Initial loss: {float(loss0):.6f}")
    print(f"Grad wrt point_world: {g0}")

    # Take one gradient step on the observation point
    alpha = 0.5
    theta1 = theta0 - alpha * g0

    print("\n=== After one grad step on point_world ===")
    print(f"Updated point_world (theta1): {theta1}")
    loss1 = float(loss_jit(theta1))
    print(f"Loss after update: {loss1:.6f}")

    # Optionally, inspect optimized voxel positions for both theta0 and theta1
    def solve_for(theta):
        def residual_x(x):
            return residual_fn_param(x, theta)
        x_opt = gauss_newton_manifold(
            residual_x,
            x_init,
            block_slices,
            manifold_types,
            cfg,
        )
        return x_opt[voxel_slice]

    v_opt0 = solve_for(theta0)
    v_opt1 = solve_for(theta1)

    print("\n=== Optimized voxel comparison ===")
    print(f"v_opt (from theta0): {v_opt0}")
    print(f"v_opt (from theta1): {v_opt1}")


if __name__ == "__main__":
    main()