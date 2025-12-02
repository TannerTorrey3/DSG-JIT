# experiments/exp07_differentiable_voxel_obs.py

import jax
import jax.numpy as jnp

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import (
    prior_residual,
    voxel_point_observation_residual,
    sigma_to_weight,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import GNConfig, gauss_newton_manifold


def build_single_voxel_graph():
    """
    Build a tiny factor graph:

      - one voxel_cell variable v in R^3
      - prior: v ~ [0, 0, 0] (weak)
      - voxel_point_obs: v ~ [1, 0, 0] (stronger, via weight)

    Returns:
      fg, x_init, block_slices, manifold_types, target
    """
    fg = FactorGraph()

    # Single voxel variable, intentionally off from the target
    voxel0 = Variable(
        id=NodeId(0),
        type="voxel_cell",
        value=jnp.array([-0.5, 0.2, 0.0], dtype=jnp.float32),
    )
    fg.add_variable(voxel0)

    # Prior on voxel at [0,0,0] (weak)
    f_prior = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)},
    )

    # Observation: voxel should align with world point [1, 0, 0]
    target = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    weight_obs = sigma_to_weight(0.05)  # strong obs
    f_obs = Factor(
        id=FactorId(1),
        type="voxel_point_obs",
        var_ids=(NodeId(0),),
        params={
            "point_world": target,
            "weight": weight_obs,
        },
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_obs)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # Pack state and build manifold metadata
    x_init, index = fg.pack_state()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # For this tiny graph, voxel is the only variable.
    # It should occupy the first 3 entries in x.
    voxel_slice = block_slices[NodeId(0)]

    # Sanity: voxel_slice should be something like slice(0, 3, None)
    if (voxel_slice.start, voxel_slice.stop - voxel_slice.start) != (0, 3):
        raise RuntimeError(f"Unexpected voxel slice: {voxel_slice}")

    # Residual function
    residual_fn = fg.build_residual_function()

    return fg, x_init, residual_fn, block_slices, manifold_types, voxel_slice, target


def main():
    fg, x_init, residual_fn, block_slices, manifold_types, voxel_slice, target = (
        build_single_voxel_graph()
    )

    # Gauss-Newton config
    cfg = GNConfig(
        max_iters=20,
        damping=1e-2,
        max_step_norm=0.1,
    )

    # Define a pure JAX function: initial state -> loss
    def solve_and_loss(x0: jnp.ndarray) -> jnp.ndarray:
        """
        Run manifold Gauss-Newton starting from x0 and compute:

            loss = || v_opt - target ||^2

        where v_opt is the optimized voxel position.
        """
        x_opt = gauss_newton_manifold(
            residual_fn,
            x0,
            block_slices,
            manifold_types,
            cfg,
        )
        v_opt = x_opt[voxel_slice]  # shape (3,)
        return jnp.sum((v_opt - target) ** 2)

    # JIT + grad
    loss_jit = jax.jit(solve_and_loss)
    grad_fn = jax.jit(jax.grad(solve_and_loss))

    # Evaluate initial loss and gradient
    loss0 = loss_jit(x_init)
    g0 = grad_fn(x_init)

    print("=== Differentiable Voxel Inference Experiment ===")
    print(f"Initial state x0: {x_init}")
    print(f"Initial loss: {float(loss0):.6f}")
    print(f"Grad wrt x0: {g0}")

    # Take one gradient step on the initial state
    alpha = 0.5
    x_init_updated = x_init - alpha * g0

    print("\n=== After one grad step on x0 ===")
    print(f"x0_updated: {x_init_updated}")

    # Solve again from updated init
    x_opt_orig = gauss_newton_manifold(
        residual_fn, x_init, block_slices, manifold_types, cfg
    )
    x_opt_updated = gauss_newton_manifold(
        residual_fn, x_init_updated, block_slices, manifold_types, cfg
    )

    v_opt_orig = x_opt_orig[voxel_slice]
    v_opt_updated = x_opt_updated[voxel_slice]

    loss_orig = jnp.sum((v_opt_orig - target) ** 2)
    loss_updated = jnp.sum((v_opt_updated - target) ** 2)

    print("\n=== Optimized voxel comparison ===")
    print(f"v_opt (from original x0): {v_opt_orig}")
    print(f"v_opt (from updated x0):  {v_opt_updated}")
    print(f"loss_orig:   {float(loss_orig):.6f}")
    print(f"loss_updated:{float(loss_updated):.6f}")


if __name__ == "__main__":
    main()