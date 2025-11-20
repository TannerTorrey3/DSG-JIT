# experiments/exp09_multi_voxel_param.py

import jax
import jax.numpy as jnp

from core.types import NodeId, FactorId, Variable, Factor
from core.factor_graph import FactorGraph
from slam.measurements import (
    prior_residual,
    voxel_point_observation_residual,
    voxel_smoothness_residual,
    sigma_to_weight,
)
from slam.manifold import build_manifold_metadata
from optimization.solvers import GNConfig, gauss_newton_manifold


def build_three_voxel_chain_with_param_obs():
    """
    Build a small 1D chain of 3 voxels along x:

        v0, v1, v2

    True (desired) positions:
        v0 = [0, 0, 0]
        v1 = [1, 0, 0]
        v2 = [2, 0, 0]

    Factors:
        - priors on v0, v2 to anchor the ends
        - voxel_smoothness between v0-v1 and v1-v2
        - voxel_point_obs for each voxel, with per-factor 'point_world' coming from theta

    We will:
      - treat the 3 observation points as a 3x3 parameter matrix theta
      - run manifold GN
      - compute loss as sum_i ||v_i - gt_i||^2
      - differentiate loss w.r.t theta
    """
    fg = FactorGraph()

    # Three voxel variables (voxel_cell), intentionally perturbed from [0,1,2]
    v0 = Variable(
        id=NodeId(0),
        type="voxel_cell",
        value=jnp.array([-0.2, 0.1, 0.0], dtype=jnp.float32),
    )
    v1 = Variable(
        id=NodeId(1),
        type="voxel_cell",
        value=jnp.array([0.9, -0.3, 0.0], dtype=jnp.float32),
    )
    v2 = Variable(
        id=NodeId(2),
        type="voxel_cell",
        value=jnp.array([2.3, 0.2, 0.0], dtype=jnp.float32),
    )

    fg.add_variable(v0)
    fg.add_variable(v1)
    fg.add_variable(v2)

    # Priors at v0 ~ [0,0,0], v2 ~ [2,0,0]
    f_prior0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)},
    )
    f_prior2 = Factor(
        id=FactorId(1),
        type="prior",
        var_ids=(NodeId(2),),
        params={"target": jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)},
    )

    # Voxel smoothness: v1-v0 ~ [1,0,0], v2-v1 ~ [1,0,0]
    # Use moderate regularization
    smooth_sigma = 0.1
    smooth_weight = sigma_to_weight(smooth_sigma)

    f_smooth01 = Factor(
    id=FactorId(2),
    type="voxel_smoothness",
    var_ids=(NodeId(0), NodeId(1)),
    params={
        # voxel_smoothness_residual expects 'offset'
        "offset": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        "weight": smooth_weight,
        },
    )
    f_smooth12 = Factor(
    id=FactorId(3),
    type="voxel_smoothness",
    var_ids=(NodeId(1), NodeId(2)),
    params={
        "offset": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
        "weight": smooth_weight,
        },
    )

    # voxel_point_obs for each voxel.
    # We'll override 'point_world' via theta in the parametric residual builder.
    obs_sigma = 0.05
    obs_weight = sigma_to_weight(obs_sigma)

    # Base (incorrect) observation points:
    base_obs0 = jnp.array([-0.5, 0.1, 0.0], dtype=jnp.float32)
    base_obs1 = jnp.array([0.7, -0.2, 0.0], dtype=jnp.float32)
    base_obs2 = jnp.array([2.4, 0.3, 0.0], dtype=jnp.float32)

    f_obs0 = Factor(
        id=FactorId(4),
        type="voxel_point_obs",
        var_ids=(NodeId(0),),
        params={
            "point_world": base_obs0,   # will be overridden by theta[0]
            "weight": obs_weight,
        },
    )
    f_obs1 = Factor(
        id=FactorId(5),
        type="voxel_point_obs",
        var_ids=(NodeId(1),),
        params={
            "point_world": base_obs1,   # will be overridden by theta[1]
            "weight": obs_weight,
        },
    )
    f_obs2 = Factor(
        id=FactorId(6),
        type="voxel_point_obs",
        var_ids=(NodeId(2),),
        params={
            "point_world": base_obs2,   # will be overridden by theta[2]
            "weight": obs_weight,
        },
    )

    for f in (f_prior0, f_prior2, f_smooth01, f_smooth12, f_obs0, f_obs1, f_obs2):
        fg.add_factor(f)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("voxel_smoothness", voxel_smoothness_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # Pack state, manifold metadata
    x_init, index = fg.pack_state()
    block_slices, manifold_types = build_manifold_metadata(fg)

    # Get slices for each voxel in the state vector
    v0_idx = index[NodeId(0)]
    v1_idx = index[NodeId(1)]
    v2_idx = index[NodeId(2)]

    # Normalize any (start, len) to slice
    def to_slice(idx):
        if isinstance(idx, slice):
            return idx
        start, length = idx
        return slice(start, start + length)

    v0_slice = to_slice(v0_idx)
    v1_slice = to_slice(v1_idx)
    v2_slice = to_slice(v2_idx)

    # Build parametric residual function: residual(x, theta) with theta.shape == (3, 3)
    residual_param_fn, _ = fg.build_residual_function_voxel_point_param_multi()

    # Ground-truth positions
    gt0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    gt1 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    gt2 = jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)

    gt_stack = jnp.stack([gt0, gt1, gt2], axis=0)

    # Initial theta from base_obs
    theta0 = jnp.stack([base_obs0, base_obs1, base_obs2], axis=0)  # shape (3, 3)

    return (
        fg,
        x_init,
        residual_param_fn,
        block_slices,
        manifold_types,
        (v0_slice, v1_slice, v2_slice),
        gt_stack,
        theta0,
    )


def main():
    (
        fg,
        x_init,
        residual_fn_param,
        block_slices,
        manifold_types,
        (v0_slice, v1_slice, v2_slice),
        gt_stack,
        theta0,
    ) = build_three_voxel_chain_with_param_obs()

    cfg = GNConfig(max_iters=30, damping=1e-2, max_step_norm=0.1)

    # Define solve-and-loss as a function of theta (3x3 observation points)
    def solve_and_loss(theta: jnp.ndarray) -> jnp.ndarray:
        """
        Run manifold GN with per-factor observation points 'theta'
        and compute loss over voxel positions:

            loss = sum_i || v_i - gt_i ||^2
        """
        def residual_x(x):
            return residual_fn_param(x, theta)

        x_opt = gauss_newton_manifold(
            residual_x,
            x_init,
            block_slices,
            manifold_types,
            cfg,
        )

        v0_opt = x_opt[v0_slice]
        v1_opt = x_opt[v1_slice]
        v2_opt = x_opt[v2_slice]

        v_stack = jnp.stack([v0_opt, v1_opt, v2_opt], axis=0)
        return jnp.sum((v_stack - gt_stack) ** 2)

    loss_jit = jax.jit(solve_and_loss)
    grad_fn = jax.jit(jax.grad(solve_and_loss))

    loss0 = float(loss_jit(theta0))
    g0 = grad_fn(theta0)

    print("=== Multi-voxel Differentiable Obs Experiment ===")
    print(f"Initial theta (obs points):\n{theta0}")
    print(f"Initial loss: {loss0:.6f}")
    print(f"Grad wrt theta:\n{g0}")

    # Take a gradient step on theta
    alpha = 0.3
    theta1 = theta0 - alpha * g0

    loss1 = float(loss_jit(theta1))

    print("\n=== After one grad step on theta ===")
    print(f"Updated theta:\n{theta1}")
    print(f"Loss after update: {loss1:.6f}")

    # Inspect optimized voxel positions for theta0 and theta1
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
        v0 = x_opt[v0_slice]
        v1 = x_opt[v1_slice]
        v2 = x_opt[v2_slice]
        return jnp.stack([v0, v1, v2], axis=0)

    v_opt0 = solve_for(theta0)
    v_opt1 = solve_for(theta1)

    print("\n=== Optimized voxel positions ===")
    print(f"v_opt (theta0):\n{v_opt0}")
    print(f"v_opt (theta1):\n{v_opt1}")
    print(f"Ground truth:\n{gt_stack}")


if __name__ == "__main__":
    main()