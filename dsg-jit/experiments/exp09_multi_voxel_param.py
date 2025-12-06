# experiments/exp09_multi_voxel_param.py

import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    voxel_point_observation_residual,
    voxel_smoothness_residual,
    sigma_to_weight,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import GNConfig, gauss_newton_manifold


def build_three_voxel_chain_with_param_obs():
    """
    Build a small 1D chain of 3 voxels along x, backed by a WorldModel:

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
    wm = WorldModel()

    # Three voxel variables (voxel_cell), intentionally perturbed from [0,1,2]
    v0_id = wm.add_variable(
        var_type="voxel_cell",
        value=jnp.array([-0.2, 0.1, 0.0], dtype=jnp.float32),
    )
    v1_id = wm.add_variable(
        var_type="voxel_cell",
        value=jnp.array([0.9, -0.3, 0.0], dtype=jnp.float32),
    )
    v2_id = wm.add_variable(
        var_type="voxel_cell",
        value=jnp.array([2.3, 0.2, 0.0], dtype=jnp.float32),
    )

    # Priors at v0 ~ [0,0,0], v2 ~ [2,0,0]
    wm.add_factor(
        f_type="prior",
        var_ids=(v0_id,),
        params={"target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)},
    )
    wm.add_factor(
        f_type="prior",
        var_ids=(v2_id,),
        params={"target": jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)},
    )

    # Voxel smoothness: v1-v0 ~ [1,0,0], v2-v1 ~ [1,0,0] with moderate regularization
    smooth_sigma = 0.1
    smooth_weight = sigma_to_weight(smooth_sigma)

    wm.add_factor(
        f_type="voxel_smoothness",
        var_ids=(v0_id, v1_id),
        params={
            "offset": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            "weight": smooth_weight,
        },
    )
    wm.add_factor(
        f_type="voxel_smoothness",
        var_ids=(v1_id, v2_id),
        params={
            "offset": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            "weight": smooth_weight,
        },
    )

    # voxel_point_obs for each voxel.
    # We'll override 'point_world' via theta in the parametric residual.
    obs_sigma = 0.05
    obs_weight = sigma_to_weight(obs_sigma)

    # Base (incorrect) observation points:
    base_obs0 = jnp.array([-0.5, 0.1, 0.0], dtype=jnp.float32)
    base_obs1 = jnp.array([0.7, -0.2, 0.0], dtype=jnp.float32)
    base_obs2 = jnp.array([2.4, 0.3, 0.0], dtype=jnp.float32)

    wm.add_factor(
        f_type="voxel_point_obs",
        var_ids=(v0_id,),
        params={
            "point_world": base_obs0,  # will be overridden by theta[0]
            "weight": obs_weight,
        },
    )
    wm.add_factor(
        f_type="voxel_point_obs",
        var_ids=(v1_id,),
        params={
            "point_world": base_obs1,  # will be overridden by theta[1]
            "weight": obs_weight,
        },
    )
    wm.add_factor(
        f_type="voxel_point_obs",
        var_ids=(v2_id,),
        params={
            "point_world": base_obs2,  # will be overridden by theta[2]
            "weight": obs_weight,
        },
    )

    # Register residuals at the WorldModel level
    wm.register_residual("prior", prior_residual)
    wm.register_residual("voxel_smoothness", voxel_smoothness_residual)
    wm.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # Pack state and manifold metadata
    x_init, index = wm.pack_state()
    packed_state = (x_init, index)
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=packed_state,
        fg=wm.fg,
    )

    # Get slices for each voxel in the state vector
    v0_slice = block_slices[v0_id]
    v1_slice = block_slices[v1_id]
    v2_slice = block_slices[v2_id]

    # Build parametric residual function: residual(x, theta) with theta.shape == (3, 3)
    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry

    def residual_param_fn(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Residual function that treats the 3 observation points as parameters.

        theta: shape (3, 3)
          theta[0] -> point_world for voxel 0
          theta[1] -> point_world for voxel 1
          theta[2] -> point_world for voxel 2
        """
        var_values = wm.unpack_state(x, index)
        res_list = []
        obs_idx = 0

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(f"No residual fn registered for factor type '{f.type}'")

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

            if f.type == "voxel_point_obs":
                # Override point_world with theta[obs_idx]
                point_world = theta[obs_idx]
                obs_idx += 1
                base_params = dict(f.params)
                base_params["point_world"] = point_world
                params = base_params
            else:
                params = f.params

            r = res_fn(stacked, params)
            w = params.get("weight", 1.0)
            res_list.append(jnp.sqrt(w) * r)

        return jnp.concatenate(res_list)

    # Ground-truth positions
    gt0 = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    gt1 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    gt2 = jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32)
    gt_stack = jnp.stack([gt0, gt1, gt2], axis=0)

    # Initial theta from base_obs
    theta0 = jnp.stack([base_obs0, base_obs1, base_obs2], axis=0)  # shape (3, 3)

    return (
        wm,
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
        wm,
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