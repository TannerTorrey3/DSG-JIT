# experiments/exp14_multi_voxel_param_and_weight.py

import jax
import jax.numpy as jnp

from core.factor_graph import FactorGraph
from core.types import NodeId, FactorId, Variable, Factor
from optimization.solvers import gradient_descent, GDConfig
from slam.measurements import (
    prior_residual,
    voxel_point_observation_residual,  # registered as "voxel_point_obs"
)

"""
4.c.2 – Joint learning of voxel_point_obs parameters (theta) AND a type weight.

We build a tiny voxel-only graph:

  - 3 voxel cells v0, v1, v2 in R^3 (x,y,z), variable type "voxel_cell3d"
  - Weak voxel priors pulling them toward [0,0,0], [1,0,0], [2,0,0]
  - 3 voxel_point_obs factors, each attached to one voxel cell, with biased observations

We then:

  - Treat the observation positions theta[k] (point_world per obs) as learnable
  - Also learn a scalar log_scale_obs for ALL voxel_point_obs residuals
  - Inner loop: solve for voxel states with these parameters
  - Outer loop: minimize MSE between optimized voxel centers and ground-truth [0,1,2] chain
"""


def build_voxel_graph(theta_init: jnp.ndarray) -> FactorGraph:
    """
    Build a tiny FactorGraph with 3 voxel cells and 3 voxel_point_obs factors.

    theta_init: (3,3) initial world points for the 3 observations.
    """
    fg = FactorGraph()

    # --- Variables: 3 voxel cells in R^3 (x, y, z) ---
    v0 = Variable(
        id=NodeId(0),
        type="voxel_cell3d",
        value=jnp.array([-0.2, 0.1, 0.0], dtype=jnp.float32),
    )
    v1 = Variable(
        id=NodeId(1),
        type="voxel_cell3d",
        value=jnp.array([0.8, -0.3, 0.0], dtype=jnp.float32),
    )
    v2 = Variable(
        id=NodeId(2),
        type="voxel_cell3d",
        value=jnp.array([2.3, 0.2, 0.0], dtype=jnp.float32),
    )

    fg.add_variable(v0)
    fg.add_variable(v1)
    fg.add_variable(v2)

    # --- Residual registrations ---
    fg.register_residual("voxel_prior", prior_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # --- Priors: weakly pull voxels toward 0,1,2 on x-axis ---
    prior_weight = 0.1  # weak prior, learning driven mostly by obs

    fg.add_factor(
        Factor(
            id=FactorId(0),
            type="voxel_prior",
            var_ids=(NodeId(0),),
            params={
                "target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                "weight": prior_weight,
            },
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(1),
            type="voxel_prior",
            var_ids=(NodeId(1),),
            params={
                "target": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                "weight": prior_weight,
            },
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(2),
            type="voxel_prior",
            var_ids=(NodeId(2),),
            params={
                "target": jnp.array([2.0, 0.0, 0.0], dtype=jnp.float32),
                "weight": prior_weight,
            },
        )
    )

    # --- Voxel point observations (one per voxel) ---
    obs_weight = 10.0  # relatively strong obs; learnable type scale will modulate

    fg.add_factor(
        Factor(
            id=FactorId(3),
            type="voxel_point_obs",
            var_ids=(NodeId(0),),
            params={
                "point_world": theta_init[0],  # will be overridden
                "weight": obs_weight,
            },
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(4),
            type="voxel_point_obs",
            var_ids=(NodeId(1),),
            params={
                "point_world": theta_init[1],
                "weight": obs_weight,
            },
        )
    )
    fg.add_factor(
        Factor(
            id=FactorId(5),
            type="voxel_point_obs",
            var_ids=(NodeId(2),),
            params={
                "point_world": theta_init[2],
                "weight": obs_weight,
            },
        )
    )

    return fg


def build_residual_param_and_weight(fg: FactorGraph):
    """
    Build a residual function:

        r(x, theta, log_scale_obs)

    where:
      - x is the flat voxel state vector
      - theta has shape (K,3) with one point_world per voxel_point_obs factor
      - log_scale_obs is a scalar controlling the strength of voxel_point_obs

    All voxel_point_obs residuals are scaled by exp(log_scale_obs).
    voxel_prior residuals keep their static 'weight' only.
    """
    factors = list(fg.factors.values())
    residual_fns = fg.residual_fns
    _, index = fg.pack_state()

    def residual(x: jnp.ndarray, theta: jnp.ndarray, log_scale_obs: jnp.ndarray) -> jnp.ndarray:
        var_values = fg.unpack_state(x, index)
        res_list = []
        obs_idx = 0

        scale_obs = jnp.exp(log_scale_obs)  # scalar

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(f"No residual fn registered for factor type '{f.type}'")

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids], axis=0)

            # Construct params, overriding point_world for voxel_point_obs
            if f.type == "voxel_point_obs":
                point_world = theta[obs_idx]  # (3,)
                obs_idx += 1
                base_params = dict(f.params)
                base_params["point_world"] = point_world
                params = base_params
            else:
                params = f.params

            r = res_fn(stacked, params)  # (k,)
            w = params.get("weight", 1.0)

            if f.type == "voxel_point_obs":
                r = scale_obs * r  # apply learnable type weight

            res_list.append(jnp.sqrt(w) * jnp.reshape(r, (-1,)))

        return jnp.concatenate(res_list, axis=0)

    return residual, index


def main():
    print("=== 4.c.2 – Joint learning of voxel obs params and type weight (exp14) ===\n")

    # --- Ground truth voxel centers ---
    gt_voxels = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    # --- Initial observation parameters theta (biased points) ---
    theta_init = jnp.array(
        [
            [-0.5, 0.1, 0.0],
            [0.7, -0.2, 0.0],
            [2.4, 0.3, 0.0],
        ],
        dtype=jnp.float32,
    )
    K = theta_init.shape[0]

    # --- Build graph ---
    fg = build_voxel_graph(theta_init)
    residual_pw, index = build_residual_param_and_weight(fg)
    x0, _ = fg.pack_state()  # initial voxel state

    # --- Inner solver: GD over x for fixed (theta, log_scale_obs) ---
    # More conservative settings to avoid blow-up when log_scale is large
    cfg_inner = GDConfig(learning_rate=0.05, max_iters=40)

    def solve_inner(theta: jnp.ndarray, log_scale_obs: jnp.ndarray) -> jnp.ndarray:
        def objective(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_pw(x, theta, log_scale_obs)
            return 0.5 * jnp.sum(r * r)

        return gradient_descent(objective, x0, cfg_inner)

    # --- Pack outer parameters into a single vector p: [theta_flat, log_scale_obs] ---
    def pack_params(theta: jnp.ndarray, log_scale_obs: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([theta.reshape(-1), jnp.array([log_scale_obs])])

    def unpack_params(p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        theta_flat = p[:-1]
        log_scale_obs = p[-1]
        theta = theta_flat.reshape((K, 3))
        return theta, log_scale_obs

    # --- Supervised loss: MSE between optimized voxels and ground truth ---
    def loss_fn(p: jnp.ndarray) -> jnp.ndarray:
        theta, log_scale_obs = unpack_params(p)
        # Clamp log_scale inside loss to keep things sane
        log_scale_obs = jnp.clip(log_scale_obs, -2.0, 2.0)  # scales in [~0.14, ~7.39]
        x_opt = solve_inner(theta, log_scale_obs)
        values = fg.unpack_state(x_opt, index)

        v0 = values[NodeId(0)]
        v1 = values[NodeId(1)]
        v2 = values[NodeId(2)]

        V = jnp.stack([v0, v1, v2], axis=0)  # (3,3)
        diff = V - gt_voxels
        base_loss = jnp.mean(jnp.sum(diff * diff, axis=1))

        # Small regularizer on log_scale_obs to keep it near 0
        reg = 1e-2 * (log_scale_obs ** 2)
        return base_loss + reg

    loss_grad = jax.grad(loss_fn)

    # --- Initialize outer parameters ---
    log_scale_init = jnp.array(0.0, dtype=jnp.float32)
    p = pack_params(theta_init, log_scale_init)

    # Compute initial loss and grad
    loss0 = float(loss_fn(p))
    g0 = loss_grad(p)
    print("Initial theta (obs points):")
    print(theta_init)
    print(f"Initial log_scale_obs: {float(log_scale_init):.6f}")
    print(f"Initial supervised loss: {loss0:.6f}")
    print("Initial grad norm:", float(jnp.linalg.norm(g0)))
    print()

    # --- Outer loop: GD on p with gradient + log_scale clipping ---
    lr_outer = 1e-2
    steps = 50
    g_clip = 1.0

    for it in range(steps):
        g = loss_grad(p)

        # Gradient norm clipping
        g_norm = jnp.linalg.norm(g) + 1e-8
        scale = jnp.minimum(1.0, g_clip / g_norm)
        g_clipped = g * scale

        p = p - lr_outer * g_clipped

        # Explicitly clamp log_scale component after update
        theta_t, log_scale_t = unpack_params(p)
        log_scale_t = jnp.clip(log_scale_t, -2.0, 2.0)
        p = pack_params(theta_t, log_scale_t)

        if it % 2 == 0 or it == steps - 1:
            loss_t = float(loss_fn(p))
            print(
                f"iter {it:02d}: loss={loss_t:.6f}, "
                f"log_scale_obs={float(log_scale_t):.6f}, ||g||={float(g_norm):.6f}"
            )

    # --- Final results ---
    theta_final, log_scale_final = unpack_params(p)
    log_scale_final = float(jnp.clip(log_scale_final, -2.0, 2.0))
    x_final = solve_inner(theta_final, log_scale_final)
    values_final = fg.unpack_state(x_final, index)

    v0_f = values_final[NodeId(0)]
    v1_f = values_final[NodeId(1)]
    v2_f = values_final[NodeId(2)]
    V_final = jnp.stack([v0_f, v1_f, v2_f], axis=0)

    print("\n=== Final results after joint learning (theta + log_scale_obs) ===")
    print("Learned theta (point_world per obs):")
    print(theta_final)
    print(f"\nLearned log_scale_obs: {log_scale_final:.6f}")
    print(f"Effective obs scale = exp(log_scale_obs): {float(jnp.exp(log_scale_final)):.6f}")

    print("\nOptimized voxel positions:")
    print("v0:", v0_f)
    print("v1:", v1_f)
    print("v2:", v2_f)

    print("\nGround truth voxel centers:")
    print(gt_voxels)


if __name__ == "__main__":
    main()