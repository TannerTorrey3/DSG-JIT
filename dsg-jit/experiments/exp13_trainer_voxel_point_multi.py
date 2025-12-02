# experiments/exp13_trainer_voxel_point_multi.py

import jax
import jax.numpy as jnp

from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.core.types import NodeId, Variable, Factor, FactorId
from dsg_jit.optimization.solvers import gauss_newton, GNConfig
from dsg_jit.slam.measurements import (
    prior_residual,
    voxel_smoothness_residual,
    voxel_point_observation_residual,
)


def build_voxel_graph():
    """
    Tiny 1D-ish voxel chain with three voxels and three observations.

    Variables:
      - v0, v1, v2: voxel_cell in R^3
    Factors:
      - prior on v0 ~ [0, 0, 0]
      - smoothness between v0-v1, v1-v2
      - three voxel_point_obs factors with (learnable) point_world theta[k]
    """

    fg = FactorGraph()

    # --- Variables: three voxel cells in R^3 ---
    v0 = Variable(NodeId(0), "voxel_cell", jnp.array([0.0, 0.0, 0.0]))
    v1 = Variable(NodeId(1), "voxel_cell", jnp.array([1.2, 0.2, 0.0]))
    v2 = Variable(NodeId(2), "voxel_cell", jnp.array([2.3, -0.1, 0.1]))

    fg.add_variable(v0)
    fg.add_variable(v1)
    fg.add_variable(v2)

    # --- Register residuals ---
    fg.register_residual("prior", prior_residual)
    fg.register_residual("voxel_smoothness", voxel_smoothness_residual)
    fg.register_residual("voxel_point_obs", voxel_point_observation_residual)

    # --- Prior on v0 at [0,0,0] ---
    f_prior = Factor(
        FactorId(0),
        "prior",
        (NodeId(0),),
        {
            "target": jnp.array([0.0, 0.0, 0.0]),
            "weight": 10.0,
        },
    )

    # --- Smoothness v0 <-> v1, v1 <-> v2 ---
    f_smooth_01 = Factor(
        FactorId(1),
        "voxel_smoothness",
        (NodeId(0), NodeId(1)),
        {
            "lambda": 1.0,
            "offset": jnp.zeros(3),
            "weight": 1.0,
        },
    )
    f_smooth_12 = Factor(
        FactorId(2),
        "voxel_smoothness",
        (NodeId(1), NodeId(2)),
        {
            "lambda": 1.0,
            "offset": jnp.zeros(3),
            "weight": 1.0,
        },
    )

    # --- Observations: one per voxel ---
    # We will NOT hard-code point_world here; they will be supplied via theta
    # using build_residual_function_voxel_point_param_multi.
    f_obs0 = Factor(
        FactorId(3),
        "voxel_point_obs",
        (NodeId(0),),
        {
            # placeholder; real point_world will be injected by param_multi builder
            "point_world": jnp.array([0.0, 0.0, 0.0]),
            "weight": 1.0,
        },
    )
    f_obs1 = Factor(
        FactorId(4),
        "voxel_point_obs",
        (NodeId(1),),
        {
            "point_world": jnp.array([1.0, 0.0, 0.0]),
            "weight": 1.0,
        },
    )
    f_obs2 = Factor(
        FactorId(5),
        "voxel_point_obs",
        (NodeId(2),),
        {
            "point_world": jnp.array([2.0, 0.0, 0.0]),
            "weight": 1.0,
        },
    )

    fg.add_factor(f_prior)
    fg.add_factor(f_smooth_01)
    fg.add_factor(f_smooth_12)
    fg.add_factor(f_obs0)
    fg.add_factor(f_obs1)
    fg.add_factor(f_obs2)

    return fg, (NodeId(0), NodeId(1), NodeId(2))


def solve_inner_voxel(fg: FactorGraph, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Inner solve: given observation parameters theta (K,3), solve for voxel variables x.

    Uses the already-implemented build_residual_function_voxel_point_param_multi
    and a standard Gauss-Newton solve.
    """
    residual_fn_param, _ = fg.build_residual_function_voxel_point_param_multi()
    x0, _ = fg.pack_state()  # initial voxels

    def residual_x(x: jnp.ndarray) -> jnp.ndarray:
        return residual_fn_param(x, theta)

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton(residual_x, x0, cfg)
    return x_opt


def main():
    print("=== 4.c.1 – Trainer-style multi-voxel obs learning ===\n")

    fg, voxel_ids = build_voxel_graph()
    v0_id, v1_id, v2_id = voxel_ids

    # Ground truth voxel centers we want the system to match (for supervision)
    gt_voxels = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    # Parameterization: one point_world per voxel_point_obs factor.
    # theta has shape (K,3). Start from slightly biased points.
    theta0 = jnp.array(
        [
            [-0.5, 0.1, 0.0],
            [0.7, -0.2, 0.0],
            [2.4, 0.3, 0.0],
        ],
        dtype=jnp.float32,
    )

    # Define supervised loss: solve for voxels given theta, then compare v's to gt
    def supervised_loss(theta: jnp.ndarray) -> jnp.ndarray:
        x_opt = solve_inner_voxel(fg, theta)
        _, index = fg.pack_state()
        values = fg.unpack_state(x_opt, index)

        v0 = values[v0_id]
        v1 = values[v1_id]
        v2 = values[v2_id]

        v_stack = jnp.stack([v0, v1, v2], axis=0)
        diff = v_stack - gt_voxels
        return 0.5 * jnp.sum(diff * diff)

    loss_fn = jax.jit(supervised_loss)
    grad_fn = jax.jit(jax.grad(supervised_loss))

    # --- Initial diagnostics ---
    loss_init = float(loss_fn(theta0))
    grad_init = grad_fn(theta0)

    print("Initial theta (obs points):")
    print(theta0)
    print(f"Initial supervised loss: {loss_init:.6f}")
    print("Initial grad wrt theta:")
    print(grad_init)

    # --- Outer gradient descent on theta ---
    lr = 0.1
    steps = 20
    theta = theta0

    for it in range(steps):
        g = grad_fn(theta)
        # Optional: simple gradient clipping to avoid blow-ups
        g_norm = jnp.linalg.norm(g)
        if g_norm > 10.0:
            g = g * (10.0 / g_norm)

        theta = theta - lr * g
        loss_t = float(loss_fn(theta))

        if it % 2 == 0 or it == steps - 1:
            print(
                f"iter {it:02d}: loss={loss_t:.6f}, "
                f"||g||={float(g_norm):.6f}"
            )

    # --- Final optimized voxels ---
    x_final = solve_inner_voxel(fg, theta)
    _, index = fg.pack_state()
    values = fg.unpack_state(x_final, index)

    v0 = values[v0_id]
    v1 = values[v1_id]
    v2 = values[v2_id]

    print("\n=== Final results after learning theta ===")
    print("Learned theta (point_world per obs):")
    print(theta)
    print("\nOptimized voxel positions:")
    print("v0:", v0)
    print("v1:", v1)
    print("v2:", v2)
    print("\nGround truth voxel centers:")
    print(gt_voxels)


if __name__ == "__main__":
    main()