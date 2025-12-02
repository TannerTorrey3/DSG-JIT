import jax
import jax.numpy as jnp

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,
    pose_place_attachment_residual,
)
from dsg_jit.optimization.solvers import GDConfig, gradient_descent


def _to_slice(idx):
    """Normalize FactorGraph index entry (slice or (start, length)) into a slice."""
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def build_problem():
    """
    Build a tiny SE(3) + place1d problem:

      Variables:
        pose0, pose1: R^6  (we only care about tx)
        place0:       R^1  (object position along x)

      Factors:
        - prior on pose0: wants pose0 == 0
        - odom_se3(pose0, pose1): biased, wants pose1.tx = 0.7
        - pose_place_attachment(pose1, place0): wants place0 ~ pose1.tx
        - prior on place0: wants place0 = 1.0

    True configuration should be:
        pose0.tx ≈ 0
        pose1.tx ≈ 1
        place0   ≈ 1

    If odom has too large weight, pose1 will sit closer to 0.7.
    If odom gets down-weighted, pose1 will move toward 1.0.

    We will learn the per-type log-scale for 'odom_se3'.
    """
    fg = FactorGraph()

    # Initial guesses (intentionally a bit off)
    pose0 = Variable(
        id=NodeId(0),
        type="pose_se3",
        value=jnp.array([0.2, -0.1, 0.0, 0.01, -0.02, 0.0], dtype=jnp.float32),
    )
    pose1 = Variable(
        id=NodeId(1),
        type="pose_se3",
        value=jnp.array([0.5, 0.1, 0.0, 0.02, 0.01, 0.0], dtype=jnp.float32),
    )
    place0 = Variable(
        id=NodeId(2),
        type="place1d",
        value=jnp.array([0.6], dtype=jnp.float32),
    )

    fg.add_variable(pose0)
    fg.add_variable(pose1)
    fg.add_variable(place0)

    # Prior on pose0: identity
    f_prior_pose0 = Factor(
        id=FactorId(0),
        type="prior",
        var_ids=(NodeId(0),),
        params={"target": jnp.zeros(6, dtype=jnp.float32)},
    )

    # Biased odometry: wants pose1.tx = 0.7
    biased_meas = jnp.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    f_odom = Factor(
        id=FactorId(1),
        type="odom_se3",
        var_ids=(NodeId(0), NodeId(1)),
        params={"measurement": biased_meas},
    )

    # Attachment: place0 ~ pose1.tx
    f_attach = Factor(
        id=FactorId(2),
        type="pose_place_attachment",
        var_ids=(NodeId(1), NodeId(2)),  # pose1, place0
        params={
            "pose_dim": jnp.array(6, dtype=jnp.int32),
            "place_dim": jnp.array(1, dtype=jnp.int32),
            "pose_coord_index": jnp.array(0, dtype=jnp.int32),  # tx
        },
    )

    # Prior on place0 at x = 1.0 (trusted external observation)
    f_prior_place = Factor(
        id=FactorId(3),
        type="prior",
        var_ids=(NodeId(2),),
        params={"target": jnp.array([1.0], dtype=jnp.float32)},
    )

    # Add all factors
    for f in (f_prior_pose0, f_odom, f_attach, f_prior_place):
        fg.add_factor(f)

    # Register residuals
    fg.register_residual("prior", prior_residual)
    fg.register_residual("odom_se3", odom_se3_residual)
    fg.register_residual("pose_place_attachment", pose_place_attachment_residual)

    # Pack state and get slices
    x_init, index = fg.pack_state()
    p0_slice = _to_slice(index[NodeId(0)])
    p1_slice = _to_slice(index[NodeId(1)])
    pl_slice = _to_slice(index[NodeId(2)])

    # Build weighted residual fn with a single learnable log_scale for 'odom_se3'
    factor_type_order = ["odom_se3"]
    residual_w = fg.build_residual_function_with_type_weights(factor_type_order)

    return fg, x_init, residual_w, (p0_slice, p1_slice, pl_slice)


def run_experiment():
    fg, x_init, residual_w, (p0_slice, p1_slice, pl_slice) = build_problem()

    # Inner solver config: optimize x for fixed log_scale
    gd_cfg = GDConfig(learning_rate=0.1, max_iters=200)

    # Supervised objective: want pose1.tx ≈ 1.0
    target_pose1_tx = 1.0

    def solve_and_loss(log_scales: jnp.ndarray) -> jnp.ndarray:
        """
        log_scales: shape (1,) -> log_scale for 'odom_se3'

        1. Optimize x via GD:
               x_opt = argmin_x || r(x, log_scales) ||^2
        2. Compute supervised loss:
               L = (pose1.tx - 1)^2
        """
        def objective_for_x(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_w(x, log_scales)
            return jnp.sum(r * r)

        x_opt = gradient_descent(objective_for_x, x_init, gd_cfg)

        pose1_opt = x_opt[p1_slice]
        pose1_tx = pose1_opt[0]

        return (pose1_tx - target_pose1_tx) ** 2

    # JIT wrappers
    solve_and_loss_jit = jax.jit(solve_and_loss)
    grad_log_jit = jax.jit(jax.grad(solve_and_loss))

    # Initial log-scale for odom: 0 -> scale = 1
    log_scale_odom = jnp.array([0.0], dtype=jnp.float32)

    print("=== Learnable Odom Type Weight Experiment ===")
    print(f"Initial log_scale_odom: {float(log_scale_odom[0]):.4f}")

    loss0 = float(solve_and_loss_jit(log_scale_odom))
    print(f"Initial loss (pose1.tx vs 1.0): {loss0:.6f}")

    # One gradient step on log_scale_odom
    grad0 = grad_log_jit(log_scale_odom)
    print(f"Grad wrt log_scale_odom: {grad0}")

    alpha = 0.5  # outer learning rate
    log_scale_odom_1 = log_scale_odom - alpha * grad0

    loss1 = float(solve_and_loss_jit(log_scale_odom_1))
    print(f"Updated log_scale_odom: {float(log_scale_odom_1[0]):.4f}")
    print(f"Loss after one weight step: {loss1:.6f}")

    # Inspect optimized poses for both weights
    def solve_x_only(log_scales: jnp.ndarray):
        def objective_for_x(x: jnp.ndarray) -> jnp.ndarray:
            r = residual_w(x, log_scales)
            return jnp.sum(r * r)

        return gradient_descent(objective_for_x, x_init, gd_cfg)

    x_opt0 = solve_x_only(log_scale_odom)
    x_opt1 = solve_x_only(log_scale_odom_1)

    p0_0 = x_opt0[p0_slice]
    p1_0 = x_opt0[p1_slice]
    pl_0 = x_opt0[pl_slice]

    p0_1 = x_opt1[p0_slice]
    p1_1 = x_opt1[p1_slice]
    pl_1 = x_opt1[pl_slice]

    print("\n--- Optimized states (before weight update) ---")
    print(f"pose0: {p0_0}")
    print(f"pose1: {p1_0}")
    print(f"place0: {pl_0}")

    print("\n--- Optimized states (after weight update) ---")
    print(f"pose0: {p0_1}")
    print(f"pose1: {p1_1}")
    print(f"place0: {pl_1}")

    print("\nInterpretation:")
    print("  - If the system learns to down-weight odom_se3,")
    print("    pose1.tx and place0 should move closer to 1.0.")
    print("  - You should see loss1 <= loss0 and pose1_tx moving toward 1.0.")


if __name__ == "__main__":
    run_experiment()