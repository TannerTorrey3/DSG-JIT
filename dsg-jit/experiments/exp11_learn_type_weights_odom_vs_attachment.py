import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,
    pose_place_attachment_residual,
)
from dsg_jit.optimization.solvers import GDConfig, gradient_descent


def _to_slice(idx):
    """Normalize an index entry (slice or (start, length)) into a slice."""
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
    wm = WorldModel()

    # Initial guesses (intentionally a bit off)
    pose0_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([0.2, -0.1, 0.0, 0.01, -0.02, 0.0], dtype=jnp.float32),
    )
    pose1_id = wm.add_variable(
        var_type="pose_se3",
        value=jnp.array([0.5, 0.1, 0.0, 0.02, 0.01, 0.0], dtype=jnp.float32),
    )
    place0_id = wm.add_variable(
        var_type="place1d",
        value=jnp.array([0.6], dtype=jnp.float32),
    )

    # Prior on pose0: identity
    wm.add_factor(
        f_type="prior",
        var_ids=(pose0_id,),
        params={"target": jnp.zeros(6, dtype=jnp.float32)},
    )

    # Biased odometry: wants pose1.tx = 0.7
    biased_meas = jnp.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    wm.add_factor(
        f_type="odom_se3",
        var_ids=(pose0_id, pose1_id),
        params={"measurement": biased_meas},
    )

    # Attachment: place0 ~ pose1.tx
    wm.add_factor(
        f_type="pose_place_attachment",
        var_ids=(pose1_id, place0_id),  # pose1, place0
        params={
            "pose_dim": jnp.array(6, dtype=jnp.int32),
            "place_dim": jnp.array(1, dtype=jnp.int32),
            "pose_coord_index": jnp.array(0, dtype=jnp.int32),  # tx
        },
    )

    # Prior on place0 at x = 1.0 (trusted external observation)
    wm.add_factor(
        f_type="prior",
        var_ids=(place0_id,),
        params={"target": jnp.array([1.0], dtype=jnp.float32)},
    )

    # Register residuals at the WorldModel level
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3", odom_se3_residual)
    wm.register_residual("pose_place_attachment", pose_place_attachment_residual)

    # Pack state and get slices
    x_init, index = wm.pack_state()
    p0_slice = _to_slice(index[pose0_id])
    p1_slice = _to_slice(index[pose1_id])
    pl_slice = _to_slice(index[place0_id])

    # Build weighted residual fn with a single learnable log_scale for 'odom_se3'
    factor_type_order = ["odom_se3"]
    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry
    type_to_idx = {t: i for i, t in enumerate(factor_type_order)}

    def residual_w(x: jnp.ndarray, log_scales: jnp.ndarray) -> jnp.ndarray:
        """
        Residual with per-type weights controlled by log_scales.

        log_scales[i] corresponds to factor_type_order[i].
        Missing types default to unit weight.
        """
        var_values = wm.unpack_state(x, index)
        res_list = []

        for f in factors:
            res_fn = residual_fns.get(f.type, None)
            if res_fn is None:
                raise ValueError(
                    f"No residual fn registered for factor type '{f.type}'"
                )

            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids], axis=0)
            r = res_fn(stacked, f.params)  # (k,)

            idx = type_to_idx.get(f.type, None)
            if idx is not None:
                scale = jnp.exp(log_scales[idx])
            else:
                scale = 1.0

            r_scaled = scale * r
            r_scaled = jnp.reshape(r_scaled, (-1,))
            res_list.append(r_scaled)

        return jnp.concatenate(res_list, axis=0)

    return wm, x_init, residual_w, (p0_slice, p1_slice, pl_slice)


def run_experiment():
    wm, x_init, residual_w, (p0_slice, p1_slice, pl_slice) = build_problem()

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