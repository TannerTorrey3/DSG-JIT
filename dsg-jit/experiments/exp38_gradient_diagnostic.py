# experiments/exp38_gradient_diagnostic.py
"""
Diagnostic: validate that IFT weight gradients point in the correct direction.

Creates a tiny problem (20 poses, 1 known outlier) and checks whether
d(outer_loss)/d(log_w_outlier) is more negative than inlier gradients.

If this fails, the math/architecture is broken.
If this succeeds, the issue is optimisation dynamics at scale.

Usage:
    python -m experiments.exp38_gradient_diagnostic
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.core.math3d import (
    compose_pose_se3,
    relative_pose_se3,
    se3_retract_left,
)
from dsg_jit.slam.measurements import sigma_to_weight


def main():
    print("=" * 60)
    print("  exp38 Gradient Diagnostic")
    print("=" * 60)
    print(f"  JAX devices: {[str(d) for d in jax.devices()]}")
    print()

    # --- Tiny problem: 20 poses, 1 outlier at edge 10 ---
    n_poses = 20
    n_meas = n_poses - 1
    outlier_idx = 10  # Which edge is the outlier

    # Create a straight-line GT trajectory (simple, clean).
    gt_poses = jnp.zeros((n_poses, 6), dtype=jnp.float32)
    for i in range(n_poses):
        gt_poses = gt_poses.at[i, 0].set(i * 0.1)  # Move 0.1m in x per step

    # GT measurements.
    gt_meas = jax.vmap(relative_pose_se3)(gt_poses[:-1], gt_poses[1:])
    print(f"  GT measurement (edge 0): {gt_meas[0]}")

    # Add small Gaussian noise.
    sigma_trans = 0.03
    sigma_rot = 0.01
    sigma = jnp.array([sigma_trans]*3 + [sigma_rot]*3, dtype=jnp.float32)
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, shape=gt_meas.shape) * sigma
    noisy_meas = gt_meas + noise

    # Inject ONE outlier at edge 10.
    outlier_perturbation = jnp.array([0.5, 0.3, -0.2, 0.1, -0.1, 0.05],
                                     dtype=jnp.float32)
    noisy_meas = noisy_meas.at[outlier_idx].set(
        noisy_meas[outlier_idx] + outlier_perturbation)

    print(f"  Outlier edge: {outlier_idx}")
    print(f"  Outlier perturbation magnitude: "
          f"trans={jnp.linalg.norm(outlier_perturbation[:3]):.3f}m, "
          f"rot={jnp.linalg.norm(outlier_perturbation[3:]):.3f}rad")
    print()

    # --- Build the bilevel problem ---
    odom_w = sigma_to_weight(sigma)
    sqrt_odom_w = jnp.sqrt(odom_w)

    # Inner: 2 gauge anchors (first/last)
    inner_anchor_idx = jnp.array([0, n_poses - 1], dtype=jnp.int32)
    inner_anchor_sigma = 0.01
    inner_anchor_w = sigma_to_weight(jnp.full(6, inner_anchor_sigma))
    sqrt_inner_anchor_w = jnp.sqrt(inner_anchor_w)

    # Outer: eval every 2nd pose (dense supervision for tiny problem)
    eval_idx = jnp.array(list(range(0, n_poses, 2)), dtype=jnp.int32)

    # Outer loss weights
    aw = 15.0
    rw_trans, rw_rot = 0.25, 0.1
    sw_trans, sw_rot = 25.0, 25.0
    weight_reg = 0.01
    gn_iters = 10
    gn_damping = 5e-3

    aw_vec = jnp.array([aw]*3 + [aw]*3, dtype=jnp.float32)
    rw_vec = jnp.array([rw_trans]*3 + [rw_rot]*3, dtype=jnp.float32)
    sw_vec = jnp.array([sw_trans]*3 + [sw_rot]*3, dtype=jnp.float32)

    _odom_res_batch = jax.vmap(
        lambda a, b, m, w: w * (relative_pose_se3(a, b) - m) * sqrt_odom_w)
    _retract_batch = jax.vmap(se3_retract_left)

    # Compute GT-derived targets.
    gt_first = gt_poses[0]
    inner_anchor_targets = jax.vmap(relative_pose_se3, in_axes=(None, 0))(
        gt_first, gt_poses[inner_anchor_idx])
    # Relative eval targets: relative poses between consecutive eval positions.
    eval_rel_targets = jax.vmap(relative_pose_se3)(
        gt_poses[eval_idx[:-1]], gt_poses[eval_idx[1:]])

    print(f"  Inner anchors: {len(inner_anchor_idx)} (positions {list(np.array(inner_anchor_idx))})")
    print(f"  Eval positions: {len(eval_idx)} (positions {list(np.array(eval_idx))})")
    print(f"  Eval pairs (relative): {len(eval_rel_targets)}")
    print()

    # --- Residual and GN solver ---

    def residual_fn(x, theta, weights):
        poses = x.reshape(n_poses, 6)
        w_broad = weights[:, None]
        r_odom = _odom_res_batch(poses[:-1], poses[1:], theta, w_broad)
        r_anch = (poses[inner_anchor_idx] - inner_anchor_targets) * sqrt_inner_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step(x, theta, weights):
        r_fn = lambda x_: residual_fn(x_, theta, weights)
        r = r_fn(x)
        J = jax.jacobian(r_fn)(x)
        n = x.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        delta = jnp.linalg.solve(H, J.T @ r)
        poses = x.reshape(n_poses, 6)
        deltas = delta.reshape(n_poses, 6)
        norms = jnp.linalg.norm(deltas, axis=1, keepdims=True)
        scales = jnp.minimum(1.0, 0.5 / (norms + 1e-9))
        deltas = deltas * scales
        return _retract_batch(poses, -deltas).ravel()

    @jax.custom_vjp
    def inner_solve(theta, x_init, weights):
        def scan_body(x, _):
            return gn_step(x, theta, weights), None
        x_star, _ = jax.lax.scan(scan_body, x_init, None, length=gn_iters)
        return x_star

    def inner_solve_fwd(theta, x_init, weights):
        x_star = inner_solve(theta, x_init, weights)
        return x_star, (x_star, theta, weights)

    def inner_solve_bwd(res, g):
        x_star, theta, weights = res
        r_fn_x = lambda x_: residual_fn(x_, theta, weights)
        J = jax.jacobian(r_fn_x)(x_star)
        n = x_star.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        u = jnp.linalg.solve(H, g)
        v = J @ u

        _, vjp_theta = jax.vjp(
            lambda t: residual_fn(x_star, t, weights), theta)
        dtheta = -vjp_theta(v)[0]

        _, vjp_weights = jax.vjp(
            lambda w: residual_fn(x_star, theta, w), weights)
        dweights = -vjp_weights(v)[0]

        return (dtheta, jnp.zeros_like(x_star), dweights)

    inner_solve.defvjp(inner_solve_fwd, inner_solve_bwd)

    # --- Outer loss (relative evaluation) ---
    log_w_init_val = 3.0
    _relative_batch = jax.vmap(relative_pose_se3)

    def outer_loss(theta, log_w, x_init):
        weights = jax.nn.sigmoid(log_w)
        x_star = inner_solve(theta, x_init, weights)
        poses_opt = x_star.reshape(n_poses, 6)

        # Relative eval: compare relative poses between consecutive eval positions
        solved_rel = _relative_batch(poses_opt[eval_idx[:-1]], poses_opt[eval_idx[1:]])
        eval_diffs = solved_rel - eval_rel_targets
        a_loss = jnp.sum(aw_vec * eval_diffs ** 2)

        # Reg loss
        dev = theta - noisy_meas
        r_loss = jnp.sum(rw_vec * dev ** 2)

        # Smoothness
        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(sw_vec * s_diffs ** 2)

        # Weight reg
        w_reg_loss = weight_reg * jnp.sum((log_w - log_w_init_val) ** 2)

        return a_loss + r_loss + s_loss + w_reg_loss

    def outer_loss_components(theta, log_w, x_init):
        """Return individual loss components for analysis."""
        weights = jax.nn.sigmoid(log_w)
        x_star = inner_solve(theta, x_init, weights)
        poses_opt = x_star.reshape(n_poses, 6)

        solved_rel = _relative_batch(poses_opt[eval_idx[:-1]], poses_opt[eval_idx[1:]])
        eval_diffs = solved_rel - eval_rel_targets
        a_loss = jnp.sum(aw_vec * eval_diffs ** 2)

        dev = theta - noisy_meas
        r_loss = jnp.sum(rw_vec * dev ** 2)

        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(sw_vec * s_diffs ** 2)

        w_reg_loss = weight_reg * jnp.sum((log_w - log_w_init_val) ** 2)

        return {
            "anchor": float(a_loss),
            "reg": float(r_loss),
            "smooth": float(s_loss),
            "w_reg": float(w_reg_loss),
            "total": float(a_loss + r_loss + s_loss + w_reg_loss),
        }

    # --- Compute gradients ---
    print("  Computing gradients (this validates IFT math)...")
    print()

    # Initial state: theta = noisy_meas, log_w = 3.0 (w ≈ 0.95)
    theta_init = noisy_meas.copy()
    log_w_init = jnp.full(n_meas, log_w_init_val, dtype=jnp.float32)

    # x_init from forward compose
    def _compose_scan(carry, meas):
        return compose_pose_se3(carry, meas), compose_pose_se3(carry, meas)
    origin = jnp.zeros(6, dtype=jnp.float32)
    _, all_poses = jax.lax.scan(_compose_scan, origin, noisy_meas)
    x_init = jnp.concatenate([origin[None], all_poses]).ravel()

    # Loss components at initialization
    components = outer_loss_components(theta_init, log_w_init, x_init)
    print("  Loss components at init (theta=noisy, w=0.95):")
    for k, v in components.items():
        print(f"    {k:8s}: {v:.4f}")
    print()

    # Gradient w.r.t. log_w
    grad_fn = jax.grad(outer_loss, argnums=(0, 1))
    g_theta, g_log_w = grad_fn(theta_init, log_w_init, x_init)

    print("  --- Weight gradients (d loss / d log_w) per edge ---")
    print(f"  {'Edge':>4}  {'Gradient':>10}  {'|grad|':>8}  {'Type':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}")

    g_w_np = np.array(g_log_w)
    for i in range(n_meas):
        edge_type = "OUTLIER" if i == outlier_idx else "inlier"
        print(f"  {i:4d}  {g_w_np[i]:+10.6f}  {abs(g_w_np[i]):8.6f}  {edge_type}")

    print()
    outlier_grad = g_w_np[outlier_idx]
    inlier_grads = np.concatenate([g_w_np[:outlier_idx], g_w_np[outlier_idx+1:]])
    mean_inlier_grad = np.mean(inlier_grads)
    max_inlier_grad_mag = np.max(np.abs(inlier_grads))

    print(f"  Outlier grad:       {outlier_grad:+.6f}")
    print(f"  Mean inlier grad:   {mean_inlier_grad:+.6f}")
    print(f"  Max |inlier grad|:  {max_inlier_grad_mag:.6f}")
    print()

    # The key test: outlier gradient should be MORE POSITIVE than inliers.
    # (Positive gradient on log_w means: increasing log_w increases loss,
    #  so the optimiser should DECREASE log_w, pushing w toward 0.)
    #
    # Wait — actually we want the outlier weight to DECREASE (w → 0).
    # Since we minimise loss: d(loss)/d(log_w) > 0 means gradient descent pushes log_w down.
    # OR: d(loss)/d(log_w) < 0 means gradient descent pushes log_w UP.
    #
    # For outlier rejection: we want the outlier's log_w to decrease.
    # This happens when d(loss)/d(log_w_outlier) > 0.
    # (Because update: log_w -= lr * grad, so positive grad → log_w decreases.)

    if outlier_grad > mean_inlier_grad + 0.001:
        print("  ✓ PASS: Outlier edge has larger positive gradient than inliers.")
        print("    → Gradient descent will push outlier weight DOWN (correct).")
    elif outlier_grad > 0 and outlier_grad > mean_inlier_grad:
        print("  ~ WEAK PASS: Outlier grad is positive and larger than mean,")
        print("    but separation is small.")
    else:
        print("  ✗ FAIL: Outlier gradient does NOT correctly point toward rejection.")
        print("    → The IFT math may be incorrect or the loss structure is wrong.")

    print()

    # --- Also check: what happens if we manually set outlier weight low? ---
    print("  --- Manual weight test: set outlier w=0.01, keep rest at 0.95 ---")
    log_w_manual = jnp.full(n_meas, log_w_init_val, dtype=jnp.float32)
    log_w_manual = log_w_manual.at[outlier_idx].set(-4.6)  # sigmoid(-4.6) ≈ 0.01

    components_high_w = outer_loss_components(theta_init, log_w_init, x_init)
    components_low_w = outer_loss_components(theta_init, log_w_manual, x_init)

    print(f"  All w=0.95:          total loss = {components_high_w['total']:.4f}")
    print(f"    anchor={components_high_w['anchor']:.4f}, "
          f"reg={components_high_w['reg']:.4f}, "
          f"smooth={components_high_w['smooth']:.4f}, "
          f"w_reg={components_high_w['w_reg']:.4f}")
    print(f"  Outlier w=0.01:      total loss = {components_low_w['total']:.4f}")
    print(f"    anchor={components_low_w['anchor']:.4f}, "
          f"reg={components_low_w['reg']:.4f}, "
          f"smooth={components_low_w['smooth']:.4f}, "
          f"w_reg={components_low_w['w_reg']:.4f}")
    print()

    if components_low_w['total'] < components_high_w['total']:
        delta = components_high_w['total'] - components_low_w['total']
        print(f"  ✓ PASS: Rejecting the outlier REDUCES total loss by {delta:.4f}")
        print(f"    anchor: {components_high_w['anchor']:.4f} → {components_low_w['anchor']:.4f} "
              f"(Δ={components_low_w['anchor'] - components_high_w['anchor']:+.4f})")
        print(f"    w_reg:  {components_high_w['w_reg']:.4f} → {components_low_w['w_reg']:.4f} "
              f"(Δ={components_low_w['w_reg'] - components_high_w['w_reg']:+.4f})")
    else:
        delta = components_low_w['total'] - components_high_w['total']
        print(f"  ✗ FAIL: Rejecting the outlier INCREASES total loss by {delta:.4f}")
        print(f"    The w_reg penalty outweighs the anchor improvement.")
        print(f"    anchor improvement: {components_high_w['anchor'] - components_low_w['anchor']:.4f}")
        print(f"    w_reg penalty:      {components_low_w['w_reg'] - components_high_w['w_reg']:.4f}")

    print()
    print("  --- Theta gradient analysis ---")
    g_theta_np = np.array(g_theta)
    g_theta_norms = np.linalg.norm(g_theta_np, axis=1)
    print(f"  Outlier edge theta grad norm: {g_theta_norms[outlier_idx]:.6f}")
    print(f"  Mean inlier theta grad norm:  {np.mean(np.concatenate([g_theta_norms[:outlier_idx], g_theta_norms[outlier_idx+1:]])):.6f}")
    print(f"  Max theta grad norm:          {np.max(g_theta_norms):.6f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
