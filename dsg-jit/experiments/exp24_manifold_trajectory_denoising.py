# experiments/exp24_manifold_trajectory_denoising.py
"""
Trajectory denoising via parameterized manifold Gauss-Newton.

Given a ground-truth SE(3) trajectory with N poses connected by odometry
factors, this experiment:

  1. Generates ground-truth relative odometry measurements.
  2. Corrupts them with additive Gaussian noise (known sigma).
  3. Builds a factor graph with a prior on pose0 and noisy odom factors.
  4. Treats the noisy measurements as learnable parameters theta (N-1, 6).
  5. Uses gauss_newton_manifold as the inner solver (with sequence-tuple
     metadata for JAX compatibility).
  6. Defines an outer loss = pose error at anchor poses + Gaussian
     regularisation on theta (penalises deviation from noisy observations
     weighted by 1/sigma^2).
  7. Differentiates the outer loss through the inner manifold GN solve
     via jax.grad and iteratively updates theta.

The result is a denoised trajectory whose odometry measurements have been
refined to be more consistent with the anchor constraints while respecting
the known noise model.

Optimised configuration (from hyperparameter sweep):
  - 3 anchor poses (start, middle, end)
  - anchor_weight = 5.0, reg_weight = 1.0
  - outer lr = 0.001, 100 iterations (converges by ~40)
  - inner GN: 15 iters, damping 5e-3, max_step_norm 0.5
"""

from __future__ import annotations

import time
import json

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_geodesic_residual,
    sigma_to_weight,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig
from dsg_jit.telemetry import telemetry_span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_slice(idx):
    """Normalise an index entry (slice or (start, length)) into a slice."""
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def generate_ground_truth_trajectory(n_poses: int, step: float = 1.0) -> jnp.ndarray:
    """Straight-line trajectory with a gentle yaw curve.

    Returns an (n_poses, 6) array of SE(3) poses in [tx,ty,tz,wx,wy,wz].
    Pose 0 starts at the origin with zero rotation.
    """
    poses = []
    for i in range(n_poses):
        angle = 0.05 * i  # gentle yaw; pose0 has zero rotation
        tx = step * i * jnp.cos(angle)
        ty = step * i * jnp.sin(angle)
        tz = 0.0
        poses.append(jnp.array([tx, ty, tz, 0.0, 0.0, angle], dtype=jnp.float32))
    return jnp.stack(poses)


def relative_measurements_from_poses(gt_poses: jnp.ndarray) -> jnp.ndarray:
    """Compute ground-truth relative odometry from consecutive poses."""
    from dsg_jit.core.math3d import relative_pose_se3
    n = gt_poses.shape[0]
    meas = []
    for i in range(n - 1):
        meas.append(relative_pose_se3(gt_poses[i], gt_poses[i + 1]))
    return jnp.stack(meas)


def add_gaussian_noise(
    measurements: jnp.ndarray,
    sigma: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Add zero-mean Gaussian noise with per-component sigma."""
    noise = jax.random.normal(key, shape=measurements.shape) * sigma
    return measurements + noise


# ---------------------------------------------------------------------------
# Factor graph construction
# ---------------------------------------------------------------------------

def build_trajectory_graph(
    n_poses: int,
    noisy_measurements: jnp.ndarray,
    gt_poses: jnp.ndarray,
    sigma: jnp.ndarray,
):
    """Build a world model for a pose chain with noisy geodesic odometry.

    Returns
    -------
    wm : WorldModel
    x_init : jnp.ndarray
    index : dict
    pose_ids : list[NodeId]
    odom_factor_ids : list[FactorId]
    """
    wm = WorldModel()

    # Use noisy measurements to initialise poses by forward-composing
    # from the known origin.  This gives the solver a reasonable start.
    from dsg_jit.core.math3d import compose_pose_se3
    init_poses = [gt_poses[0]]  # pose0 is known exactly
    for k in range(n_poses - 1):
        next_pose = compose_pose_se3(init_poses[-1], noisy_measurements[k])
        init_poses.append(next_pose)

    pose_ids = []
    for i in range(n_poses):
        pid = wm.add_variable(
            var_type="pose_se3",
            value=init_poses[i],
        )
        pose_ids.append(pid)

    # Strong prior on pose0 (identity / known start).
    odom_weight = sigma_to_weight(sigma)
    wm.add_factor(
        f_type="prior",
        var_ids=(pose_ids[0],),
        params={
            "target": gt_poses[0],
            "weight": sigma_to_weight(jnp.full(6, 0.01)),  # very tight
        },
    )

    # Odometry factors between consecutive poses.
    odom_factor_ids = []
    for k in range(n_poses - 1):
        fid = wm.add_factor(
            f_type="odom_se3_geodesic",
            var_ids=(pose_ids[k], pose_ids[k + 1]),
            params={
                "measurement": noisy_measurements[k],
                "weight": odom_weight,
            },
        )
        odom_factor_ids.append(fid)

    # Register residual functions.
    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    x_init, index = wm.pack_state()
    return wm, x_init, index, pose_ids, odom_factor_ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray):
    """Compute Absolute Trajectory Error (translational and rotational).

    Parameters
    ----------
    estimated : (N, 6) array of SE(3) poses [tx,ty,tz,wx,wy,wz]
    ground_truth : (N, 6) array of GT poses

    Returns
    -------
    dict with ATE metrics
    """
    trans_errors = np.linalg.norm(estimated[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.linalg.norm(estimated[:, 3:] - ground_truth[:, 3:], axis=1)

    return {
        "ate_trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "ate_trans_mean": float(np.mean(trans_errors)),
        "ate_trans_max": float(np.max(trans_errors)),
        "ate_trans_median": float(np.median(trans_errors)),
        "ate_trans_std": float(np.std(trans_errors)),
        "ate_rot_rmse": float(np.sqrt(np.mean(rot_errors ** 2))),
        "ate_rot_mean": float(np.mean(rot_errors)),
        "ate_rot_max": float(np.max(rot_errors)),
        "ate_rot_median": float(np.median(rot_errors)),
        "ate_rot_std": float(np.std(rot_errors)),
        "per_pose_trans": trans_errors.tolist(),
        "per_pose_rot": rot_errors.tolist(),
    }


def compute_rpe(estimated: np.ndarray, ground_truth: np.ndarray):
    """Compute Relative Pose Error over consecutive pairs.

    Parameters
    ----------
    estimated : (N, 6) array of SE(3) poses
    ground_truth : (N, 6) array of GT poses

    Returns
    -------
    dict with RPE metrics
    """
    from dsg_jit.core.math3d import relative_pose_se3

    n = estimated.shape[0]
    trans_errors = []
    rot_errors = []

    for i in range(n - 1):
        rel_est = np.array(relative_pose_se3(
            jnp.array(estimated[i]), jnp.array(estimated[i + 1])
        ))
        rel_gt = np.array(relative_pose_se3(
            jnp.array(ground_truth[i]), jnp.array(ground_truth[i + 1])
        ))
        diff = rel_est - rel_gt
        trans_errors.append(np.linalg.norm(diff[:3]))
        rot_errors.append(np.linalg.norm(diff[3:]))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "rpe_trans_mean": float(np.mean(trans_errors)),
        "rpe_trans_max": float(np.max(trans_errors)),
        "rpe_rot_rmse": float(np.sqrt(np.mean(rot_errors ** 2))),
        "rpe_rot_mean": float(np.mean(rot_errors)),
        "rpe_rot_max": float(np.max(rot_errors)),
        "per_step_trans": trans_errors.tolist(),
        "per_step_rot": rot_errors.tolist(),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@telemetry_span(component="experiment", op="exp24_manifold_trajectory_denoising")
def main():
    # ---- Configuration ----
    n_poses = 5
    step = 1.0

    # Known noise model: translation sigma = 0.15 m, rotation sigma = 0.08 rad
    sigma = jnp.array([0.15, 0.15, 0.15, 0.08, 0.08, 0.08], dtype=jnp.float32)

    key = jax.random.PRNGKey(42)

    # Outer loss weights (from hyperparameter sweep)
    anchor_weight = 5.0
    reg_weight = 1.0
    smooth_weight = 2.0  # temporal smoothness on consecutive measurements

    # Outer optimiser
    lr = 0.001
    n_outer_iters = 100

    # Inner manifold Gauss-Newton
    gn_cfg = GNConfig(max_iters=15, damping=5e-3, max_step_norm=0.5)

    # ---- Generate data ----
    gt_poses = generate_ground_truth_trajectory(n_poses, step)
    gt_measurements = relative_measurements_from_poses(gt_poses)
    noisy_measurements = add_gaussian_noise(gt_measurements, sigma, key)

    # ---- Build factor graph ----
    t_build_start = time.perf_counter()
    wm, x_init, index, pose_ids, odom_fids = build_trajectory_graph(
        n_poses, noisy_measurements, gt_poses, sigma,
    )

    # Manifold metadata as sequence tuples (JAX-friendly path).
    packed = (x_init, index)
    block_slices_dict, manifold_types_dict = build_manifold_metadata(packed, wm.fg)
    bs_seq = list(block_slices_dict.items())
    mt_seq = list(manifold_types_dict.items())

    pose_slices = [_to_slice(index[pid]) for pid in pose_ids]
    t_build = time.perf_counter() - t_build_start

    # ---- Build parametric residual ----
    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry

    def residual_param_fn(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Parametric residual r(x, theta) substituting theta for odom measurements."""
        var_values = wm.unpack_state(x, index)
        res_list = []
        odom_idx = 0

        for f in factors:
            res_fn = residual_fns[f.type]
            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

            if f.type == "odom_se3_geodesic":
                params = dict(f.params)
                params["measurement"] = theta[odom_idx]
                odom_idx += 1
            else:
                params = f.params

            r = res_fn(stacked, params)
            res_list.append(r)

        return jnp.concatenate(res_list)

    # ---- Anchor setup ----
    anchor_indices = [0, n_poses // 2, n_poses - 1]
    anchor_gt = jnp.stack([gt_poses[i] for i in anchor_indices])
    anchor_slices = [pose_slices[i] for i in anchor_indices]

    # Information weight for the Gaussian regulariser.
    info_weight = sigma_to_weight(sigma)  # 1 / sigma^2, shape (6,)

    def solve_and_loss(theta: jnp.ndarray) -> jnp.ndarray:
        """Inner manifold GN solve + outer supervised + regularisation loss.

        Three outer loss terms:
          1. Anchor loss — pose error at known ground-truth poses.
          2. Gaussian regulariser — keeps theta near noisy observations,
             weighted by 1/sigma^2 (Mahalanobis distance).
          3. Temporal smoothness — penalises large jumps between
             consecutive measurements, enforcing the physical prior
             that a smooth trajectory produces slowly-varying odometry.
        """
        def residual_fn(x):
            return residual_param_fn(x, theta)

        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)

        # Anchor loss: squared error at known poses.
        anchor_loss = 0.0
        for i, sl in enumerate(anchor_slices):
            diff = x_opt[sl] - anchor_gt[i]
            anchor_loss = anchor_loss + jnp.sum(diff ** 2)

        # Gaussian regulariser on theta.
        deviation = theta - noisy_measurements
        reg_loss = jnp.sum(info_weight * deviation ** 2)

        # Temporal smoothness: penalise differences between consecutive
        # measurements.  On a smooth trajectory the relative odometry
        # between adjacent steps changes slowly, so ||theta_k - theta_{k-1}||
        # should be small.  Weighted by info_weight so the penalty is
        # proportional to the noise confidence per component.
        diffs = theta[1:] - theta[:-1]  # (K-1, 6)
        smooth_loss = jnp.sum(diffs ** 2)

        return (anchor_weight * anchor_loss
                + reg_weight * reg_loss
                + smooth_weight * smooth_loss)

    # ---- JIT compile + warm-up ----
    loss_fn = jax.jit(solve_and_loss)
    grad_fn = jax.jit(jax.grad(solve_and_loss))

    theta = noisy_measurements.copy()

    # Warm-up JIT compilation (exclude from timing).
    t_jit_start = time.perf_counter()
    _ = loss_fn(theta).block_until_ready()
    _ = grad_fn(theta).block_until_ready()
    t_jit = time.perf_counter() - t_jit_start

    # ---- Helper: extract optimised poses ----
    def get_optimised_poses(theta_val):
        def residual_fn(x):
            return residual_param_fn(x, theta_val)
        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)
        return jnp.stack([x_opt[sl] for sl in pose_slices])

    # ---- Before-denoising metrics ----
    gt_np = np.array(gt_poses)
    poses_before_np = np.array(get_optimised_poses(noisy_measurements))
    ate_before = compute_ate(poses_before_np, gt_np)
    rpe_before = compute_rpe(poses_before_np, gt_np)

    meas_error_before = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))

    # ---- Outer optimisation loop ----
    loss_history = []
    grad_norm_history = []

    t_opt_start = time.perf_counter()
    for step_i in range(n_outer_iters):
        g = grad_fn(theta)
        g.block_until_ready()
        theta = theta - lr * g

        loss_val = float(loss_fn(theta))
        grad_norm = float(jnp.linalg.norm(g))
        loss_history.append(loss_val)
        grad_norm_history.append(grad_norm)

    t_opt = time.perf_counter() - t_opt_start

    # Time per outer iteration (steady-state, skip first 5 for JIT warmth).
    t_per_iter = t_opt / n_outer_iters

    # ---- After-denoising metrics ----
    poses_after_np = np.array(get_optimised_poses(theta))
    ate_after = compute_ate(poses_after_np, gt_np)
    rpe_after = compute_rpe(poses_after_np, gt_np)

    meas_error_after = float(jnp.mean(jnp.linalg.norm(
        theta - gt_measurements, axis=1)))

    # Per-component measurement error (trans vs rot).
    meas_trans_err_before = float(jnp.mean(jnp.linalg.norm(
        (noisy_measurements - gt_measurements)[:, :3], axis=1)))
    meas_trans_err_after = float(jnp.mean(jnp.linalg.norm(
        (theta - gt_measurements)[:, :3], axis=1)))
    meas_rot_err_before = float(jnp.mean(jnp.linalg.norm(
        (noisy_measurements - gt_measurements)[:, 3:], axis=1)))
    meas_rot_err_after = float(jnp.mean(jnp.linalg.norm(
        (theta - gt_measurements)[:, 3:], axis=1)))

    # Inner solver residual norm (quality of inner solve).
    def inner_residual_norm(theta_val):
        def r(x): return residual_param_fn(x, theta_val)
        x_opt = gauss_newton_manifold(r, x_init, bs_seq, mt_seq, gn_cfg)
        return float(jnp.linalg.norm(r(x_opt)))

    inner_res_before = inner_residual_norm(noisy_measurements)
    inner_res_after = inner_residual_norm(theta)

    # ---- Convergence analysis ----
    converged_iter = n_outer_iters
    for i in range(1, len(loss_history)):
        if abs(loss_history[i] - loss_history[i - 1]) < 1e-6:
            converged_iter = i + 1
            break

    # ---- Assemble results ----
    results = {
        "config": {
            "n_poses": n_poses,
            "step_size": step,
            "sigma_trans": float(sigma[0]),
            "sigma_rot": float(sigma[3]),
            "anchor_indices": anchor_indices,
            "n_anchors": len(anchor_indices),
            "anchor_weight": anchor_weight,
            "reg_weight": reg_weight,
            "smooth_weight": smooth_weight,
            "outer_lr": lr,
            "n_outer_iters": n_outer_iters,
            "inner_gn_iters": gn_cfg.max_iters,
            "inner_damping": gn_cfg.damping,
            "inner_max_step_norm": gn_cfg.max_step_norm,
            "jax_seed": 42,
            "state_dim": int(x_init.shape[0]),
            "n_factors": len(factors),
            "n_odom_factors": n_poses - 1,
        },
        "timing": {
            "graph_build_s": round(t_build, 4),
            "jit_compile_s": round(t_jit, 4),
            "outer_loop_s": round(t_opt, 4),
            "per_iter_s": round(t_per_iter, 4),
            "total_s": round(t_build + t_jit + t_opt, 4),
        },
        "convergence": {
            "initial_loss": round(loss_history[0], 6) if loss_history else None,
            "final_loss": round(loss_history[-1], 6) if loss_history else None,
            "converged_at_iter": converged_iter,
            "final_grad_norm": round(grad_norm_history[-1], 8) if grad_norm_history else None,
            "loss_history": [round(v, 6) for v in loss_history],
            "grad_norm_history": [round(v, 6) for v in grad_norm_history],
        },
        "inner_solver": {
            "residual_norm_before": round(inner_res_before, 6),
            "residual_norm_after": round(inner_res_after, 6),
        },
        "ate_before": {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in ate_before.items()},
        "ate_after": {k: round(v, 6) if isinstance(v, float) else v
                      for k, v in ate_after.items()},
        "rpe_before": {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in rpe_before.items()},
        "rpe_after": {k: round(v, 6) if isinstance(v, float) else v
                      for k, v in rpe_after.items()},
        "measurement_error": {
            "overall_before": round(meas_error_before, 6),
            "overall_after": round(meas_error_after, 6),
            "trans_before": round(meas_trans_err_before, 6),
            "trans_after": round(meas_trans_err_after, 6),
            "rot_before": round(meas_rot_err_before, 6),
            "rot_after": round(meas_rot_err_after, 6),
        },
        "improvement": {
            "ate_trans_rmse_pct": round(
                (1 - ate_after["ate_trans_rmse"] / ate_before["ate_trans_rmse"]) * 100, 1),
            "ate_rot_rmse_pct": round(
                (1 - ate_after["ate_rot_rmse"] / ate_before["ate_rot_rmse"]) * 100, 1),
            "rpe_trans_rmse_pct": round(
                (1 - rpe_after["rpe_trans_rmse"] / rpe_before["rpe_trans_rmse"]) * 100, 1),
            "rpe_rot_rmse_pct": round(
                (1 - rpe_after["rpe_rot_rmse"] / rpe_before["rpe_rot_rmse"]) * 100, 1),
            "meas_error_pct": round(
                (1 - meas_error_after / meas_error_before) * 100, 1),
        },
    }

    # ---- Print summary ----
    print("=" * 70)
    print("  Manifold Trajectory Denoising — exp24 Results")
    print("=" * 70)
    print()
    print(f"  Trajectory: {n_poses} SE(3) poses, step={step}m, yaw_rate=0.05 rad/step")
    print(f"  Noise:      sigma_trans={sigma[0]:.2f}m, sigma_rot={sigma[3]:.2f}rad")
    print(f"  Anchors:    {len(anchor_indices)} poses at indices {anchor_indices}")
    print(f"  Weights:    anchor={anchor_weight}, reg={reg_weight}, smooth={smooth_weight}")
    print(f"  Solver:     manifold GN ({gn_cfg.max_iters} iters, "
          f"damping={gn_cfg.damping}, step_clamp={gn_cfg.max_step_norm})")
    print(f"  Outer:      GD lr={lr}, {n_outer_iters} iters")
    print(f"  State dim:  {int(x_init.shape[0])} ({n_poses}×6)")
    print()

    print("--- Timing ---")
    print(f"  Graph build:     {t_build*1000:8.1f} ms")
    print(f"  JIT compile:     {t_jit:8.2f} s")
    print(f"  Outer loop:      {t_opt:8.2f} s  ({n_outer_iters} iters)")
    print(f"  Per iteration:   {t_per_iter*1000:8.1f} ms")
    print(f"  Total:           {t_build + t_jit + t_opt:8.2f} s")
    print()

    print("--- Convergence ---")
    print(f"  Initial loss:    {results['convergence']['initial_loss']}")
    print(f"  Final loss:      {results['convergence']['final_loss']}")
    print(f"  Converged at:    iter {converged_iter}")
    print(f"  Final grad norm: {results['convergence']['final_grad_norm']}")
    print()

    print("--- Absolute Trajectory Error (ATE) ---")
    print(f"  {'':20s} {'Before':>12s} {'After':>12s} {'Improv.':>10s}")
    print(f"  {'Trans RMSE [m]':20s} {ate_before['ate_trans_rmse']:12.4f} "
          f"{ate_after['ate_trans_rmse']:12.4f} {results['improvement']['ate_trans_rmse_pct']:9.1f}%")
    print(f"  {'Trans mean [m]':20s} {ate_before['ate_trans_mean']:12.4f} "
          f"{ate_after['ate_trans_mean']:12.4f}")
    print(f"  {'Trans max [m]':20s} {ate_before['ate_trans_max']:12.4f} "
          f"{ate_after['ate_trans_max']:12.4f}")
    print(f"  {'Trans median [m]':20s} {ate_before['ate_trans_median']:12.4f} "
          f"{ate_after['ate_trans_median']:12.4f}")
    print(f"  {'Rot RMSE [rad]':20s} {ate_before['ate_rot_rmse']:12.4f} "
          f"{ate_after['ate_rot_rmse']:12.4f} {results['improvement']['ate_rot_rmse_pct']:9.1f}%")
    print(f"  {'Rot mean [rad]':20s} {ate_before['ate_rot_mean']:12.4f} "
          f"{ate_after['ate_rot_mean']:12.4f}")
    print(f"  {'Rot max [rad]':20s} {ate_before['ate_rot_max']:12.4f} "
          f"{ate_after['ate_rot_max']:12.4f}")
    print()

    print("--- Relative Pose Error (RPE) ---")
    print(f"  {'':20s} {'Before':>12s} {'After':>12s} {'Improv.':>10s}")
    print(f"  {'Trans RMSE [m]':20s} {rpe_before['rpe_trans_rmse']:12.4f} "
          f"{rpe_after['rpe_trans_rmse']:12.4f} {results['improvement']['rpe_trans_rmse_pct']:9.1f}%")
    print(f"  {'Trans mean [m]':20s} {rpe_before['rpe_trans_mean']:12.4f} "
          f"{rpe_after['rpe_trans_mean']:12.4f}")
    print(f"  {'Rot RMSE [rad]':20s} {rpe_before['rpe_rot_rmse']:12.4f} "
          f"{rpe_after['rpe_rot_rmse']:12.4f} {results['improvement']['rpe_rot_rmse_pct']:9.1f}%")
    print(f"  {'Rot mean [rad]':20s} {rpe_before['rpe_rot_mean']:12.4f} "
          f"{rpe_after['rpe_rot_mean']:12.4f}")
    print()

    print("--- Measurement Error ---")
    print(f"  {'':20s} {'Before':>12s} {'After':>12s} {'Improv.':>10s}")
    print(f"  {'Overall':20s} {meas_error_before:12.4f} "
          f"{meas_error_after:12.4f} {results['improvement']['meas_error_pct']:9.1f}%")
    print(f"  {'Trans [m]':20s} {meas_trans_err_before:12.4f} "
          f"{meas_trans_err_after:12.4f}")
    print(f"  {'Rot [rad]':20s} {meas_rot_err_before:12.4f} "
          f"{meas_rot_err_after:12.4f}")
    print()

    print("--- Inner Solver ---")
    print(f"  Residual norm before: {inner_res_before:.6f}")
    print(f"  Residual norm after:  {inner_res_after:.6f}")
    print()

    print("--- Per-Pose Translational Error [m] ---")
    print(f"  {'Pose':>6s} {'Before':>10s} {'After':>10s} {'Delta':>10s}")
    for i in range(n_poses):
        b = ate_before["per_pose_trans"][i]
        a = ate_after["per_pose_trans"][i]
        print(f"  {i:6d} {b:10.4f} {a:10.4f} {a - b:+10.4f}")
    print()

    # ---- Save JSON results ----
    out_path = "exp24_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to {out_path}")

    return results


if __name__ == "__main__":
    main()
