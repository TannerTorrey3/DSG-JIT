# experiments/exp30_joint_denoise_reweight.py
"""
Joint measurement denoising and per-edge weight learning via bilevel optimisation.

Combines the best of exp28 (measurement correction) and exp29 (learned weights)
into a single bilevel loop.  The outer optimizer jointly learns:
  - delta_theta: additive corrections to noisy measurements
  - log_weights: per-edge confidence weights for the inner solver

The inner manifold-aware Gauss-Newton solver uses SPARSE anchors (every 25 poses)
while the outer loss is evaluated at DENSE HELD-OUT anchors (every 5 poses).
This split provides real gradient signal: the solver must generalise beyond
the anchors it sees, and the learned weights + corrected measurements must
help it do so.

Key improvements over exp28/exp29:
  - Joint optimisation: weights tell the denoiser which edges matter;
    corrected measurements stabilise weight learning.
  - Split anchors: outer loss at held-out positions, not redundant with inner.
  - Information-weighted smoothness: translation and rotation contribute
    proportionally to their noise levels.
  - Correction parameterisation: delta_theta centered at zero is better
    conditioned for Adam than raw theta.
  - Taper-weighted overlap: Hann window blending instead of naive averaging.

Usage:
    # Synthetic (default 500 poses)
    python -m experiments.exp30_joint_denoise_reweight

    # KITTI
    python -m experiments.exp30_joint_denoise_reweight \\
        --kitti-root /path/to/kitti/odometry --seq 07

    # Force CPU
    JAX_PLATFORM_NAME=cpu python -m experiments.exp30_joint_denoise_reweight --max-frames 51
"""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.core.math3d import (
    compose_pose_se3,
    relative_pose_se3,
    se3_retract_left,
)
from dsg_jit.slam.measurements import sigma_to_weight


# ---------------------------------------------------------------------------
# GPU / device setup
# ---------------------------------------------------------------------------

def _print_device_info():
    devices = jax.devices()
    print(f"  JAX devices:    {[str(d) for d in devices]}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("  WARNING: running on CPU. Set JAX_PLATFORM_NAME=gpu or "
              "install jax[cuda] for GPU acceleration.")


# ---------------------------------------------------------------------------
# Data loading (shared with exp28/29)
# ---------------------------------------------------------------------------

def load_kitti_poses(kitti_root: str, seq: str, max_frames: int | None = None):
    try:
        from dsg_jit.datasets.kitti_odometry import load_kitti_odometry_sequence
        from dsg_jit.datasets.kitti_utils import kitti_frames_to_poses6d
        frames = load_kitti_odometry_sequence(
            kitti_root, seq, load_right=False, load_velodyne=False,
            with_poses=True, max_frames=max_frames)
        if not frames or frames[0].T_w_cam0 is None:
            return None, None
        poses = kitti_frames_to_poses6d(frames)
        return poses, {"source": f"kitti_seq{seq}", "n_frames": len(frames)}
    except (FileNotFoundError, Exception) as e:
        print(f"  Could not load KITTI data: {e}")
        return None, None


def generate_kitti_like_trajectory(n_poses: int) -> tuple[jnp.ndarray, dict]:
    """Constant-speed curved trajectory with KITTI-like characteristics."""
    poses = []
    x, y, heading = 0.0, 0.0, 0.0
    curvature = np.full(n_poses, 0.002)
    for frac in [0.25, 0.5, 0.75]:
        s = int(frac * n_poses)
        curvature[s:s + min(50, n_poses // 10)] = 0.02
    for i in range(n_poses):
        poses.append(jnp.array([x, y, 0.0, 0.0, 0.0, heading],
                               dtype=jnp.float32))
        x += float(jnp.cos(heading))
        y += float(jnp.sin(heading))
        heading += curvature[i]
    return jnp.stack(poses), {"source": "synthetic", "n_frames": n_poses}


# ---------------------------------------------------------------------------
# Metrics (same as exp28/29)
# ---------------------------------------------------------------------------

def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray) -> dict:
    te = np.linalg.norm(estimated[:, :3] - ground_truth[:, :3], axis=1)
    re = np.linalg.norm(estimated[:, 3:] - ground_truth[:, 3:], axis=1)
    return {
        "ate_trans_rmse": float(np.sqrt(np.mean(te ** 2))),
        "ate_trans_mean": float(np.mean(te)),
        "ate_trans_max": float(np.max(te)),
        "ate_rot_rmse": float(np.sqrt(np.mean(re ** 2))),
        "ate_rot_mean": float(np.mean(re)),
    }


def compute_rpe(estimated: np.ndarray, ground_truth: np.ndarray) -> dict:
    n = estimated.shape[0]
    te, re = [], []
    for i in range(n - 1):
        rel_e = np.array(relative_pose_se3(
            jnp.array(estimated[i]), jnp.array(estimated[i + 1])))
        rel_g = np.array(relative_pose_se3(
            jnp.array(ground_truth[i]), jnp.array(ground_truth[i + 1])))
        d = rel_e - rel_g
        te.append(np.linalg.norm(d[:3]))
        re.append(np.linalg.norm(d[3:]))
    te, re = np.array(te), np.array(re)
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(te ** 2))),
        "rpe_trans_mean": float(np.mean(te)),
        "rpe_rot_rmse": float(np.sqrt(np.mean(re ** 2))),
        "rpe_rot_mean": float(np.mean(re)),
    }


def compute_kitti_metric(estimated: np.ndarray, ground_truth: np.ndarray,
                         step_size: int = 10) -> dict:
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    n = estimated.shape[0]
    t_errs, r_errs = [], []
    for start in range(0, n, step_size):
        for length in lengths:
            end = start + length
            if end >= n:
                continue
            gt_rel = np.array(relative_pose_se3(
                jnp.array(ground_truth[start]), jnp.array(ground_truth[end])))
            est_rel = np.array(relative_pose_se3(
                jnp.array(estimated[start]), jnp.array(estimated[end])))
            path_len = float(np.linalg.norm(gt_rel[:3]))
            if path_len < 1.0:
                continue
            t_errs.append(float(np.linalg.norm(est_rel[:3] - gt_rel[:3]))
                          / path_len * 100.0)
            r_errs.append(np.degrees(float(np.linalg.norm(est_rel[3:] - gt_rel[3:])))
                          / path_len)
    if not t_errs:
        return {"kitti_trans_err_pct": 0.0, "kitti_rot_err_degm": 0.0}
    return {
        "kitti_trans_err_pct": float(np.mean(t_errs)),
        "kitti_rot_err_degm": float(np.mean(r_errs)),
    }


def reconstruct_trajectory(first_pose, measurements):
    poses = [first_pose]
    for k in range(measurements.shape[0]):
        poses.append(compose_pose_se3(poses[-1], measurements[k]))
    return np.array(jnp.stack(poses))


# ---------------------------------------------------------------------------
# Adam optimiser (minimal, no external dependency)
# ---------------------------------------------------------------------------

def adam_init(theta: jnp.ndarray):
    return (jnp.zeros_like(theta), jnp.zeros_like(theta), 0)


def adam_step(grad, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = b1 * m + (1.0 - b1) * grad
    v = b2 * v + (1.0 - b2) * grad ** 2
    m_hat = m / (1.0 - b1 ** t)
    v_hat = v / (1.0 - b2 ** t)
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return update, (m, v, t)


# ---------------------------------------------------------------------------
# Joint bilevel denoiser + weight learner
# ---------------------------------------------------------------------------

def build_joint_denoiser(
    n_poses: int,
    inner_anchor_positions: list[int],
    outer_anchor_positions: list[int],
    sigma: jnp.ndarray,
    *,
    gn_iters: int = 10,
    gn_damping: float = 5e-3,
    aw_trans: float = 10.0,
    aw_rot: float = 10.0,
    rw_trans: float = 0.1,
    rw_rot: float = 0.1,
    sw_trans: float = 0.0,
    sw_rot: float = 0.0,
    weight_reg: float = 0.001,
):
    """Build a JIT-compiled joint denoiser and weight learner.

    The inner GN solver uses sparse anchors and per-edge learned weights.
    The outer loss evaluates at held-out dense anchors, forcing the learned
    corrections and weights to generalise.

    Translation and rotation components have **independent** loss weights
    so each can be tuned without interfering with the other.

    Parameters
    ----------
    n_poses : int
        Number of poses in the window.
    inner_anchor_positions : list[int]
        Sparse anchor indices used by the inner GN solver.
    outer_anchor_positions : list[int]
        Dense anchor indices used ONLY in the outer loss (held out).
    sigma : jnp.ndarray
        Noise standard deviations (6,).
    gn_iters : int
        Inner Gauss-Newton iterations.
    aw_trans, aw_rot : float
        Anchor loss weight for translation / rotation components.
    rw_trans, rw_rot : float
        Measurement correction regularisation for translation / rotation.
    sw_trans, sw_rot : float
        Temporal smoothness weight for translation / rotation.
        Default 0 — the inner GN already smooths implicitly; explicit
        smoothness on measurements destroys RPE.
    weight_reg : float
        Regularisation pulling log-weights toward zero (uniform).

    Returns
    -------
    grad_fn, loss_fn : callables
        grad_fn(delta_theta, log_weights, x_init,
                inner_targets, outer_targets, noisy_meas) -> (grad_dt, grad_lw)
        loss_fn(...) -> scalar
    """
    n_meas = n_poses - 1
    base_odom_w = sigma_to_weight(sigma)           # (6,) baseline information
    anchor_w = sigma_to_weight(jnp.full(6, 0.01))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    inner_idx = jnp.array(inner_anchor_positions, dtype=jnp.int32)
    outer_idx = jnp.array(outer_anchor_positions, dtype=jnp.int32)

    # Per-component outer loss weights: [aw_trans]*3 + [aw_rot]*3
    anchor_w_vec = jnp.array(
        [aw_trans] * 3 + [aw_rot] * 3, dtype=jnp.float32)
    reg_w_vec = jnp.array(
        [rw_trans] * 3 + [rw_rot] * 3, dtype=jnp.float32)
    smooth_w_vec = jnp.array(
        [sw_trans] * 3 + [sw_rot] * 3, dtype=jnp.float32)

    # Vectorised residual with per-edge learned weights.
    def _odom_res_single(pose_a, pose_b, meas, log_w):
        w = jnp.exp(log_w)
        sqrt_info = jnp.sqrt(base_odom_w * w)
        return (relative_pose_se3(pose_a, pose_b) - meas) * sqrt_info

    _odom_res_batch = jax.vmap(_odom_res_single)

    def residual_fn(x, theta, log_weights, inner_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_res_batch(
            poses[:-1], poses[1:], theta, log_weights)
        r_anch = (poses[inner_idx] - inner_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    _retract_batch = jax.vmap(se3_retract_left)
    max_step_per_pose = 0.5

    def gn_step(x, theta, log_weights, inner_targets):
        def r_fn(x_):
            return residual_fn(x_, theta, log_weights, inner_targets)
        r = r_fn(x)
        J = jax.jacobian(r_fn)(x)
        n = x.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        delta = jnp.linalg.solve(H, J.T @ r)

        poses = x.reshape(n_poses, 6)
        deltas = delta.reshape(n_poses, 6)
        norms = jnp.linalg.norm(deltas, axis=1, keepdims=True)
        scales = jnp.minimum(1.0, max_step_per_pose / (norms + 1e-9))
        deltas = deltas * scales
        new_poses = _retract_batch(poses, -deltas)
        return new_poses.ravel()

    def outer_loss(delta_theta, log_weights, x_init,
                   inner_targets, outer_targets, noisy_meas):
        theta = noisy_meas + delta_theta  # corrected measurements

        # Inner solve: GN with learned weights + sparse inner anchors.
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, theta, log_weights, inner_targets)

        poses_opt = x.reshape(n_poses, 6)

        # 1. Validation anchor loss at HELD-OUT positions.
        #    Translation and rotation weighted independently.
        diffs = poses_opt[outer_idx] - outer_targets      # (n_outer, 6)
        a_loss = jnp.sum(anchor_w_vec * diffs ** 2)

        # 2. Measurement correction regularisation.
        #    Penalise large corrections; trans/rot independent.
        r_loss = jnp.sum(reg_w_vec * delta_theta ** 2)

        # 3. Temporal smoothness (default OFF — inner GN smooths implicitly).
        #    When enabled, penalises large changes between consecutive corrected
        #    measurements.  Only useful if measurements have structured noise
        #    (e.g., periodic bias).
        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(smooth_w_vec * s_diffs ** 2)

        # 4. Weight regularisation (toward uniform, exp(0) = 1).
        w_loss = jnp.sum(log_weights ** 2)

        return a_loss + r_loss + s_loss + weight_reg * w_loss

    grad_fn = jax.jit(jax.grad(outer_loss, argnums=(0, 1)))
    loss_fn = jax.jit(outer_loss)
    return grad_fn, loss_fn


# ---------------------------------------------------------------------------
# PGO solvers for downstream evaluation
# ---------------------------------------------------------------------------

def build_pgo_solver(n_poses, anchor_indices, sigma, gn_iters=20, damping=1e-3):
    """Standard PGO with uniform weights (baseline)."""
    odom_w = sigma_to_weight(sigma)
    anchor_w = sigma_to_weight(jnp.full(6, 0.01))
    sqrt_odom_w = jnp.sqrt(odom_w)
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_indices, dtype=jnp.int32)

    _odom_single = jax.vmap(
        lambda a, b, m: (relative_pose_se3(a, b) - m) * sqrt_odom_w)
    _retract_batch = jax.vmap(se3_retract_left)

    def residual_fn(x, measurements, anchor_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_single(poses[:-1], poses[1:], measurements)
        r_anch = (poses[anchor_idx] - anchor_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step(x, measurements, anchor_targets):
        def r_fn(x_):
            return residual_fn(x_, measurements, anchor_targets)
        r = r_fn(x)
        J = jax.jacobian(r_fn)(x)
        n = x.shape[0]
        H = J.T @ J + damping * jnp.eye(n)
        delta = jnp.linalg.solve(H, J.T @ r)
        poses = x.reshape(n_poses, 6)
        deltas = delta.reshape(n_poses, 6)
        norms = jnp.linalg.norm(deltas, axis=1, keepdims=True)
        scales = jnp.minimum(1.0, 0.5 / (norms + 1e-9))
        deltas = deltas * scales
        return _retract_batch(poses, -deltas).ravel()

    @jax.jit
    def solve(measurements, x_init, anchor_targets):
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, measurements, anchor_targets)
        return x

    return solve


def build_weighted_pgo_solver(n_poses, anchor_indices, sigma,
                               gn_iters=20, damping=1e-3):
    """PGO solver that uses per-edge learned weights."""
    base_odom_w = sigma_to_weight(sigma)
    anchor_w = sigma_to_weight(jnp.full(6, 0.01))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_indices, dtype=jnp.int32)

    def _odom_single(pose_a, pose_b, meas, log_w):
        w = jnp.exp(log_w)
        sqrt_info = jnp.sqrt(base_odom_w * w)
        return (relative_pose_se3(pose_a, pose_b) - meas) * sqrt_info

    _odom_batch = jax.vmap(_odom_single)
    _retract_batch = jax.vmap(se3_retract_left)

    def residual_fn(x, measurements, log_weights, anchor_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_batch(poses[:-1], poses[1:], measurements, log_weights)
        r_anch = (poses[anchor_idx] - anchor_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step(x, measurements, log_weights, anchor_targets):
        def r_fn(x_):
            return residual_fn(x_, measurements, log_weights, anchor_targets)
        r = r_fn(x)
        J = jax.jacobian(r_fn)(x)
        n = x.shape[0]
        H = J.T @ J + damping * jnp.eye(n)
        delta = jnp.linalg.solve(H, J.T @ r)
        poses = x.reshape(n_poses, 6)
        deltas = delta.reshape(n_poses, 6)
        norms = jnp.linalg.norm(deltas, axis=1, keepdims=True)
        scales = jnp.minimum(1.0, 0.5 / (norms + 1e-9))
        deltas = deltas * scales
        return _retract_batch(poses, -deltas).ravel()

    @jax.jit
    def solve(measurements, log_weights, x_init, anchor_targets):
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, measurements, log_weights, anchor_targets)
        return x

    return solve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Joint measurement denoising + weight learning")
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--seq", type=str, default="07")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=50,
                        help="Poses per solve window (default: 50)")
    parser.add_argument("--inner-spacing", type=int, default=25,
                        help="Inner (solver) anchor spacing (default: 25)")
    parser.add_argument("--outer-spacing", type=int, default=5,
                        help="Outer (validation) anchor spacing (default: 5)")
    parser.add_argument("--sigma-trans", type=float, default=0.10)
    parser.add_argument("--sigma-rot", type=float, default=0.05)
    parser.add_argument("--lr-meas", type=float, default=1e-3,
                        help="Adam LR for measurement corrections (default: 1e-3)")
    parser.add_argument("--lr-weight", type=float, default=5e-3,
                        help="Adam LR for log-weights (default: 5e-3)")
    parser.add_argument("--n-outer-iters", type=int, default=200,
                        help="Outer Adam iterations per window (default: 200)")
    parser.add_argument("--gn-iters", type=int, default=10,
                        help="Inner GN iterations (default: 10)")
    parser.add_argument("--aw-trans", type=float, default=10.0,
                        help="Anchor weight, translation (default: 10.0)")
    parser.add_argument("--aw-rot", type=float, default=10.0,
                        help="Anchor weight, rotation (default: 10.0)")
    parser.add_argument("--rw-trans", type=float, default=0.1,
                        help="Measurement reg weight, translation (default: 0.1)")
    parser.add_argument("--rw-rot", type=float, default=0.1,
                        help="Measurement reg weight, rotation (default: 0.1)")
    parser.add_argument("--sw-trans", type=float, default=0.0,
                        help="Smoothness weight, translation (default: 0.0)")
    parser.add_argument("--sw-rot", type=float, default=0.0,
                        help="Smoothness weight, rotation (default: 0.0)")
    parser.add_argument("--wreg", type=float, default=0.001,
                        help="Weight regularisation (default: 0.001)")
    parser.add_argument("--pgo-spacing", type=int, default=50,
                        help="PGO eval anchor spacing (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="exp30_results.json")
    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)
    window_size = args.window_size
    overlap = min(10, window_size // 5)
    stride = window_size - overlap

    print("=" * 70)
    print("  exp30 -- Joint Measurement Denoising + Weight Learning")
    print("=" * 70)
    _print_device_info()
    print()

    # ---- Load or generate trajectory ----
    gt_poses, data_info = None, None
    if args.kitti_root:
        print(f"Loading KITTI sequence {args.seq}...", flush=True)
        gt_poses, data_info = load_kitti_poses(
            args.kitti_root, args.seq, args.max_frames)

    if gt_poses is None:
        n_synth = args.max_frames or 500
        print(f"Generating synthetic trajectory ({n_synth} poses)...", flush=True)
        gt_poses, data_info = generate_kitti_like_trajectory(n_synth)

    n_poses_total = gt_poses.shape[0]
    gt_np = np.array(gt_poses)
    n_meas_total = n_poses_total - 1

    # ---- Compute measurements and add noise ----
    print("Computing relative measurements...", flush=True)
    gt_measurements = jnp.stack([
        relative_pose_se3(gt_poses[i], gt_poses[i + 1])
        for i in range(n_meas_total)
    ])

    key = jax.random.PRNGKey(args.seed)
    noise = jax.random.normal(key, shape=gt_measurements.shape) * sigma
    noisy_measurements = gt_measurements + noise

    # ---- Plan windows ----
    actual_window = min(window_size, n_poses_total)
    if n_poses_total <= window_size:
        windows = [(0, n_poses_total)]
    else:
        windows = []
        for start in range(0, n_poses_total - actual_window + 1, stride):
            windows.append((start, start + actual_window))
        if windows[-1][1] < n_poses_total:
            windows.append((n_poses_total - actual_window, n_poses_total))

    # Split anchors: inner (sparse, for GN stability) and outer (dense, held-out).
    all_anchors_in_window = list(range(0, actual_window, args.outer_spacing))
    if all_anchors_in_window[-1] != actual_window - 1:
        all_anchors_in_window.append(actual_window - 1)

    inner_anchors = list(range(0, actual_window, args.inner_spacing))
    if inner_anchors[-1] != actual_window - 1:
        inner_anchors.append(actual_window - 1)

    outer_anchors = [a for a in all_anchors_in_window if a not in inner_anchors]

    # PGO anchor positions (global, sparse — for downstream eval).
    pgo_anchor_indices = list(range(0, n_poses_total, args.pgo_spacing))
    if pgo_anchor_indices[-1] != n_poses_total - 1:
        pgo_anchor_indices.append(n_poses_total - 1)

    anchor_density = len(all_anchors_in_window) / actual_window * 100

    traj_len = float(np.sum(np.linalg.norm(
        np.diff(gt_np[:, :3], axis=0), axis=1)))

    print(f"  Data source:    {data_info['source']}")
    print(f"  Poses:          {n_poses_total}")
    print(f"  Trajectory len: {traj_len:.1f} m")
    print(f"  Noise:          sigma_t={args.sigma_trans}m, sigma_r={args.sigma_rot}rad")
    print(f"  Window size:    {actual_window} poses, stride={stride}, overlap={overlap}")
    print(f"  Windows:        {len(windows)}")
    print(f"  Anchors/window: {len(all_anchors_in_window)} total "
          f"({len(inner_anchors)} inner + {len(outer_anchors)} outer)")
    print(f"  Anchor density: {anchor_density:.1f}% (learning)")
    print(f"  PGO anchors:    {len(pgo_anchor_indices)} (global, spacing={args.pgo_spacing})")
    print(f"  Weights:        aw_t={args.aw_trans}, aw_r={args.aw_rot}, "
          f"rw_t={args.rw_trans}, rw_r={args.rw_rot}, "
          f"sw_t={args.sw_trans}, sw_r={args.sw_rot}, wreg={args.wreg}")
    print(f"  Inner GN iters: {args.gn_iters}")
    print(f"  Outer iters:    {args.n_outer_iters} "
          f"(Adam, lr_meas={args.lr_meas}, lr_weight={args.lr_weight})")
    print()

    # ---- Baseline ----
    noisy_poses_np = reconstruct_trajectory(gt_poses[0], noisy_measurements)
    meas_err_baseline = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))
    ate_noisy = compute_ate(noisy_poses_np, gt_np)
    print(f"Baseline (dead reckoning):")
    print(f"  ATE trans RMSE:  {ate_noisy['ate_trans_rmse']:.4f} m")
    print(f"  Meas error:      {meas_err_baseline:.4f}")
    print()

    # ---- Build and JIT-compile joint denoiser ----
    print(f"JIT-compiling joint denoiser ({actual_window} poses, "
          f"{len(inner_anchors)} inner + {len(outer_anchors)} outer anchors)...",
          flush=True)
    t_jit_start = time.perf_counter()

    grad_fn, loss_fn = build_joint_denoiser(
        actual_window, inner_anchors, outer_anchors, sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        aw_trans=args.aw_trans, aw_rot=args.aw_rot,
        rw_trans=args.rw_trans, rw_rot=args.rw_rot,
        sw_trans=args.sw_trans, sw_rot=args.sw_rot,
        weight_reg=args.wreg)

    # Warm-up with dummy data.
    n_meas_window = actual_window - 1
    dummy_dt = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_lw = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_inner = jnp.zeros((len(inner_anchors), 6), dtype=jnp.float32)
    dummy_outer = jnp.zeros((len(outer_anchors), 6), dtype=jnp.float32)
    dummy_meas = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    g_dt, g_lw = grad_fn(dummy_dt, dummy_lw, dummy_x,
                          dummy_inner, dummy_outer, dummy_meas)
    g_dt.block_until_ready()
    g_lw.block_until_ready()

    t_jit = time.perf_counter() - t_jit_start
    print(f"JIT compilation: {t_jit:.1f}s")
    print()

    # ---- Joint optimisation window by window ----
    print("Joint denoising + weight learning...", flush=True)
    t_learn_start = time.perf_counter()

    # Accumulators for taper-weighted overlap blending.
    meas_accum = np.zeros((n_meas_total, 6), dtype=np.float64)
    weight_accum = np.zeros((n_meas_total, 6), dtype=np.float64)
    taper_accum = np.zeros(n_meas_total, dtype=np.float64)

    for wi, (w_start, w_end) in enumerate(windows):
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1

        # Extract window measurements (fixed — never modified by inner solver).
        w_noisy = jnp.array(noisy_measurements[w_start:w_start + w_n_meas])

        # Anchor targets in LOCAL coordinates.
        gt_first = gt_poses[w_start]
        w_inner_targets = jnp.stack([
            relative_pose_se3(gt_first, gt_poses[w_start + p])
            for p in inner_anchors
        ])
        w_outer_targets = jnp.stack([
            relative_pose_se3(gt_first, gt_poses[w_start + p])
            for p in outer_anchors
        ])

        # x_init: forward-compose from origin using noisy measurements.
        origin = jnp.zeros(6, dtype=jnp.float32)
        init_poses = [origin]
        for k in range(w_n_meas):
            init_poses.append(compose_pose_se3(init_poses[-1], w_noisy[k]))
        x_init = jnp.concatenate(init_poses)

        # Initialise: zero corrections, uniform weights.
        delta_theta = jnp.zeros((w_n_meas, 6), dtype=jnp.float32)
        log_w = jnp.zeros((w_n_meas, 6), dtype=jnp.float32)
        adam_state_dt = adam_init(delta_theta)
        adam_state_lw = adam_init(log_w)

        initial_loss = float(loss_fn(delta_theta, log_w, x_init,
                                      w_inner_targets, w_outer_targets, w_noisy))

        for it in range(args.n_outer_iters):
            g_dt, g_lw = grad_fn(delta_theta, log_w, x_init,
                                  w_inner_targets, w_outer_targets, w_noisy)
            g_dt.block_until_ready()

            if jnp.any(jnp.isnan(g_dt)) or jnp.any(jnp.isnan(g_lw)):
                print(f"  Window {wi}: NaN gradient at iter {it}, stopping early")
                break

            update_dt, adam_state_dt = adam_step(g_dt, adam_state_dt, lr=args.lr_meas)
            update_lw, adam_state_lw = adam_step(g_lw, adam_state_lw, lr=args.lr_weight)
            delta_theta = delta_theta - update_dt
            log_w = log_w - update_lw

        # Taper window: Hann-like weighting (center of window gets full weight).
        taper = np.array(
            0.5 * (1.0 - jnp.cos(2.0 * math.pi * jnp.arange(w_n_meas)
                                   / max(w_n_meas, 1))))

        # Accumulate taper-weighted results.
        theta_corrected = np.array(w_noisy + delta_theta)
        log_w_np = np.array(log_w)
        for i in range(w_n_meas):
            gi = w_start + i
            if gi < n_meas_total:
                tw = taper[i]
                meas_accum[gi] += tw * theta_corrected[i]
                weight_accum[gi] += tw * log_w_np[i]
                taper_accum[gi] += tw

        elapsed = time.perf_counter() - t_learn_start
        final_loss = float(loss_fn(delta_theta, log_w, x_init,
                                    w_inner_targets, w_outer_targets, w_noisy))
        w_corr_norm = float(jnp.mean(jnp.linalg.norm(delta_theta, axis=1)))
        w_mean_weight = float(jnp.mean(jnp.exp(log_w)))
        gt_meas_window = gt_measurements[w_start:w_start + w_n_meas]
        w_meas_err = float(jnp.mean(jnp.linalg.norm(
            w_noisy + delta_theta - gt_meas_window, axis=1)))
        w_orig_err = float(jnp.mean(jnp.linalg.norm(
            w_noisy - gt_meas_window, axis=1)))
        loss_reduction = (1 - final_loss / max(initial_loss, 1e-9)) * 100
        print(f"  Window {wi+1}/{len(windows)} "
              f"(poses {w_start}-{w_end-1}): "
              f"loss {initial_loss:.1f}->{final_loss:.1f} ({loss_reduction:.1f}%), "
              f"meas {w_orig_err:.4f}->{w_meas_err:.4f} "
              f"({(1 - w_meas_err/max(w_orig_err, 1e-9))*100:.1f}%), "
              f"corr={w_corr_norm:.4f}, "
              f"w_mean={w_mean_weight:.3f}, "
              f"elapsed={elapsed:.1f}s", flush=True)

    # Normalise taper-weighted accumulators.
    denoised_measurements = np.array(noisy_measurements).copy()
    global_log_weights = np.zeros((n_meas_total, 6), dtype=np.float32)
    for i in range(n_meas_total):
        if taper_accum[i] > 0:
            denoised_measurements[i] = meas_accum[i] / taper_accum[i]
            global_log_weights[i] = weight_accum[i] / taper_accum[i]

    t_learn = time.perf_counter() - t_learn_start
    denoised_meas_jnp = jnp.array(denoised_measurements)
    learned_log_weights = jnp.array(global_log_weights)
    learned_weights = jnp.exp(learned_log_weights)

    # ---- Measurement error ----
    meas_err_after = float(jnp.mean(jnp.linalg.norm(
        denoised_meas_jnp - gt_measurements, axis=1)))
    meas_trans_before = float(jnp.mean(jnp.linalg.norm(
        (noisy_measurements - gt_measurements)[:, :3], axis=1)))
    meas_trans_after = float(jnp.mean(jnp.linalg.norm(
        (denoised_meas_jnp - gt_measurements)[:, :3], axis=1)))
    meas_rot_before = float(jnp.mean(jnp.linalg.norm(
        (noisy_measurements - gt_measurements)[:, 3:], axis=1)))
    meas_rot_after = float(jnp.mean(jnp.linalg.norm(
        (denoised_meas_jnp - gt_measurements)[:, 3:], axis=1)))

    print(f"\nJoint optimisation complete: {len(windows)} windows, {t_learn:.1f}s")
    print(f"  Meas error: {meas_err_baseline:.4f} -> {meas_err_after:.4f} "
          f"({(1 - meas_err_after/meas_err_baseline)*100:.1f}%)")
    print(f"  Trans:      {meas_trans_before:.4f} -> {meas_trans_after:.4f} "
          f"({(1 - meas_trans_after/meas_trans_before)*100:.1f}%)")
    print(f"  Rot:        {meas_rot_before:.4f} -> {meas_rot_after:.4f} "
          f"({(1 - meas_rot_after/meas_rot_before)*100:.1f}%)")

    # Weight statistics.
    print(f"\n  Learned weight stats:")
    print(f"    Mean:  {float(jnp.mean(learned_weights)):.4f}")
    print(f"    Std:   {float(jnp.std(learned_weights)):.4f}")
    print(f"    Min:   {float(jnp.min(learned_weights)):.4f}")
    print(f"    Max:   {float(jnp.max(learned_weights)):.4f}")
    comp_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    for c in range(6):
        wc = learned_weights[:, c]
        print(f"    {comp_names[c]}: mean={float(jnp.mean(wc)):.4f}, "
              f"std={float(jnp.std(wc)):.4f}")
    print()

    # ---- Downstream PGO evaluation (3-way comparison) ----
    print("Building JIT-compiled PGO solvers...", flush=True)

    pgo_uniform = build_pgo_solver(
        n_poses_total, pgo_anchor_indices, sigma, gn_iters=20, damping=1e-3)
    pgo_weighted = build_weighted_pgo_solver(
        n_poses_total, pgo_anchor_indices, sigma, gn_iters=20, damping=1e-3)

    # PGO init: forward-compose from GT first pose.
    init_poses_pgo = [gt_poses[0]]
    for k in range(n_meas_total):
        init_poses_pgo.append(compose_pose_se3(init_poses_pgo[-1],
                                                noisy_measurements[k]))
    x_init_pgo = jnp.concatenate(init_poses_pgo)
    anchor_targets_pgo = jnp.stack([gt_poses[i] for i in pgo_anchor_indices])

    # Warm up solvers.
    t_pgo_jit_start = time.perf_counter()
    _ = pgo_uniform(noisy_measurements, x_init_pgo,
                     anchor_targets_pgo).block_until_ready()
    uniform_log_w = jnp.zeros((n_meas_total, 6), dtype=jnp.float32)
    _ = pgo_weighted(noisy_measurements, uniform_log_w, x_init_pgo,
                      anchor_targets_pgo).block_until_ready()
    t_pgo_jit = time.perf_counter() - t_pgo_jit_start
    print(f"PGO JIT compile: {t_pgo_jit:.1f}s")

    # 1. PGO with noisy measurements + uniform weights (baseline).
    print("Running PGO: noisy + uniform weights...", flush=True)
    t0 = time.perf_counter()
    x_pgo_noisy = pgo_uniform(noisy_measurements, x_init_pgo,
                               anchor_targets_pgo).block_until_ready()
    t_pgo_noisy = time.perf_counter() - t0
    pgo_noisy_poses = np.array(x_pgo_noisy.reshape(n_poses_total, 6))

    # 2. PGO with denoised measurements + uniform weights.
    print("Running PGO: denoised + uniform weights...", flush=True)
    init_poses_den = [gt_poses[0]]
    for k in range(n_meas_total):
        init_poses_den.append(compose_pose_se3(init_poses_den[-1],
                                                denoised_meas_jnp[k]))
    x_init_den = jnp.concatenate(init_poses_den)
    t0 = time.perf_counter()
    x_pgo_den_uniform = pgo_uniform(denoised_meas_jnp, x_init_den,
                                     anchor_targets_pgo).block_until_ready()
    t_pgo_den_uniform = time.perf_counter() - t0
    pgo_den_uniform_poses = np.array(x_pgo_den_uniform.reshape(n_poses_total, 6))

    # 3. PGO with denoised measurements + learned weights.
    print("Running PGO: denoised + learned weights...", flush=True)
    t0 = time.perf_counter()
    x_pgo_den_weighted = pgo_weighted(denoised_meas_jnp, learned_log_weights,
                                       x_init_den,
                                       anchor_targets_pgo).block_until_ready()
    t_pgo_den_weighted = time.perf_counter() - t0
    pgo_den_weighted_poses = np.array(x_pgo_den_weighted.reshape(n_poses_total, 6))

    # ---- Compute all metrics ----
    results_pgo = {}
    for label, poses_arr in [
        ("pgo_noisy_uniform", pgo_noisy_poses),
        ("pgo_denoised_uniform", pgo_den_uniform_poses),
        ("pgo_denoised_weighted", pgo_den_weighted_poses),
    ]:
        ate = compute_ate(poses_arr, gt_np)
        rpe = compute_rpe(poses_arr, gt_np)
        kitti = compute_kitti_metric(poses_arr, gt_np)
        results_pgo[label] = {
            **{k: round(v, 6) for k, v in ate.items()},
            **{k: round(v, 6) for k, v in rpe.items()},
            **{k: round(v, 4) for k, v in kitti.items()},
        }

    # Weight-noise correlation analysis.
    noise_per_edge = np.array(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1))
    mean_weight_per_edge = np.array(jnp.mean(learned_weights, axis=1))
    corr = float(np.corrcoef(noise_per_edge, mean_weight_per_edge)[0, 1])

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print()
    print("--- Measurement Error ---")
    print(f"  {'':20s} {'Before':>10s} {'After':>10s} {'Improv.':>10s}")
    print(f"  {'Overall':20s} {meas_err_baseline:10.4f} {meas_err_after:10.4f} "
          f"{(1 - meas_err_after/meas_err_baseline)*100:9.1f}%")
    print(f"  {'Translation':20s} {meas_trans_before:10.4f} {meas_trans_after:10.4f} "
          f"{(1 - meas_trans_after/meas_trans_before)*100:9.1f}%")
    print(f"  {'Rotation':20s} {meas_rot_before:10.4f} {meas_rot_after:10.4f} "
          f"{(1 - meas_rot_after/meas_rot_before)*100:9.1f}%")

    print()
    print("--- Downstream: PGO Trajectory Quality (3-way comparison) ---")
    hdr = (f"  {'':20s} {'noisy+unif':>12s} {'denoised+unif':>14s} "
           f"{'denoised+wt':>12s} {'full improv':>12s}")
    print(hdr)
    for label, k in [
        ("ATE trans [m]", "ate_trans_rmse"),
        ("ATE rot [rad]", "ate_rot_rmse"),
        ("RPE trans [m]", "rpe_trans_rmse"),
        ("RPE rot [rad]", "rpe_rot_rmse"),
        ("KITTI trans [%]", "kitti_trans_err_pct"),
        ("KITTI rot [deg/m]", "kitti_rot_err_degm"),
    ]:
        v1 = results_pgo["pgo_noisy_uniform"][k]
        v2 = results_pgo["pgo_denoised_uniform"][k]
        v3 = results_pgo["pgo_denoised_weighted"][k]
        imp = (1 - v3 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:14.4f} {v3:12.4f} {imp:11.1f}%")

    print()
    print("--- Weight-Noise Correlation ---")
    print(f"  Pearson(noise_magnitude, learned_weight) = {corr:.4f}")
    print(f"  {'GOOD' if corr < -0.1 else 'WEAK' if abs(corr) < 0.1 else 'UNEXPECTED'}: "
          f"{'negative' if corr < 0 else 'positive'} correlation "
          f"({'noisy edges get lower weight' if corr < 0 else 'noisy edges get HIGHER weight'})")
    top10_noisy = np.argsort(noise_per_edge)[-10:]
    top10_clean = np.argsort(noise_per_edge)[:10]
    print(f"  Top-10 noisiest edges: mean weight = "
          f"{float(np.mean(mean_weight_per_edge[top10_noisy])):.4f}")
    print(f"  Top-10 cleanest edges: mean weight = "
          f"{float(np.mean(mean_weight_per_edge[top10_clean])):.4f}")

    print()
    print("--- Timing ---")
    print(f"  Joint denoiser JIT:  {t_jit:8.1f} s")
    print(f"  PGO JIT:             {t_pgo_jit:8.1f} s")
    print(f"  Joint optimisation:  {t_learn:8.1f} s  ({len(windows)} windows)")
    print(f"  PGO (noisy+uniform): {t_pgo_noisy:8.1f} s")
    print(f"  PGO (den+uniform):   {t_pgo_den_uniform:8.1f} s")
    print(f"  PGO (den+weighted):  {t_pgo_den_weighted:8.1f} s")
    print()

    # ---- Save results ----
    results = {
        "config": {
            "data_source": data_info["source"],
            "n_poses": n_poses_total,
            "trajectory_length_m": round(traj_len, 1),
            "window_size": actual_window,
            "stride": stride,
            "overlap": overlap,
            "n_windows": len(windows),
            "inner_spacing": args.inner_spacing,
            "outer_spacing": args.outer_spacing,
            "inner_anchors_per_window": len(inner_anchors),
            "outer_anchors_per_window": len(outer_anchors),
            "anchor_density_pct": round(anchor_density, 1),
            "n_pgo_anchors": len(pgo_anchor_indices),
            "sigma_trans": args.sigma_trans,
            "sigma_rot": args.sigma_rot,
            "gn_iters": args.gn_iters,
            "outer_iters": args.n_outer_iters,
            "lr_meas": args.lr_meas,
            "lr_weight": args.lr_weight,
            "aw_trans": args.aw_trans,
            "aw_rot": args.aw_rot,
            "rw_trans": args.rw_trans,
            "rw_rot": args.rw_rot,
            "sw_trans": args.sw_trans,
            "sw_rot": args.sw_rot,
            "weight_reg": args.wreg,
            "optimizer": "adam",
            "overlap_blending": "hann_taper",
        },
        "measurement_error": {
            "before": round(meas_err_baseline, 6),
            "after": round(meas_err_after, 6),
            "improvement_pct": round(
                (1 - meas_err_after / meas_err_baseline) * 100, 1),
            "trans_before": round(meas_trans_before, 6),
            "trans_after": round(meas_trans_after, 6),
            "rot_before": round(meas_rot_before, 6),
            "rot_after": round(meas_rot_after, 6),
        },
        "learned_weight_stats": {
            "mean": round(float(jnp.mean(learned_weights)), 4),
            "std": round(float(jnp.std(learned_weights)), 4),
            "min": round(float(jnp.min(learned_weights)), 4),
            "max": round(float(jnp.max(learned_weights)), 4),
            "noise_weight_correlation": round(corr, 4),
        },
        **{k: v for k, v in results_pgo.items()},
        "timing": {
            "joint_denoiser_jit_s": round(t_jit, 2),
            "pgo_jit_s": round(t_pgo_jit, 2),
            "joint_optimisation_s": round(t_learn, 2),
            "pgo_noisy_uniform_s": round(t_pgo_noisy, 2),
            "pgo_denoised_uniform_s": round(t_pgo_den_uniform, 2),
            "pgo_denoised_weighted_s": round(t_pgo_den_weighted, 2),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")

    return results


if __name__ == "__main__":
    main()
