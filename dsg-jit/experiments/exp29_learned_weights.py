# experiments/exp29_learned_weights.py
"""
Learned per-measurement information weights via bilevel optimisation.

Instead of modifying measurement *values* (exp24–28), this experiment learns
per-edge *confidence weights* that tell the PGO solver which odometry
measurements to trust.  Measurements stay fixed — they can never get worse.

The bilevel structure:
  - Inner: manifold-aware Gauss-Newton on the pose graph, where each
    odometry residual is scaled by a learned per-edge weight vector.
  - Outer: Adam optimises the weights to minimise anchor loss at sparse
    GT positions (simulating GPS fixes).

Key insight: the learned weights themselves reveal the noise model —
edges with low weight are noisy.  PGO with these weights IS the
denoised result.

Usage:
    # Synthetic (default 500 poses, 2% anchors)
    python -m experiments.exp29_learned_weights

    # Custom anchor density
    python -m experiments.exp29_learned_weights --anchor-spacing 50

    # KITTI
    python -m experiments.exp29_learned_weights \\
        --kitti-root /path/to/kitti/odometry --seq 07
"""

from __future__ import annotations

import argparse
import json
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
# Data loading (shared with exp28)
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
# Metrics (same as exp28)
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
# Bilevel weight learner
# ---------------------------------------------------------------------------

def build_weight_learner(
    n_poses: int,
    inner_anchor_positions: list[int],
    outer_anchor_positions: list[int],
    sigma: jnp.ndarray,
    *,
    gn_iters: int = 10,
    gn_damping: float = 5e-3,
    anchor_weight: float = 10.0,
    weight_reg: float = 0.01,
):
    """Build a JIT-compiled bilevel weight learner with split anchors.

    The key insight: inner GN solver uses one set of anchors, outer loss
    evaluates at a DIFFERENT set (held out from the solver).  This forces
    the learned weights to help the solver generalise to unseen positions,
    providing actual gradient signal.

    Parameters
    ----------
    n_poses : int
        Number of poses in the window.
    inner_anchor_positions : list[int]
        Anchor indices used by the inner GN solver.
    outer_anchor_positions : list[int]
        Anchor indices used ONLY in the outer loss (held out from solver).
    sigma : jnp.ndarray
        Noise standard deviations (6,).
    gn_iters : int
        Inner Gauss-Newton iterations.
    anchor_weight : float
        Weight on anchor loss in outer objective.
    weight_reg : float
        Regularisation pulling log-weights toward zero (uniform).

    Returns
    -------
    grad_fn, loss_fn : callables
        grad_fn(log_weights, measurements, x_init,
                inner_anchor_targets, outer_anchor_targets) -> grad
        loss_fn(...) -> scalar
    """
    n_meas = n_poses - 1
    base_odom_w = sigma_to_weight(sigma)      # (6,) baseline information
    anchor_w = sigma_to_weight(jnp.full(6, 0.01))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    inner_idx = jnp.array(inner_anchor_positions, dtype=jnp.int32)
    outer_idx = jnp.array(outer_anchor_positions, dtype=jnp.int32)
    anchor_info_w = sigma_to_weight(sigma)

    # Vectorised residual with per-edge learned weights.
    def _odom_res_single(pose_a, pose_b, meas, log_w):
        w = jnp.exp(log_w)  # per-component weight multiplier
        sqrt_info = jnp.sqrt(base_odom_w * w)
        return (relative_pose_se3(pose_a, pose_b) - meas) * sqrt_info

    _odom_res_batch = jax.vmap(_odom_res_single)

    def residual_fn(x, measurements, log_weights, inner_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_res_batch(
            poses[:-1], poses[1:], measurements, log_weights)
        r_anch = (poses[inner_idx] - inner_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    _retract_batch = jax.vmap(se3_retract_left)
    max_step_per_pose = 0.5

    def gn_step(x, measurements, log_weights, inner_targets):
        def r_fn(x_):
            return residual_fn(x_, measurements, log_weights, inner_targets)
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

    def outer_loss(log_weights, measurements, x_init,
                   inner_targets, outer_targets):
        # Inner solve: GN with learned weights + inner anchors only.
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, measurements, log_weights, inner_targets)

        poses_opt = x.reshape(n_poses, 6)

        # Outer loss: evaluate at HELD-OUT anchor positions.
        # The solver never saw these — weights must help generalisation.
        diffs = poses_opt[outer_idx] - outer_targets
        a_loss = jnp.sum(anchor_info_w * diffs ** 2)

        # Weight regularisation: pull log-weights toward 0 (uniform).
        w_reg = jnp.sum(log_weights ** 2)

        return anchor_weight * a_loss + weight_reg * w_reg

    grad_fn = jax.jit(jax.grad(outer_loss))
    loss_fn = jax.jit(outer_loss)
    return grad_fn, loss_fn


# ---------------------------------------------------------------------------
# PGO solver with learned weights (for final evaluation)
# ---------------------------------------------------------------------------

def build_weighted_pgo_solver(
    n_poses: int,
    anchor_indices: list[int],
    sigma: jnp.ndarray,
    gn_iters: int = 20,
    damping: float = 1e-3,
):
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
# Standard (uniform-weight) PGO solver for baseline comparison
# ---------------------------------------------------------------------------

def build_pgo_solver(n_poses, anchor_indices, sigma, gn_iters=20, damping=1e-3):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Learned per-measurement weights via bilevel optimisation")
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--seq", type=str, default="07")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=50,
                        help="Poses per solve window (default: 50)")
    parser.add_argument("--anchor-spacing", type=int, default=5,
                        help="GT anchor every N poses for learning (default: 5 = 20%%)")
    parser.add_argument("--sigma-trans", type=float, default=0.10)
    parser.add_argument("--sigma-rot", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--n-outer-iters", type=int, default=150,
                        help="Outer Adam iterations per window (default: 150)")
    parser.add_argument("--gn-iters", type=int, default=10,
                        help="Inner GN iterations (default: 10)")
    parser.add_argument("--aw", type=float, default=10.0,
                        help="Anchor weight (default: 10.0)")
    parser.add_argument("--wreg", type=float, default=0.01,
                        help="Weight regularisation (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="exp29_results.json")
    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)
    window_size = args.window_size
    overlap = min(10, window_size // 5)
    stride = window_size - overlap

    print("=" * 70)
    print("  exp29 -- Learned Per-Measurement Weights")
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

    # All anchor positions within each window (at anchor_spacing).
    all_anchors_in_window = list(range(0, actual_window, args.anchor_spacing))
    if all_anchors_in_window[-1] != actual_window - 1:
        all_anchors_in_window.append(actual_window - 1)

    # Split into inner (solver) and outer (validation) anchors.
    # Even-indexed anchors go to inner solver, odd-indexed to outer loss.
    # First and last always go to inner (boundary constraints).
    inner_anchors = []
    outer_anchors = []
    for i, a in enumerate(all_anchors_in_window):
        if a == 0 or a == actual_window - 1:
            inner_anchors.append(a)
        elif i % 2 == 0:
            inner_anchors.append(a)
        else:
            outer_anchors.append(a)

    # Ensure we have at least 1 outer anchor.
    if len(outer_anchors) == 0 and len(inner_anchors) > 2:
        # Move middle inner anchor to outer.
        mid = len(inner_anchors) // 2
        outer_anchors.append(inner_anchors.pop(mid))
        outer_anchors.sort()

    # PGO anchor positions (global, sparse — for downstream eval).
    # Use sparser anchors for PGO to test generalisation.
    pgo_spacing = max(args.anchor_spacing, 10)  # at least as sparse as learning
    pgo_anchor_indices = list(range(0, n_poses_total, pgo_spacing))
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
    print(f"  PGO anchors:    {len(pgo_anchor_indices)} (global, spacing={pgo_spacing})")
    print(f"  Weights:        aw={args.aw}, wreg={args.wreg}")
    print(f"  Inner GN iters: {args.gn_iters}")
    print(f"  Outer iters:    {args.n_outer_iters} (Adam, lr={args.lr})")
    print()

    # ---- Baseline (uniform weights) ----
    noisy_poses_np = reconstruct_trajectory(gt_poses[0], noisy_measurements)
    meas_err_baseline = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))
    ate_noisy = compute_ate(noisy_poses_np, gt_np)
    print(f"Baseline (dead reckoning):")
    print(f"  ATE trans RMSE:  {ate_noisy['ate_trans_rmse']:.4f} m")
    print(f"  Meas error:      {meas_err_baseline:.4f}")
    print()

    # ---- Build and JIT-compile weight learner ----
    print(f"JIT-compiling weight learner ({actual_window} poses)...", flush=True)
    t_jit_start = time.perf_counter()

    grad_fn, loss_fn = build_weight_learner(
        actual_window, inner_anchors, outer_anchors, sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        anchor_weight=args.aw, weight_reg=args.wreg)

    # Warm-up with dummy data.
    n_meas_window = actual_window - 1
    dummy_meas = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_log_w = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_inner = jnp.zeros((len(inner_anchors), 6), dtype=jnp.float32)
    dummy_outer = jnp.zeros((len(outer_anchors), 6), dtype=jnp.float32)
    _ = grad_fn(dummy_log_w, dummy_meas, dummy_x,
                dummy_inner, dummy_outer).block_until_ready()

    t_jit = time.perf_counter() - t_jit_start
    print(f"JIT compilation: {t_jit:.1f}s")
    print()

    # ---- Learn weights window by window ----
    print("Learning per-measurement weights...", flush=True)
    t_learn_start = time.perf_counter()

    # Global log-weights: accumulate from overlapping windows.
    global_log_weights = np.zeros((n_meas_total, 6), dtype=np.float32)
    weight_counts = np.zeros(n_meas_total, dtype=np.int32)
    weight_accum = np.zeros((n_meas_total, 6), dtype=np.float32)

    for wi, (w_start, w_end) in enumerate(windows):
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1

        # Extract window measurements (FIXED — never modified).
        w_meas = jnp.array(noisy_measurements[w_start:w_start + w_n_meas])

        # Anchor targets in LOCAL coordinates (split inner/outer).
        gt_first = gt_poses[w_start]
        w_inner_targets = jnp.stack([
            relative_pose_se3(gt_first, gt_poses[w_start + p])
            for p in inner_anchors
        ])
        w_outer_targets = jnp.stack([
            relative_pose_se3(gt_first, gt_poses[w_start + p])
            for p in outer_anchors
        ])

        # x_init: forward-compose from origin.
        origin = jnp.zeros(6, dtype=jnp.float32)
        init_poses = [origin]
        for k in range(w_n_meas):
            init_poses.append(compose_pose_se3(init_poses[-1], w_meas[k]))
        x_init = jnp.concatenate(init_poses)

        # Initialise log-weights to 0 (= uniform weighting, exp(0)=1).
        log_w = jnp.zeros((w_n_meas, 6), dtype=jnp.float32)
        adam_state = adam_init(log_w)

        for it in range(args.n_outer_iters):
            g = grad_fn(log_w, w_meas, x_init,
                        w_inner_targets, w_outer_targets)
            g.block_until_ready()

            if jnp.any(jnp.isnan(g)):
                print(f"  Window {wi}: NaN gradient at iter {it}, stopping early")
                break

            update, adam_state = adam_step(g, adam_state, lr=args.lr)
            log_w = log_w - update

        # Accumulate learned weights (for overlapping regions, average).
        log_w_np = np.array(log_w)
        for i in range(w_n_meas):
            gi = w_start + i
            if gi < n_meas_total:
                weight_accum[gi] += log_w_np[i]
                weight_counts[gi] += 1

        elapsed = time.perf_counter() - t_learn_start
        final_loss = float(loss_fn(log_w, w_meas, x_init,
                                    w_inner_targets, w_outer_targets))
        w_mean = float(jnp.mean(jnp.exp(log_w)))
        w_std = float(jnp.std(jnp.exp(log_w)))
        w_min = float(jnp.min(jnp.exp(log_w)))
        w_max = float(jnp.max(jnp.exp(log_w)))
        print(f"  Window {wi+1}/{len(windows)} "
              f"(poses {w_start}-{w_end-1}): "
              f"loss={final_loss:.4f}, "
              f"w: mean={w_mean:.3f} std={w_std:.3f} "
              f"[{w_min:.3f}, {w_max:.3f}], "
              f"elapsed={elapsed:.1f}s", flush=True)

    # Average overlapping log-weights.
    for i in range(n_meas_total):
        if weight_counts[i] > 0:
            global_log_weights[i] = weight_accum[i] / weight_counts[i]

    t_learn = time.perf_counter() - t_learn_start
    learned_log_weights = jnp.array(global_log_weights)
    learned_weights = jnp.exp(learned_log_weights)

    # Weight statistics.
    print(f"\nWeight learning complete: {len(windows)} windows, {t_learn:.1f}s")
    print(f"  Global weight stats:")
    print(f"    Mean:  {float(jnp.mean(learned_weights)):.4f}")
    print(f"    Std:   {float(jnp.std(learned_weights)):.4f}")
    print(f"    Min:   {float(jnp.min(learned_weights)):.4f}")
    print(f"    Max:   {float(jnp.max(learned_weights)):.4f}")
    # Per-component (trans x,y,z  rot x,y,z).
    comp_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    for c in range(6):
        wc = learned_weights[:, c]
        print(f"    {comp_names[c]}: mean={float(jnp.mean(wc)):.4f}, "
              f"std={float(jnp.std(wc)):.4f}")
    print()

    # ---- Downstream evaluation: PGO with learned weights vs uniform ----
    print("Building JIT-compiled PGO solvers...", flush=True)

    # Uniform-weight PGO (baseline).
    pgo_uniform = build_pgo_solver(
        n_poses_total, pgo_anchor_indices, sigma, gn_iters=20, damping=1e-3)

    # Weighted PGO (with learned weights).
    pgo_weighted = build_weighted_pgo_solver(
        n_poses_total, pgo_anchor_indices, sigma, gn_iters=20, damping=1e-3)

    # PGO init: forward-compose from GT first pose.
    init_poses_pgo = [gt_poses[0]]
    for k in range(n_meas_total):
        init_poses_pgo.append(compose_pose_se3(init_poses_pgo[-1],
                                                noisy_measurements[k]))
    x_init_pgo = jnp.concatenate(init_poses_pgo)
    anchor_targets_pgo = jnp.stack([gt_poses[i] for i in pgo_anchor_indices])

    # Warm up both solvers.
    t_pgo_jit_start = time.perf_counter()
    _ = pgo_uniform(noisy_measurements, x_init_pgo,
                     anchor_targets_pgo).block_until_ready()
    uniform_log_w = jnp.zeros((n_meas_total, 6), dtype=jnp.float32)
    _ = pgo_weighted(noisy_measurements, uniform_log_w, x_init_pgo,
                      anchor_targets_pgo).block_until_ready()
    t_pgo_jit = time.perf_counter() - t_pgo_jit_start
    print(f"PGO JIT compile: {t_pgo_jit:.1f}s")

    # PGO with uniform weights (baseline).
    print("Running PGO with uniform weights...", flush=True)
    t0 = time.perf_counter()
    x_pgo_uniform = pgo_uniform(noisy_measurements, x_init_pgo,
                                 anchor_targets_pgo).block_until_ready()
    t_pgo_uniform = time.perf_counter() - t0
    pgo_uniform_poses = np.array(x_pgo_uniform.reshape(n_poses_total, 6))

    # PGO with learned weights.
    print("Running PGO with learned weights...", flush=True)
    t0 = time.perf_counter()
    x_pgo_weighted = pgo_weighted(noisy_measurements, learned_log_weights,
                                   x_init_pgo,
                                   anchor_targets_pgo).block_until_ready()
    t_pgo_weighted = time.perf_counter() - t0
    pgo_weighted_poses = np.array(x_pgo_weighted.reshape(n_poses_total, 6))

    # ---- Compute all metrics ----
    ate_uniform = compute_ate(pgo_uniform_poses, gt_np)
    rpe_uniform = compute_rpe(pgo_uniform_poses, gt_np)
    kitti_uniform = compute_kitti_metric(pgo_uniform_poses, gt_np)

    ate_weighted = compute_ate(pgo_weighted_poses, gt_np)
    rpe_weighted = compute_rpe(pgo_weighted_poses, gt_np)
    kitti_weighted = compute_kitti_metric(pgo_weighted_poses, gt_np)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print()
    print("--- Downstream: PGO Trajectory Quality ---")
    hdr = f"  {'':20s} {'PGO uniform':>12s} {'PGO learned':>12s} {'Improv.':>10s}"
    print(hdr)
    for label, k in [
        ("ATE trans [m]", "ate_trans_rmse"),
        ("ATE rot [rad]", "ate_rot_rmse"),
    ]:
        v1, v2 = ate_uniform[k], ate_weighted[k]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")
    for label, k in [
        ("RPE trans [m]", "rpe_trans_rmse"),
        ("RPE rot [rad]", "rpe_rot_rmse"),
    ]:
        v1, v2 = rpe_uniform[k], rpe_weighted[k]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")
    for label, k in [
        ("KITTI trans [%]", "kitti_trans_err_pct"),
        ("KITTI rot [deg/m]", "kitti_rot_err_degm"),
    ]:
        v1, v2 = kitti_uniform[k], kitti_weighted[k]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")

    # Weight-noise correlation analysis.
    print()
    print("--- Weight-Noise Correlation ---")
    noise_per_edge = np.array(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1))
    mean_weight_per_edge = np.array(jnp.mean(learned_weights, axis=1))
    # Pearson correlation: negative = weights correctly down-weight noisy edges.
    corr = float(np.corrcoef(noise_per_edge, mean_weight_per_edge)[0, 1])
    print(f"  Pearson(noise_magnitude, learned_weight) = {corr:.4f}")
    print(f"  {'GOOD' if corr < -0.1 else 'WEAK' if abs(corr) < 0.1 else 'UNEXPECTED'}: "
          f"{'negative' if corr < 0 else 'positive'} correlation "
          f"({'noisy edges get lower weight' if corr < 0 else 'noisy edges get HIGHER weight'})")

    # Top-10 noisiest edges: what weights did they get?
    top10_noisy = np.argsort(noise_per_edge)[-10:]
    top10_clean = np.argsort(noise_per_edge)[:10]
    print(f"  Top-10 noisiest edges: mean weight = {float(np.mean(mean_weight_per_edge[top10_noisy])):.4f}")
    print(f"  Top-10 cleanest edges: mean weight = {float(np.mean(mean_weight_per_edge[top10_clean])):.4f}")

    print()
    print("--- Timing ---")
    print(f"  Weight learner JIT:  {t_jit:8.1f} s")
    print(f"  PGO JIT:             {t_pgo_jit:8.1f} s")
    print(f"  Weight learning:     {t_learn:8.1f} s  ({len(windows)} windows)")
    print(f"  PGO (uniform):       {t_pgo_uniform:8.1f} s")
    print(f"  PGO (learned):       {t_pgo_weighted:8.1f} s")
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
            "anchor_spacing": args.anchor_spacing,
            "anchor_density_pct": round(anchor_density, 1),
            "inner_anchors_per_window": len(inner_anchors),
            "outer_anchors_per_window": len(outer_anchors),
            "n_pgo_anchors": len(pgo_anchor_indices),
            "sigma_trans": args.sigma_trans,
            "sigma_rot": args.sigma_rot,
            "gn_iters": args.gn_iters,
            "outer_iters": args.n_outer_iters,
            "outer_lr": args.lr,
            "anchor_weight": args.aw,
            "weight_reg": args.wreg,
            "optimizer": "adam",
        },
        "learned_weight_stats": {
            "mean": round(float(jnp.mean(learned_weights)), 4),
            "std": round(float(jnp.std(learned_weights)), 4),
            "min": round(float(jnp.min(learned_weights)), 4),
            "max": round(float(jnp.max(learned_weights)), 4),
            "noise_weight_correlation": round(corr, 4),
        },
        "downstream_pgo_uniform": {
            **{k: round(v, 6) for k, v in ate_uniform.items()},
            **{k: round(v, 6) for k, v in rpe_uniform.items()},
            **{k: round(v, 4) for k, v in kitti_uniform.items()},
        },
        "downstream_pgo_learned": {
            **{k: round(v, 6) for k, v in ate_weighted.items()},
            **{k: round(v, 6) for k, v in rpe_weighted.items()},
            **{k: round(v, 4) for k, v in kitti_weighted.items()},
        },
        "timing": {
            "weight_learner_jit_s": round(t_jit, 2),
            "pgo_jit_s": round(t_pgo_jit, 2),
            "weight_learning_s": round(t_learn, 2),
            "pgo_uniform_s": round(t_pgo_uniform, 2),
            "pgo_learned_s": round(t_pgo_weighted, 2),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")

    return results


if __name__ == "__main__":
    main()
