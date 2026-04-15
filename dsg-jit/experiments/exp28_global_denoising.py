# experiments/exp28_global_denoising.py
"""
Global SE(3) measurement denoising via bilevel optimisation on GPU.

Unlike exp27 (independent 5-pose segments), this experiment performs a
*global* solve over up to 500 poses at once.  For trajectories longer
than 500 poses, an incremental windowed approach processes overlapping
500-pose chunks.

The bilevel structure:
  - Inner: manifold-aware Gauss-Newton on the full 500-pose graph
  - Outer: Adam on all 499 measurements to minimise anchor + reg + smooth loss

Designed for GPU execution.  JIT compilation of the global solver takes
several minutes but subsequent calls reuse the compiled kernel.

Usage:
    # Synthetic (default 500 poses)
    python -m experiments.exp28_global_denoising

    # KITTI
    python -m experiments.exp28_global_denoising \\
        --kitti-root /path/to/kitti/odometry --seq 07

    # Force CPU (for testing)
    JAX_PLATFORM_NAME=cpu python -m experiments.exp28_global_denoising --max-frames 21
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
# Data loading (shared with exp27)
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
# Metrics (same as exp27)
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
# Global bilevel denoising solver
# ---------------------------------------------------------------------------

def build_global_denoiser(
    n_poses: int,
    anchor_positions: list[int],
    sigma: jnp.ndarray,
    *,
    gn_iters: int = 3,
    gn_damping: float = 5e-3,
    anchor_weight: float = 5.0,
    reg_weight: float = 0.1,
    smooth_weight: float = 2.0,
):
    """Build a JIT-compiled bilevel denoiser for ``n_poses`` SE(3) poses.

    Returns ``grad_fn(theta, x_init, anchor_targets, noisy_meas) -> grad``.
    """
    n_meas = n_poses - 1
    odom_w = sigma_to_weight(sigma)
    anchor_w = sigma_to_weight(jnp.full(6, 0.01))
    sqrt_odom_w = jnp.sqrt(odom_w)
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_positions, dtype=jnp.int32)
    n_anchors = len(anchor_positions)

    # Vectorised residual components.
    def _odom_res_single(pose_a, pose_b, meas):
        return (relative_pose_se3(pose_a, pose_b) - meas) * sqrt_odom_w

    _odom_res_batch = jax.vmap(_odom_res_single)

    def _anchor_res(poses, targets):
        return (poses[anchor_idx] - targets) * sqrt_anchor_w

    def residual_fn(x, theta, anchor_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_res_batch(poses[:-1], poses[1:], theta)  # (n_meas, 6)
        r_anch = _anchor_res(poses, anchor_targets)              # (n_anchors, 6)
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    _retract_batch = jax.vmap(se3_retract_left)

    # Per-pose step clamp: limit each pose's update independently.
    max_step_per_pose = 0.5

    def gn_step(x, theta, anchor_targets):
        def r_fn(x_):
            return residual_fn(x_, theta, anchor_targets)

        r = r_fn(x)
        J = jax.jacobian(r_fn)(x)
        n = x.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        delta = jnp.linalg.solve(H, J.T @ r)

        # Per-pose step clamp (not global — avoids choking large systems).
        poses = x.reshape(n_poses, 6)
        deltas = delta.reshape(n_poses, 6)
        norms = jnp.linalg.norm(deltas, axis=1, keepdims=True)
        scales = jnp.minimum(1.0, max_step_per_pose / (norms + 1e-9))
        deltas = deltas * scales

        # Vectorised manifold retraction.
        new_poses = _retract_batch(poses, -deltas)
        return new_poses.ravel()

    # Information-weighted anchor loss: scale by 1/sigma so translation
    # and rotation contribute proportionally to their noise levels.
    anchor_info_w = sigma_to_weight(sigma)  # [1/σ²] per component

    def outer_loss(theta, x_init, anchor_targets, noisy_meas):
        # Inner solve: unrolled GN.
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, theta, anchor_targets)

        poses_opt = x.reshape(n_poses, 6)

        # Anchor loss (information-weighted: balances translation/rotation).
        diffs = poses_opt[anchor_idx] - anchor_targets  # (n_anchors, 6)
        a_loss = jnp.sum(anchor_info_w * diffs ** 2)

        # Regularisation: keep theta near original noisy measurements.
        dev = theta - noisy_meas
        r_loss = jnp.sum(odom_w * dev ** 2)

        # Temporal smoothness on theta (information-weighted).
        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(odom_w * s_diffs ** 2)

        return anchor_weight * a_loss + reg_weight * r_loss + smooth_weight * s_loss

    grad_fn = jax.jit(jax.grad(outer_loss))
    loss_fn = jax.jit(outer_loss)
    return grad_fn, loss_fn


# ---------------------------------------------------------------------------
# JIT-compiled PGO for downstream evaluation (from exp27, vectorised)
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
        # Per-pose step clamp.
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
    parser = argparse.ArgumentParser(description="Global SE(3) denoising")
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--seq", type=str, default="07")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=500,
                        help="Max poses per global solve window (default: 500)")
    parser.add_argument("--anchor-spacing", type=int, default=50,
                        help="GT anchor every N poses within each window")
    parser.add_argument("--sigma-trans", type=float, default=0.10)
    parser.add_argument("--sigma-rot", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--n-outer-iters", type=int, default=100,
                        help="Outer Adam iterations per window (default: 100)")
    parser.add_argument("--gn-iters", type=int, default=5,
                        help="Inner GN iterations (default: 5, keep low for memory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="exp28_results.json")
    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)
    window_size = args.window_size
    overlap = min(50, window_size // 10)
    stride = window_size - overlap

    print("=" * 70)
    print("  exp28 -- Global SE(3) Measurement Denoising")
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
        # Ensure last window covers the end.
        if windows[-1][1] < n_poses_total:
            windows.append((n_poses_total - actual_window, n_poses_total))

    # Anchor positions within each window (fixed for JIT reuse).
    anchor_pos_in_window = list(range(0, actual_window, args.anchor_spacing))
    if anchor_pos_in_window[-1] != actual_window - 1:
        anchor_pos_in_window.append(actual_window - 1)

    # PGO anchor positions (global, for downstream eval).
    pgo_anchor_indices = list(range(0, n_poses_total, args.anchor_spacing))
    if pgo_anchor_indices[-1] != n_poses_total - 1:
        pgo_anchor_indices.append(n_poses_total - 1)

    traj_len = float(np.sum(np.linalg.norm(
        np.diff(gt_np[:, :3], axis=0), axis=1)))

    print(f"  Data source:    {data_info['source']}")
    print(f"  Poses:          {n_poses_total}")
    print(f"  Trajectory len: {traj_len:.1f} m")
    print(f"  Noise:          sigma_t={args.sigma_trans}m, sigma_r={args.sigma_rot}rad")
    print(f"  Window size:    {actual_window} poses, stride={stride}, overlap={overlap}")
    print(f"  Windows:        {len(windows)}")
    print(f"  Anchors/window: {len(anchor_pos_in_window)} (every {args.anchor_spacing})")
    print(f"  PGO anchors:    {len(pgo_anchor_indices)} (global)")
    print(f"  Inner GN iters: {args.gn_iters}")
    print(f"  Outer iters:    {args.n_outer_iters} (Adam, lr={args.lr})")
    print()

    # ---- Baseline ----
    noisy_poses_np = reconstruct_trajectory(gt_poses[0], noisy_measurements)
    meas_err_before = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))
    ate_noisy = compute_ate(noisy_poses_np, gt_np)
    print(f"Baseline:")
    print(f"  ATE trans RMSE:  {ate_noisy['ate_trans_rmse']:.4f} m")
    print(f"  Meas error:      {meas_err_before:.4f}")
    print()

    # ---- Build and JIT-compile global denoiser ----
    print(f"JIT-compiling global denoiser ({actual_window} poses)...", flush=True)
    t_jit_start = time.perf_counter()

    grad_fn, loss_fn = build_global_denoiser(
        actual_window, anchor_pos_in_window, sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3)

    # Warm-up with dummy data.
    n_meas_window = actual_window - 1
    dummy_theta = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_anchors = jnp.zeros((len(anchor_pos_in_window), 6), dtype=jnp.float32)
    _ = grad_fn(dummy_theta, dummy_x, dummy_anchors, dummy_theta).block_until_ready()

    t_jit = time.perf_counter() - t_jit_start
    print(f"JIT compilation: {t_jit:.1f}s")
    print()

    # ---- Denoise measurements window by window ----
    print("Denoising measurements...", flush=True)
    t_denoise_start = time.perf_counter()

    denoised_measurements = np.array(noisy_measurements).copy()
    # Track how many times each measurement has been denoised (for averaging).
    commit_counts = np.zeros(n_meas_total, dtype=np.int32)
    accum = np.zeros_like(denoised_measurements)

    for wi, (w_start, w_end) in enumerate(windows):
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1

        # Extract window data.
        w_noisy = jnp.array(noisy_measurements[w_start:w_start + w_n_meas])

        # Anchor targets in LOCAL coordinates (relative to first pose).
        gt_first = gt_poses[w_start]
        w_anchor_targets = jnp.stack([
            relative_pose_se3(gt_first, gt_poses[w_start + p])
            for p in anchor_pos_in_window
        ])

        # x_init: forward-compose from origin using noisy measurements.
        origin = jnp.zeros(6, dtype=jnp.float32)
        init_poses = [origin]
        for k in range(w_n_meas):
            init_poses.append(compose_pose_se3(init_poses[-1], w_noisy[k]))
        x_init = jnp.concatenate(init_poses)

        # Outer optimisation with Adam.
        theta = w_noisy.copy()
        adam_state = adam_init(theta)

        for it in range(args.n_outer_iters):
            g = grad_fn(theta, x_init, w_anchor_targets, w_noisy)
            g.block_until_ready()

            if jnp.any(jnp.isnan(g)):
                print(f"  Window {wi}: NaN gradient at iter {it}, stopping early")
                break

            update, adam_state = adam_step(g, adam_state, lr=args.lr)
            theta = theta - update

        # Accumulate results (for overlapping regions, we average).
        theta_np = np.array(theta)
        for i in range(w_n_meas):
            gi = w_start + i
            if gi < n_meas_total:
                accum[gi] += theta_np[i]
                commit_counts[gi] += 1

        elapsed = time.perf_counter() - t_denoise_start
        w_meas_err = float(jnp.mean(jnp.linalg.norm(
            theta - gt_measurements[w_start:w_start + w_n_meas], axis=1)))
        w_orig_err = float(jnp.mean(jnp.linalg.norm(
            w_noisy - gt_measurements[w_start:w_start + w_n_meas], axis=1)))
        print(f"  Window {wi+1}/{len(windows)} "
              f"(poses {w_start}-{w_end-1}): "
              f"meas_err {w_orig_err:.4f} -> {w_meas_err:.4f} "
              f"({(1 - w_meas_err/w_orig_err)*100:.1f}%), "
              f"elapsed={elapsed:.1f}s", flush=True)

    # Average overlapping regions.
    for i in range(n_meas_total):
        if commit_counts[i] > 0:
            denoised_measurements[i] = accum[i] / commit_counts[i]

    t_denoise = time.perf_counter() - t_denoise_start
    denoised_meas_jnp = jnp.array(denoised_measurements)

    # Measurement error.
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

    print(f"\nDenoising complete: {len(windows)} windows, {t_denoise:.1f}s")
    print(f"  Meas error: {meas_err_before:.4f} -> {meas_err_after:.4f} "
          f"({(1 - meas_err_after/meas_err_before)*100:.1f}%)")
    print(f"  Trans:      {meas_trans_before:.4f} -> {meas_trans_after:.4f} "
          f"({(1 - meas_trans_after/meas_trans_before)*100:.1f}%)")
    print(f"  Rot:        {meas_rot_before:.4f} -> {meas_rot_after:.4f} "
          f"({(1 - meas_rot_after/meas_rot_before)*100:.1f}%)")
    print()

    # ---- Downstream PGO evaluation ----
    print("Building JIT-compiled PGO solver...", flush=True)
    pgo_solve = build_pgo_solver(
        n_poses_total, pgo_anchor_indices, sigma, gn_iters=20, damping=1e-3)

    # Warm-up PGO.
    t_pgo_jit_start = time.perf_counter()
    init_x = jnp.concatenate(
        [gt_poses[0]] + [compose_pose_se3(gt_poses[0], noisy_measurements[0])]
        * (n_poses_total - 1))  # dummy, just for shape
    # Proper init via forward compose.
    init_poses_pgo = [gt_poses[0]]
    for k in range(n_meas_total):
        init_poses_pgo.append(compose_pose_se3(init_poses_pgo[-1],
                                                noisy_measurements[k]))
    x_init_pgo = jnp.concatenate(init_poses_pgo)
    anchor_targets_pgo = jnp.stack([gt_poses[i] for i in pgo_anchor_indices])
    _ = pgo_solve(noisy_measurements, x_init_pgo, anchor_targets_pgo).block_until_ready()
    t_pgo_jit = time.perf_counter() - t_pgo_jit_start
    print(f"PGO JIT compile: {t_pgo_jit:.1f}s")

    # PGO with noisy measurements.
    print("Running PGO with noisy measurements...", flush=True)
    t0 = time.perf_counter()
    x_pgo_noisy = pgo_solve(noisy_measurements, x_init_pgo,
                             anchor_targets_pgo).block_until_ready()
    t_pgo_noisy = time.perf_counter() - t0
    pgo_noisy_poses = np.array(x_pgo_noisy.reshape(n_poses_total, 6))

    # PGO with denoised measurements.
    print("Running PGO with denoised measurements...", flush=True)
    init_poses_den = [gt_poses[0]]
    for k in range(n_meas_total):
        init_poses_den.append(compose_pose_se3(init_poses_den[-1],
                                                denoised_meas_jnp[k]))
    x_init_den = jnp.concatenate(init_poses_den)
    t0 = time.perf_counter()
    x_pgo_den = pgo_solve(denoised_meas_jnp, x_init_den,
                           anchor_targets_pgo).block_until_ready()
    t_pgo_denoised = time.perf_counter() - t0
    pgo_denoised_poses = np.array(x_pgo_den.reshape(n_poses_total, 6))

    # ---- Compute all metrics ----
    ate_pgo_noisy = compute_ate(pgo_noisy_poses, gt_np)
    rpe_pgo_noisy = compute_rpe(pgo_noisy_poses, gt_np)
    kitti_pgo_noisy = compute_kitti_metric(pgo_noisy_poses, gt_np)

    ate_pgo_den = compute_ate(pgo_denoised_poses, gt_np)
    rpe_pgo_den = compute_rpe(pgo_denoised_poses, gt_np)
    kitti_pgo_den = compute_kitti_metric(pgo_denoised_poses, gt_np)

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print()
    print("--- Measurement Error ---")
    print(f"  {'':20s} {'Before':>10s} {'After':>10s} {'Improv.':>10s}")
    print(f"  {'Overall':20s} {meas_err_before:10.4f} {meas_err_after:10.4f} "
          f"{(1 - meas_err_after/meas_err_before)*100:9.1f}%")
    print(f"  {'Translation':20s} {meas_trans_before:10.4f} {meas_trans_after:10.4f} "
          f"{(1 - meas_trans_after/meas_trans_before)*100:9.1f}%")
    print(f"  {'Rotation':20s} {meas_rot_before:10.4f} {meas_rot_after:10.4f} "
          f"{(1 - meas_rot_after/meas_rot_before)*100:9.1f}%")

    print()
    print("--- Downstream: PGO Trajectory Quality ---")
    hdr = f"  {'':20s} {'PGO+noisy':>12s} {'PGO+denoised':>12s} {'Improv.':>10s}"
    print(hdr)
    for label, k in [
        ("ATE trans [m]", "ate_trans_rmse"),
        ("ATE rot [rad]", "ate_rot_rmse"),
    ]:
        v1, v2 = ate_pgo_noisy[k], ate_pgo_den[k]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")
    for label, k in [
        ("RPE trans [m]", "rpe_trans_rmse"),
        ("RPE rot [rad]", "rpe_rot_rmse"),
    ]:
        v1, v2 = rpe_pgo_noisy[k], rpe_pgo_den[k]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")
    for label, k in [
        ("KITTI trans [%]", "kitti_trans_err_pct"),
        ("KITTI rot [deg/m]", "kitti_rot_err_degm"),
    ]:
        v1, v2 = kitti_pgo_noisy[k], kitti_pgo_den[k]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")

    print()
    print("--- Timing ---")
    print(f"  Denoiser JIT:        {t_jit:8.1f} s")
    print(f"  PGO JIT:             {t_pgo_jit:8.1f} s")
    print(f"  Denoising:           {t_denoise:8.1f} s  ({len(windows)} windows)")
    print(f"  PGO (noisy):         {t_pgo_noisy:8.1f} s")
    print(f"  PGO (denoised):      {t_pgo_denoised:8.1f} s")
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
            "anchors_per_window": len(anchor_pos_in_window),
            "n_pgo_anchors": len(pgo_anchor_indices),
            "sigma_trans": args.sigma_trans,
            "sigma_rot": args.sigma_rot,
            "gn_iters": args.gn_iters,
            "outer_iters": args.n_outer_iters,
            "outer_lr": args.lr,
            "optimizer": "adam",
        },
        "measurement_error": {
            "before": round(meas_err_before, 6),
            "after": round(meas_err_after, 6),
            "improvement_pct": round(
                (1 - meas_err_after / meas_err_before) * 100, 1),
            "trans_before": round(meas_trans_before, 6),
            "trans_after": round(meas_trans_after, 6),
            "rot_before": round(meas_rot_before, 6),
            "rot_after": round(meas_rot_after, 6),
        },
        "downstream_pgo_noisy": {
            **{k: round(v, 6) for k, v in ate_pgo_noisy.items()},
            **{k: round(v, 6) for k, v in rpe_pgo_noisy.items()},
            **{k: round(v, 4) for k, v in kitti_pgo_noisy.items()},
        },
        "downstream_pgo_denoised": {
            **{k: round(v, 6) for k, v in ate_pgo_den.items()},
            **{k: round(v, 6) for k, v in rpe_pgo_den.items()},
            **{k: round(v, 4) for k, v in kitti_pgo_den.items()},
        },
        "timing": {
            "denoiser_jit_s": round(t_jit, 2),
            "pgo_jit_s": round(t_pgo_jit, 2),
            "denoising_s": round(t_denoise, 2),
            "pgo_noisy_s": round(t_pgo_noisy, 2),
            "pgo_denoised_s": round(t_pgo_denoised, 2),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")

    return results


if __name__ == "__main__":
    main()
