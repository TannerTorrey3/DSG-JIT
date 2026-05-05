# experiments/exp34_jit_outer_loop.py
"""
JIT-compiled outer optimization loop for maximum GPU throughput.

Based on exp32's dense solver (GPU-friendly) with the outer Adam loop
wrapped in jax.lax.fori_loop. This eliminates Python dispatch overhead
between iterations — the entire optimization runs as a single fused
GPU kernel with no host synchronization.

Key difference from exp32: the outer loop (grad → adam step → update)
is compiled into one kernel instead of 100 separate Python-dispatched
calls.

Usage:
    python -m experiments.exp34_jit_outer_loop \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences --seq 00
    python -m experiments.exp34_jit_outer_loop \
        --sequences-dir /path/to/sequences --seq 00 --n-outer-iters 50
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.core.math3d import (
    compose_pose_se3,
    relative_pose_se3,
    se3_retract_left,
    so3_log,
)
from dsg_jit.slam.measurements import sigma_to_weight


# ---------------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------------

def _print_device_info():
    devices = jax.devices()
    print(f"  JAX devices:    {[str(d) for d in devices]}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("  WARNING: running on CPU.")


# ---------------------------------------------------------------------------
# KITTI pose loading
# ---------------------------------------------------------------------------

def load_kitti_poses(path: str, n_poses: int | None = None) -> tuple[jnp.ndarray, dict]:
    """Load KITTI-format poses.txt and convert to 6D [tx,ty,tz,wx,wy,wz] vectors."""
    raw = np.loadtxt(path).reshape(-1, 3, 4)
    total_available = len(raw)
    if n_poses is not None:
        raw = raw[:n_poses]

    poses = []
    for i in range(len(raw)):
        R = raw[i, :3, :3]
        t = raw[i, :3, 3]
        w = np.array(so3_log(jnp.array(R, dtype=jnp.float32)))
        pose_vec = jnp.array(
            [t[0], t[1], t[2], w[0], w[1], w[2]], dtype=jnp.float32
        )
        poses.append(pose_vec)

    n = len(poses)
    return jnp.stack(poses), {
        "source": f"kitti:{path}",
        "n_frames": n,
        "total_available": total_available,
    }


def find_sequences(sequences_dir: str, seq_filter: str | None = None) -> list[tuple[str, str]]:
    """Find all valid sequence directories containing poses.txt."""
    sequences = []
    for entry in sorted(os.listdir(sequences_dir)):
        seq_path = os.path.join(sequences_dir, entry)
        poses_path = os.path.join(seq_path, "poses.txt")
        if os.path.isdir(seq_path) and os.path.isfile(poses_path):
            if seq_filter is None or entry in seq_filter.split(","):
                sequences.append((entry, poses_path))
    return sequences


# ---------------------------------------------------------------------------
# Per-pose metrics
# ---------------------------------------------------------------------------

def reconstruct_trajectory(start_pose: jnp.ndarray, measurements: jnp.ndarray) -> np.ndarray:
    """Forward-compose measurements from a starting pose to build a trajectory."""
    poses = [np.array(start_pose)]
    current = start_pose
    for i in range(measurements.shape[0]):
        current = compose_pose_se3(current, measurements[i])
        poses.append(np.array(current))
    return np.stack(poses)


def compute_per_pose_meas_error(
    measurements: jnp.ndarray,
    gt_measurements: jnp.ndarray,
) -> dict:
    """Per-edge measurement error split into trans and rot."""
    diff = measurements - gt_measurements
    trans_err = np.array(jnp.linalg.norm(diff[:, :3], axis=1))
    rot_err = np.array(jnp.linalg.norm(diff[:, 3:], axis=1))
    return {
        "per_edge_trans": trans_err,
        "per_edge_rot": rot_err,
        "mean_trans": float(np.mean(trans_err)),
        "mean_rot": float(np.mean(rot_err)),
        "rmse_trans": float(np.sqrt(np.mean(trans_err ** 2))),
        "rmse_rot": float(np.sqrt(np.mean(rot_err ** 2))),
    }


# ---------------------------------------------------------------------------
# Bilevel denoiser with JIT-compiled outer loop
# ---------------------------------------------------------------------------

def build_denoiser(
    n_poses: int,
    anchor_positions: list[int],
    sigma: jnp.ndarray,
    *,
    gn_iters: int = 10,
    gn_damping: float = 5e-3,
    aw_trans: float = 5.0,
    aw_rot: float = 5.0,
    rw_trans: float = 0.25,
    rw_rot: float = 0.1,
    smooth_weight: float = 25.0,
    inner_anchor_sigma: float = 0.01,
    n_outer_iters: int = 100,
    lr: float = 1e-3,
):
    """Build a fully JIT-compiled denoiser with fused outer loop.

    The outer Adam optimization is wrapped in lax.fori_loop so the
    entire window solve (n_outer_iters × grad + adam step) runs as a
    single GPU kernel without Python dispatch overhead.
    """
    n_meas = n_poses - 1
    odom_w = sigma_to_weight(sigma)
    sqrt_odom_w = jnp.sqrt(odom_w)
    anchor_w = sigma_to_weight(jnp.full(6, inner_anchor_sigma))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_positions, dtype=jnp.int32)

    anchor_w_vec = jnp.array(
        [aw_trans] * 3 + [aw_rot] * 3, dtype=jnp.float32)

    reg_w_vec = odom_w * jnp.array(
        [rw_trans] * 3 + [rw_rot] * 3, dtype=jnp.float32)

    _odom_res_batch = jax.vmap(
        lambda a, b, m: (relative_pose_se3(a, b) - m) * sqrt_odom_w)
    _retract_batch = jax.vmap(se3_retract_left)
    max_step_per_pose = 0.5

    def residual_fn(x, theta, anchor_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_res_batch(poses[:-1], poses[1:], theta)
        r_anch = (poses[anchor_idx] - anchor_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step(x, theta, anchor_targets):
        def r_fn(x_):
            return residual_fn(x_, theta, anchor_targets)
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
        return _retract_batch(poses, -deltas).ravel()

    def outer_loss(theta, x_init, anchor_targets, noisy_meas):
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, theta, anchor_targets)

        poses_opt = x.reshape(n_poses, 6)

        diffs = poses_opt[anchor_idx] - anchor_targets
        a_loss = jnp.sum(anchor_w_vec * diffs ** 2)

        dev = theta - noisy_meas
        r_loss = jnp.sum(reg_w_vec * dev ** 2)

        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(s_diffs ** 2)

        return a_loss + r_loss + smooth_weight * s_loss

    grad_fn = jax.grad(outer_loss)

    # --- JIT-compiled full optimization (fused outer loop) ---
    def fused_optimize(theta_init, x_init, anchor_targets, noisy_meas):
        """Run the full outer optimization as a single compiled kernel."""

        # Adam state: (m, v, t, theta)
        m0 = jnp.zeros_like(theta_init)
        v0 = jnp.zeros_like(theta_init)

        def adam_body(i, state):
            theta, m, v = state
            g = grad_fn(theta, x_init, anchor_targets, noisy_meas)

            # Adam update.
            t = (i + 1).astype(jnp.float32)
            m_new = 0.9 * m + 0.1 * g
            v_new = 0.999 * v + 0.001 * g ** 2
            m_hat = m_new / (1.0 - 0.9 ** t)
            v_hat = v_new / (1.0 - 0.999 ** t)
            update = lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            theta_new = theta - update

            return (theta_new, m_new, v_new)

        init_state = (theta_init, m0, v0)
        final_state = jax.lax.fori_loop(0, n_outer_iters, adam_body, init_state)
        theta_opt = final_state[0]
        return theta_opt

    fused_optimize_jit = jax.jit(fused_optimize)

    # Also provide the un-fused version for comparison.
    grad_fn_jit = jax.jit(grad_fn)
    loss_fn_jit = jax.jit(outer_loss)

    return fused_optimize_jit, grad_fn_jit, loss_fn_jit


# ---------------------------------------------------------------------------
# Denoise a single sequence
# ---------------------------------------------------------------------------

def denoise_sequence(
    gt_poses: jnp.ndarray,
    args,
    sigma: jnp.ndarray,
    seq_id: str = "??",
) -> dict:
    """Run the full denoising pipeline on a single sequence."""
    n_poses_total = gt_poses.shape[0]
    n_meas_total = n_poses_total - 1
    window_size = args.window_size
    overlap = min(10, window_size // 5)
    stride = window_size - overlap

    # Compute GT measurements.
    gt_measurements = jnp.stack([
        relative_pose_se3(gt_poses[i], gt_poses[i + 1])
        for i in range(n_meas_total)
    ])

    # Add calibrated noise.
    key = jax.random.PRNGKey(args.seed)
    noise = jax.random.normal(key, shape=gt_measurements.shape) * sigma
    noisy_measurements = gt_measurements + noise

    # Plan windows.
    actual_window = min(window_size, n_poses_total)
    if n_poses_total <= window_size:
        windows = [(0, n_poses_total)]
    else:
        windows = []
        for start in range(0, n_poses_total - actual_window + 1, stride):
            windows.append((start, start + actual_window))
        if windows[-1][1] < n_poses_total:
            windows.append((n_poses_total - actual_window, n_poses_total))

    # Anchor positions.
    anchor_pos_in_window = list(range(0, actual_window, args.anchor_spacing))
    if anchor_pos_in_window[-1] != actual_window - 1:
        anchor_pos_in_window.append(actual_window - 1)

    anchor_density = len(anchor_pos_in_window) / actual_window * 100

    gt_np = np.array(gt_poses)
    traj_len = float(np.sum(np.linalg.norm(
        np.diff(gt_np[:, :3], axis=0), axis=1)))

    print(f"\n  Sequence {seq_id}: {n_poses_total} poses, "
          f"{traj_len:.1f}m trajectory")
    print(f"  Windows: {len(windows)} (size={actual_window}, "
          f"stride={stride}, overlap={overlap})")
    print(f"  Anchors/window: {len(anchor_pos_in_window)} "
          f"({anchor_density:.1f}% density)")
    print(f"  Outer loop: JIT-fused ({args.n_outer_iters} iters)")

    # Baseline error.
    baseline = compute_per_pose_meas_error(noisy_measurements, gt_measurements)

    # Build denoiser.
    t_jit_start = time.perf_counter()
    fused_opt, grad_fn, loss_fn = build_denoiser(
        actual_window, anchor_pos_in_window, sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        aw_trans=args.aw_trans, aw_rot=args.aw_rot,
        rw_trans=args.rw_trans, rw_rot=args.rw_rot,
        smooth_weight=args.sw,
        inner_anchor_sigma=args.inner_anchor_sigma,
        n_outer_iters=args.n_outer_iters,
        lr=args.lr)

    # Warm-up the fused optimizer.
    n_meas_window = actual_window - 1
    dummy_theta = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_anchors = jnp.zeros((len(anchor_pos_in_window), 6),
                               dtype=jnp.float32)
    _ = fused_opt(dummy_theta, dummy_x, dummy_anchors,
                  dummy_theta).block_until_ready()
    t_jit = time.perf_counter() - t_jit_start
    print(f"  JIT: {t_jit:.1f}s")

    # Commit ranges (midpoint-of-overlap).
    commit_ranges = []
    for wi in range(len(windows)):
        w_start, w_end = windows[wi]
        w_first_edge = w_start
        w_last_edge = w_end - 2

        if wi == 0:
            commit_start = w_first_edge
        else:
            prev_last_edge = windows[wi - 1][1] - 2
            commit_start = (w_first_edge + prev_last_edge) // 2 + 1

        if wi == len(windows) - 1:
            commit_end = w_last_edge
        else:
            next_first_edge = windows[wi + 1][0]
            commit_end = (next_first_edge + w_last_edge) // 2

        commit_ranges.append((commit_start, commit_end))

    # Denoise window by window using the fused optimizer.
    t_denoise_start = time.perf_counter()
    denoised_measurements = np.array(noisy_measurements).copy()
    window_times = []

    for wi, (w_start, w_end) in enumerate(windows):
        t_win_start = time.perf_counter()
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1

        w_noisy = jnp.array(noisy_measurements[w_start:w_start + w_n_meas])

        # Anchor targets in local coordinates.
        gt_first = gt_poses[w_start]
        w_anchor_targets = jnp.stack([
            relative_pose_se3(gt_first, gt_poses[w_start + p])
            for p in anchor_pos_in_window
        ])

        # x_init: forward-compose from origin.
        origin = jnp.zeros(6, dtype=jnp.float32)
        init_poses = [origin]
        for k in range(w_n_meas):
            init_poses.append(compose_pose_se3(init_poses[-1], w_noisy[k]))
        x_init = jnp.concatenate(init_poses)

        # Run fused optimization (single GPU kernel).
        theta_opt = fused_opt(w_noisy, x_init, w_anchor_targets, w_noisy)
        theta_opt.block_until_ready()

        # Commit.
        theta_np = np.array(theta_opt)
        commit_start, commit_end = commit_ranges[wi]
        for gi in range(commit_start, commit_end + 1):
            local_i = gi - w_start
            if 0 <= local_i < w_n_meas and gi < n_meas_total:
                denoised_measurements[gi] = theta_np[local_i]

        t_win = time.perf_counter() - t_win_start
        window_times.append(t_win)

        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            elapsed = time.perf_counter() - t_denoise_start
            avg_hz = (wi + 1) * actual_window / elapsed
            print(f"    Window {wi+1}/{len(windows)} done "
                  f"({elapsed:.1f}s, {avg_hz:.1f} poses/s)", flush=True)

    t_denoise = time.perf_counter() - t_denoise_start
    denoised_meas_jnp = jnp.array(denoised_measurements)

    # Compute results.
    after = compute_per_pose_meas_error(denoised_meas_jnp, gt_measurements)

    delta_trans = baseline['per_edge_trans'] - after['per_edge_trans']
    delta_rot = baseline['per_edge_rot'] - after['per_edge_rot']

    n_trans_improved = int(np.sum(delta_trans > 0))
    n_rot_improved = int(np.sum(delta_rot > 0))

    trans_improv_pct = (1 - after['rmse_trans'] / baseline['rmse_trans']) * 100
    rot_improv_pct = (1 - after['rmse_rot'] / baseline['rmse_rot']) * 100

    # Throughput stats.
    poses_per_sec = n_poses_total / t_denoise
    avg_window_time = np.mean(window_times)
    p95_window_time = np.percentile(window_times, 95)

    print(f"\n  Results: Trans RMSE {baseline['rmse_trans']:.4f} -> "
          f"{after['rmse_trans']:.4f} ({trans_improv_pct:+.1f}%), "
          f"Rot RMSE {baseline['rmse_rot']:.4f} -> "
          f"{after['rmse_rot']:.4f} ({rot_improv_pct:+.1f}%)")
    print(f"  Edges improved: trans {n_trans_improved}/{n_meas_total}, "
          f"rot {n_rot_improved}/{n_meas_total}")
    print(f"\n  --- Throughput ---")
    print(f"  Total denoise time: {t_denoise:.1f}s")
    print(f"  Poses/sec: {poses_per_sec:.1f}")
    print(f"  Window time: avg={avg_window_time:.3f}s, "
          f"p95={p95_window_time:.3f}s")
    print(f"  Real-time factor (vs 10Hz): {poses_per_sec/10:.2f}x")
    print(f"  Real-time factor (vs 100Hz): {poses_per_sec/100:.2f}x")

    # Reconstruct trajectories for visualization.
    start_pose = gt_poses[0]
    gt_traj = reconstruct_trajectory(start_pose, gt_measurements)
    noisy_traj = reconstruct_trajectory(start_pose, noisy_measurements)
    denoised_traj = reconstruct_trajectory(start_pose, denoised_meas_jnp)

    return {
        "sequence": seq_id,
        "n_poses": n_poses_total,
        "trajectory_length_m": round(traj_len, 1),
        "n_windows": len(windows),
        "anchor_density_pct": round(anchor_density, 1),
        "outer_iters": args.n_outer_iters,
        "baseline": {
            "trans_rmse": round(baseline['rmse_trans'], 6),
            "rot_rmse": round(baseline['rmse_rot'], 6),
        },
        "denoised": {
            "trans_rmse": round(after['rmse_trans'], 6),
            "rot_rmse": round(after['rmse_rot'], 6),
        },
        "improvement_pct": {
            "trans_rmse": round(trans_improv_pct, 1),
            "rot_rmse": round(rot_improv_pct, 1),
            "combined": round((trans_improv_pct + rot_improv_pct) / 2, 1),
        },
        "per_edge": {
            "trans_improved": n_trans_improved,
            "rot_improved": n_rot_improved,
            "total_edges": n_meas_total,
        },
        "throughput": {
            "poses_per_sec": round(poses_per_sec, 1),
            "avg_window_time_s": round(avg_window_time, 4),
            "p95_window_time_s": round(p95_window_time, 4),
            "jit_s": round(t_jit, 2),
            "denoise_s": round(t_denoise, 2),
            "realtime_factor_10hz": round(poses_per_sec / 10, 2),
            "realtime_factor_100hz": round(poses_per_sec / 100, 2),
        },
        "trajectories": {
            "gt": gt_traj.tolist(),
            "noisy": noisy_traj.tolist(),
            "denoised": denoised_traj.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp34: JIT-fused outer loop denoiser")
    parser.add_argument("--sequences-dir", type=str, required=True,
                        help="Path to SemanticKITTI sequences directory")
    parser.add_argument("--seq", type=str, default=None,
                        help="Comma-separated sequence IDs (default: all)")
    parser.add_argument("--n-poses", type=int, default=None,
                        help="Limit poses per sequence (default: all)")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--anchor-spacing", type=int, default=100)
    parser.add_argument("--sigma-trans", type=float, default=0.10)
    parser.add_argument("--sigma-rot", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-outer-iters", type=int, default=100)
    parser.add_argument("--gn-iters", type=int, default=10)
    parser.add_argument("--aw-trans", type=float, default=5.0)
    parser.add_argument("--aw-rot", type=float, default=5.0)
    parser.add_argument("--rw-trans", type=float, default=0.25)
    parser.add_argument("--rw-rot", type=float, default=0.1)
    parser.add_argument("--sw", type=float, default=35.0,
                        help="Smoothness weight (default: 35.0)")
    parser.add_argument("--inner-anchor-sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/data/tkocher/exp_res")
    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)

    print("=" * 70)
    print("  exp34 -- JIT-Fused Outer Loop Denoiser")
    print("=" * 70)
    _print_device_info()
    print()
    print(f"  Config: window={args.window_size}, "
          f"anchor_spacing={args.anchor_spacing}")
    print(f"  Weights: rw_t={args.rw_trans}, rw_r={args.rw_rot}, "
          f"sw={args.sw}")
    print(f"  Outer loop: lax.fori_loop, {args.n_outer_iters} iters, "
          f"lr={args.lr}")
    print(f"  Inner GN: {args.gn_iters} iters")
    print()

    # Find sequences.
    sequences = find_sequences(args.sequences_dir, args.seq)
    if not sequences:
        print(f"ERROR: No sequences found in {args.sequences_dir}")
        return

    print(f"Found {len(sequences)} sequences: {[s[0] for s in sequences]}")

    # Run.
    all_results = []
    t_total_start = time.perf_counter()

    for seq_id, poses_path in sequences:
        gt_poses, data_info = load_kitti_poses(poses_path, args.n_poses)
        if gt_poses.shape[0] < args.window_size:
            print(f"\n  Sequence {seq_id}: skipping ({gt_poses.shape[0]} "
                  f"< {args.window_size} poses)")
            continue
        result = denoise_sequence(gt_poses, args, sigma, seq_id=seq_id)
        all_results.append(result)

    t_total = time.perf_counter() - t_total_start

    # Summary.
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    if all_results:
        print(f"  {'Seq':>4s} {'Poses':>6s} {'Poses/s':>8s} "
              f"{'RT@10Hz':>8s} {'RT@100Hz':>9s} "
              f"{'T_RMSE%':>8s} {'R_RMSE%':>8s} {'Combined':>9s}")
        print(f"  {'-'*4} {'-'*6} {'-'*8} "
              f"{'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*9}")

        for r in all_results:
            tp = r['throughput']
            imp = r['improvement_pct']
            print(f"  {r['sequence']:>4s} {r['n_poses']:>6d} "
                  f"{tp['poses_per_sec']:>7.1f} "
                  f"{tp['realtime_factor_10hz']:>7.2f}x "
                  f"{tp['realtime_factor_100hz']:>8.2f}x "
                  f"{imp['trans_rmse']:>+7.1f}% "
                  f"{imp['rot_rmse']:>+7.1f}% "
                  f"{imp['combined']:>+8.1f}%")

    # Save results.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, f"exp34_{timestamp}.json")

    output = {
        "config": {
            "window_size": args.window_size,
            "anchor_spacing": args.anchor_spacing,
            "n_outer_iters": args.n_outer_iters,
            "gn_iters": args.gn_iters,
            "lr": args.lr,
            "rw_trans": args.rw_trans,
            "rw_rot": args.rw_rot,
            "sw": args.sw,
            "aw_trans": args.aw_trans,
            "aw_rot": args.aw_rot,
            "inner_anchor_sigma": args.inner_anchor_sigma,
            "outer_loop": "lax.fori_loop (fused)",
        },
        "sequences": all_results,
        "total_time_s": round(t_total, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
