# experiments/exp32_kitti_full_denoise.py
"""
Full-scale KITTI/SemanticKITTI denoising evaluation.

Runs the optimized bilevel denoiser (from exp31 findings) across all
sequences in a SemanticKITTI dataset directory. Uses real trajectory
geometry with calibrated synthetic noise.

Key configuration (from exp31 sweep):
  - Low rotation regularisation (rw_rot=0.1) — let optimizer correct freely
  - High temporal smoothness (sw=25) — dominant constraint
  - 2% GT anchor density — sparse but sufficient
  - Window size 65 with midpoint-of-overlap commit

Usage:
    python -m experiments.exp32_kitti_full_denoise \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences
    python -m experiments.exp32_kitti_full_denoise \
        --sequences-dir /path/to/sequences --seq 00 --n-poses 500
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
    """Load KITTI-format poses.txt and convert to 6D [tx,ty,tz,wx,wy,wz] vectors.

    Each line is a flattened 3x4 [R|t] matrix (12 floats, row-major).
    """
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
    """Find all valid sequence directories containing poses.txt.

    Returns list of (sequence_id, poses_path) tuples.
    """
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

def compute_per_pose_meas_error(
    measurements: jnp.ndarray,
    gt_measurements: jnp.ndarray,
) -> dict:
    """Per-edge measurement error split into trans and rot."""
    diff = measurements - gt_measurements
    trans_err = np.array(jnp.linalg.norm(diff[:, :3], axis=1))
    rot_err = np.array(jnp.linalg.norm(diff[:, 3:], axis=1))
    overall_err = np.array(jnp.linalg.norm(diff, axis=1))
    return {
        "per_edge_trans": trans_err,
        "per_edge_rot": rot_err,
        "per_edge_overall": overall_err,
        "mean_trans": float(np.mean(trans_err)),
        "mean_rot": float(np.mean(rot_err)),
        "mean_overall": float(np.mean(overall_err)),
        "rmse_trans": float(np.sqrt(np.mean(trans_err ** 2))),
        "rmse_rot": float(np.sqrt(np.mean(rot_err ** 2))),
    }


# ---------------------------------------------------------------------------
# Adam optimiser
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
# Bilevel denoiser
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
):
    """Build a JIT-compiled bilevel denoiser with optimized defaults."""
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

    grad_fn = jax.jit(jax.grad(outer_loss))
    loss_fn = jax.jit(outer_loss)
    return grad_fn, loss_fn


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

    # Anchor positions — use global spacing for consistent density.
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

    # Baseline error.
    baseline = compute_per_pose_meas_error(noisy_measurements, gt_measurements)

    # Build denoiser (JIT compile once per window size).
    t_jit_start = time.perf_counter()
    grad_fn, loss_fn = build_denoiser(
        actual_window, anchor_pos_in_window, sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        aw_trans=args.aw_trans, aw_rot=args.aw_rot,
        rw_trans=args.rw_trans, rw_rot=args.rw_rot,
        smooth_weight=args.sw,
        inner_anchor_sigma=args.inner_anchor_sigma)

    # Warm-up.
    n_meas_window = actual_window - 1
    dummy_theta = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_anchors = jnp.zeros((len(anchor_pos_in_window), 6),
                               dtype=jnp.float32)
    _ = grad_fn(dummy_theta, dummy_x, dummy_anchors,
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

    # Denoise window by window.
    t_denoise_start = time.perf_counter()
    denoised_measurements = np.array(noisy_measurements).copy()

    for wi, (w_start, w_end) in enumerate(windows):
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

        # Outer optimisation.
        theta = w_noisy.copy()
        adam_state = adam_init(theta)

        for it in range(args.n_outer_iters):
            g = grad_fn(theta, x_init, w_anchor_targets, w_noisy)
            g.block_until_ready()

            if jnp.any(jnp.isnan(g)):
                print(f"    Window {wi}: NaN at iter {it}, stopping")
                break

            update, adam_state = adam_step(g, adam_state, lr=args.lr)
            theta = theta - update

        # Commit.
        theta_np = np.array(theta)
        commit_start, commit_end = commit_ranges[wi]
        for gi in range(commit_start, commit_end + 1):
            local_i = gi - w_start
            if 0 <= local_i < w_n_meas and gi < n_meas_total:
                denoised_measurements[gi] = theta_np[local_i]

        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            elapsed = time.perf_counter() - t_denoise_start
            print(f"    Window {wi+1}/{len(windows)} done "
                  f"({elapsed:.1f}s elapsed)", flush=True)

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

    print(f"  Results: Trans RMSE {baseline['rmse_trans']:.4f} -> "
          f"{after['rmse_trans']:.4f} ({trans_improv_pct:+.1f}%), "
          f"Rot RMSE {baseline['rmse_rot']:.4f} -> "
          f"{after['rmse_rot']:.4f} ({rot_improv_pct:+.1f}%)")
    print(f"  Edges improved: trans {n_trans_improved}/{n_meas_total}, "
          f"rot {n_rot_improved}/{n_meas_total}")
    print(f"  Timing: JIT {t_jit:.1f}s, denoise {t_denoise:.1f}s "
          f"({len(windows)} windows)")

    return {
        "sequence": seq_id,
        "n_poses": n_poses_total,
        "trajectory_length_m": round(traj_len, 1),
        "n_windows": len(windows),
        "anchor_density_pct": round(anchor_density, 1),
        "baseline": {
            "trans_rmse": round(baseline['rmse_trans'], 6),
            "rot_rmse": round(baseline['rmse_rot'], 6),
            "trans_mean": round(baseline['mean_trans'], 6),
            "rot_mean": round(baseline['mean_rot'], 6),
        },
        "denoised": {
            "trans_rmse": round(after['rmse_trans'], 6),
            "rot_rmse": round(after['rmse_rot'], 6),
            "trans_mean": round(after['mean_trans'], 6),
            "rot_mean": round(after['mean_rot'], 6),
        },
        "improvement_pct": {
            "trans_rmse": round(trans_improv_pct, 1),
            "rot_rmse": round(rot_improv_pct, 1),
            "trans_mean": round(
                (1 - after['mean_trans'] / baseline['mean_trans']) * 100, 1),
            "rot_mean": round(
                (1 - after['mean_rot'] / baseline['mean_rot']) * 100, 1),
        },
        "per_edge": {
            "trans_improved": n_trans_improved,
            "rot_improved": n_rot_improved,
            "total_edges": n_meas_total,
        },
        "timing": {
            "jit_s": round(t_jit, 2),
            "denoise_s": round(t_denoise, 2),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp32: Full-scale KITTI denoising evaluation")
    parser.add_argument("--sequences-dir", type=str, required=True,
                        help="Path to SemanticKITTI sequences directory")
    parser.add_argument("--seq", type=str, default=None,
                        help="Comma-separated sequence IDs to run "
                        "(default: all with poses.txt)")
    parser.add_argument("--n-poses", type=int, default=None,
                        help="Limit poses per sequence (default: use all)")
    parser.add_argument("--window-size", type=int, default=65,
                        help="Poses per solve window (default: 65)")
    parser.add_argument("--anchor-spacing", type=int, default=65,
                        help="GT anchor every N poses (default: 65, ~2%% density)")
    parser.add_argument("--sigma-trans", type=float, default=0.10)
    parser.add_argument("--sigma-rot", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-outer-iters", type=int, default=100)
    parser.add_argument("--gn-iters", type=int, default=10)
    parser.add_argument("--aw-trans", type=float, default=5.0)
    parser.add_argument("--aw-rot", type=float, default=5.0)
    parser.add_argument("--rw-trans", type=float, default=0.25)
    parser.add_argument("--rw-rot", type=float, default=0.1)
    parser.add_argument("--sw", type=float, default=25.0,
                        help="Smoothness weight (default: 25.0)")
    parser.add_argument("--inner-anchor-sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/data/tkocher/exp_res",
                        help="Directory for result files (default: /data/tkocher/exp_res)")
    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)

    print("=" * 70)
    print("  exp32 -- Full-Scale KITTI Denoising Evaluation")
    print("=" * 70)
    _print_device_info()
    print()
    print(f"  Config: window={args.window_size}, anchor_spacing={args.anchor_spacing}")
    print(f"  Weights: rw_t={args.rw_trans}, rw_r={args.rw_rot}, sw={args.sw}")
    print(f"  Noise: sigma_t={args.sigma_trans}, sigma_r={args.sigma_rot}")
    print()

    # Find sequences.
    sequences = find_sequences(args.sequences_dir, args.seq)
    if not sequences:
        print(f"ERROR: No sequences found in {args.sequences_dir}")
        return

    print(f"Found {len(sequences)} sequences: "
          f"{[s[0] for s in sequences]}")

    # Run denoising on each sequence.
    all_results = []
    t_total_start = time.perf_counter()

    for seq_id, poses_path in sequences:
        gt_poses, data_info = load_kitti_poses(poses_path, args.n_poses)
        n_loaded = gt_poses.shape[0]

        if n_loaded < args.window_size:
            print(f"\n  Sequence {seq_id}: only {n_loaded} poses, "
                  f"skipping (need >= {args.window_size})")
            continue

        result = denoise_sequence(gt_poses, args, sigma, seq_id=seq_id)
        all_results.append(result)

    t_total = time.perf_counter() - t_total_start

    # Aggregate summary.
    print()
    print("=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)
    print()

    if all_results:
        print(f"  {'Seq':>4s} {'Poses':>6s} {'Length':>8s} "
              f"{'T_RMSE%':>8s} {'R_RMSE%':>8s} "
              f"{'T_edges':>8s} {'R_edges':>8s} {'Time':>7s}")
        print(f"  {'-'*4:>4s} {'-'*6:>6s} {'-'*8:>8s} "
              f"{'-'*8:>8s} {'-'*8:>8s} "
              f"{'-'*8:>8s} {'-'*8:>8s} {'-'*7:>7s}")

        total_edges = 0
        total_trans_imp = 0
        total_rot_imp = 0
        weighted_trans_pct = 0.0
        weighted_rot_pct = 0.0

        for r in all_results:
            n_edges = r['per_edge']['total_edges']
            total_edges += n_edges
            total_trans_imp += r['per_edge']['trans_improved']
            total_rot_imp += r['per_edge']['rot_improved']
            weighted_trans_pct += r['improvement_pct']['trans_rmse'] * n_edges
            weighted_rot_pct += r['improvement_pct']['rot_rmse'] * n_edges

            print(f"  {r['sequence']:>4s} "
                  f"{r['n_poses']:>6d} "
                  f"{r['trajectory_length_m']:>7.1f}m "
                  f"{r['improvement_pct']['trans_rmse']:>+7.1f}% "
                  f"{r['improvement_pct']['rot_rmse']:>+7.1f}% "
                  f"{r['per_edge']['trans_improved']:>4d}/"
                  f"{n_edges:<4d}"
                  f"{r['per_edge']['rot_improved']:>4d}/"
                  f"{n_edges:<4d}"
                  f"{r['timing']['denoise_s']:>6.1f}s")

        avg_trans = weighted_trans_pct / total_edges
        avg_rot = weighted_rot_pct / total_edges
        combined = (avg_trans + avg_rot) / 2

        print()
        print(f"  Weighted average: Trans {avg_trans:+.1f}%, "
              f"Rot {avg_rot:+.1f}%, Combined {combined:+.1f}%")
        print(f"  Total edges: {total_edges}, "
              f"trans improved: {total_trans_imp} "
              f"({total_trans_imp/total_edges*100:.1f}%), "
              f"rot improved: {total_rot_imp} "
              f"({total_rot_imp/total_edges*100:.1f}%)")
        print(f"  Total time: {t_total:.1f}s")

    # Save results.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, f"exp32_{timestamp}.json")

    output = {
        "config": {
            "window_size": args.window_size,
            "anchor_spacing": args.anchor_spacing,
            "sigma_trans": args.sigma_trans,
            "sigma_rot": args.sigma_rot,
            "rw_trans": args.rw_trans,
            "rw_rot": args.rw_rot,
            "sw": args.sw,
            "gn_iters": args.gn_iters,
            "outer_iters": args.n_outer_iters,
            "lr": args.lr,
            "aw_trans": args.aw_trans,
            "aw_rot": args.aw_rot,
            "inner_anchor_sigma": args.inner_anchor_sigma,
        },
        "sequences": all_results,
        "aggregate": {
            "n_sequences": len(all_results),
            "total_edges": total_edges,
            "weighted_trans_rmse_pct": round(avg_trans, 1),
            "weighted_rot_rmse_pct": round(avg_rot, 1),
            "combined_pct": round(combined, 1),
            "total_trans_improved": total_trans_imp,
            "total_rot_improved": total_rot_imp,
            "total_time_s": round(t_total, 1),
        } if all_results else {},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
