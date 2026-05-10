# experiments/exp37_downstream_eval.py
"""
Downstream trajectory evaluation for the exp36 real-time denoiser.

Quantifies end-to-end trajectory improvement using standard SLAM metrics
(ATE, RPE, KITTI) across multiple noise levels.  This demonstrates that
per-edge measurement denoising translates to meaningful trajectory-level
gains — the key result for publication.

Pipeline (per sequence, per noise level, per seed):
    1. Load KITTI GT poses → compute GT relative measurements
    2. Inject noise at specified σ level
    3. Run exp36 denoiser on noisy measurements
    4. Forward-compose noisy and denoised trajectories from GT first pose
    5. Compute ATE/RPE/KITTI metrics for both vs GT trajectory
    6. Report improvement percentages

Usage:
    python -m experiments.exp37_downstream_eval \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences --seq 06 \
        --sigma-levels 0.03
    python -m experiments.exp37_downstream_eval \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences \
        --n-seeds 5
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


# ---------------------------------------------------------------------------
# KITTI loading
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
# Forward composition (vectorised)
# ---------------------------------------------------------------------------

def _forward_compose_scan(start_pose: jnp.ndarray, measurements: jnp.ndarray) -> jnp.ndarray:
    """Forward-compose measurements via lax.scan (single JIT dispatch)."""
    def scan_body(carry, meas):
        new_pose = compose_pose_se3(carry, meas)
        return new_pose, new_pose
    _, poses = jax.lax.scan(scan_body, start_pose, measurements)
    return jnp.concatenate([start_pose[None], poses])

_forward_compose_jit = jax.jit(_forward_compose_scan)


def reconstruct_trajectory(start_pose: jnp.ndarray, measurements: jnp.ndarray) -> np.ndarray:
    """Forward-compose measurements from a starting pose to build a trajectory."""
    return np.array(_forward_compose_jit(start_pose, measurements))


# ---------------------------------------------------------------------------
# Trajectory metrics (from exp28)
# ---------------------------------------------------------------------------

def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Absolute Trajectory Error: per-pose error vs ground truth."""
    te = np.linalg.norm(estimated[:, :3] - ground_truth[:, :3], axis=1)
    re = np.linalg.norm(estimated[:, 3:] - ground_truth[:, 3:], axis=1)
    return {
        "ate_trans_rmse": float(np.sqrt(np.mean(te ** 2))),
        "ate_trans_mean": float(np.mean(te)),
        "ate_trans_max": float(np.max(te)),
        "ate_rot_rmse": float(np.sqrt(np.mean(re ** 2))),
        "ate_rot_mean": float(np.mean(re)),
    }


def compute_rpe(denoised_meas: np.ndarray, gt_meas: np.ndarray) -> dict:
    """Relative Pose Error: direct measurement comparison (no trajectory round-trip)."""
    d = denoised_meas - gt_meas
    te = np.linalg.norm(d[:, :3], axis=1)
    re = np.linalg.norm(d[:, 3:], axis=1)
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(te ** 2))),
        "rpe_trans_mean": float(np.mean(te)),
        "rpe_rot_rmse": float(np.sqrt(np.mean(re ** 2))),
        "rpe_rot_mean": float(np.mean(re)),
    }


def compute_kitti_metric(estimated: np.ndarray, ground_truth: np.ndarray,
                         step_size: int = 10) -> dict:
    """KITTI-style metric: translation error (%) and rotation error (deg/m)."""
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    n = estimated.shape[0]

    # Build all (start, end) pairs, then batch compute.
    starts, ends = [], []
    for start in range(0, n, step_size):
        for length in lengths:
            end = start + length
            if end >= n:
                continue
            starts.append(start)
            ends.append(end)

    if not starts:
        return {"kitti_trans_err_pct": 0.0, "kitti_rot_err_degm": 0.0}

    starts = np.array(starts)
    ends = np.array(ends)

    # Batch relative pose computation.
    gt_jnp = jnp.array(ground_truth)
    est_jnp = jnp.array(estimated)
    gt_rels = np.array(jax.vmap(relative_pose_se3)(gt_jnp[starts], gt_jnp[ends]))
    est_rels = np.array(jax.vmap(relative_pose_se3)(est_jnp[starts], est_jnp[ends]))

    path_lens = np.linalg.norm(gt_rels[:, :3], axis=1)
    valid = path_lens >= 1.0

    if not np.any(valid):
        return {"kitti_trans_err_pct": 0.0, "kitti_rot_err_degm": 0.0}

    t_errs = (np.linalg.norm(est_rels[valid, :3] - gt_rels[valid, :3], axis=1)
              / path_lens[valid] * 100.0)
    r_errs = (np.degrees(np.linalg.norm(est_rels[valid, 3:] - gt_rels[valid, 3:], axis=1))
              / path_lens[valid])

    return {
        "kitti_trans_err_pct": float(np.mean(t_errs)),
        "kitti_rot_err_degm": float(np.mean(r_errs)),
    }


# ---------------------------------------------------------------------------
# Noise estimation (from exp36)
# ---------------------------------------------------------------------------

def estimate_noise_model(measurements: np.ndarray) -> dict:
    """Batch MAD noise estimation."""
    diffs = measurements[1:] - measurements[:-1]
    mad = np.median(np.abs(diffs - np.median(diffs, axis=0)), axis=0)
    sigma_noise = mad / (np.sqrt(2) * 0.6745)
    var_diffs = np.var(diffs, axis=0)
    var_process = np.maximum(var_diffs - 2 * sigma_noise ** 2, 1e-12)
    sigma_process = np.sqrt(var_process)

    sigma_noise_trans = float(np.mean(sigma_noise[:3]))
    sigma_noise_rot = float(np.mean(sigma_noise[3:]))
    sigma_process_trans = float(np.mean(sigma_process[:3]))
    sigma_process_rot = float(np.mean(sigma_process[3:]))

    return {
        "sigma_noise_per_comp": sigma_noise.tolist(),
        "sigma_process_per_comp": sigma_process.tolist(),
        "sigma_noise_trans": sigma_noise_trans,
        "sigma_noise_rot": sigma_noise_rot,
        "sigma_process_trans": sigma_process_trans,
        "sigma_process_rot": sigma_process_rot,
    }


def estimate_noise_online(
    measurements: np.ndarray,
    alpha: float = 0.05,
    init_size: int = 20,
) -> dict:
    """Online noise estimation via EMA and sign-based running median."""
    diffs = measurements[1:] - measurements[:-1]
    n_diffs = len(diffs)
    actual_init = min(init_size, n_diffs)

    init_diffs = diffs[:actual_init]
    running_median = np.median(init_diffs, axis=0)
    running_mad = np.median(np.abs(init_diffs - running_median), axis=0)
    running_var = np.var(init_diffs, axis=0)

    for i in range(actual_init, n_diffs):
        d = diffs[i]
        step = alpha * np.maximum(running_mad, 1e-10)
        running_median += step * np.sign(d - running_median)
        abs_dev = np.abs(d - running_median)
        running_mad = (1.0 - alpha) * running_mad + alpha * abs_dev
        running_var = (1.0 - alpha) * running_var + alpha * (d - running_median) ** 2

    sigma_noise = running_mad / (np.sqrt(2) * 0.6745)
    var_process = np.maximum(running_var - 2 * sigma_noise ** 2, 1e-12)
    sigma_process = np.sqrt(var_process)

    sigma_noise_trans = float(np.mean(sigma_noise[:3]))
    sigma_noise_rot = float(np.mean(sigma_noise[3:]))
    sigma_process_trans = float(np.mean(sigma_process[:3]))
    sigma_process_rot = float(np.mean(sigma_process[3:]))

    return {
        "sigma_noise_per_comp": sigma_noise.tolist(),
        "sigma_process_per_comp": sigma_process.tolist(),
        "sigma_noise_trans": sigma_noise_trans,
        "sigma_noise_rot": sigma_noise_rot,
        "sigma_process_trans": sigma_process_trans,
        "sigma_process_rot": sigma_process_rot,
    }


# ---------------------------------------------------------------------------
# Auto-weights from noise model (from exp36)
# ---------------------------------------------------------------------------

def compute_auto_weights(noise_model: dict, *, base_sw: float = 1.0,
                         base_rw: float = 1.0) -> dict:
    """Derive sw and rw from the estimated noise model."""
    sigma_p_t = noise_model["sigma_process_trans"]
    sigma_p_r = noise_model["sigma_process_rot"]
    sigma_n_t = noise_model["sigma_noise_trans"]
    sigma_n_r = noise_model["sigma_noise_rot"]

    sw_trans = base_sw / max(sigma_p_t ** 2, 1e-12)
    sw_rot = base_sw / max(sigma_p_r ** 2, 1e-12)
    rw_trans = base_rw / max(sigma_n_t ** 2, 1e-12)
    rw_rot = base_rw / max(sigma_n_r ** 2, 1e-12)

    snr_trans = sigma_p_t / max(sigma_n_t, 1e-12)
    snr_rot = sigma_p_r / max(sigma_n_r, 1e-12)

    return {
        "sw_trans": sw_trans,
        "sw_rot": sw_rot,
        "rw_trans": rw_trans,
        "rw_rot": rw_rot,
        "snr_trans": snr_trans,
        "snr_rot": snr_rot,
    }


# ---------------------------------------------------------------------------
# Bilevel denoiser — IFT + two-phase optimisation (from exp36)
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
    sw_trans: float = 25.0,
    sw_rot: float = 25.0,
    inner_anchor_sigma: float = 0.01,
    n_trans_iters: int = 50,
    n_rot_iters: int = 20,
    lr: float = 1e-3,
):
    """Build a JIT-compiled denoiser with IFT backward pass and two-phase Adam."""
    n_meas = n_poses - 1
    odom_w = sigma_to_weight(sigma)
    sqrt_odom_w = jnp.sqrt(odom_w)
    anchor_w = sigma_to_weight(jnp.full(6, inner_anchor_sigma))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_positions, dtype=jnp.int32)

    anchor_w_vec = jnp.array(
        [aw_trans] * 3 + [aw_rot] * 3, dtype=jnp.float32)

    reg_w_vec = jnp.array(
        [rw_trans] * 3 + [rw_rot] * 3, dtype=jnp.float32)

    sw_vec = jnp.array(
        [sw_trans] * 3 + [sw_rot] * 3, dtype=jnp.float32)

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

    @jax.custom_vjp
    def inner_solve(theta, x_init, anchor_targets):
        def scan_body(x, _):
            return gn_step(x, theta, anchor_targets), None
        x_star, _ = jax.lax.scan(scan_body, x_init, None, length=gn_iters)
        return x_star

    def inner_solve_fwd(theta, x_init, anchor_targets):
        x_star = inner_solve(theta, x_init, anchor_targets)
        return x_star, (x_star, theta, anchor_targets)

    def inner_solve_bwd(res, g):
        x_star, theta, anchor_targets = res
        r_fn_x = lambda x_: residual_fn(x_, theta, anchor_targets)
        J = jax.jacobian(r_fn_x)(x_star)
        n = x_star.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        u = jnp.linalg.solve(H, g)
        _, vjp_fn = jax.vjp(
            lambda t: residual_fn(x_star, t, anchor_targets), theta)
        v = J @ u
        dtheta = -vjp_fn(v)[0]
        return (dtheta, jnp.zeros_like(x_star), jnp.zeros_like(anchor_targets))

    inner_solve.defvjp(inner_solve_fwd, inner_solve_bwd)

    def outer_loss(theta, x_init, anchor_targets, noisy_meas):
        x_star = inner_solve(theta, x_init, anchor_targets)
        poses_opt = x_star.reshape(n_poses, 6)

        diffs = poses_opt[anchor_idx] - anchor_targets
        a_loss = jnp.sum(anchor_w_vec * diffs ** 2)

        dev = theta - noisy_meas
        r_loss = jnp.sum(reg_w_vec * dev ** 2)

        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(sw_vec * s_diffs ** 2)

        return a_loss + r_loss + s_loss

    grad_fn = jax.grad(outer_loss)

    trans_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    rot_mask = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    def fused_optimize(theta_init, x_init, anchor_targets, noisy_meas):
        """Two-phase Adam: translation first, then rotation (fresh state)."""
        m0 = jnp.zeros_like(theta_init)
        v0 = jnp.zeros_like(theta_init)

        def trans_adam_body(i, state):
            theta, m, v = state
            g = grad_fn(theta, x_init, anchor_targets, noisy_meas)
            g = g * trans_mask

            t = (i + 1).astype(jnp.float32)
            m_new = 0.9 * m + 0.1 * g
            v_new = 0.999 * v + 0.001 * g ** 2
            m_hat = m_new / (1.0 - 0.9 ** t)
            v_hat = v_new / (1.0 - 0.999 ** t)
            update = lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return (theta - update, m_new, v_new)

        state = jax.lax.fori_loop(
            0, n_trans_iters, trans_adam_body, (theta_init, m0, v0))
        theta_after_trans = state[0]

        m0_rot = jnp.zeros_like(theta_after_trans)
        v0_rot = jnp.zeros_like(theta_after_trans)

        def rot_adam_body(i, state):
            theta, m, v = state
            g = grad_fn(theta, x_init, anchor_targets, noisy_meas)
            g = g * rot_mask

            t = (i + 1).astype(jnp.float32)
            m_new = 0.9 * m + 0.1 * g
            v_new = 0.999 * v + 0.001 * g ** 2
            m_hat = m_new / (1.0 - 0.9 ** t)
            v_hat = v_new / (1.0 - 0.999 ** t)
            update = lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return (theta - update, m_new, v_new)

        state = jax.lax.fori_loop(
            0, n_rot_iters, rot_adam_body, (theta_after_trans, m0_rot, v0_rot))
        return state[0]

    fused_optimize_jit = jax.jit(fused_optimize)

    return fused_optimize_jit


# ---------------------------------------------------------------------------
# Core evaluation: denoise + compute trajectory metrics
# ---------------------------------------------------------------------------

def evaluate_sequence(
    gt_poses: jnp.ndarray,
    sigma: jnp.ndarray,
    seq_id: str,
    seed: int,
    fused_opt,
    *,
    window_size: int = 100,
    anchor_spacing: int = 100,
) -> dict:
    """Denoise a sequence and compute trajectory-level metrics.

    The fused_opt function must be pre-built and JIT-warmed to avoid
    recompilation on every call.
    """
    n_poses_total = gt_poses.shape[0]
    n_meas_total = n_poses_total - 1

    # Compute GT measurements.
    gt_measurements = jax.vmap(relative_pose_se3)(gt_poses[:-1], gt_poses[1:])

    # Inject noise.
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, shape=gt_measurements.shape) * sigma
    noisy_measurements = gt_measurements + noise

    # Plan windows.
    overlap = min(10, window_size // 5)
    stride = window_size - overlap
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
    anchor_pos_in_window = list(range(0, actual_window, anchor_spacing))
    if anchor_pos_in_window[-1] != actual_window - 1:
        anchor_pos_in_window.append(actual_window - 1)

    # Commit ranges.
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

    # Denoise.
    denoised_measurements = np.array(noisy_measurements).copy()

    for wi, (w_start, w_end) in enumerate(windows):
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1
        w_noisy = jnp.array(noisy_measurements[w_start:w_start + w_n_meas])

        # Anchor targets.
        gt_first = gt_poses[w_start]
        anchor_global = gt_poses[w_start + jnp.array(anchor_pos_in_window)]
        w_anchor_targets = jax.vmap(relative_pose_se3, in_axes=(None, 0))(
            gt_first, anchor_global)

        # x_init.
        origin = jnp.zeros(6, dtype=jnp.float32)
        x_init = _forward_compose_jit(origin, w_noisy).ravel()

        # Optimise.
        theta_opt = fused_opt(w_noisy, x_init, w_anchor_targets, w_noisy)
        theta_opt.block_until_ready()

        # Commit.
        theta_np = np.array(theta_opt)
        commit_start, commit_end = commit_ranges[wi]
        for gi in range(commit_start, commit_end + 1):
            local_i = gi - w_start
            if 0 <= local_i < w_n_meas and gi < n_meas_total:
                denoised_measurements[gi] = theta_np[local_i]

    # Reconstruct trajectories.
    start_pose = gt_poses[0]
    gt_meas_np = np.array(gt_measurements)
    noisy_meas_np = np.array(noisy_measurements)
    denoised_meas_np = np.array(denoised_measurements)

    gt_traj = reconstruct_trajectory(start_pose, gt_measurements)
    noisy_traj = reconstruct_trajectory(start_pose, jnp.array(noisy_measurements))
    denoised_traj = reconstruct_trajectory(start_pose, jnp.array(denoised_measurements))

    # Compute metrics.
    # RPE: direct measurement comparison (no trajectory round-trip).
    # ATE/KITTI: trajectory-level (requires forward-composed poses).
    noisy_metrics = {
        **compute_ate(noisy_traj, gt_traj),
        **compute_rpe(noisy_meas_np, gt_meas_np),
        **compute_kitti_metric(noisy_traj, gt_traj),
    }
    denoised_metrics = {
        **compute_ate(denoised_traj, gt_traj),
        **compute_rpe(denoised_meas_np, gt_meas_np),
        **compute_kitti_metric(denoised_traj, gt_traj),
    }

    # Improvement percentages.
    improvement = {}
    for key in ["ate_trans_rmse", "ate_rot_rmse",
                "rpe_trans_rmse", "rpe_rot_rmse",
                "kitti_trans_err_pct", "kitti_rot_err_degm"]:
        noisy_val = noisy_metrics[key]
        denoised_val = denoised_metrics[key]
        if noisy_val > 1e-12:
            improvement[key] = round(
                (noisy_val - denoised_val) / noisy_val * 100, 2)
        else:
            improvement[key] = 0.0

    return {
        "sequence": seq_id,
        "seed": seed,
        "n_poses": n_poses_total,
        "sigma_trans": float(sigma[0]),
        "sigma_rot": float(sigma[3]),
        "noisy": noisy_metrics,
        "denoised": denoised_metrics,
        "improvement_pct": improvement,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp37: downstream trajectory evaluation for exp36 denoiser")
    parser.add_argument("--sequences-dir", type=str, required=True,
                        help="Path to SemanticKITTI sequences directory")
    parser.add_argument("--seq", type=str, default=None,
                        help="Comma-separated sequence IDs (default: all)")
    parser.add_argument("--n-poses", type=int, default=None,
                        help="Limit poses per sequence (default: all)")
    parser.add_argument("--sigma-levels", type=str,
                        default="0.01,0.03,0.05,0.10",
                        help="Comma-separated σ_trans levels to sweep")
    parser.add_argument("--sigma-rot-ratio", type=float, default=1.0/3.0,
                        help="σ_rot = σ_trans * ratio (default: 1/3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of noise seeds to run")
    parser.add_argument("--output-dir", type=str,
                        default="/data/tkocher/exp_res")

    # Denoiser config (matching exp36 defaults)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--anchor-spacing", type=int, default=100)
    parser.add_argument("--gn-iters", type=int, default=10)
    parser.add_argument("--n-trans-iters", type=int, default=50)
    parser.add_argument("--n-rot-iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aw-trans", type=float, default=5.0)
    parser.add_argument("--aw-rot", type=float, default=5.0)
    parser.add_argument("--inner-anchor-sigma", type=float, default=0.01)
    parser.add_argument("--base-sw", type=float, default=1.0)
    parser.add_argument("--base-rw", type=float, default=1.0)

    args = parser.parse_args()

    sigma_levels = [float(s) for s in args.sigma_levels.split(",")]
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    print("=" * 70)
    print("  exp37 -- Downstream Trajectory Evaluation")
    print("=" * 70)
    _print_device_info()
    print()
    print(f"  σ_trans levels: {sigma_levels}")
    print(f"  σ_rot ratio:    {args.sigma_rot_ratio:.4f}")
    print(f"  Seeds:          {args.n_seeds} ({seeds[0]}..{seeds[-1]})")
    print(f"  Denoiser:       window={args.window_size}, "
          f"iters={args.n_trans_iters}+{args.n_rot_iters}, "
          f"gn={args.gn_iters}")
    print()

    # Find and load sequences.
    sequences = find_sequences(args.sequences_dir, args.seq)
    if not sequences:
        print(f"ERROR: No sequences found in {args.sequences_dir}")
        return

    seq_data = []
    for seq_id, poses_path in sequences:
        gt_poses, data_info = load_kitti_poses(poses_path, args.n_poses)
        if gt_poses.shape[0] < args.window_size:
            print(f"  Sequence {seq_id}: skipping ({gt_poses.shape[0]} "
                  f"< {args.window_size} poses)")
            continue
        seq_data.append((seq_id, gt_poses))

    print(f"  Loaded {len(seq_data)} sequences: "
          f"{[s[0] for s in seq_data]}")
    print()

    # Output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp37_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "sigma_levels": sigma_levels,
        "sigma_rot_ratio": args.sigma_rot_ratio,
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "window_size": args.window_size,
        "anchor_spacing": args.anchor_spacing,
        "gn_iters": args.gn_iters,
        "n_trans_iters": args.n_trans_iters,
        "n_rot_iters": args.n_rot_iters,
        "lr": args.lr,
        "aw_trans": args.aw_trans,
        "aw_rot": args.aw_rot,
        "inner_anchor_sigma": args.inner_anchor_sigma,
        "base_sw": args.base_sw,
        "base_rw": args.base_rw,
    }

    all_results = []
    t_total_start = time.perf_counter()

    # Precompute window/anchor layout (constant across all evaluations).
    actual_window = min(args.window_size, min(gp.shape[0] for _, gp in seq_data))
    anchor_pos_in_window = list(range(0, actual_window, args.anchor_spacing))
    if anchor_pos_in_window[-1] != actual_window - 1:
        anchor_pos_in_window.append(actual_window - 1)

    for sigma_trans in sigma_levels:
        sigma_rot = sigma_trans * args.sigma_rot_ratio
        sigma = jnp.array(
            [sigma_trans] * 3 + [sigma_rot] * 3, dtype=jnp.float32)

        print(f"{'='*70}")
        print(f"  Noise level: σ_t={sigma_trans:.4f}, "
              f"σ_r={sigma_rot:.4f}")
        print(f"{'='*70}")

        # Build denoiser ONCE per noise level.
        # Use the injected sigma as inner_sigma (known noise level).
        # Auto-weights: estimate from a representative sequence to get sw/rw.
        rep_gt = seq_data[0][1]
        rep_meas = jax.vmap(relative_pose_se3)(rep_gt[:-1], rep_gt[1:])
        rep_key = jax.random.PRNGKey(seeds[0])
        rep_noisy = rep_meas + jax.random.normal(
            rep_key, shape=rep_meas.shape) * sigma
        noise_model = estimate_noise_online(np.array(rep_noisy))
        auto_w = compute_auto_weights(
            noise_model, base_sw=args.base_sw, base_rw=args.base_rw)

        print(f"  Building denoiser (JIT compile)...", end="", flush=True)
        t_jit_start = time.perf_counter()

        fused_opt = build_denoiser(
            actual_window, anchor_pos_in_window, sigma,
            gn_iters=args.gn_iters, gn_damping=5e-3,
            aw_trans=args.aw_trans, aw_rot=args.aw_rot,
            rw_trans=auto_w["rw_trans"], rw_rot=auto_w["rw_rot"],
            sw_trans=auto_w["sw_trans"], sw_rot=auto_w["sw_rot"],
            inner_anchor_sigma=args.inner_anchor_sigma,
            n_trans_iters=args.n_trans_iters,
            n_rot_iters=args.n_rot_iters,
            lr=args.lr)

        # Warm-up JIT.
        n_meas_window = actual_window - 1
        dummy_theta = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
        dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
        dummy_anchors = jnp.zeros(
            (len(anchor_pos_in_window), 6), dtype=jnp.float32)
        _ = fused_opt(
            dummy_theta, dummy_x, dummy_anchors, dummy_theta
        ).block_until_ready()

        t_jit = time.perf_counter() - t_jit_start
        print(f" done ({t_jit:.1f}s)")
        print()

        level_results = []

        for seq_id, gt_poses in seq_data:
            for seed in seeds:
                print(f"    Seq {seq_id}, seed {seed}...", end="", flush=True)
                t0 = time.perf_counter()

                result = evaluate_sequence(
                    gt_poses, sigma, seq_id, seed, fused_opt,
                    window_size=args.window_size,
                    anchor_spacing=args.anchor_spacing,
                )

                dt = time.perf_counter() - t0
                imp = result["improvement_pct"]
                print(f" done ({dt:.1f}s) "
                      f"ATE_t:{imp['ate_trans_rmse']:+.1f}% "
                      f"ATE_r:{imp['ate_rot_rmse']:+.1f}% "
                      f"RPE_t:{imp['rpe_trans_rmse']:+.1f}% "
                      f"KITTI_t:{imp['kitti_trans_err_pct']:+.1f}%")

                level_results.append(result)
                all_results.append(result)

        # Per-level summary.
        if level_results:
            ate_t_vals = [r["improvement_pct"]["ate_trans_rmse"]
                         for r in level_results]
            ate_r_vals = [r["improvement_pct"]["ate_rot_rmse"]
                         for r in level_results]
            rpe_t_vals = [r["improvement_pct"]["rpe_trans_rmse"]
                         for r in level_results]
            rpe_r_vals = [r["improvement_pct"]["rpe_rot_rmse"]
                         for r in level_results]
            kitti_t_vals = [r["improvement_pct"]["kitti_trans_err_pct"]
                           for r in level_results]
            kitti_r_vals = [r["improvement_pct"]["kitti_rot_err_degm"]
                           for r in level_results]

            print(f"\n  Level summary (σ_t={sigma_trans}):")
            print(f"    ATE  trans: {np.mean(ate_t_vals):+.1f}% "
                  f"± {np.std(ate_t_vals):.1f}")
            print(f"    ATE  rot:   {np.mean(ate_r_vals):+.1f}% "
                  f"± {np.std(ate_r_vals):.1f}")
            print(f"    RPE  trans: {np.mean(rpe_t_vals):+.1f}% "
                  f"± {np.std(rpe_t_vals):.1f}")
            print(f"    RPE  rot:   {np.mean(rpe_r_vals):+.1f}% "
                  f"± {np.std(rpe_r_vals):.1f}")
            print(f"    KITTI trans: {np.mean(kitti_t_vals):+.1f}% "
                  f"± {np.std(kitti_t_vals):.1f}")
            print(f"    KITTI rot:  {np.mean(kitti_r_vals):+.1f}% "
                  f"± {np.std(kitti_r_vals):.1f}")
            print()

    t_total = time.perf_counter() - t_total_start

    # Final summary table.
    print()
    print("=" * 70)
    print("  SUMMARY — Improvement (%) by noise level")
    print("=" * 70)
    print()

    header = (f"  {'σ_t':>6s} {'σ_r':>6s}  "
              f"{'ATE_t':>7s} {'ATE_r':>7s}  "
              f"{'RPE_t':>7s} {'RPE_r':>7s}  "
              f"{'KITTI_t':>8s} {'KITTI_r':>8s}")
    sep = f"  {'-'*6} {'-'*6}  {'-'*7} {'-'*7}  {'-'*7} {'-'*7}  {'-'*8} {'-'*8}"
    print(header)
    print(sep)

    summary_rows = []
    for sigma_trans in sigma_levels:
        sigma_rot = sigma_trans * args.sigma_rot_ratio
        level_res = [r for r in all_results
                     if abs(r["sigma_trans"] - sigma_trans) < 1e-6]
        if not level_res:
            continue

        ate_t = np.mean([r["improvement_pct"]["ate_trans_rmse"]
                         for r in level_res])
        ate_r = np.mean([r["improvement_pct"]["ate_rot_rmse"]
                         for r in level_res])
        rpe_t = np.mean([r["improvement_pct"]["rpe_trans_rmse"]
                         for r in level_res])
        rpe_r = np.mean([r["improvement_pct"]["rpe_rot_rmse"]
                         for r in level_res])
        kitti_t = np.mean([r["improvement_pct"]["kitti_trans_err_pct"]
                           for r in level_res])
        kitti_r = np.mean([r["improvement_pct"]["kitti_rot_err_degm"]
                           for r in level_res])

        row = (f"  {sigma_trans:>6.4f} {sigma_rot:>6.4f}  "
               f"{ate_t:>+6.1f}% {ate_r:>+6.1f}%  "
               f"{rpe_t:>+6.1f}% {rpe_r:>+6.1f}%  "
               f"{kitti_t:>+7.1f}% {kitti_r:>+7.1f}%")
        print(row)
        summary_rows.append({
            "sigma_trans": sigma_trans,
            "sigma_rot": sigma_rot,
            "ate_trans_pct": round(ate_t, 2),
            "ate_rot_pct": round(ate_r, 2),
            "rpe_trans_pct": round(rpe_t, 2),
            "rpe_rot_pct": round(rpe_r, 2),
            "kitti_trans_pct": round(kitti_t, 2),
            "kitti_rot_pct": round(kitti_r, 2),
        })

    print()
    print(f"  Total time: {t_total:.1f}s")

    # Save results.
    output = {
        "config": config,
        "summary": summary_rows,
        "results": all_results,
        "total_time_s": round(t_total, 1),
    }
    output_path = os.path.join(run_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save summary text.
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("exp37: Downstream Trajectory Evaluation\n")
        f.write(f"{'='*70}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for sigma_trans in sigma_levels:
            sigma_rot = sigma_trans * args.sigma_rot_ratio
            level_res = [r for r in all_results
                         if abs(r["sigma_trans"] - sigma_trans) < 1e-6]
            if not level_res:
                continue
            ate_t = np.mean([r["improvement_pct"]["ate_trans_rmse"]
                             for r in level_res])
            ate_r = np.mean([r["improvement_pct"]["ate_rot_rmse"]
                             for r in level_res])
            rpe_t = np.mean([r["improvement_pct"]["rpe_trans_rmse"]
                             for r in level_res])
            rpe_r = np.mean([r["improvement_pct"]["rpe_rot_rmse"]
                             for r in level_res])
            kitti_t = np.mean([r["improvement_pct"]["kitti_trans_err_pct"]
                               for r in level_res])
            kitti_r = np.mean([r["improvement_pct"]["kitti_rot_err_degm"]
                               for r in level_res])
            f.write(f"  {sigma_trans:>6.4f} {sigma_rot:>6.4f}  "
                    f"{ate_t:>+6.1f}% {ate_r:>+6.1f}%  "
                    f"{rpe_t:>+6.1f}% {rpe_r:>+6.1f}%  "
                    f"{kitti_t:>+7.1f}% {kitti_r:>+7.1f}%\n")
        f.write(f"\nTotal time: {t_total:.1f}s\n")

    print(f"\n  Results: {output_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Run dir: {run_dir}/")


if __name__ == "__main__":
    main()
