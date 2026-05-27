# experiments/exp40_windowed_denoise.py
"""
Real-time differentiable SE(3) measurement denoiser with windowed
noise decomposition.

Evolution of exp39 that fixes the Allan variance model mismatch on
structured (non-random-walk) trajectories.  The Allan approach assumes
σ²_A(τ) grows linearly with τ (random walk), but real odometry on
curved trajectories produces quadratic growth at large τ, inflating
σ_process and making regularisation dominate smoothness.

Fix: hybrid estimator that combines:
    - σ_noise:   Online MAD on consecutive differences (proven robust,
                  ~1.2× accuracy across all noise levels)
    - σ_process: Sliding-window batch variance decomposition
                 σ²_process = Var(diffs_in_window) - 2·σ²_noise
                 This mirrors the batch MAD formula but streams over
                 a causal window, avoiding both the EMA adaptive-center
                 collapse (exp36) and the Allan random-walk mismatch
                 (exp39).

The window size W controls the bias-variance trade-off:
    - Small W (~20):  fast adaptation, higher variance, may underestimate
                      σ_process in very smooth segments
    - Large W (~200): stable estimate, may blur regime changes
    - Default W=50:   ~5 seconds at 10Hz, long enough to capture process
                      variation, short enough for local adaptation

Changes from exp39:
    1. New `estimate_noise_windowed()` — sliding-window hybrid estimator
    2. Default --noise-method changed to 'windowed'
    3. Retains allan/ema/batch as fallback options for ablation
    4. New --var-window-size parameter

Usage:
    python -m experiments.exp40_windowed_denoise \\
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences --seq 00

    # Compare methods:
    python -m experiments.exp40_windowed_denoise \\
        --sequences-dir /path/to/sequences --noise-method windowed  # default
    python -m experiments.exp40_windowed_denoise \\
        --sequences-dir /path/to/sequences --noise-method allan     # exp39
    python -m experiments.exp40_windowed_denoise \\
        --sequences-dir /path/to/sequences --noise-method ema       # exp36
    python -m experiments.exp40_windowed_denoise \\
        --sequences-dir /path/to/sequences --noise-method batch     # non-causal
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

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
# Noise estimation — batch MAD
# ---------------------------------------------------------------------------

def estimate_noise_batch(measurements: np.ndarray) -> dict:
    """Batch noise estimation using MAD on full sequence."""
    diffs = measurements[1:] - measurements[:-1]

    mad = np.median(np.abs(diffs - np.median(diffs, axis=0)), axis=0)
    sigma_noise = mad / (np.sqrt(2) * 0.6745)

    var_diffs = np.var(diffs, axis=0)
    var_process = np.maximum(var_diffs - 2 * sigma_noise ** 2, 1e-12)
    sigma_process = np.sqrt(var_process)

    return {
        "sigma_noise_per_comp": sigma_noise.tolist(),
        "sigma_process_per_comp": sigma_process.tolist(),
        "sigma_noise_trans": float(np.mean(sigma_noise[:3])),
        "sigma_noise_rot": float(np.mean(sigma_noise[3:])),
        "sigma_process_trans": float(np.mean(sigma_process[:3])),
        "sigma_process_rot": float(np.mean(sigma_process[3:])),
        "method": "batch_mad",
    }


# ---------------------------------------------------------------------------
# Noise estimation — online EMA (exp36 fallback)
# ---------------------------------------------------------------------------

def estimate_noise_ema(
    measurements: np.ndarray,
    alpha: float = 0.05,
    init_size: int = 20,
) -> dict:
    """Online EMA noise estimation (exp36). WARNING: σ_process collapses."""
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

    return {
        "sigma_noise_per_comp": sigma_noise.tolist(),
        "sigma_process_per_comp": sigma_process.tolist(),
        "sigma_noise_trans": float(np.mean(sigma_noise[:3])),
        "sigma_noise_rot": float(np.mean(sigma_noise[3:])),
        "sigma_process_trans": float(np.mean(sigma_process[:3])),
        "sigma_process_rot": float(np.mean(sigma_process[3:])),
        "method": "online_ema",
    }


# ---------------------------------------------------------------------------
# Noise estimation — Allan variance (exp39 fallback)
# ---------------------------------------------------------------------------

def estimate_noise_allan(
    measurements: np.ndarray,
    alpha: float = 0.05,
    init_size: int = 20,
    cluster_sizes: tuple[int, ...] = (1, 2, 5, 10, 25),
) -> dict:
    """Allan variance decomposition (exp39). WARNING: overestimates σ_process
    on curved trajectories due to random-walk model mismatch."""
    n_meas, n_comp = measurements.shape

    # MAD for σ_noise
    diffs = measurements[1:] - measurements[:-1]
    n_diffs = len(diffs)
    actual_init = min(init_size, n_diffs)

    init_diffs = diffs[:actual_init]
    running_median = np.median(init_diffs, axis=0)
    running_mad = np.median(np.abs(init_diffs - running_median), axis=0)

    for i in range(actual_init, n_diffs):
        d = diffs[i]
        step = alpha * np.maximum(running_mad, 1e-10)
        running_median += step * np.sign(d - running_median)
        abs_dev = np.abs(d - running_median)
        running_mad = (1.0 - alpha) * running_mad + alpha * abs_dev

    sigma_noise = running_mad / (np.sqrt(2) * 0.6745)

    # Allan variance at multiple scales
    allan_var = {}
    for tau in cluster_sizes:
        if n_meas < 2 * tau:
            allan_var[tau] = np.full(n_comp, np.nan)
            continue
        cumsum = np.zeros((n_meas + 1, n_comp), dtype=np.float64)
        cumsum[1:] = np.cumsum(measurements.astype(np.float64), axis=0)
        n_clusters = n_meas - tau + 1
        cluster_sums = cumsum[tau:] - cumsum[:n_clusters]
        cluster_means = cluster_sums / tau
        n_pairs = n_clusters - tau
        if n_pairs < 1:
            allan_var[tau] = np.full(n_comp, np.nan)
            continue
        cluster_diffs = cluster_means[tau:] - cluster_means[:n_pairs]
        allan_var[tau] = 0.5 * np.mean(cluster_diffs ** 2, axis=0)

    # Fit σ_process from Allan curve
    valid_taus = [t for t in cluster_sizes if not np.any(np.isnan(allan_var[t]))]
    if len(valid_taus) >= 2:
        sigma_noise_sq = sigma_noise ** 2
        process_estimates = []
        for tau in valid_taus:
            residual = allan_var[tau] - sigma_noise_sq / tau
            process_estimates.append(3.0 * residual / tau)
        process_estimates = np.array(process_estimates)
        sigma_process_sq = np.median(process_estimates, axis=0)
        sigma_process_sq = np.maximum(sigma_process_sq, 1e-12)
        sigma_process = np.sqrt(sigma_process_sq)
    else:
        var_diffs = np.var(diffs, axis=0)
        var_process = np.maximum(var_diffs - 2 * sigma_noise ** 2, 1e-12)
        sigma_process = np.sqrt(var_process)

    return {
        "sigma_noise_per_comp": sigma_noise.tolist(),
        "sigma_process_per_comp": sigma_process.tolist(),
        "sigma_noise_trans": float(np.mean(sigma_noise[:3])),
        "sigma_noise_rot": float(np.mean(sigma_noise[3:])),
        "sigma_process_trans": float(np.mean(sigma_process[:3])),
        "sigma_process_rot": float(np.mean(sigma_process[3:])),
        "method": "allan",
    }


# ---------------------------------------------------------------------------
# Noise estimation — windowed hybrid (NEW — default)
# ---------------------------------------------------------------------------

def estimate_noise_windowed(
    measurements: np.ndarray,
    alpha: float = 0.05,
    init_size: int = 20,
    var_window: int = 50,
) -> dict:
    """Online noise estimation via MAD + sliding-window variance.

    Hybrid estimator combining:
        1. Online MAD on consecutive differences → σ_noise
           (robust, proven ~1.2× accuracy)
        2. Sliding-window batch variance → σ_process
           σ²_process = Var(diffs_in_window) - 2·σ²_noise

    The sliding window computes the SAME formula as batch MAD for
    σ_process, but restricted to a causal window of the last W
    differences.  This avoids:
        - EMA adaptive-center collapse (exp36): variance is computed
          around the window mean, not an adaptive median
        - Allan random-walk mismatch (exp39): no multi-scale model
          assumption, just the direct variance decomposition

    The window mean adapts slowly (changes by O(1/W) per step), so it
    tracks global trends without chasing the signal — unlike the EMA
    median which adapts at rate α ≈ 0.05.

    Parameters
    ----------
    measurements : (N, 6) array
    alpha : EMA decay for MAD updates
    init_size : batch init size for MAD
    var_window : sliding window size W for variance estimation.
        Default 50 (~5s at 10Hz).  Larger W → more stable σ_process
        estimate but slower adaptation to regime changes.
    """
    diffs = measurements[1:] - measurements[:-1]
    n_diffs = len(diffs)
    n_comp = diffs.shape[1]
    actual_init = min(init_size, n_diffs)

    # --- Phase 1: Online MAD for σ_noise ---
    init_diffs = diffs[:actual_init]
    running_median = np.median(init_diffs, axis=0)
    running_mad = np.median(np.abs(init_diffs - running_median), axis=0)

    for i in range(actual_init, n_diffs):
        d = diffs[i]
        step = alpha * np.maximum(running_mad, 1e-10)
        running_median += step * np.sign(d - running_median)
        abs_dev = np.abs(d - running_median)
        running_mad = (1.0 - alpha) * running_mad + alpha * abs_dev

    sigma_noise = running_mad / (np.sqrt(2) * 0.6745)
    sigma_noise_sq = sigma_noise ** 2

    # --- Phase 2: Sliding-window variance for σ_process ---
    #
    # Stream through diffs maintaining a deque of the last W values.
    # At each step, compute Var(window) using Welch's online algorithm
    # with insert/remove updates.
    #
    # For efficiency, maintain running sum and sum-of-squares:
    #   Var = (sum_sq / n) - (sum / n)²
    # Update on insert/remove: O(1) per step.

    W = min(var_window, n_diffs)

    if n_diffs <= W:
        # Sequence shorter than window — use full batch.
        var_diffs = np.var(diffs, axis=0)
    else:
        # Sliding window with O(1) updates.
        window = deque(maxlen=W)
        running_sum = np.zeros(n_comp, dtype=np.float64)
        running_sum_sq = np.zeros(n_comp, dtype=np.float64)

        # Fill initial window.
        for i in range(W):
            d = diffs[i].astype(np.float64)
            window.append(d)
            running_sum += d
            running_sum_sq += d ** 2

        # Stream remaining diffs, tracking the variance.
        # We keep the variance from the LAST window position (end of
        # sequence) as the final estimate.  This is causal: it uses
        # only measurements up to that point.
        for i in range(W, n_diffs):
            old_d = window[0]
            new_d = diffs[i].astype(np.float64)
            running_sum += new_d - old_d
            running_sum_sq += new_d ** 2 - old_d ** 2
            window.append(new_d)

        # Variance of final window.
        mean = running_sum / W
        var_diffs = (running_sum_sq / W - mean ** 2).astype(np.float64)

        # Bessel's correction for unbiased estimate.
        var_diffs = var_diffs * W / (W - 1)

    var_process = np.maximum(var_diffs - 2 * sigma_noise_sq, 1e-12)
    sigma_process = np.sqrt(var_process).astype(np.float64)

    # --- Aggregate ---
    sigma_noise_trans = float(np.mean(sigma_noise[:3]))
    sigma_noise_rot = float(np.mean(sigma_noise[3:]))
    sigma_process_trans = float(np.mean(sigma_process[:3]))
    sigma_process_rot = float(np.mean(sigma_process[3:]))

    return {
        "sigma_noise_per_comp": sigma_noise.tolist(),
        "sigma_process_per_comp": [float(x) for x in sigma_process],
        "sigma_noise_trans": sigma_noise_trans,
        "sigma_noise_rot": sigma_noise_rot,
        "sigma_process_trans": sigma_process_trans,
        "sigma_process_rot": sigma_process_rot,
        "var_window": W,
        "method": "windowed",
    }


# ---------------------------------------------------------------------------
# Auto-weights from noise model
# ---------------------------------------------------------------------------

def compute_auto_weights(noise_model: dict, *, base_sw: float = 1.0,
                         base_rw: float = 1.0) -> dict:
    """Derive sw and rw from the estimated noise model.

    sw  =  base_sw / σ²_process   (smoothness prior)
    rw  =  base_rw / σ²_noise     (observation fidelity)
    """
    sn_t = noise_model["sigma_noise_trans"]
    sn_r = noise_model["sigma_noise_rot"]
    sp_t = noise_model["sigma_process_trans"]
    sp_r = noise_model["sigma_process_rot"]

    sw_trans = base_sw / max(sp_t ** 2, 1e-12)
    sw_rot = base_sw / max(sp_r ** 2, 1e-12)
    rw_trans = base_rw / max(sn_t ** 2, 1e-12)
    rw_rot = base_rw / max(sn_r ** 2, 1e-12)

    snr_trans = sp_t / max(sn_t, 1e-12)
    snr_rot = sp_r / max(sn_r, 1e-12)

    return {
        "sw_trans": sw_trans,
        "sw_rot": sw_rot,
        "rw_trans": rw_trans,
        "rw_rot": rw_rot,
        "snr_trans": snr_trans,
        "snr_rot": snr_rot,
    }


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
# Bilevel denoiser — IFT + two-phase optimisation
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

    # --- Inner PGO residual and GN step ---

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

    # --- IFT inner solve via custom_vjp ---

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

    # --- Outer loss ---

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

    # --- Two-phase gradient masks ---
    trans_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    rot_mask = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    def fused_optimize(theta_init, x_init, anchor_targets, noisy_meas):
        """Two-phase Adam: translation first, then rotation (fresh state)."""

        # Phase 1 — Translation only
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

        # Phase 2 — Rotation only (fresh Adam state)
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
    seed: int = 42,
) -> dict:
    """Run the full denoising pipeline on a single sequence."""
    n_poses_total = gt_poses.shape[0]
    n_meas_total = n_poses_total - 1
    window_size = args.window_size
    overlap = min(10, window_size // 5)
    stride = window_size - overlap

    # Compute GT measurements (vectorised, single dispatch).
    gt_measurements = jax.vmap(relative_pose_se3)(gt_poses[:-1], gt_poses[1:])

    # Add calibrated noise.
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, shape=gt_measurements.shape) * sigma
    noisy_measurements = gt_measurements + noise

    # --- Noise estimation ---
    noisy_np = np.array(noisy_measurements)

    if args.noise_method == "batch":
        noise_model = estimate_noise_batch(noisy_np)
        noise_label = "batch MAD"
    elif args.noise_method == "ema":
        noise_model = estimate_noise_ema(
            noisy_np, alpha=args.noise_ema_alpha, init_size=args.noise_init_size)
        noise_label = f"online EMA (α={args.noise_ema_alpha})"
    elif args.noise_method == "allan":
        noise_model = estimate_noise_allan(
            noisy_np, alpha=args.noise_ema_alpha,
            init_size=args.noise_init_size,
            cluster_sizes=tuple(args.allan_cluster_sizes))
        noise_label = f"Allan (τ={args.allan_cluster_sizes})"
    else:  # windowed (default)
        noise_model = estimate_noise_windowed(
            noisy_np, alpha=args.noise_ema_alpha,
            init_size=args.noise_init_size,
            var_window=args.var_window_size)
        noise_label = f"windowed (W={args.var_window_size})"

    print(f"\n  Sequence {seq_id}: {n_poses_total} poses")
    print(f"  --- Noise estimation ({noise_label}) ---")
    print(f"  σ_noise:   trans={noise_model['sigma_noise_trans']:.5f}, "
          f"rot={noise_model['sigma_noise_rot']:.5f}")
    print(f"  σ_process: trans={noise_model['sigma_process_trans']:.5f}, "
          f"rot={noise_model['sigma_process_rot']:.5f}")

    if args.auto_weights:
        auto_w = compute_auto_weights(
            noise_model,
            base_sw=args.base_sw,
            base_rw=args.base_rw,
        )
        sw_trans = auto_w["sw_trans"]
        sw_rot = auto_w["sw_rot"]
        rw_trans = auto_w["rw_trans"]
        rw_rot = auto_w["rw_rot"]
        snr_trans = auto_w["snr_trans"]
        snr_rot = auto_w["snr_rot"]
        sw_rw_ratio_t = sw_trans / max(rw_trans, 1e-12)
        sw_rw_ratio_r = sw_rot / max(rw_rot, 1e-12)
        print(f"  --- Auto weights (base_sw={args.base_sw}, "
              f"base_rw={args.base_rw}) ---")
        print(f"  SNR: trans={snr_trans:.4f}, rot={snr_rot:.4f}")
        print(f"  sw/rw ratio: trans={sw_rw_ratio_t:.2f}, "
              f"rot={sw_rw_ratio_r:.2f}")
    else:
        sw_trans = args.sw_trans
        sw_rot = args.sw_rot
        rw_trans = args.rw_trans
        rw_rot = args.rw_rot
        snr_trans = None
        snr_rot = None
        sw_rw_ratio_t = sw_trans / max(rw_trans, 1e-12)
        sw_rw_ratio_r = sw_rot / max(rw_rot, 1e-12)
        print(f"  --- Manual weights ---")

    print(f"  sw_trans={sw_trans:.2f}, sw_rot={sw_rot:.2f}")
    print(f"  rw_trans={rw_trans:.2f}, rw_rot={rw_rot:.2f}")
    print(f"  lr={args.lr}")

    # Use estimated sigma for the inner solver information matrix.
    inner_sigma = jnp.array(
        noise_model["sigma_noise_per_comp"], dtype=jnp.float32)

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

    n_total_iters = args.n_trans_iters + args.n_rot_iters
    print(f"\n  Trajectory: {traj_len:.1f}m")
    print(f"  Windows: {len(windows)} (size={actual_window}, "
          f"stride={stride}, overlap={overlap})")
    print(f"  Anchors/window: {len(anchor_pos_in_window)} "
          f"({anchor_density:.1f}% density)")
    print(f"  Outer loop: Two-phase ({args.n_trans_iters} trans + "
          f"{args.n_rot_iters} rot = {n_total_iters} total)")
    print(f"  Inner GN: {args.gn_iters} iters (IFT backward)")

    # Baseline error.
    baseline = compute_per_pose_meas_error(noisy_measurements, gt_measurements)

    # Build denoiser.
    t_jit_start = time.perf_counter()
    fused_opt, grad_fn, loss_fn = build_denoiser(
        actual_window, anchor_pos_in_window, inner_sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        aw_trans=args.aw_trans, aw_rot=args.aw_rot,
        rw_trans=rw_trans, rw_rot=rw_rot,
        sw_trans=sw_trans, sw_rot=sw_rot,
        inner_anchor_sigma=args.inner_anchor_sigma,
        n_trans_iters=args.n_trans_iters,
        n_rot_iters=args.n_rot_iters,
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

        # Anchor targets in local coordinates (vectorised).
        gt_first = gt_poses[w_start]
        anchor_global = gt_poses[w_start + jnp.array(anchor_pos_in_window)]
        w_anchor_targets = jax.vmap(relative_pose_se3, in_axes=(None, 0))(
            gt_first, anchor_global)

        # x_init: forward-compose from origin (single JIT dispatch).
        origin = jnp.zeros(6, dtype=jnp.float32)
        x_init = _forward_compose_jit(origin, w_noisy).ravel()

        # Run fused two-phase optimization (single compiled kernel).
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
        "outer_iters": {
            "trans": args.n_trans_iters,
            "rot": args.n_rot_iters,
            "total": n_total_iters,
        },
        "noise_model": noise_model,
        "noise_method": noise_label,
        "weights_used": {
            "sw_trans": round(sw_trans, 4),
            "sw_rot": round(sw_rot, 4),
            "rw_trans": round(rw_trans, 4),
            "rw_rot": round(rw_rot, 4),
            "sw_rw_ratio_trans": round(sw_rw_ratio_t, 4),
            "sw_rw_ratio_rot": round(sw_rw_ratio_r, 4),
            "lr": args.lr,
            "auto": args.auto_weights,
            "snr_trans": round(snr_trans, 4) if snr_trans is not None else None,
            "snr_rot": round(snr_rot, 4) if snr_rot is not None else None,
        },
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
        description="exp40: real-time SE(3) denoiser with windowed "
                    "noise decomposition")
    parser.add_argument("--sequences-dir", type=str, required=True,
                        help="Path to SemanticKITTI sequences directory")
    parser.add_argument("--seq", type=str, default=None,
                        help="Comma-separated sequence IDs (default: all)")
    parser.add_argument("--n-poses", type=int, default=None,
                        help="Limit poses per sequence (default: all)")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--anchor-spacing", type=int, default=100)
    parser.add_argument("--sigma-trans", type=float, default=0.03)
    parser.add_argument("--sigma-rot", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-trans-iters", type=int, default=50,
                        help="Phase 1 (translation) Adam iterations")
    parser.add_argument("--n-rot-iters", type=int, default=20,
                        help="Phase 2 (rotation) Adam iterations")
    parser.add_argument("--gn-iters", type=int, default=10)
    parser.add_argument("--aw-trans", type=float, default=5.0)
    parser.add_argument("--aw-rot", type=float, default=5.0)
    parser.add_argument("--inner-anchor-sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of noise seeds to run (for statistics)")
    parser.add_argument("--output-dir", type=str,
                        default="/data/tkocher/exp_res")

    # Noise estimation method
    parser.add_argument("--noise-method", type=str, default="windowed",
                        choices=["windowed", "allan", "ema", "batch"],
                        help="Noise estimation method: 'windowed' (default), "
                             "'allan' (exp39), 'ema' (exp36), 'batch'")

    # Windowed estimator controls
    parser.add_argument("--var-window-size", type=int, default=50,
                        help="Sliding window size for variance estimation "
                             "(default: 50)")

    # Allan variance controls (for --noise-method allan)
    parser.add_argument("--allan-cluster-sizes", type=int, nargs="+",
                        default=[1, 2, 5, 10, 25],
                        help="Cluster sizes τ for Allan variance")

    # Auto-tuning controls
    parser.add_argument("--auto-weights", action="store_true", default=True,
                        help="Auto-derive sw and rw from noise model (default)")
    parser.add_argument("--no-auto-weights", dest="auto_weights",
                        action="store_false",
                        help="Use manual sw/rw values instead")
    parser.add_argument("--base-sw", type=float, default=1.0,
                        help="Base smoothness scale for auto weights")
    parser.add_argument("--base-rw", type=float, default=1.0,
                        help="Base regularization scale for auto weights")

    # Online noise estimation controls (for ema/allan/windowed)
    parser.add_argument("--noise-ema-alpha", type=float, default=0.05,
                        help="EMA decay rate for online MAD")
    parser.add_argument("--noise-init-size", type=int, default=20,
                        help="Number of initial diffs for batch init")

    # Manual weight overrides (used when --no-auto-weights)
    parser.add_argument("--sw-trans", type=float, default=35.0)
    parser.add_argument("--sw-rot", type=float, default=35.0)
    parser.add_argument("--rw-trans", type=float, default=0.25)
    parser.add_argument("--rw-rot", type=float, default=0.1)

    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)

    n_total_iters = args.n_trans_iters + args.n_rot_iters

    print("=" * 70)
    print("  exp40 -- SE(3) Denoiser with Windowed Noise Decomposition")
    print("=" * 70)
    _print_device_info()
    print()
    print(f"  Config: window={args.window_size}, "
          f"anchor_spacing={args.anchor_spacing}")
    print(f"  Noise injection: σ_t={args.sigma_trans}, σ_r={args.sigma_rot}")
    print(f"  Noise estimation: {args.noise_method.upper()}")
    if args.noise_method == "windowed":
        print(f"  Variance window: {args.var_window_size}")
    elif args.noise_method == "allan":
        print(f"  Allan cluster sizes: {args.allan_cluster_sizes}")
    if args.auto_weights:
        print(f"  Weights: AUTO (base_sw={args.base_sw}, "
              f"base_rw={args.base_rw})")
    else:
        print(f"  Weights: MANUAL sw_t={args.sw_trans}, sw_r={args.sw_rot}, "
              f"rw_t={args.rw_trans}, rw_r={args.rw_rot}")
    print(f"  Outer loop: Two-phase ({args.n_trans_iters} trans + "
          f"{args.n_rot_iters} rot = {n_total_iters}), lr={args.lr}")
    print(f"  Inner GN: {args.gn_iters} iters (IFT backward)")
    print()

    # Find sequences.
    sequences = find_sequences(args.sequences_dir, args.seq)
    if not sequences:
        print(f"ERROR: No sequences found in {args.sequences_dir}")
        return

    print(f"Found {len(sequences)} sequences: {[s[0] for s in sequences]}")

    n_seeds = args.n_seeds
    seeds = list(range(args.seed, args.seed + n_seeds))
    if n_seeds > 1:
        print(f"  Seeds: {n_seeds} (seed {seeds[0]}..{seeds[-1]})")

    # Pre-load all GT poses.
    seq_data = []
    for seq_id, poses_path in sequences:
        gt_poses, data_info = load_kitti_poses(poses_path, args.n_poses)
        if gt_poses.shape[0] < args.window_size:
            print(f"\n  Sequence {seq_id}: skipping ({gt_poses.shape[0]} "
                  f"< {args.window_size} poses)")
            continue
        seq_data.append((seq_id, gt_poses))

    # Output directory for this run.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp40_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "window_size": args.window_size,
        "anchor_spacing": args.anchor_spacing,
        "n_trans_iters": args.n_trans_iters,
        "n_rot_iters": args.n_rot_iters,
        "n_total_iters": n_total_iters,
        "gn_iters": args.gn_iters,
        "lr": args.lr,
        "sigma_trans": args.sigma_trans,
        "sigma_rot": args.sigma_rot,
        "auto_weights": args.auto_weights,
        "base_sw": args.base_sw,
        "base_rw": args.base_rw,
        "aw_trans": args.aw_trans,
        "aw_rot": args.aw_rot,
        "inner_anchor_sigma": args.inner_anchor_sigma,
        "outer_loop": f"two-phase ({args.n_trans_iters} trans + {args.n_rot_iters} rot)",
        "inner_backward": "IFT (custom_vjp)",
        "noise_method": args.noise_method,
        "var_window_size": args.var_window_size if args.noise_method == "windowed" else None,
        "noise_ema_alpha": args.noise_ema_alpha,
        "noise_init_size": args.noise_init_size,
        "allan_cluster_sizes": args.allan_cluster_sizes if args.noise_method == "allan" else None,
        "n_seeds": n_seeds,
        "seeds": seeds,
    }

    # Run all seeds, saving each independently.
    all_seed_results = []
    t_total_start = time.perf_counter()
    summary_lines = []

    for si, seed in enumerate(seeds):
        if n_seeds > 1:
            print(f"\n{'='*70}")
            print(f"  Seed {si+1}/{n_seeds} (seed={seed})")
            print(f"{'='*70}")

        seed_results = []
        for seq_id, gt_poses in seq_data:
            result = denoise_sequence(
                gt_poses, args, sigma, seq_id=seq_id, seed=seed)
            result["seed"] = seed
            if si > 0 and "trajectories" in result:
                del result["trajectories"]
            seed_results.append(result)

        all_seed_results.append(seed_results)

        # Save per-seed JSON file.
        seed_output = {
            "config": config,
            "seed": seed,
            "sequences": seed_results,
        }
        seed_path = os.path.join(run_dir, f"seed_{seed:04d}.json")
        with open(seed_path, "w") as f:
            json.dump(seed_output, f, indent=2)

        # Build results table for this seed.
        table_header = (
            f"{'Seq':>4s} {'Poses':>6s} {'Poses/s':>8s} "
            f"{'RT@10Hz':>8s} {'RT@100Hz':>9s} "
            f"{'T_RMSE%':>8s} {'R_RMSE%':>8s} {'Combined':>9s} "
            f"{'sw/rw_t':>8s}")
        table_sep = (
            f"{'-'*4} {'-'*6} {'-'*8} "
            f"{'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")

        summary_lines.append(f"{'='*70}")
        summary_lines.append(f"  RESULTS — Seed {seed}")
        summary_lines.append(f"{'='*70}")
        summary_lines.append("")
        summary_lines.append(f"  {table_header}")
        summary_lines.append(f"  {table_sep}")

        print()
        print(f"  {table_header}")
        print(f"  {table_sep}")

        for r in seed_results:
            tp = r['throughput']
            imp = r['improvement_pct']
            sw_rw_t = r['weights_used']['sw_rw_ratio_trans']
            row = (f"  {r['sequence']:>4s} {r['n_poses']:>6d} "
                   f"{tp['poses_per_sec']:>7.1f} "
                   f"{tp['realtime_factor_10hz']:>7.2f}x "
                   f"{tp['realtime_factor_100hz']:>8.2f}x "
                   f"{imp['trans_rmse']:>+7.1f}% "
                   f"{imp['rot_rmse']:>+7.1f}% "
                   f"{imp['combined']:>+8.1f}% "
                   f"{sw_rw_t:>8.1f}")
            print(row)
            summary_lines.append(row)

        summary_lines.append("")
        print(f"  Saved: {seed_path}")

    t_total = time.perf_counter() - t_total_start

    # Aggregate statistics across seeds.
    if n_seeds > 1:
        print()
        print("=" * 70)
        print(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
        print("=" * 70)
        print()

        agg_header = (
            f"{'Seq':>4s} {'Poses':>6s} "
            f"{'T_mean':>8s} {'T_std':>7s} "
            f"{'R_mean':>8s} {'R_std':>7s} "
            f"{'C_mean':>8s} {'C_std':>7s}")
        agg_sep = (
            f"{'-'*4} {'-'*6} "
            f"{'-'*8} {'-'*7} "
            f"{'-'*8} {'-'*7} "
            f"{'-'*8} {'-'*7}")

        print(f"  {agg_header}")
        print(f"  {agg_sep}")

        summary_lines.append(f"{'='*70}")
        summary_lines.append(f"  AGGREGATE RESULTS ({n_seeds} seeds)")
        summary_lines.append(f"{'='*70}")
        summary_lines.append("")
        summary_lines.append(f"  {agg_header}")
        summary_lines.append(f"  {agg_sep}")

        aggregate = []
        for si_seq in range(len(seq_data)):
            seq_id = seq_data[si_seq][0]
            n_poses = seq_data[si_seq][1].shape[0]
            t_vals = [all_seed_results[si][si_seq]['improvement_pct']['trans_rmse']
                      for si in range(n_seeds)]
            r_vals = [all_seed_results[si][si_seq]['improvement_pct']['rot_rmse']
                      for si in range(n_seeds)]
            c_vals = [all_seed_results[si][si_seq]['improvement_pct']['combined']
                      for si in range(n_seeds)]

            stats = {
                "sequence": seq_id,
                "n_poses": n_poses,
                "trans_mean": round(float(np.mean(t_vals)), 2),
                "trans_std": round(float(np.std(t_vals)), 2),
                "rot_mean": round(float(np.mean(r_vals)), 2),
                "rot_std": round(float(np.std(r_vals)), 2),
                "combined_mean": round(float(np.mean(c_vals)), 2),
                "combined_std": round(float(np.std(c_vals)), 2),
            }
            aggregate.append(stats)

            row = (f"  {seq_id:>4s} {n_poses:>6d} "
                   f"{stats['trans_mean']:>+7.1f}% {stats['trans_std']:>6.1f} "
                   f"{stats['rot_mean']:>+7.1f}% {stats['rot_std']:>6.1f} "
                   f"{stats['combined_mean']:>+7.1f}% "
                   f"{stats['combined_std']:>6.1f}")
            print(row)
            summary_lines.append(row)

        # Save aggregate JSON.
        agg_path = os.path.join(run_dir, "aggregate.json")
        with open(agg_path, "w") as f:
            json.dump({"config": config, "aggregate": aggregate,
                       "total_time_s": round(t_total, 1)}, f, indent=2)
        print(f"\n  Aggregate: {agg_path}")

    # Write combined results text file.
    results_path = os.path.join(run_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n  All results: {run_dir}/")
    print(f"  Summary:     {results_path}")
    print(f"  Total time:  {t_total:.1f}s")


if __name__ == "__main__":
    main()
