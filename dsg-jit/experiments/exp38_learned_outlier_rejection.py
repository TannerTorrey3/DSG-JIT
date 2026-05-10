# experiments/exp38_learned_outlier_rejection.py
"""
Learned outlier rejection through differentiable pose graph optimisation.

Extends exp36's bilevel IFT denoiser with per-edge confidence weights
learned end-to-end through the Gauss-Newton solver.  Given a stream of
odometry measurements with unknown outliers, the system simultaneously:

    1. Denoises clean edges (additive corrections θ)
    2. Rejects corrupted edges (learned weights w → 0 for outliers)

No outlier labels are required — weights are learned purely from anchor
supervision by differentiating through the inner GN solver via IFT.

Novelty vs prior work:
    - GNC (Yang et al., 2020): fixed kernel, manual annealing, no gradients
    - Switchable constraints (Sünderhauf & Protzel, 2012): binary, no E2E learning
    - Roessle et al. (ICCV 2023): learned confidence for feature matching only
    - THIS: continuous weights through full GN, real-time via IFT, self-tuned

Usage:
    python -m experiments.exp38_learned_outlier_rejection \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences --seq 06 \
        --outlier-ratio 0.1
    python -m experiments.exp38_learned_outlier_rejection \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences \
        --outlier-ratio 0.2 --outlier-mag 0.5
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
# Noise estimation (online EMA from exp36)
# ---------------------------------------------------------------------------

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


def compute_auto_weights(noise_model: dict, *, base_sw: float = 1.0,
                         base_rw: float = 1.0,
                         sw_ratio: float = 3.0) -> dict:
    """Derive sw and rw from the estimated noise model.

    sw is capped at sw_ratio * rw to prevent smoothness from dominating.
    """
    sigma_p_t = noise_model["sigma_process_trans"]
    sigma_p_r = noise_model["sigma_process_rot"]
    sigma_n_t = noise_model["sigma_noise_trans"]
    sigma_n_r = noise_model["sigma_noise_rot"]

    rw_trans = base_rw / max(sigma_n_t ** 2, 1e-12)
    rw_rot = base_rw / max(sigma_n_r ** 2, 1e-12)

    sw_trans = min(base_sw / max(sigma_p_t ** 2, 1e-12), sw_ratio * rw_trans)
    sw_rot = min(base_sw / max(sigma_p_r ** 2, 1e-12), sw_ratio * rw_rot)

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
# Per-edge metrics
# ---------------------------------------------------------------------------

def compute_per_edge_error(measurements: jnp.ndarray,
                           gt_measurements: jnp.ndarray) -> dict:
    """Per-edge measurement error split into trans and rot."""
    diff = measurements - gt_measurements
    trans_err = np.array(jnp.linalg.norm(diff[:, :3], axis=1))
    rot_err = np.array(jnp.linalg.norm(diff[:, 3:], axis=1))
    return {
        "per_edge_trans": trans_err,
        "per_edge_rot": rot_err,
        "rmse_trans": float(np.sqrt(np.mean(trans_err ** 2))),
        "rmse_rot": float(np.sqrt(np.mean(rot_err ** 2))),
    }


# ---------------------------------------------------------------------------
# Outlier injection
# ---------------------------------------------------------------------------

def inject_outliers(
    measurements: jnp.ndarray,
    key: jax.random.PRNGKey,
    outlier_ratio: float = 0.1,
    outlier_mag_trans: float = 0.5,
    outlier_mag_rot: float = 0.2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inject random outliers into a fraction of measurements.

    Returns:
        corrupted: measurements with outliers injected
        outlier_mask: boolean array, True for corrupted edges
    """
    n_meas = measurements.shape[0]
    key1, key2 = jax.random.split(key)

    # Select which edges are outliers.
    outlier_mask = jax.random.uniform(key1, (n_meas,)) < outlier_ratio

    # Generate large random perturbations for outliers.
    outlier_sigma = jnp.array(
        [outlier_mag_trans] * 3 + [outlier_mag_rot] * 3, dtype=jnp.float32)
    perturbation = jax.random.normal(key2, shape=measurements.shape) * outlier_sigma

    # Apply perturbation only to outlier edges.
    corrupted = measurements + perturbation * outlier_mask[:, None]

    return corrupted, outlier_mask


# ---------------------------------------------------------------------------
# Bilevel solver with learned outlier weights
# ---------------------------------------------------------------------------

def build_robust_denoiser(
    n_poses: int,
    inner_anchor_positions: list[int],
    eval_positions: list[int],
    sigma: jnp.ndarray,
    *,
    gn_iters: int = 10,
    gn_damping: float = 5e-3,
    aw_trans: float = 15.0,
    aw_rot: float = 15.0,
    rw_trans: float = 0.25,
    rw_rot: float = 0.1,
    sw_trans: float = 25.0,
    sw_rot: float = 25.0,
    inner_anchor_sigma: float = 0.01,
    n_trans_iters: int = 50,
    n_rot_iters: int = 20,
    lr: float = 1e-3,
    weight_lr: float = 1e-2,
    weight_reg: float = 0.01,
    n_weight_iters: int = 30,
):
    """Build a JIT-compiled robust denoiser with learned per-edge weights.

    Key architecture: inner and outer anchors are SEPARATED.
      - Inner solver uses sparse gauge-fix anchors (first/last) so the
        pose graph is measurement-dominated. Outliers corrupt the trajectory.
      - Outer loss evaluates at dense GT positions to detect corruption.
        This asymmetry creates strong gradient signal for weight learning.

    Learns simultaneously:
        1. Measurement corrections θ (additive, 6D per edge)
        2. Per-edge confidence weights w = sigmoid(log_w) ∈ (0, 1)
    """
    n_meas = n_poses - 1
    odom_w = sigma_to_weight(sigma)
    sqrt_odom_w = jnp.sqrt(odom_w)

    # Inner anchors: sparse, for gauge fixing only
    inner_anchor_w = sigma_to_weight(jnp.full(6, inner_anchor_sigma))
    sqrt_inner_anchor_w = jnp.sqrt(inner_anchor_w)
    inner_anchor_idx = jnp.array(inner_anchor_positions, dtype=jnp.int32)
    n_inner_anchors = len(inner_anchor_positions)

    # Outer eval positions: dense, for supervision
    eval_idx = jnp.array(eval_positions, dtype=jnp.int32)

    anchor_w_vec = jnp.array(
        [aw_trans] * 3 + [aw_rot] * 3, dtype=jnp.float32)
    reg_w_vec = jnp.array(
        [rw_trans] * 3 + [rw_rot] * 3, dtype=jnp.float32)
    sw_vec = jnp.array(
        [sw_trans] * 3 + [sw_rot] * 3, dtype=jnp.float32)

    _odom_res_batch = jax.vmap(
        lambda a, b, m, w: w * (relative_pose_se3(a, b) - m) * sqrt_odom_w)
    _retract_batch = jax.vmap(se3_retract_left)
    max_step_per_pose = 0.5

    # --- Inner PGO residual with learned weights ---
    # Only uses sparse gauge-fix anchors (first/last pose).

    def residual_fn(x, theta, inner_anchor_targets, weights):
        """Weighted residual: w_i scales each odometry residual."""
        poses = x.reshape(n_poses, 6)
        w_broad = weights[:, None]  # (n_meas, 1)
        r_odom = _odom_res_batch(poses[:-1], poses[1:], theta, w_broad)
        r_anch = (poses[inner_anchor_idx] - inner_anchor_targets) * sqrt_inner_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step(x, theta, inner_anchor_targets, weights):
        def r_fn(x_):
            return residual_fn(x_, theta, inner_anchor_targets, weights)
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

    # --- IFT inner solve ---

    @jax.custom_vjp
    def inner_solve(theta, x_init, inner_anchor_targets, weights):
        def scan_body(x, _):
            return gn_step(x, theta, inner_anchor_targets, weights), None
        x_star, _ = jax.lax.scan(scan_body, x_init, None, length=gn_iters)
        return x_star

    def inner_solve_fwd(theta, x_init, inner_anchor_targets, weights):
        x_star = inner_solve(theta, x_init, inner_anchor_targets, weights)
        return x_star, (x_star, theta, inner_anchor_targets, weights)

    def inner_solve_bwd(res, g):
        x_star, theta, inner_anchor_targets, weights = res
        r_fn_x = lambda x_: residual_fn(x_, theta, inner_anchor_targets, weights)
        J = jax.jacobian(r_fn_x)(x_star)
        n = x_star.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        u = jnp.linalg.solve(H, g)

        # Gradient w.r.t. theta
        _, vjp_theta = jax.vjp(
            lambda t: residual_fn(x_star, t, inner_anchor_targets, weights), theta)
        v = J @ u
        dtheta = -vjp_theta(v)[0]

        # Gradient w.r.t. weights
        _, vjp_weights = jax.vjp(
            lambda w: residual_fn(x_star, theta, inner_anchor_targets, w), weights)
        dweights = -vjp_weights(v)[0]

        return (dtheta, jnp.zeros_like(x_star),
                jnp.zeros_like(inner_anchor_targets), dweights)

    inner_solve.defvjp(inner_solve_fwd, inner_solve_bwd)

    # --- Outer loss ---
    # Uses RELATIVE evaluation between consecutive eval positions.
    # This localises the error signal: only the eval pair straddling an
    # outlier sees large error, concentrating gradient on the actual outlier
    # rather than spreading it across the entire downstream chain.

    log_w_init_val = 3.0
    _relative_batch = jax.vmap(relative_pose_se3)

    def outer_loss(theta, log_w, x_init, inner_anchor_targets,
                   eval_rel_targets, noisy_meas):
        weights = jax.nn.sigmoid(log_w)
        x_star = inner_solve(theta, x_init, inner_anchor_targets, weights)
        poses_opt = x_star.reshape(n_poses, 6)

        # Relative evaluation: compare relative poses between consecutive
        # eval positions in solved trajectory vs GT.
        # This is LOCAL — each pair only depends on edges between those positions.
        solved_rel = _relative_batch(poses_opt[eval_idx[:-1]], poses_opt[eval_idx[1:]])
        eval_diffs = solved_rel - eval_rel_targets
        a_loss = jnp.sum(anchor_w_vec * eval_diffs ** 2)

        # Regularisation: corrected measurements near original
        dev = theta - noisy_meas
        r_loss = jnp.sum(reg_w_vec * dev ** 2)

        # Smoothness
        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(sw_vec * s_diffs ** 2)

        # Weight regularisation: L2 toward initial value (w ≈ 1).
        w_reg_loss = weight_reg * jnp.sum((log_w - log_w_init_val) ** 2)

        return a_loss + r_loss + s_loss + w_reg_loss

    grad_fn = jax.grad(outer_loss, argnums=(0, 1))

    # --- Joint optimisation: theta + weights together ---
    trans_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    rot_mask = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    def fused_optimize(theta_init, log_w_init, x_init,
                       inner_anchor_targets, eval_rel_targets, noisy_meas):
        """Joint optimisation: theta (two-phase) + weights updated together."""

        # Phase 1 — Translation + weights jointly
        m_theta = jnp.zeros_like(theta_init)
        v_theta = jnp.zeros_like(theta_init)
        m_w = jnp.zeros_like(log_w_init)
        v_w = jnp.zeros_like(log_w_init)

        def trans_body(i, state):
            theta, log_w, mt, vt, mw, vw = state
            g_theta, g_w = grad_fn(theta, log_w, x_init,
                                   inner_anchor_targets, eval_rel_targets, noisy_meas)
            g_t = g_theta * trans_mask
            t = (i + 1).astype(jnp.float32)

            # Update theta (translation only)
            mt_new = 0.9 * mt + 0.1 * g_t
            vt_new = 0.999 * vt + 0.001 * g_t ** 2
            mt_hat = mt_new / (1.0 - 0.9 ** t)
            vt_hat = vt_new / (1.0 - 0.999 ** t)
            theta_update = lr * mt_hat / (jnp.sqrt(vt_hat) + 1e-8)

            # Update weights
            mw_new = 0.9 * mw + 0.1 * g_w
            vw_new = 0.999 * vw + 0.001 * g_w ** 2
            mw_hat = mw_new / (1.0 - 0.9 ** t)
            vw_hat = vw_new / (1.0 - 0.999 ** t)
            w_update = weight_lr * mw_hat / (jnp.sqrt(vw_hat) + 1e-8)

            return (theta - theta_update, log_w - w_update,
                    mt_new, vt_new, mw_new, vw_new)

        state = jax.lax.fori_loop(
            0, n_trans_iters, trans_body,
            (theta_init, log_w_init, m_theta, v_theta, m_w, v_w))
        theta_t, log_w_t = state[0], state[1]

        # Phase 2 — Rotation + weights jointly (fresh Adam for theta, continue for w)
        m_theta_r = jnp.zeros_like(theta_t)
        v_theta_r = jnp.zeros_like(theta_t)
        mw_cont = state[4]
        vw_cont = state[5]

        def rot_body(i, state):
            theta, log_w, mt, vt, mw, vw = state
            g_theta, g_w = grad_fn(theta, log_w, x_init,
                                   inner_anchor_targets, eval_rel_targets, noisy_meas)
            g_r = g_theta * rot_mask
            t = (i + 1).astype(jnp.float32)

            # Update theta (rotation only)
            mt_new = 0.9 * mt + 0.1 * g_r
            vt_new = 0.999 * vt + 0.001 * g_r ** 2
            mt_hat = mt_new / (1.0 - 0.9 ** t)
            vt_hat = vt_new / (1.0 - 0.999 ** t)
            theta_update = lr * mt_hat / (jnp.sqrt(vt_hat) + 1e-8)

            # Update weights (continuing)
            t_w = (i + 1 + n_trans_iters).astype(jnp.float32)
            mw_new = 0.9 * mw + 0.1 * g_w
            vw_new = 0.999 * vw + 0.001 * g_w ** 2
            mw_hat = mw_new / (1.0 - 0.9 ** t_w)
            vw_hat = vw_new / (1.0 - 0.999 ** t_w)
            w_update = weight_lr * mw_hat / (jnp.sqrt(vw_hat) + 1e-8)

            return (theta - theta_update, log_w - w_update,
                    mt_new, vt_new, mw_new, vw_new)

        state = jax.lax.fori_loop(
            0, n_rot_iters, rot_body,
            (theta_t, log_w_t, m_theta_r, v_theta_r, mw_cont, vw_cont))

        return state[0], state[1]

    fused_optimize_jit = jax.jit(fused_optimize)

    return fused_optimize_jit


# ---------------------------------------------------------------------------
# Evaluation of a single sequence
# ---------------------------------------------------------------------------

def evaluate_sequence(
    gt_poses: jnp.ndarray,
    args,
    sigma: jnp.ndarray,
    seq_id: str = "??",
    seed: int = 42,
) -> dict:
    """Run the full robust denoising pipeline on a single sequence."""
    n_poses_total = gt_poses.shape[0]
    n_meas_total = n_poses_total - 1
    window_size = args.window_size
    overlap = min(10, window_size // 5)
    stride = window_size - overlap

    # Compute GT measurements.
    gt_measurements = jax.vmap(relative_pose_se3)(gt_poses[:-1], gt_poses[1:])

    # Add calibrated noise.
    key = jax.random.PRNGKey(seed)
    key_noise, key_outlier = jax.random.split(key)
    noise = jax.random.normal(key_noise, shape=gt_measurements.shape) * sigma
    noisy_measurements = gt_measurements + noise

    # Inject outliers.
    corrupted_measurements, outlier_mask = inject_outliers(
        noisy_measurements, key_outlier,
        outlier_ratio=args.outlier_ratio,
        outlier_mag_trans=args.outlier_mag_trans,
        outlier_mag_rot=args.outlier_mag_rot,
    )
    outlier_mask_np = np.array(outlier_mask)
    n_outliers = int(np.sum(outlier_mask_np))
    n_inliers = n_meas_total - n_outliers

    # Noise estimation.
    noisy_np = np.array(corrupted_measurements)
    noise_model = estimate_noise_online(noisy_np)
    auto_w = compute_auto_weights(
        noise_model, base_sw=args.base_sw, base_rw=args.base_rw,
        sw_ratio=args.sw_ratio)

    inner_sigma = jnp.array(
        noise_model["sigma_noise_per_comp"], dtype=jnp.float32)

    print(f"\n  Sequence {seq_id}: {n_poses_total} poses, "
          f"{n_outliers} outliers ({args.outlier_ratio*100:.0f}%)")
    print(f"  σ_noise: trans={noise_model['sigma_noise_trans']:.5f}, "
          f"rot={noise_model['sigma_noise_rot']:.5f}")
    print(f"  sw_trans={auto_w['sw_trans']:.2f}, rw_trans={auto_w['rw_trans']:.2f}")

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

    # Inner anchors: first and last pose only (gauge fix).
    inner_anchor_pos = [0, actual_window - 1]

    # Outer eval positions: dense (every anchor_spacing poses).
    eval_pos_in_window = list(range(0, actual_window, args.anchor_spacing))
    if eval_pos_in_window[-1] != actual_window - 1:
        eval_pos_in_window.append(actual_window - 1)

    print(f"  Inner anchors: {len(inner_anchor_pos)} (gauge fix)")
    print(f"  Eval positions: {len(eval_pos_in_window)} (every {args.anchor_spacing})")

    # Build denoiser.
    t_jit_start = time.perf_counter()
    fused_opt = build_robust_denoiser(
        actual_window, inner_anchor_pos, eval_pos_in_window, inner_sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        aw_trans=args.aw_trans, aw_rot=args.aw_rot,
        rw_trans=auto_w["rw_trans"], rw_rot=auto_w["rw_rot"],
        sw_trans=auto_w["sw_trans"], sw_rot=auto_w["sw_rot"],
        inner_anchor_sigma=args.inner_anchor_sigma,
        n_trans_iters=args.n_trans_iters,
        n_rot_iters=args.n_rot_iters,
        lr=args.lr,
        weight_lr=args.weight_lr,
        weight_reg=args.weight_reg,
        n_weight_iters=args.n_weight_iters,
    )

    # Warm-up.
    n_meas_window = actual_window - 1
    n_eval_pairs = len(eval_pos_in_window) - 1
    dummy_theta = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_inner_anchors = jnp.zeros((len(inner_anchor_pos), 6), dtype=jnp.float32)
    dummy_eval_rel = jnp.zeros((n_eval_pairs, 6), dtype=jnp.float32)
    dummy_log_w = jnp.full(n_meas_window, 3.0, dtype=jnp.float32)
    _ = fused_opt(dummy_theta, dummy_log_w, dummy_x, dummy_inner_anchors,
                  dummy_eval_rel, dummy_theta)
    jax.block_until_ready(_)
    t_jit = time.perf_counter() - t_jit_start
    print(f"  JIT: {t_jit:.1f}s")

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

    # Denoise + learn weights.
    t_denoise_start = time.perf_counter()
    denoised_measurements = np.array(corrupted_measurements).copy()
    learned_weights = np.ones(n_meas_total, dtype=np.float32)
    window_times = []

    for wi, (w_start, w_end) in enumerate(windows):
        t_win_start = time.perf_counter()
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1
        w_noisy = jnp.array(corrupted_measurements[w_start:w_start + w_n_meas])

        # Inner anchor targets (gauge fix: first and last).
        gt_first = gt_poses[w_start]
        inner_anchor_global = gt_poses[w_start + jnp.array(inner_anchor_pos)]
        w_inner_anchor_targets = jax.vmap(relative_pose_se3, in_axes=(None, 0))(
            gt_first, inner_anchor_global)

        # Outer eval: RELATIVE poses between consecutive eval positions.
        eval_global = gt_poses[w_start + jnp.array(eval_pos_in_window)]
        w_eval_rel_targets = jax.vmap(relative_pose_se3)(
            eval_global[:-1], eval_global[1:])

        # x_init.
        origin = jnp.zeros(6, dtype=jnp.float32)
        x_init = _forward_compose_jit(origin, w_noisy).ravel()

        # Initial log_weights: w ≈ 0.95 (trust all), gradient pushes outliers down.
        log_w_init = jnp.full(w_n_meas, 3.0, dtype=jnp.float32)

        # Optimise.
        theta_opt, log_w_opt = fused_opt(
            w_noisy, log_w_init, x_init, w_inner_anchor_targets,
            w_eval_rel_targets, w_noisy)
        jax.block_until_ready(theta_opt)

        # Commit.
        theta_np = np.array(theta_opt)
        w_np = float(jax.nn.sigmoid(log_w_opt).mean())  # for logging
        weights_np = np.array(jax.nn.sigmoid(log_w_opt))
        commit_start, commit_end = commit_ranges[wi]
        for gi in range(commit_start, commit_end + 1):
            local_i = gi - w_start
            if 0 <= local_i < w_n_meas and gi < n_meas_total:
                denoised_measurements[gi] = theta_np[local_i]
                learned_weights[gi] = weights_np[local_i]

        t_win = time.perf_counter() - t_win_start
        window_times.append(t_win)

        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            elapsed = time.perf_counter() - t_denoise_start
            avg_hz = (wi + 1) * actual_window / elapsed
            print(f"    Window {wi+1}/{len(windows)} done "
                  f"({elapsed:.1f}s, {avg_hz:.1f} poses/s, "
                  f"avg_w={w_np:.3f})", flush=True)

    t_denoise = time.perf_counter() - t_denoise_start
    poses_per_sec = n_poses_total / t_denoise

    # --- Metrics ---
    # Baseline: corrupted vs GT
    baseline = compute_per_edge_error(
        jnp.array(corrupted_measurements), gt_measurements)
    # After denoising
    after = compute_per_edge_error(
        jnp.array(denoised_measurements), gt_measurements)

    trans_improv = (1 - after['rmse_trans'] / baseline['rmse_trans']) * 100
    rot_improv = (1 - after['rmse_rot'] / baseline['rmse_rot']) * 100

    # Weight analysis
    w_outlier = learned_weights[outlier_mask_np]
    w_inlier = learned_weights[~outlier_mask_np]

    # Classification accuracy at threshold 0.5
    predicted_outlier = learned_weights < 0.5
    tp = int(np.sum(predicted_outlier & outlier_mask_np))
    fp = int(np.sum(predicted_outlier & ~outlier_mask_np))
    fn = int(np.sum(~predicted_outlier & outlier_mask_np))
    tn = int(np.sum(~predicted_outlier & ~outlier_mask_np))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / n_meas_total

    print(f"\n  --- Results ---")
    print(f"  Trans RMSE: {baseline['rmse_trans']:.5f} → "
          f"{after['rmse_trans']:.5f} ({trans_improv:+.1f}%)")
    print(f"  Rot RMSE:   {baseline['rmse_rot']:.5f} → "
          f"{after['rmse_rot']:.5f} ({rot_improv:+.1f}%)")
    print(f"  Throughput: {poses_per_sec:.1f} poses/sec")
    print(f"\n  --- Outlier Detection ---")
    print(f"  Mean weight (outliers): {np.mean(w_outlier):.4f}")
    print(f"  Mean weight (inliers):  {np.mean(w_inlier):.4f}")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, "
          f"F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

    return {
        "sequence": seq_id,
        "seed": seed,
        "n_poses": n_poses_total,
        "n_outliers": n_outliers,
        "n_inliers": n_inliers,
        "outlier_ratio_actual": round(n_outliers / n_meas_total, 4),
        "baseline": {
            "trans_rmse": round(baseline['rmse_trans'], 6),
            "rot_rmse": round(baseline['rmse_rot'], 6),
        },
        "denoised": {
            "trans_rmse": round(after['rmse_trans'], 6),
            "rot_rmse": round(after['rmse_rot'], 6),
        },
        "improvement_pct": {
            "trans_rmse": round(trans_improv, 2),
            "rot_rmse": round(rot_improv, 2),
            "combined": round((trans_improv + rot_improv) / 2, 2),
        },
        "weight_stats": {
            "outlier_mean_w": round(float(np.mean(w_outlier)), 4),
            "outlier_std_w": round(float(np.std(w_outlier)), 4),
            "outlier_median_w": round(float(np.median(w_outlier)), 4),
            "inlier_mean_w": round(float(np.mean(w_inlier)), 4),
            "inlier_std_w": round(float(np.std(w_inlier)), 4),
            "inlier_median_w": round(float(np.median(w_inlier)), 4),
        },
        "classification": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "throughput": {
            "poses_per_sec": round(poses_per_sec, 1),
            "jit_s": round(t_jit, 2),
            "denoise_s": round(t_denoise, 2),
        },
        "learned_weights": learned_weights.tolist(),
        "outlier_mask": outlier_mask_np.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp38: learned outlier rejection through differentiable PGO")
    parser.add_argument("--sequences-dir", type=str, required=True)
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--n-poses", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--output-dir", type=str,
                        default="/data/tkocher/exp_res")

    # Noise
    parser.add_argument("--sigma-trans", type=float, default=0.03)
    parser.add_argument("--sigma-rot", type=float, default=0.01)

    # Outliers
    parser.add_argument("--outlier-ratio", type=float, default=0.1,
                        help="Fraction of edges to corrupt (default: 10%%)")
    parser.add_argument("--outlier-mag-trans", type=float, default=0.5,
                        help="Outlier translation magnitude (metres)")
    parser.add_argument("--outlier-mag-rot", type=float, default=0.2,
                        help="Outlier rotation magnitude (radians)")

    # Denoiser config
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--anchor-spacing", type=int, default=10)
    parser.add_argument("--gn-iters", type=int, default=10)
    parser.add_argument("--n-trans-iters", type=int, default=50)
    parser.add_argument("--n-rot-iters", type=int, default=20)
    parser.add_argument("--n-weight-iters", type=int, default=0,
                        help="(deprecated, weights now joint with theta)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-lr", type=float, default=1e-2)
    parser.add_argument("--weight-reg", type=float, default=0.01,
                        help="Regularisation toward w=1 (trust measurements)")
    parser.add_argument("--aw-trans", type=float, default=15.0)
    parser.add_argument("--aw-rot", type=float, default=15.0)
    parser.add_argument("--inner-anchor-sigma", type=float, default=0.01)
    parser.add_argument("--base-sw", type=float, default=1.0)
    parser.add_argument("--base-rw", type=float, default=1.0)
    parser.add_argument("--sw-ratio", type=float, default=3.0)

    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    n_total_iters = args.n_trans_iters + args.n_rot_iters

    print("=" * 70)
    print("  exp38 -- Learned Outlier Rejection via Differentiable PGO")
    print("=" * 70)
    _print_device_info()
    print()
    print(f"  Noise: σ_t={args.sigma_trans}, σ_r={args.sigma_rot}")
    print(f"  Outliers: {args.outlier_ratio*100:.0f}% ratio, "
          f"mag_t={args.outlier_mag_trans}, mag_r={args.outlier_mag_rot}")
    print(f"  Denoiser: window={args.window_size}, "
          f"eval_spacing={args.anchor_spacing}")
    print(f"  Architecture: inner=2 gauge anchors, "
          f"outer=dense eval every {args.anchor_spacing}")
    print(f"  Optimiser: {args.n_trans_iters} trans + {args.n_rot_iters} rot "
          f"= {n_total_iters} (joint θ+w)")
    print(f"  LR: theta={args.lr}, weights={args.weight_lr}")
    print(f"  Weight reg: {args.weight_reg}, aw={args.aw_trans}, init w≈0.95")
    print(f"  Seeds: {args.n_seeds}")
    print()

    # Find sequences.
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

    print(f"  Loaded {len(seq_data)} sequences: {[s[0] for s in seq_data]}")

    # Output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp38_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "sigma_trans": args.sigma_trans,
        "sigma_rot": args.sigma_rot,
        "outlier_ratio": args.outlier_ratio,
        "outlier_mag_trans": args.outlier_mag_trans,
        "outlier_mag_rot": args.outlier_mag_rot,
        "window_size": args.window_size,
        "anchor_spacing": args.anchor_spacing,
        "gn_iters": args.gn_iters,
        "n_trans_iters": args.n_trans_iters,
        "n_rot_iters": args.n_rot_iters,
        "n_weight_iters": args.n_weight_iters,
        "lr": args.lr,
        "weight_lr": args.weight_lr,
        "weight_reg": args.weight_reg,
        "sw_ratio": args.sw_ratio,
        "n_seeds": args.n_seeds,
        "seeds": seeds,
    }

    all_results = []
    t_total_start = time.perf_counter()

    for seed in seeds:
        if args.n_seeds > 1:
            print(f"\n{'='*70}")
            print(f"  Seed {seed}")
            print(f"{'='*70}")

        for seq_id, gt_poses in seq_data:
            result = evaluate_sequence(gt_poses, args, sigma, seq_id, seed)
            all_results.append(result)

    t_total = time.perf_counter() - t_total_start

    # Summary.
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()

    header = (f"  {'Seq':>4s} {'Seed':>5s} {'Out%':>5s} "
              f"{'T_imp%':>7s} {'R_imp%':>7s} "
              f"{'w_out':>6s} {'w_in':>6s} "
              f"{'Prec':>5s} {'Rec':>5s} {'F1':>5s} {'Hz':>6s}")
    print(header)
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*7} {'-'*7} "
          f"{'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")

    for r in all_results:
        imp = r['improvement_pct']
        ws = r['weight_stats']
        cl = r['classification']
        print(f"  {r['sequence']:>4s} {r['seed']:>5d} "
              f"{r['outlier_ratio_actual']*100:>4.0f}% "
              f"{imp['trans_rmse']:>+6.1f}% {imp['rot_rmse']:>+6.1f}% "
              f"{ws['outlier_mean_w']:>5.3f} {ws['inlier_mean_w']:>5.3f} "
              f"{cl['precision']:>5.3f} {cl['recall']:>5.3f} "
              f"{cl['f1']:>5.3f} {r['throughput']['poses_per_sec']:>5.0f}")

    # Aggregate.
    if len(all_results) > 1:
        t_vals = [r['improvement_pct']['trans_rmse'] for r in all_results]
        r_vals = [r['improvement_pct']['rot_rmse'] for r in all_results]
        f1_vals = [r['classification']['f1'] for r in all_results]
        w_out_vals = [r['weight_stats']['outlier_mean_w'] for r in all_results]
        w_in_vals = [r['weight_stats']['inlier_mean_w'] for r in all_results]

        print(f"\n  Aggregate ({len(all_results)} runs):")
        print(f"    Trans improvement: {np.mean(t_vals):+.1f}% ± {np.std(t_vals):.1f}")
        print(f"    Rot improvement:   {np.mean(r_vals):+.1f}% ± {np.std(r_vals):.1f}")
        print(f"    F1 score:          {np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f}")
        print(f"    Mean w (outliers): {np.mean(w_out_vals):.4f}")
        print(f"    Mean w (inliers):  {np.mean(w_in_vals):.4f}")

    print(f"\n  Total time: {t_total:.1f}s")

    # Save.
    output = {
        "config": config,
        "results": all_results,
        "total_time_s": round(t_total, 1),
    }
    output_path = os.path.join(run_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results: {output_path}")
    print(f"  Run dir: {run_dir}/")


if __name__ == "__main__":
    main()
