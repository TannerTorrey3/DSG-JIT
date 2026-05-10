# experiments/exp38_learned_outlier_rejection.py
"""
Robust outlier rejection via residual-based weighting in streaming IFT denoiser.

Architecture (per window):
    Phase A — Outlier detection:
        1. Run GN solve on corrupted measurements with GT anchors → x*
        2. Compute per-edge residuals: r_i = relative_pose(x*[i], x*[i+1]) - m_i
        3. Normalise: chi_i = ||r_i|| / sigma_estimated
        4. Welsch kernel: w_i = exp(-chi_i² / (2 * c²)), c from noise model

    Phase B — Weighted IFT denoising (exp36 architecture):
        5. Run bilevel IFT with weighted inner solver (outlier edges ignored)
        6. Two-phase Adam on theta (translation, then rotation)
        7. Commit denoised measurements for inlier edges

Novelty vs prior work:
    - GNC (Yang et al., 2020): requires multiple annealing passes, manual kernel schedule
    - Switchable constraints: binary switch, not continuous, no denoising
    - THIS: single-pass, real-time, kernel scale auto-tuned from online noise
      estimate, integrated with IFT denoising in a streaming framework

Usage:
    python -m experiments.exp38_learned_outlier_rejection \
        --sequences-dir /data/afnan/SemanticKitti/dataset/sequences --seq 06 \
        --outlier-ratio 0.1
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
    """Derive sw and rw from the estimated noise model."""
    sigma_p_t = noise_model["sigma_process_trans"]
    sigma_p_r = noise_model["sigma_process_rot"]
    sigma_n_t = noise_model["sigma_noise_trans"]
    sigma_n_r = noise_model["sigma_noise_rot"]

    rw_trans = base_rw / max(sigma_n_t ** 2, 1e-12)
    rw_rot = base_rw / max(sigma_n_r ** 2, 1e-12)

    sw_trans = min(base_sw / max(sigma_p_t ** 2, 1e-12), sw_ratio * rw_trans)
    sw_rot = min(base_sw / max(sigma_p_r ** 2, 1e-12), sw_ratio * rw_rot)

    return {
        "sw_trans": sw_trans,
        "sw_rot": sw_rot,
        "rw_trans": rw_trans,
        "rw_rot": rw_rot,
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
    """Inject random outliers into a fraction of measurements."""
    n_meas = measurements.shape[0]
    key1, key2 = jax.random.split(key)

    outlier_mask = jax.random.uniform(key1, (n_meas,)) < outlier_ratio
    outlier_sigma = jnp.array(
        [outlier_mag_trans] * 3 + [outlier_mag_rot] * 3, dtype=jnp.float32)
    perturbation = jax.random.normal(key2, shape=measurements.shape) * outlier_sigma
    corrupted = measurements + perturbation * outlier_mask[:, None]

    return corrupted, outlier_mask


# ---------------------------------------------------------------------------
# Robust denoiser: residual-based weights + IFT denoising
# ---------------------------------------------------------------------------

def build_robust_denoiser(
    n_poses: int,
    anchor_positions: list[int],
    sigma: jnp.ndarray,
    *,
    gn_iters: int = 10,
    gn_damping: float = 5e-3,
    kernel_scale: float = 3.0,
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
    """Build a JIT-compiled robust denoiser.

    Two-phase architecture:
        A) Initial GN solve → per-edge residuals → Welsch kernel weights
        B) Weighted IFT denoiser (exp36-style) with outlier edges suppressed

    The kernel scale `c` controls sensitivity: edges with normalised residual
    > c are strongly downweighted.  Derived from the noise model automatically.
    """
    n_meas = n_poses - 1
    odom_w = sigma_to_weight(sigma)
    sqrt_odom_w = jnp.sqrt(odom_w)
    anchor_w = sigma_to_weight(jnp.full(6, inner_anchor_sigma))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_positions, dtype=jnp.int32)

    # Sigma per component for normalisation (from noise model)
    sigma_per_comp = sigma  # already per-component [6]

    anchor_w_vec = jnp.array(
        [aw_trans] * 3 + [aw_rot] * 3, dtype=jnp.float32)
    reg_w_vec = jnp.array(
        [rw_trans] * 3 + [rw_rot] * 3, dtype=jnp.float32)
    sw_vec = jnp.array(
        [sw_trans] * 3 + [sw_rot] * 3, dtype=jnp.float32)

    _odom_res_batch_unweighted = jax.vmap(
        lambda a, b, m: (relative_pose_se3(a, b) - m) * sqrt_odom_w)
    _odom_res_batch_weighted = jax.vmap(
        lambda a, b, m, w: w * (relative_pose_se3(a, b) - m) * sqrt_odom_w)
    _relative_batch = jax.vmap(relative_pose_se3)
    _retract_batch = jax.vmap(se3_retract_left)
    max_step_per_pose = 0.5

    # ===== PHASE A: Initial GN solve (unweighted) for residual computation =====

    def residual_unweighted(x, meas, anchor_targets):
        poses = x.reshape(n_poses, 6)
        r_odom = _odom_res_batch_unweighted(poses[:-1], poses[1:], meas)
        r_anch = (poses[anchor_idx] - anchor_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step_unweighted(x, meas, anchor_targets):
        r_fn = lambda x_: residual_unweighted(x_, meas, anchor_targets)
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

    def initial_solve(meas, x_init, anchor_targets):
        """Run GN to convergence on raw measurements (no custom_vjp needed)."""
        def scan_body(x, _):
            return gn_step_unweighted(x, meas, anchor_targets), None
        x_star, _ = jax.lax.scan(scan_body, x_init, None, length=gn_iters)
        return x_star

    def compute_weights(x_star, meas):
        """Compute per-edge Welsch kernel weights from residuals."""
        poses = x_star.reshape(n_poses, 6)
        # Per-edge residual: difference between fitted relative pose and measurement
        fitted_rel = _relative_batch(poses[:-1], poses[1:])
        residual = fitted_rel - meas  # (n_meas, 6)

        # Normalise by estimated sigma per component, then take norm
        normalised = residual / sigma_per_comp  # element-wise
        chi = jnp.linalg.norm(normalised, axis=1)  # (n_meas,)

        # Welsch kernel: w = exp(-chi²/(2c²))
        weights = jnp.exp(-chi ** 2 / (2.0 * kernel_scale ** 2))
        return weights, chi

    # ===== PHASE B: Weighted IFT denoiser (exp36 architecture) =====

    def residual_weighted(x, theta, anchor_targets, weights):
        poses = x.reshape(n_poses, 6)
        w_broad = weights[:, None]
        r_odom = _odom_res_batch_weighted(poses[:-1], poses[1:], theta, w_broad)
        r_anch = (poses[anchor_idx] - anchor_targets) * sqrt_anchor_w
        return jnp.concatenate([r_odom.ravel(), r_anch.ravel()])

    def gn_step_weighted(x, theta, anchor_targets, weights):
        r_fn = lambda x_: residual_weighted(x_, theta, anchor_targets, weights)
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

    # IFT inner solve (only theta gradient, weights are fixed input)
    @jax.custom_vjp
    def inner_solve(theta, x_init, anchor_targets, weights):
        def scan_body(x, _):
            return gn_step_weighted(x, theta, anchor_targets, weights), None
        x_star, _ = jax.lax.scan(scan_body, x_init, None, length=gn_iters)
        return x_star

    def inner_solve_fwd(theta, x_init, anchor_targets, weights):
        x_star = inner_solve(theta, x_init, anchor_targets, weights)
        return x_star, (x_star, theta, anchor_targets, weights)

    def inner_solve_bwd(res, g):
        x_star, theta, anchor_targets, weights = res
        r_fn_x = lambda x_: residual_weighted(x_, theta, anchor_targets, weights)
        J = jax.jacobian(r_fn_x)(x_star)
        n = x_star.shape[0]
        H = J.T @ J + gn_damping * jnp.eye(n)
        u = jnp.linalg.solve(H, g)

        _, vjp_fn = jax.vjp(
            lambda t: residual_weighted(x_star, t, anchor_targets, weights), theta)
        v = J @ u
        dtheta = -vjp_fn(v)[0]

        return (dtheta, jnp.zeros_like(x_star),
                jnp.zeros_like(anchor_targets), jnp.zeros_like(weights))

    inner_solve.defvjp(inner_solve_fwd, inner_solve_bwd)

    # Outer loss (same as exp36 — no weight learning)
    def outer_loss(theta, x_init, anchor_targets, weights, noisy_meas):
        x_star = inner_solve(theta, x_init, anchor_targets, weights)
        poses_opt = x_star.reshape(n_poses, 6)

        diffs = poses_opt[anchor_idx] - anchor_targets
        a_loss = jnp.sum(anchor_w_vec * diffs ** 2)

        dev = theta - noisy_meas
        r_loss = jnp.sum(reg_w_vec * dev ** 2)

        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(sw_vec * s_diffs ** 2)

        return a_loss + r_loss + s_loss

    grad_fn = jax.grad(outer_loss)

    # Two-phase Adam (exp36 architecture)
    trans_mask = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    rot_mask = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    def fused_optimize(noisy_meas, x_init, anchor_targets, weights):
        """Two-phase Adam with fixed weights: translation first, rotation second."""

        # Phase 1 — Translation
        m0 = jnp.zeros_like(noisy_meas)
        v0 = jnp.zeros_like(noisy_meas)

        def trans_body(i, state):
            theta, m, v = state
            g = grad_fn(theta, x_init, anchor_targets, weights, noisy_meas)
            g = g * trans_mask
            t = (i + 1).astype(jnp.float32)
            m_new = 0.9 * m + 0.1 * g
            v_new = 0.999 * v + 0.001 * g ** 2
            m_hat = m_new / (1.0 - 0.9 ** t)
            v_hat = v_new / (1.0 - 0.999 ** t)
            update = lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return (theta - update, m_new, v_new)

        state = jax.lax.fori_loop(
            0, n_trans_iters, trans_body, (noisy_meas, m0, v0))
        theta_t = state[0]

        # Phase 2 — Rotation
        m0_r = jnp.zeros_like(theta_t)
        v0_r = jnp.zeros_like(theta_t)

        def rot_body(i, state):
            theta, m, v = state
            g = grad_fn(theta, x_init, anchor_targets, weights, noisy_meas)
            g = g * rot_mask
            t = (i + 1).astype(jnp.float32)
            m_new = 0.9 * m + 0.1 * g
            v_new = 0.999 * v + 0.001 * g ** 2
            m_hat = m_new / (1.0 - 0.9 ** t)
            v_hat = v_new / (1.0 - 0.999 ** t)
            update = lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return (theta - update, m_new, v_new)

        state = jax.lax.fori_loop(
            0, n_rot_iters, rot_body, (theta_t, m0_r, v0_r))
        return state[0]

    # ===== Combined: Phase A → Phase B =====

    def robust_denoise(noisy_meas, x_init, anchor_targets):
        """Full pipeline: detect outliers, then denoise with weights."""
        # Phase A: initial solve → residuals → weights
        x_star_init = initial_solve(noisy_meas, x_init, anchor_targets)
        weights, chi_scores = compute_weights(x_star_init, noisy_meas)

        # Phase B: weighted IFT denoiser
        theta_opt = fused_optimize(noisy_meas, x_init, anchor_targets, weights)

        return theta_opt, weights, chi_scores

    robust_denoise_jit = jax.jit(robust_denoise)
    return robust_denoise_jit


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
    print(f"  Kernel scale: {args.kernel_scale}")

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

    # Anchor positions (same as exp36 — dense for inner solver).
    anchor_pos_in_window = list(range(0, actual_window, args.anchor_spacing))
    if anchor_pos_in_window[-1] != actual_window - 1:
        anchor_pos_in_window.append(actual_window - 1)

    # Build denoiser.
    t_jit_start = time.perf_counter()
    robust_denoise = build_robust_denoiser(
        actual_window, anchor_pos_in_window, inner_sigma,
        gn_iters=args.gn_iters, gn_damping=5e-3,
        kernel_scale=args.kernel_scale,
        aw_trans=args.aw_trans, aw_rot=args.aw_rot,
        rw_trans=auto_w["rw_trans"], rw_rot=auto_w["rw_rot"],
        sw_trans=auto_w["sw_trans"], sw_rot=auto_w["sw_rot"],
        inner_anchor_sigma=args.inner_anchor_sigma,
        n_trans_iters=args.n_trans_iters,
        n_rot_iters=args.n_rot_iters,
        lr=args.lr,
    )

    # Warm-up.
    n_meas_window = actual_window - 1
    dummy_meas = jnp.zeros((n_meas_window, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(actual_window * 6, dtype=jnp.float32)
    dummy_anchors = jnp.zeros((len(anchor_pos_in_window), 6), dtype=jnp.float32)
    _ = robust_denoise(dummy_meas, dummy_x, dummy_anchors)
    jax.block_until_ready(_[0])
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

    # Denoise window by window.
    t_denoise_start = time.perf_counter()
    denoised_measurements = np.array(corrupted_measurements).copy()
    all_weights = np.ones(n_meas_total, dtype=np.float32)
    all_chi = np.zeros(n_meas_total, dtype=np.float32)

    for wi, (w_start, w_end) in enumerate(windows):
        t_win_start = time.perf_counter()
        w_n_poses = w_end - w_start
        w_n_meas = w_n_poses - 1
        w_noisy = jnp.array(corrupted_measurements[w_start:w_start + w_n_meas])

        # Anchor targets from GT (same as exp36).
        gt_first = gt_poses[w_start]
        anchor_global = gt_poses[w_start + jnp.array(anchor_pos_in_window)]
        w_anchor_targets = jax.vmap(relative_pose_se3, in_axes=(None, 0))(
            gt_first, anchor_global)

        # x_init: forward-compose from origin.
        origin = jnp.zeros(6, dtype=jnp.float32)
        x_init = _forward_compose_jit(origin, w_noisy).ravel()

        # Run robust denoiser (Phase A + Phase B).
        theta_opt, weights, chi_scores = robust_denoise(
            w_noisy, x_init, w_anchor_targets)
        jax.block_until_ready(theta_opt)

        # Commit.
        theta_np = np.array(theta_opt)
        weights_np = np.array(weights)
        chi_np = np.array(chi_scores)
        commit_start, commit_end = commit_ranges[wi]
        for gi in range(commit_start, commit_end + 1):
            local_i = gi - w_start
            if 0 <= local_i < w_n_meas and gi < n_meas_total:
                denoised_measurements[gi] = theta_np[local_i]
                all_weights[gi] = weights_np[local_i]
                all_chi[gi] = chi_np[local_i]

        t_win = time.perf_counter() - t_win_start

        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            elapsed = time.perf_counter() - t_denoise_start
            avg_hz = (wi + 1) * actual_window / elapsed
            mean_w = float(weights_np.mean())
            n_rejected = int((weights_np < 0.5).sum())
            print(f"    Window {wi+1}/{len(windows)} done "
                  f"({elapsed:.1f}s, {avg_hz:.1f} poses/s, "
                  f"avg_w={mean_w:.3f}, rejected={n_rejected}/{w_n_meas})",
                  flush=True)

    t_denoise = time.perf_counter() - t_denoise_start
    poses_per_sec = n_poses_total / t_denoise

    # --- Metrics ---
    baseline = compute_per_edge_error(
        jnp.array(corrupted_measurements), gt_measurements)
    after = compute_per_edge_error(
        jnp.array(denoised_measurements), gt_measurements)

    trans_improv = (1 - after['rmse_trans'] / baseline['rmse_trans']) * 100
    rot_improv = (1 - after['rmse_rot'] / baseline['rmse_rot']) * 100

    # Weight analysis.
    w_outlier = all_weights[outlier_mask_np]
    w_inlier = all_weights[~outlier_mask_np]

    # Classification at threshold 0.5.
    predicted_outlier = all_weights < 0.5
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
    print(f"  Mean chi (outliers):    {np.mean(all_chi[outlier_mask_np]):.2f}")
    print(f"  Mean chi (inliers):     {np.mean(all_chi[~outlier_mask_np]):.2f}")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, "
          f"F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

    return {
        "sequence": seq_id,
        "seed": seed,
        "n_poses": n_poses_total,
        "n_outliers": n_outliers,
        "n_inliers": n_meas_total - n_outliers,
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
            "inlier_mean_w": round(float(np.mean(w_inlier)), 4),
            "inlier_std_w": round(float(np.std(w_inlier)), 4),
            "outlier_mean_chi": round(float(np.mean(all_chi[outlier_mask_np])), 2),
            "inlier_mean_chi": round(float(np.mean(all_chi[~outlier_mask_np])), 2),
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
        "weights": all_weights.tolist(),
        "chi_scores": all_chi.tolist(),
        "outlier_mask": outlier_mask_np.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp38: robust outlier rejection via residual-based "
                    "weighting in streaming IFT denoiser")
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
    parser.add_argument("--anchor-spacing", type=int, default=100)
    parser.add_argument("--gn-iters", type=int, default=10)
    parser.add_argument("--n-trans-iters", type=int, default=50)
    parser.add_argument("--n-rot-iters", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernel-scale", type=float, default=3.0,
                        help="Welsch kernel scale (in sigma units). "
                             "Edges with chi > c are strongly downweighted.")
    parser.add_argument("--aw-trans", type=float, default=5.0)
    parser.add_argument("--aw-rot", type=float, default=5.0)
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
    print("  exp38 -- Robust Outlier Rejection via Residual-Weighted IFT")
    print("=" * 70)
    _print_device_info()
    print()
    print(f"  Noise: σ_t={args.sigma_trans}, σ_r={args.sigma_rot}")
    print(f"  Outliers: {args.outlier_ratio*100:.0f}% ratio, "
          f"mag_t={args.outlier_mag_trans}, mag_r={args.outlier_mag_rot}")
    print(f"  Denoiser: window={args.window_size}, "
          f"anchor_spacing={args.anchor_spacing}")
    print(f"  Kernel: Welsch, scale={args.kernel_scale}σ")
    print(f"  Optimiser: {args.n_trans_iters} trans + {args.n_rot_iters} rot "
          f"= {n_total_iters}")
    print(f"  LR: {args.lr}")
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
        "lr": args.lr,
        "kernel_scale": args.kernel_scale,
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
              f"{'chi_out':>7s} {'chi_in':>6s} "
              f"{'Prec':>5s} {'Rec':>5s} {'F1':>5s} {'Hz':>6s}")
    print(header)
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*7} {'-'*7} "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*6}")

    for r in all_results:
        imp = r['improvement_pct']
        ws = r['weight_stats']
        cl = r['classification']
        print(f"  {r['sequence']:>4s} {r['seed']:>5d} "
              f"{r['outlier_ratio_actual']*100:>4.0f}% "
              f"{imp['trans_rmse']:>+6.1f}% {imp['rot_rmse']:>+6.1f}% "
              f"{ws['outlier_mean_w']:>5.3f} {ws['inlier_mean_w']:>5.3f} "
              f"{ws['outlier_mean_chi']:>6.1f} {ws['inlier_mean_chi']:>5.1f} "
              f"{cl['precision']:>5.3f} {cl['recall']:>5.3f} "
              f"{cl['f1']:>5.3f} {r['throughput']['poses_per_sec']:>5.0f}")

    # Aggregate.
    if len(all_results) > 1:
        t_vals = [r['improvement_pct']['trans_rmse'] for r in all_results]
        r_vals = [r['improvement_pct']['rot_rmse'] for r in all_results]
        f1_vals = [r['classification']['f1'] for r in all_results]
        print(f"\n  Aggregate ({len(all_results)} runs):")
        print(f"    Trans improvement: {np.mean(t_vals):+.1f}% ± {np.std(t_vals):.1f}")
        print(f"    Rot improvement:   {np.mean(r_vals):+.1f}% ± {np.std(r_vals):.1f}")
        print(f"    F1 score:          {np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f}")

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
