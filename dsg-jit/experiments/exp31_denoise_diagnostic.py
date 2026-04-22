# experiments/exp31_denoise_diagnostic.py
"""
Per-pose denoising diagnostic — isolate what causes rotation RMSE blowup.

Builds on exp28's direct bilevel denoising (simpler than exp30's joint
approach) but applies exp24's proven loss design:
  - UNWEIGHTED anchor loss (geometric, not info-weighted)
  - Info-weighted regularisation (statistical prior)
  - Unweighted temporal smoothness

Key diagnostic additions:
  - Per-pose measurement error (trans and rot separately) before/after
  - Per-window convergence tracking
  - Separate trans/rot error reporting at every pose index
  - No PGO evaluation, no KITTI, no weight learning — pure denoising signal

Hypothesis: exp28's info-weighted anchor loss over-weights rotation at
anchor positions (since sigma_rot < sigma_trans => 4x rotation anchor
weight), causing the optimiser to distort rotation corrections between
anchors.  This experiment tests that hypothesis by using exp24's
unweighted anchor loss at scale.

Usage:
    python -m experiments.exp31_denoise_diagnostic
    python -m experiments.exp31_denoise_diagnostic --rw-trans 0.5 --rw-rot 0.05
    python -m experiments.exp31_denoise_diagnostic --n-poses 200 --anchor-spacing 50
    JAX_PLATFORM_NAME=cpu python -m experiments.exp31_denoise_diagnostic --n-poses 21
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
# Device info
# ---------------------------------------------------------------------------

def _print_device_info():
    devices = jax.devices()
    print(f"  JAX devices:    {[str(d) for d in devices]}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("  WARNING: running on CPU.")


# ---------------------------------------------------------------------------
# Synthetic trajectory (same as exp28/30)
# ---------------------------------------------------------------------------

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


def reconstruct_trajectory(first_pose, measurements):
    poses = [first_pose]
    for k in range(measurements.shape[0]):
        poses.append(compose_pose_se3(poses[-1], measurements[k]))
    return np.array(jnp.stack(poses))


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
# Bilevel denoiser (exp28 structure + exp24 loss design)
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
    rw_trans: float = 1.0,
    rw_rot: float = 0.1,
    smooth_weight: float = 2.0,
    inner_anchor_sigma: float = 0.01,
):
    """Build a JIT-compiled bilevel denoiser.

    Loss design:
      - Anchor loss: separate trans/rot weights.
      - Regularisation: info-weighted (1/sigma^2) with separate trans/rot
        multipliers so rotation corrections are not over-constrained.
      - Smoothness: UNWEIGHTED temporal diff.
      - Inner anchor stiffness controlled by inner_anchor_sigma.
        Lower = stiffer (poses pinned harder at anchors).
        Higher = softer (inner solver can distribute error more smoothly
        near anchors, reducing rotation distortion).
    """
    n_meas = n_poses - 1
    odom_w = sigma_to_weight(sigma)
    sqrt_odom_w = jnp.sqrt(odom_w)
    anchor_w = sigma_to_weight(jnp.full(6, inner_anchor_sigma))
    sqrt_anchor_w = jnp.sqrt(anchor_w)
    anchor_idx = jnp.array(anchor_positions, dtype=jnp.int32)

    # Per-component anchor weight.
    anchor_w_vec = jnp.array(
        [aw_trans] * 3 + [aw_rot] * 3, dtype=jnp.float32)

    # Per-component reg weight: info-weighted base * separate trans/rot scale.
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

        # 1. Anchor loss — separate trans/rot weights.
        #    Weakening aw_rot reduces the rotation pull at anchor positions,
        #    preventing the optimizer from distorting rotation corrections
        #    at anchor-adjacent edges.
        diffs = poses_opt[anchor_idx] - anchor_targets
        a_loss = jnp.sum(anchor_w_vec * diffs ** 2)

        # 2. Regularisation — info-weighted with separate trans/rot scaling.
        #    Rotation gets a lower multiplier so the optimiser has freedom
        #    to correct rotation noise without being pulled back.
        dev = theta - noisy_meas
        r_loss = jnp.sum(reg_w_vec * dev ** 2)

        # 3. Temporal smoothness — UNWEIGHTED (exp24 design).
        s_diffs = theta[1:] - theta[:-1]
        s_loss = jnp.sum(s_diffs ** 2)

        return a_loss + r_loss + smooth_weight * s_loss

    grad_fn = jax.jit(jax.grad(outer_loss))
    loss_fn = jax.jit(outer_loss)
    return grad_fn, loss_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="exp31: Per-pose denoising diagnostic")
    parser.add_argument("--n-poses", type=int, default=100,
                        help="Total poses in trajectory (default: 100)")
    parser.add_argument("--window-size", type=int, default=50,
                        help="Poses per solve window (default: 50)")
    parser.add_argument("--anchor-spacing", type=int, default=25,
                        help="GT anchor every N poses within each window (default: 25)")
    parser.add_argument("--sigma-trans", type=float, default=0.10)
    parser.add_argument("--sigma-rot", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--n-outer-iters", type=int, default=100,
                        help="Outer Adam iterations per window (default: 100)")
    parser.add_argument("--gn-iters", type=int, default=10,
                        help="Inner GN iterations (default: 10)")
    parser.add_argument("--aw-trans", type=float, default=5.0,
                        help="Anchor weight, translation (default: 5.0)")
    parser.add_argument("--aw-rot", type=float, default=5.0,
                        help="Anchor weight, rotation (default: 5.0)")
    parser.add_argument("--rw-trans", type=float, default=1.0,
                        help="Reg weight, translation (default: 1.0)")
    parser.add_argument("--rw-rot", type=float, default=0.1,
                        help="Reg weight, rotation (default: 0.1)")
    parser.add_argument("--sw", type=float, default=2.0,
                        help="Smoothness weight (default: 2.0)")
    parser.add_argument("--inner-anchor-sigma", type=float, default=0.01,
                        help="Inner solver anchor stiffness sigma (default: 0.01, "
                        "higher = softer anchors)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="exp31_results.json")
    args = parser.parse_args()

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)
    window_size = args.window_size
    overlap = min(10, window_size // 5)
    stride = window_size - overlap

    print("=" * 70)
    print("  exp31 -- Per-Pose Denoising Diagnostic")
    print("=" * 70)
    _print_device_info()
    print()

    # ---- Generate trajectory ----
    n_poses_total = args.n_poses
    print(f"Generating synthetic trajectory ({n_poses_total} poses)...",
          flush=True)
    gt_poses, data_info = generate_kitti_like_trajectory(n_poses_total)
    gt_np = np.array(gt_poses)
    n_meas_total = n_poses_total - 1

    # ---- Compute measurements and add noise ----
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

    # Anchor positions within each window.
    anchor_pos_in_window = list(range(0, actual_window, args.anchor_spacing))
    if anchor_pos_in_window[-1] != actual_window - 1:
        anchor_pos_in_window.append(actual_window - 1)

    anchor_density = len(anchor_pos_in_window) / actual_window * 100

    traj_len = float(np.sum(np.linalg.norm(
        np.diff(gt_np[:, :3], axis=0), axis=1)))

    print(f"  Poses:          {n_poses_total}")
    print(f"  Trajectory len: {traj_len:.1f} m")
    print(f"  Noise:          sigma_t={args.sigma_trans}m, "
          f"sigma_r={args.sigma_rot}rad")
    print(f"  Window size:    {actual_window} poses, stride={stride}, "
          f"overlap={overlap}")
    print(f"  Windows:        {len(windows)}")
    print(f"  Anchors/window: {len(anchor_pos_in_window)} "
          f"(every {args.anchor_spacing}, {anchor_density:.0f}% density)")
    print(f"  Weights:        aw_t={args.aw_trans}, aw_r={args.aw_rot}, "
          f"rw_t={args.rw_trans}, rw_r={args.rw_rot}, sw={args.sw}")
    print(f"  Inner anchor:   sigma={args.inner_anchor_sigma}")
    print(f"  Loss design:    anchor=separate trans/rot, "
          f"reg=info-weighted (separate trans/rot), smooth=UNWEIGHTED")
    print(f"  Inner GN iters: {args.gn_iters}")
    print(f"  Outer iters:    {args.n_outer_iters} (Adam, lr={args.lr})")
    print()

    # ---- Baseline per-pose error ----
    baseline = compute_per_pose_meas_error(noisy_measurements, gt_measurements)
    print(f"Baseline measurement error:")
    print(f"  Trans mean: {baseline['mean_trans']:.4f} m   "
          f"RMSE: {baseline['rmse_trans']:.4f} m")
    print(f"  Rot mean:   {baseline['mean_rot']:.4f} rad  "
          f"RMSE: {baseline['rmse_rot']:.4f} rad")
    print()

    # ---- Build and JIT-compile denoiser ----
    print(f"JIT-compiling denoiser ({actual_window} poses, "
          f"{len(anchor_pos_in_window)} anchors)...", flush=True)
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
    print(f"JIT compilation: {t_jit:.1f}s")
    print()

    # ---- Precompute commit ranges (midpoint-of-overlap) ----
    # Each window solves the full window for context but only commits
    # corrections from its assigned region.  Overlap regions are split at
    # their midpoint so every edge is covered exactly once and each edge
    # comes from the window whose centre is closest (best solve quality).
    commit_ranges = []
    for wi in range(len(windows)):
        w_start, w_end = windows[wi]
        w_first_edge = w_start
        w_last_edge = w_end - 2  # last edge index this window covers

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

    print(f"  Commit ranges:  {['%d-%d' % (s, e) for s, e in commit_ranges]}")
    print()

    # ---- Denoise window by window ----
    print("Denoising measurements...", flush=True)
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
        initial_loss = float(loss_fn(theta, x_init, w_anchor_targets, w_noisy))

        for it in range(args.n_outer_iters):
            g = grad_fn(theta, x_init, w_anchor_targets, w_noisy)
            g.block_until_ready()

            if jnp.any(jnp.isnan(g)):
                print(f"  Window {wi}: NaN gradient at iter {it}, "
                      f"stopping early")
                break

            update, adam_state = adam_step(g, adam_state, lr=args.lr)
            theta = theta - update

        # Commit only this window's assigned edges.
        theta_np = np.array(theta)
        commit_start, commit_end = commit_ranges[wi]
        for gi in range(commit_start, commit_end + 1):
            local_i = gi - w_start
            if 0 <= local_i < w_n_meas and gi < n_meas_total:
                denoised_measurements[gi] = theta_np[local_i]

        # Per-window diagnostics.
        final_loss = float(loss_fn(theta, x_init, w_anchor_targets, w_noisy))
        gt_meas_window = gt_measurements[w_start:w_start + w_n_meas]
        w_err_before = compute_per_pose_meas_error(w_noisy, gt_meas_window)
        w_err_after = compute_per_pose_meas_error(
            jnp.array(theta_np), gt_meas_window)

        elapsed = time.perf_counter() - t_denoise_start
        loss_red = (1 - final_loss / max(initial_loss, 1e-9)) * 100
        t_imp = (1 - w_err_after['mean_trans']
                 / max(w_err_before['mean_trans'], 1e-9)) * 100
        r_imp = (1 - w_err_after['mean_rot']
                 / max(w_err_before['mean_rot'], 1e-9)) * 100
        print(f"  Window {wi+1}/{len(windows)} "
              f"(poses {w_start}-{w_end-1}): "
              f"loss {initial_loss:.1f}->{final_loss:.1f} "
              f"({loss_red:.1f}%), "
              f"trans {w_err_before['mean_trans']:.4f}"
              f"->{w_err_after['mean_trans']:.4f} ({t_imp:+.1f}%), "
              f"rot {w_err_before['mean_rot']:.4f}"
              f"->{w_err_after['mean_rot']:.4f} ({r_imp:+.1f}%), "
              f"elapsed={elapsed:.1f}s", flush=True)

    t_denoise = time.perf_counter() - t_denoise_start
    denoised_meas_jnp = jnp.array(denoised_measurements)

    # ---- Per-pose error analysis ----
    after = compute_per_pose_meas_error(denoised_meas_jnp, gt_measurements)

    # Per-edge delta (positive = improvement, negative = degradation).
    delta_trans = baseline['per_edge_trans'] - after['per_edge_trans']
    delta_rot = baseline['per_edge_rot'] - after['per_edge_rot']

    # Count improved / degraded edges.
    n_trans_improved = int(np.sum(delta_trans > 0))
    n_rot_improved = int(np.sum(delta_rot > 0))
    n_trans_degraded = int(np.sum(delta_trans < 0))
    n_rot_degraded = int(np.sum(delta_rot < 0))

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print()
    print("--- Global Measurement Error ---")
    print(f"  {'':20s} {'Before':>10s} {'After':>10s} {'Improv.':>10s}")
    print(f"  {'Trans mean [m]':20s} "
          f"{baseline['mean_trans']:10.4f} {after['mean_trans']:10.4f} "
          f"{(1 - after['mean_trans']/baseline['mean_trans'])*100:9.1f}%")
    print(f"  {'Trans RMSE [m]':20s} "
          f"{baseline['rmse_trans']:10.4f} {after['rmse_trans']:10.4f} "
          f"{(1 - after['rmse_trans']/baseline['rmse_trans'])*100:9.1f}%")
    print(f"  {'Rot mean [rad]':20s} "
          f"{baseline['mean_rot']:10.4f} {after['mean_rot']:10.4f} "
          f"{(1 - after['mean_rot']/baseline['mean_rot'])*100:9.1f}%")
    print(f"  {'Rot RMSE [rad]':20s} "
          f"{baseline['rmse_rot']:10.4f} {after['rmse_rot']:10.4f} "
          f"{(1 - after['rmse_rot']/baseline['rmse_rot'])*100:9.1f}%")

    print()
    print("--- Per-Edge Improvement Distribution ---")
    print(f"  Translation: {n_trans_improved}/{n_meas_total} improved, "
          f"{n_trans_degraded}/{n_meas_total} degraded")
    print(f"  Rotation:    {n_rot_improved}/{n_meas_total} improved, "
          f"{n_rot_degraded}/{n_meas_total} degraded")

    # Edges near anchors vs far from anchors.
    global_anchor_positions = set()
    for w_start, w_end in windows:
        for p in anchor_pos_in_window:
            global_anchor_positions.add(w_start + p)

    near_anchor = []  # edges within 2 of any anchor
    far_from_anchor = []
    for i in range(n_meas_total):
        near = any(abs(i - a) <= 2 for a in global_anchor_positions)
        if near:
            near_anchor.append(i)
        else:
            far_from_anchor.append(i)

    if near_anchor and far_from_anchor:
        near_idx = np.array(near_anchor)
        far_idx = np.array(far_from_anchor)
        print()
        print("--- Anchor Proximity Analysis ---")
        print(f"  Edges near anchors (within 2): {len(near_anchor)}")
        print(f"  Edges far from anchors:        {len(far_from_anchor)}")
        print()
        print(f"  {'':25s} {'Near anchor':>12s} {'Far from':>12s}")
        print(f"  {'Trans delta (mean) [m]':25s} "
              f"{np.mean(delta_trans[near_idx]):+12.4f} "
              f"{np.mean(delta_trans[far_idx]):+12.4f}")
        print(f"  {'Rot delta (mean) [rad]':25s} "
              f"{np.mean(delta_rot[near_idx]):+12.4f} "
              f"{np.mean(delta_rot[far_idx]):+12.4f}")
        print(f"  {'Trans after (mean) [m]':25s} "
              f"{np.mean(after['per_edge_trans'][near_idx]):12.4f} "
              f"{np.mean(after['per_edge_trans'][far_idx]):12.4f}")
        print(f"  {'Rot after (mean) [rad]':25s} "
              f"{np.mean(after['per_edge_rot'][near_idx]):12.4f} "
              f"{np.mean(after['per_edge_rot'][far_idx]):12.4f}")

    # Per-edge detail (sampled if too many).
    print()
    print("--- Per-Edge Measurement Error (sampled) ---")
    sample_step = max(1, n_meas_total // 30)
    print(f"  {'Edge':>6s} {'T_before':>10s} {'T_after':>10s} "
          f"{'T_delta':>10s} {'R_before':>10s} {'R_after':>10s} "
          f"{'R_delta':>10s} {'anchor?':>8s}")
    for i in range(0, n_meas_total, sample_step):
        is_anchor = 'Y' if i in global_anchor_positions else ''
        print(f"  {i:6d} "
              f"{baseline['per_edge_trans'][i]:10.4f} "
              f"{after['per_edge_trans'][i]:10.4f} "
              f"{delta_trans[i]:+10.4f} "
              f"{baseline['per_edge_rot'][i]:10.4f} "
              f"{after['per_edge_rot'][i]:10.4f} "
              f"{delta_rot[i]:+10.4f} "
              f"{is_anchor:>8s}")

    print()
    print("--- Timing ---")
    print(f"  JIT compilation: {t_jit:8.1f} s")
    print(f"  Denoising:       {t_denoise:8.1f} s  ({len(windows)} windows)")
    print()

    # ---- Save results ----
    results = {
        "config": {
            "n_poses": n_poses_total,
            "trajectory_length_m": round(traj_len, 1),
            "window_size": actual_window,
            "stride": stride,
            "overlap": overlap,
            "n_windows": len(windows),
            "anchor_spacing": args.anchor_spacing,
            "anchors_per_window": len(anchor_pos_in_window),
            "anchor_density_pct": round(anchor_density, 1),
            "sigma_trans": args.sigma_trans,
            "sigma_rot": args.sigma_rot,
            "gn_iters": args.gn_iters,
            "outer_iters": args.n_outer_iters,
            "outer_lr": args.lr,
            "aw_trans": args.aw_trans,
            "aw_rot": args.aw_rot,
            "rw_trans": args.rw_trans,
            "rw_rot": args.rw_rot,
            "smooth_weight": args.sw,
            "inner_anchor_sigma": args.inner_anchor_sigma,
        },
        "measurement_error": {
            "trans_before": round(baseline['mean_trans'], 6),
            "trans_after": round(after['mean_trans'], 6),
            "trans_improvement_pct": round(
                (1 - after['mean_trans'] / baseline['mean_trans']) * 100, 1),
            "rot_before": round(baseline['mean_rot'], 6),
            "rot_after": round(after['mean_rot'], 6),
            "rot_improvement_pct": round(
                (1 - after['mean_rot'] / baseline['mean_rot']) * 100, 1),
            "rmse_trans_before": round(baseline['rmse_trans'], 6),
            "rmse_trans_after": round(after['rmse_trans'], 6),
            "rmse_rot_before": round(baseline['rmse_rot'], 6),
            "rmse_rot_after": round(after['rmse_rot'], 6),
        },
        "per_edge_distribution": {
            "trans_improved": n_trans_improved,
            "trans_degraded": n_trans_degraded,
            "rot_improved": n_rot_improved,
            "rot_degraded": n_rot_degraded,
            "total_edges": n_meas_total,
        },
        "per_edge_trans_before": [round(float(v), 6)
                                  for v in baseline['per_edge_trans']],
        "per_edge_trans_after": [round(float(v), 6)
                                 for v in after['per_edge_trans']],
        "per_edge_rot_before": [round(float(v), 6)
                                for v in baseline['per_edge_rot']],
        "per_edge_rot_after": [round(float(v), 6)
                               for v in after['per_edge_rot']],
        "timing": {
            "jit_s": round(t_jit, 2),
            "denoising_s": round(t_denoise, 2),
        },
    }

    if near_anchor and far_from_anchor:
        near_idx = np.array(near_anchor)
        far_idx = np.array(far_from_anchor)
        results["anchor_proximity"] = {
            "n_near": len(near_anchor),
            "n_far": len(far_from_anchor),
            "near_trans_delta_mean": round(
                float(np.mean(delta_trans[near_idx])), 6),
            "far_trans_delta_mean": round(
                float(np.mean(delta_trans[far_idx])), 6),
            "near_rot_delta_mean": round(
                float(np.mean(delta_rot[near_idx])), 6),
            "far_rot_delta_mean": round(
                float(np.mean(delta_rot[far_idx])), 6),
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")

    return results


if __name__ == "__main__":
    main()
