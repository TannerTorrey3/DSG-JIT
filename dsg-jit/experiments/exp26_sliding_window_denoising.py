# experiments/exp26_sliding_window_denoising.py
"""
Scaled trajectory denoising: 1000 poses via 200 independent 5-pose chains.

Each chain has GT anchors at both endpoints (poses 0 and 4), matching the
exp24/exp25 setup that achieved 86.8% measurement error reduction.  In
practice these anchors could come from GPS fixes, loop closures, or
known landmarks — any absolute position reference.

Strategy:
  - 1000 poses split into 200 non-overlapping 5-pose segments
  - GT anchors at every 5th pose (segment boundaries)
  - Each segment uses the same JIT-compiled loss (single trace reuse)
  - 3 anchors per segment: poses 0, 2, 4 (all GT)
  - No overlap, no stitching, no anchor drift

Optimal hyperparameters (from exp25 sweep on 5-pose windows):
  - n_anchors = 3, anchor_weight = 5.0, reg_weight = 0.1, smooth_weight = 2.0
  - outer lr = 0.002, 150 iterations
  - inner GN: 10 iters, damping 5e-3, max_step_norm 0.5
"""

from __future__ import annotations

import time
import json

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_geodesic_residual,
    sigma_to_weight,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_slice(idx):
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def generate_ground_truth_trajectory(n_poses: int, step: float = 1.0,
                                     yaw_rate: float = 0.05) -> jnp.ndarray:
    """Constant-speed curved trajectory matching exp25's curvature.

    Each step moves exactly ``step`` metres forward in the current heading,
    then rotates by ``yaw_rate`` rad.  This produces equally-spaced poses
    (constant arc length) so that noise sigma and solver parameters remain
    valid across all segments.

    yaw_rate=0.05 matches exp25's curvature per step.
    """
    poses = []
    tx, ty, heading = 0.0, 0.0, 0.0
    for i in range(n_poses):
        poses.append(jnp.array([tx, ty, 0.0, 0.0, 0.0, heading],
                               dtype=jnp.float32))
        # Advance: step forward in current heading, then turn
        tx += step * float(jnp.cos(heading))
        ty += step * float(jnp.sin(heading))
        heading += yaw_rate
    return jnp.stack(poses)


def relative_measurements_from_poses(gt_poses: jnp.ndarray) -> jnp.ndarray:
    from dsg_jit.core.math3d import relative_pose_se3
    n = gt_poses.shape[0]
    meas = []
    for i in range(n - 1):
        meas.append(relative_pose_se3(gt_poses[i], gt_poses[i + 1]))
    return jnp.stack(meas)


def add_gaussian_noise(measurements, sigma, key):
    noise = jax.random.normal(key, shape=measurements.shape) * sigma
    return measurements + noise


def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray):
    trans_errors = np.linalg.norm(estimated[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.linalg.norm(estimated[:, 3:] - ground_truth[:, 3:], axis=1)
    return {
        "ate_trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "ate_trans_mean": float(np.mean(trans_errors)),
        "ate_trans_max": float(np.max(trans_errors)),
        "ate_rot_rmse": float(np.sqrt(np.mean(rot_errors ** 2))),
        "ate_rot_mean": float(np.mean(rot_errors)),
        "ate_rot_max": float(np.max(rot_errors)),
    }


def compute_rpe(estimated: np.ndarray, ground_truth: np.ndarray):
    from dsg_jit.core.math3d import relative_pose_se3
    n = estimated.shape[0]
    trans_errors, rot_errors = [], []
    for i in range(n - 1):
        rel_est = np.array(relative_pose_se3(
            jnp.array(estimated[i]), jnp.array(estimated[i + 1])))
        rel_gt = np.array(relative_pose_se3(
            jnp.array(ground_truth[i]), jnp.array(ground_truth[i + 1])))
        diff = rel_est - rel_gt
        trans_errors.append(np.linalg.norm(diff[:3]))
        rot_errors.append(np.linalg.norm(diff[3:]))
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "rpe_trans_mean": float(np.mean(trans_errors)),
        "rpe_rot_rmse": float(np.sqrt(np.mean(rot_errors ** 2))),
        "rpe_rot_mean": float(np.mean(rot_errors)),
    }


# ---------------------------------------------------------------------------
# Build a canonical 5-pose graph (structure only, data will vary)
# ---------------------------------------------------------------------------

def build_canonical_window(sigma: jnp.ndarray):
    """Build a canonical 5-pose factor graph whose structure is fixed.

    Returns everything needed to construct the JIT-compilable loss.
    """
    n_poses = 5
    wm = WorldModel()

    placeholder = jnp.zeros(6, dtype=jnp.float32)

    pose_ids = []
    for i in range(n_poses):
        pid = wm.add_variable(var_type="pose_se3", value=placeholder)
        pose_ids.append(pid)

    odom_weight = sigma_to_weight(sigma)
    prior_weight = sigma_to_weight(jnp.full(6, 0.01))

    wm.add_factor(
        f_type="prior",
        var_ids=(pose_ids[0],),
        params={"target": placeholder, "weight": prior_weight},
    )

    for k in range(n_poses - 1):
        wm.add_factor(
            f_type="odom_se3_geodesic",
            var_ids=(pose_ids[k], pose_ids[k + 1]),
            params={"measurement": placeholder, "weight": odom_weight},
        )

    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    x_init, index = wm.pack_state()
    packed = (x_init, index)
    block_slices_dict, manifold_types_dict = build_manifold_metadata(packed, wm.fg)
    bs_seq = list(block_slices_dict.items())
    mt_seq = list(manifold_types_dict.items())
    pose_slices = [_to_slice(index[pid]) for pid in pose_ids]

    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry

    factor_info = []
    for f in factors:
        var_slices = [_to_slice(index[vid]) for vid in f.var_ids]
        factor_info.append((f.type, var_slices))

    return bs_seq, mt_seq, pose_slices, factor_info, residual_fns, x_init.shape[0]


def build_segment_loss_fn(
    bs_seq, mt_seq, pose_slices, factor_info, residual_fns,
    state_dim: int, gn_cfg: GNConfig, sigma: jnp.ndarray,
):
    """Build a JIT-compilable loss for one 5-pose segment with 3 GT anchors.

    Arguments passed at runtime (all window-varying data):
      - theta: (4, 6) learnable measurements
      - x_init: (30,) initial state
      - anchor_first: (6,) GT anchor at pose 0
      - anchor_mid: (6,) GT anchor at pose 2
      - anchor_last: (6,) GT anchor at pose 4
      - noisy_meas: (4, 6) original noisy measurements (for regulariser)
      - aw, rw, sw: loss weights
    """
    info_weight = sigma_to_weight(sigma)
    prior_weight = sigma_to_weight(jnp.full(6, 0.01))

    anchor_sl_first = pose_slices[0]
    anchor_sl_mid = pose_slices[2]
    anchor_sl_last = pose_slices[4]

    def residual_fn_parametric(x, theta, anchor_first):
        res_list = []
        odom_idx = 0
        for f_type, var_slices in factor_info:
            stacked = jnp.concatenate([x[sl] for sl in var_slices])
            res_fn = residual_fns[f_type]
            if f_type == "odom_se3_geodesic":
                params = {"measurement": theta[odom_idx], "weight": info_weight}
                odom_idx += 1
            else:  # prior
                params = {"target": anchor_first, "weight": prior_weight}
            r = res_fn(stacked, params)
            res_list.append(r)
        return jnp.concatenate(res_list)

    def solve_and_loss(theta, x_init, anchor_first, anchor_mid, anchor_last,
                       noisy_meas, aw, rw, sw):
        def residual_fn(x):
            return residual_fn_parametric(x, theta, anchor_first)

        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)

        # 3-anchor loss (all GT).
        diff_first = x_opt[anchor_sl_first] - anchor_first
        diff_mid = x_opt[anchor_sl_mid] - anchor_mid
        diff_last = x_opt[anchor_sl_last] - anchor_last
        anchor_loss = (jnp.sum(diff_first ** 2)
                       + jnp.sum(diff_mid ** 2)
                       + jnp.sum(diff_last ** 2))

        # Regulariser: keep theta near noisy observations.
        deviation = theta - noisy_meas
        reg_loss = jnp.sum(info_weight * deviation ** 2)

        # Temporal smoothness.
        diffs = theta[1:] - theta[:-1]
        smooth_loss = jnp.sum(diffs ** 2)

        return aw * anchor_loss + rw * reg_loss + sw * smooth_loss

    return solve_and_loss


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    # ---- Configuration ----
    n_poses = 1001  # 200 segments of 5 poses (sharing boundary poses)
    step = 1.0
    sigma = jnp.array([0.15, 0.15, 0.15, 0.08, 0.08, 0.08], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    segment_size = 5  # poses per segment
    segment_stride = segment_size - 1  # 4 measurements per segment

    # Optimal hyperparameters from exp25 sweep (3-anchor, 5-pose).
    anchor_weight = 5.0
    reg_weight = 0.1
    smooth_weight = 2.0
    lr = 0.002
    n_outer_iters = 150

    gn_cfg = GNConfig(max_iters=10, damping=5e-3, max_step_norm=0.5)

    n_measurements = n_poses - 1
    n_segments = n_measurements // segment_stride

    # GT anchors at every segment boundary (every 4 poses).
    anchor_spacing = segment_stride
    gt_anchor_indices = list(range(0, n_poses, anchor_spacing))
    if gt_anchor_indices[-1] != n_poses - 1:
        gt_anchor_indices.append(n_poses - 1)

    print("=" * 70)
    print("  exp26 -- Scaled Trajectory Denoising (Independent Segments)")
    print("=" * 70)
    print(f"  Trajectory:     {n_poses} SE(3) poses, step={step}m")
    print(f"  Noise:          sigma_trans={sigma[0]:.2f}m, sigma_rot={sigma[3]:.2f}rad")
    print(f"  Segments:       {n_segments} x {segment_size}-pose chains")
    print(f"  GT anchors:     every {anchor_spacing} poses ({len(gt_anchor_indices)} total)")
    print(f"  Anchors/seg:    3 (poses 0, 2, 4 — all GT)")
    print(f"  Weights:        aw={anchor_weight}, rw={reg_weight}, sw={smooth_weight}")
    print(f"  Outer:          lr={lr}, {n_outer_iters} iters/segment")
    print(f"  Inner GN:       {gn_cfg.max_iters} iters, damping={gn_cfg.damping}")
    print()

    # ---- Generate data ----
    print("Generating trajectory...", flush=True)
    gt_poses = generate_ground_truth_trajectory(n_poses, step)
    gt_measurements = relative_measurements_from_poses(gt_poses)
    noisy_measurements = add_gaussian_noise(gt_measurements, sigma, key)
    gt_np = np.array(gt_poses)

    # ---- Before-denoising baseline (forward-compose from noisy) ----
    from dsg_jit.core.math3d import compose_pose_se3
    noisy_poses = [gt_poses[0]]
    for k in range(n_measurements):
        noisy_poses.append(compose_pose_se3(noisy_poses[-1], noisy_measurements[k]))
    noisy_poses_np = np.array(jnp.stack(noisy_poses))

    ate_before = compute_ate(noisy_poses_np, gt_np)
    rpe_before = compute_rpe(noisy_poses_np, gt_np)
    meas_error_before = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))

    print(f"Baseline ATE trans RMSE: {ate_before['ate_trans_rmse']:.4f} m")
    print(f"Baseline ATE rot RMSE:   {ate_before['ate_rot_rmse']:.4f} rad")
    print(f"Baseline meas error:     {meas_error_before:.4f}")
    print()

    # ---- Build canonical segment and JIT-compile ONCE ----
    print("Building canonical segment and JIT compiling...", flush=True)
    t_jit_start = time.perf_counter()

    (bs_seq, mt_seq, pose_slices, factor_info,
     residual_fns, state_dim) = build_canonical_window(sigma)

    solve_and_loss = build_segment_loss_fn(
        bs_seq, mt_seq, pose_slices, factor_info, residual_fns,
        state_dim, gn_cfg, sigma)

    grad_fn = jax.jit(jax.grad(solve_and_loss))

    # Warm-up with dummy data to trigger compilation.
    dummy_theta = jnp.zeros((4, 6), dtype=jnp.float32)
    dummy_x_init = jnp.zeros(state_dim, dtype=jnp.float32)
    dummy_anchor = jnp.zeros(6, dtype=jnp.float32)
    _aw = jnp.float32(anchor_weight)
    _rw = jnp.float32(reg_weight)
    _sw = jnp.float32(smooth_weight)

    _ = grad_fn(dummy_theta, dummy_x_init, dummy_anchor, dummy_anchor,
                dummy_anchor, dummy_theta, _aw, _rw, _sw).block_until_ready()

    t_jit = time.perf_counter() - t_jit_start
    print(f"JIT compilation: {t_jit:.1f}s")
    print()

    # ---- Process all segments independently ----
    denoised_measurements = np.array(noisy_measurements).copy()

    print("Processing segments...", flush=True)
    t_seg_start = time.perf_counter()

    from dsg_jit.core.math3d import relative_pose_se3

    for seg in range(n_segments):
        meas_start = seg * segment_stride
        meas_end = meas_start + segment_stride
        pose_start = meas_start  # first pose index

        # 4 measurements for this segment
        seg_meas = jnp.array(denoised_measurements[meas_start:meas_end])

        # 3 GT anchors in LOCAL coordinates (relative to anchor_first).
        # This keeps all state values near zero for numerical stability.
        gt_first = gt_poses[pose_start]
        anchor_first = jnp.zeros(6, dtype=jnp.float32)  # origin
        anchor_mid = relative_pose_se3(gt_first, gt_poses[pose_start + 2])
        anchor_last = relative_pose_se3(gt_first, gt_poses[pose_start + 4])

        # Build x_init in local frame: forward-compose from origin
        init_poses_list = [anchor_first]
        for k in range(segment_stride):
            init_poses_list.append(
                compose_pose_se3(init_poses_list[-1], seg_meas[k]))
        x_init = jnp.concatenate(init_poses_list)

        # Outer optimisation
        theta = seg_meas.copy()
        for _ in range(n_outer_iters):
            g = grad_fn(theta, x_init, anchor_first, anchor_mid, anchor_last,
                        seg_meas, _aw, _rw, _sw)
            g.block_until_ready()
            theta = theta - lr * g

        # Commit all 4 denoised measurements
        for i in range(segment_stride):
            denoised_measurements[meas_start + i] = np.array(theta[i])

        if (seg + 1) % 50 == 0 or seg == 0:
            elapsed = time.perf_counter() - t_seg_start
            print(f"  Segment {seg+1}/{n_segments} "
                  f"(meas {meas_start}-{meas_end}) "
                  f"elapsed={elapsed:.1f}s", flush=True)

    t_seg = time.perf_counter() - t_seg_start
    per_seg_ms = t_seg / max(1, n_segments) * 1000
    print(f"\nAll segments complete: {n_segments} segments, "
          f"{t_seg:.1f}s total")
    print(f"Per segment: {per_seg_ms:.1f}ms")
    print()

    # ---- Reconstruct denoised trajectory ----
    denoised_meas_jnp = jnp.array(denoised_measurements)
    denoised_poses = [gt_poses[0]]
    for k in range(n_measurements):
        denoised_poses.append(compose_pose_se3(
            denoised_poses[-1], denoised_meas_jnp[k]))
    denoised_poses_np = np.array(jnp.stack(denoised_poses))

    # ---- After-denoising metrics ----
    ate_after = compute_ate(denoised_poses_np, gt_np)
    rpe_after = compute_rpe(denoised_poses_np, gt_np)
    meas_error_after = float(jnp.mean(jnp.linalg.norm(
        denoised_meas_jnp - gt_measurements, axis=1)))

    # Per-component measurement error.
    meas_trans_before = float(jnp.mean(jnp.linalg.norm(
        (noisy_measurements - gt_measurements)[:, :3], axis=1)))
    meas_trans_after = float(jnp.mean(jnp.linalg.norm(
        (denoised_meas_jnp - gt_measurements)[:, :3], axis=1)))
    meas_rot_before = float(jnp.mean(jnp.linalg.norm(
        (noisy_measurements - gt_measurements)[:, 3:], axis=1)))
    meas_rot_after = float(jnp.mean(jnp.linalg.norm(
        (denoised_meas_jnp - gt_measurements)[:, 3:], axis=1)))

    # ---- Print summary ----
    print("--- Absolute Trajectory Error (ATE) ---")
    print(f"  {'':20s} {'Before':>12s} {'After':>12s} {'Improv.':>10s}")
    print(f"  {'Trans RMSE [m]':20s} {ate_before['ate_trans_rmse']:12.4f} "
          f"{ate_after['ate_trans_rmse']:12.4f} "
          f"{(1 - ate_after['ate_trans_rmse']/ate_before['ate_trans_rmse'])*100:9.1f}%")
    print(f"  {'Trans mean [m]':20s} {ate_before['ate_trans_mean']:12.4f} "
          f"{ate_after['ate_trans_mean']:12.4f}")
    print(f"  {'Trans max [m]':20s} {ate_before['ate_trans_max']:12.4f} "
          f"{ate_after['ate_trans_max']:12.4f}")
    print(f"  {'Rot RMSE [rad]':20s} {ate_before['ate_rot_rmse']:12.4f} "
          f"{ate_after['ate_rot_rmse']:12.4f} "
          f"{(1 - ate_after['ate_rot_rmse']/ate_before['ate_rot_rmse'])*100:9.1f}%")
    print(f"  {'Rot mean [rad]':20s} {ate_before['ate_rot_mean']:12.4f} "
          f"{ate_after['ate_rot_mean']:12.4f}")
    print()

    print("--- Relative Pose Error (RPE) ---")
    print(f"  {'':20s} {'Before':>12s} {'After':>12s} {'Improv.':>10s}")
    print(f"  {'Trans RMSE [m]':20s} {rpe_before['rpe_trans_rmse']:12.4f} "
          f"{rpe_after['rpe_trans_rmse']:12.4f} "
          f"{(1 - rpe_after['rpe_trans_rmse']/rpe_before['rpe_trans_rmse'])*100:9.1f}%")
    print(f"  {'Rot RMSE [rad]':20s} {rpe_before['rpe_rot_rmse']:12.4f} "
          f"{rpe_after['rpe_rot_rmse']:12.4f} "
          f"{(1 - rpe_after['rpe_rot_rmse']/rpe_before['rpe_rot_rmse'])*100:9.1f}%")
    print()

    print("--- Measurement Error ---")
    print(f"  {'':20s} {'Before':>12s} {'After':>12s} {'Improv.':>10s}")
    print(f"  {'Overall':20s} {meas_error_before:12.4f} "
          f"{meas_error_after:12.4f} "
          f"{(1 - meas_error_after/meas_error_before)*100:9.1f}%")
    print(f"  {'Translation [m]':20s} {meas_trans_before:12.4f} "
          f"{meas_trans_after:12.4f} "
          f"{(1 - meas_trans_after/meas_trans_before)*100:9.1f}%")
    print(f"  {'Rotation [rad]':20s} {meas_rot_before:12.4f} "
          f"{meas_rot_after:12.4f} "
          f"{(1 - meas_rot_after/meas_rot_before)*100:9.1f}%")
    print()

    print("--- Timing ---")
    print(f"  JIT compile:     {t_jit:8.1f} s")
    print(f"  Segments:        {t_seg:8.1f} s  ({n_segments} segments)")
    print(f"  Per segment:     {per_seg_ms:8.1f} ms")
    print(f"  Total:           {t_jit + t_seg:8.1f} s")
    print()

    # ---- Save results ----
    results = {
        "config": {
            "n_poses": n_poses,
            "segment_size": segment_size,
            "n_segments": n_segments,
            "anchor_spacing": anchor_spacing,
            "n_gt_anchors": len(gt_anchor_indices),
            "sigma_trans": float(sigma[0]),
            "sigma_rot": float(sigma[3]),
            "anchor_weight": anchor_weight,
            "reg_weight": reg_weight,
            "smooth_weight": smooth_weight,
            "outer_lr": lr,
            "n_outer_iters": n_outer_iters,
            "inner_gn_iters": gn_cfg.max_iters,
            "inner_damping": gn_cfg.damping,
            "inner_max_step_norm": gn_cfg.max_step_norm,
        },
        "timing": {
            "jit_compile_s": round(t_jit, 2),
            "segments_s": round(t_seg, 2),
            "per_segment_ms": round(per_seg_ms, 1),
            "total_s": round(t_jit + t_seg, 2),
        },
        "baseline": {
            **{k: round(v, 6) for k, v in ate_before.items()},
            **{k: round(v, 6) for k, v in rpe_before.items()},
            "meas_error": round(meas_error_before, 6),
            "meas_trans_error": round(meas_trans_before, 6),
            "meas_rot_error": round(meas_rot_before, 6),
        },
        "denoised": {
            **{k: round(v, 6) for k, v in ate_after.items()},
            **{k: round(v, 6) for k, v in rpe_after.items()},
            "meas_error": round(meas_error_after, 6),
            "meas_trans_error": round(meas_trans_after, 6),
            "meas_rot_error": round(meas_rot_after, 6),
        },
        "improvement": {
            "ate_trans_rmse_pct": round(
                (1 - ate_after["ate_trans_rmse"] / ate_before["ate_trans_rmse"]) * 100, 1),
            "ate_rot_rmse_pct": round(
                (1 - ate_after["ate_rot_rmse"] / ate_before["ate_rot_rmse"]) * 100, 1),
            "rpe_trans_rmse_pct": round(
                (1 - rpe_after["rpe_trans_rmse"] / rpe_before["rpe_trans_rmse"]) * 100, 1),
            "rpe_rot_rmse_pct": round(
                (1 - rpe_after["rpe_rot_rmse"] / rpe_before["rpe_rot_rmse"]) * 100, 1),
            "meas_error_pct": round(
                (1 - meas_error_after / meas_error_before) * 100, 1),
            "meas_trans_pct": round(
                (1 - meas_trans_after / meas_trans_before) * 100, 1),
            "meas_rot_pct": round(
                (1 - meas_rot_after / meas_rot_before) * 100, 1),
        },
    }

    out_path = "exp26_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results written to {out_path}")

    return results


if __name__ == "__main__":
    main()
