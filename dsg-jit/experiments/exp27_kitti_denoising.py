# experiments/exp27_kitti_denoising.py
"""
Measurement denoising on KITTI odometry trajectories.

Demonstrates bilevel SE(3) measurement denoising on real trajectory geometry
from the KITTI odometry benchmark.  Ground-truth relative poses are corrupted
with synthetic noise, then denoised via differentiable Gauss-Newton.

Phase 1 (this script): GT relative poses + synthetic noise + sparse GT anchors.
Phase 2 (future):      Real VO measurements + raw OXTS GPS anchors.

Usage:
    python -m experiments.exp27_kitti_denoising \\
        --kitti-root /path/to/kitti/odometry \\
        --seq 07 \\
        --anchor-spacing 50 \\
        --sigma-trans 0.10 \\
        --sigma-rot 0.05

    If --kitti-root is not provided or the data is missing, the experiment
    falls back to generating a KITTI-like trajectory synthetically so the
    code can be tested without the dataset.
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
    so3_log,
)
from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_geodesic_residual,
    sigma_to_weight,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kitti_poses(kitti_root: str, seq: str, max_frames: int | None = None):
    """Load KITTI GT poses and convert to 6D vectors.

    Returns (poses_6d, sequence_info) or (None, None) if data is unavailable.
    """
    try:
        from dsg_jit.datasets.kitti_odometry import load_kitti_odometry_sequence
        from dsg_jit.datasets.kitti_utils import kitti_frames_to_poses6d

        frames = load_kitti_odometry_sequence(
            kitti_root, seq,
            load_right=False, load_velodyne=False,
            with_poses=True, max_frames=max_frames,
        )
        if not frames or frames[0].T_w_cam0 is None:
            return None, None

        poses = kitti_frames_to_poses6d(frames)
        info = {
            "source": "kitti_odometry",
            "sequence": seq,
            "n_frames": len(frames),
        }
        return poses, info
    except (FileNotFoundError, Exception) as e:
        print(f"  Could not load KITTI data: {e}")
        return None, None


def generate_kitti_like_trajectory(n_poses: int) -> tuple[jnp.ndarray, dict]:
    """Generate a trajectory with KITTI-like characteristics.

    Mimics a suburban driving trajectory: mostly forward motion at ~10m/s
    (1m/frame at 10Hz) with gentle turns and a few sharper corners.
    """
    key = jax.random.PRNGKey(0)
    poses = []
    x, y, heading = 0.0, 0.0, 0.0

    # Pre-generate curvature variations (gentle driving with a few turns).
    keys = jax.random.split(key, 3)
    base_curvature = 0.002  # gentle curve
    # Add a few sharper turns.
    curvature = np.full(n_poses, base_curvature)
    # Simulate turns at ~25%, 50%, 75% of trajectory.
    for frac in [0.25, 0.5, 0.75]:
        turn_start = int(frac * n_poses)
        turn_len = min(50, n_poses // 10)
        curvature[turn_start:turn_start + turn_len] = 0.02  # sharper turn

    step = 1.0  # ~10m/s at 10Hz = 1m/frame
    for i in range(n_poses):
        poses.append(jnp.array([x, y, 0.0, 0.0, 0.0, heading],
                               dtype=jnp.float32))
        x += step * float(jnp.cos(heading))
        y += step * float(jnp.sin(heading))
        heading += curvature[i]

    info = {
        "source": "synthetic_kitti_like",
        "n_frames": n_poses,
    }
    return jnp.stack(poses), info


# ---------------------------------------------------------------------------
# Factor graph construction (canonical 5-pose segment, same as exp26)
# ---------------------------------------------------------------------------

def _to_slice(idx):
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def build_canonical_segment(sigma: jnp.ndarray, n_poses: int = 5):
    """Build a canonical n-pose factor graph for one segment."""
    wm = WorldModel()
    placeholder = jnp.zeros(6, dtype=jnp.float32)

    pose_ids = []
    for _ in range(n_poses):
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
    """Build JIT-compilable loss for one 5-pose segment with 3 GT anchors."""
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
            else:
                params = {"target": anchor_first, "weight": prior_weight}
            r = res_fn(stacked, params)
            res_list.append(r)
        return jnp.concatenate(res_list)

    def solve_and_loss(theta, x_init, anchor_first, anchor_mid, anchor_last,
                       noisy_meas, aw, rw, sw):
        def residual_fn(x):
            return residual_fn_parametric(x, theta, anchor_first)

        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)

        diff_first = x_opt[anchor_sl_first] - anchor_first
        diff_mid = x_opt[anchor_sl_mid] - anchor_mid
        diff_last = x_opt[anchor_sl_last] - anchor_last
        anchor_loss = (jnp.sum(diff_first ** 2)
                       + jnp.sum(diff_mid ** 2)
                       + jnp.sum(diff_last ** 2))

        deviation = theta - noisy_meas
        reg_loss = jnp.sum(info_weight * deviation ** 2)

        diffs = theta[1:] - theta[:-1]
        smooth_loss = jnp.sum(diffs ** 2)

        return aw * anchor_loss + rw * reg_loss + sw * smooth_loss

    return solve_and_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray) -> dict:
    trans_errors = np.linalg.norm(estimated[:, :3] - ground_truth[:, :3], axis=1)
    rot_errors = np.linalg.norm(estimated[:, 3:] - ground_truth[:, 3:], axis=1)
    return {
        "ate_trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "ate_trans_mean": float(np.mean(trans_errors)),
        "ate_trans_max": float(np.max(trans_errors)),
        "ate_rot_rmse": float(np.sqrt(np.mean(rot_errors ** 2))),
        "ate_rot_mean": float(np.mean(rot_errors)),
    }


def compute_rpe(estimated: np.ndarray, ground_truth: np.ndarray) -> dict:
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


def compute_kitti_metric(estimated: np.ndarray, ground_truth: np.ndarray,
                         step_size: int = 10) -> dict:
    """Compute KITTI-style translational and rotational error.

    Evaluates over subsequences of varying length (100-800m at 10Hz/1m per
    frame ≈ 100-800 frames).  Returns average translational error (%) and
    rotational error (deg/m).
    """
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    n = estimated.shape[0]

    trans_errors, rot_errors = [], []
    for start in range(0, n, step_size):
        for length in lengths:
            end = start + length
            if end >= n:
                continue

            # Relative transform: GT.
            gt_rel = np.array(relative_pose_se3(
                jnp.array(ground_truth[start]), jnp.array(ground_truth[end])))
            est_rel = np.array(relative_pose_se3(
                jnp.array(estimated[start]), jnp.array(estimated[end])))

            # Translational error (% of path length).
            path_len = float(np.linalg.norm(gt_rel[:3]))
            if path_len < 1.0:
                continue
            t_err = float(np.linalg.norm(est_rel[:3] - gt_rel[:3]))
            trans_errors.append(t_err / path_len * 100.0)

            # Rotational error (deg/m).
            r_err = float(np.linalg.norm(est_rel[3:] - gt_rel[3:]))
            rot_errors.append(np.degrees(r_err) / path_len)

    if not trans_errors:
        return {"kitti_trans_err_pct": 0.0, "kitti_rot_err_degm": 0.0}

    return {
        "kitti_trans_err_pct": float(np.mean(trans_errors)),
        "kitti_rot_err_degm": float(np.mean(rot_errors)),
    }


def reconstruct_trajectory(first_pose: jnp.ndarray,
                           measurements: jnp.ndarray) -> np.ndarray:
    """Forward-compose measurements from a starting pose."""
    poses = [first_pose]
    for k in range(measurements.shape[0]):
        poses.append(compose_pose_se3(poses[-1], measurements[k]))
    return np.array(jnp.stack(poses))


# ---------------------------------------------------------------------------
# Downstream evaluation: PGO with original vs denoised measurements
# ---------------------------------------------------------------------------

def build_pgo_solver(
    n_poses: int,
    anchor_indices: list[int],
    sigma: jnp.ndarray,
    gn_iters: int = 20,
    damping: float = 1e-3,
):
    """Build a JIT-compiled PGO solver.

    Returns a function ``solve(measurements, x_init, anchor_targets) -> x_opt``
    that is fully JIT-compiled.  The graph structure (number of poses, anchor
    positions) is baked in at trace time; only measurements, initial state,
    and anchor targets vary at runtime.
    """
    from dsg_jit.core.math3d import se3_retract_left

    odom_weight = sigma_to_weight(sigma)
    anchor_weight = sigma_to_weight(jnp.full(6, 0.01))
    sqrt_odom_w = jnp.sqrt(odom_weight)
    sqrt_anchor_w = jnp.sqrt(anchor_weight)

    anchor_idx_array = jnp.array(anchor_indices, dtype=jnp.int32)

    def residual_fn(x, measurements, anchor_targets):
        # x is (n_poses * 6,) flat state.
        poses = x.reshape(n_poses, 6)

        # Odometry residuals: relative_pose(pose_i, pose_{i+1}) - meas_i.
        res_odom = []
        for k in range(n_poses - 1):
            xi_est = relative_pose_se3(poses[k], poses[k + 1])
            r = (xi_est - measurements[k]) * sqrt_odom_w
            res_odom.append(r)

        # Anchor residuals: pose[anchor_idx] - anchor_target.
        res_anchor = []
        for j in range(len(anchor_indices)):
            idx = anchor_indices[j]
            r = (poses[idx] - anchor_targets[j]) * sqrt_anchor_w
            res_anchor.append(r)

        return jnp.concatenate(res_odom + res_anchor)

    def gn_step(x, measurements, anchor_targets):
        def r_fn(x_):
            return residual_fn(x_, measurements, anchor_targets)

        r = r_fn(x)
        J = jax.jacobian(r_fn)(x)

        H = J.T @ J
        g = J.T @ r
        n = x.shape[0]
        H_damped = H + damping * jnp.eye(n)
        delta = jnp.linalg.solve(H_damped, g)

        # Step clamp.
        step_norm = jnp.linalg.norm(delta)
        scale = jnp.minimum(1.0, 1.0 / (step_norm + 1e-9))
        delta = scale * delta

        # Manifold retraction: apply SE(3) update per pose.
        poses = x.reshape(n_poses, 6)
        deltas = delta.reshape(n_poses, 6)
        new_poses = []
        for i in range(n_poses):
            new_poses.append(se3_retract_left(poses[i], -deltas[i]))
        return jnp.concatenate(new_poses)

    @jax.jit
    def solve(measurements, x_init, anchor_targets):
        x = x_init
        for _ in range(gn_iters):
            x = gn_step(x, measurements, anchor_targets)
        return x

    return solve


def run_pgo_with_measurements(
    measurements: jnp.ndarray,
    first_pose: jnp.ndarray,
    gt_poses: jnp.ndarray,
    anchor_indices: list[int],
    sigma: jnp.ndarray,
    pgo_solve_fn=None,
) -> tuple[np.ndarray, float]:
    """Run PGO and return (optimised_poses, elapsed_seconds).

    If ``pgo_solve_fn`` is provided it is reused (avoids recompilation).
    """
    n_poses = measurements.shape[0] + 1

    # Build initial trajectory from forward composition.
    init_poses = [first_pose]
    for k in range(n_poses - 1):
        init_poses.append(compose_pose_se3(init_poses[-1], measurements[k]))
    x_init = jnp.concatenate(init_poses)

    anchor_targets = jnp.stack([gt_poses[i] for i in anchor_indices])

    t0 = time.perf_counter()
    x_opt = pgo_solve_fn(measurements, x_init, anchor_targets)
    x_opt.block_until_ready()
    elapsed = time.perf_counter() - t0

    return np.array(x_opt.reshape(n_poses, 6)), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measurement denoising on KITTI trajectories")
    parser.add_argument("--kitti-root", type=str, default=None,
                        help="Path to KITTI odometry dataset root")
    parser.add_argument("--seq", type=str, default="07",
                        help="KITTI sequence ID (default: 07)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames (for testing)")
    parser.add_argument("--anchor-spacing", type=int, default=50,
                        help="GT anchor every N poses (default: 50)")
    parser.add_argument("--sigma-trans", type=float, default=0.10,
                        help="Translation noise std (m, default: 0.10)")
    parser.add_argument("--sigma-rot", type=float, default=0.05,
                        help="Rotation noise std (rad, default: 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="exp27_results.json")
    args = parser.parse_args()

    # ---- Hyperparameters (from exp25 sweep) ----
    anchor_weight = 5.0
    reg_weight = 0.1
    smooth_weight = 2.0
    lr = 0.002
    n_outer_iters = 150
    segment_size = 5
    segment_stride = segment_size - 1  # 4 measurements per segment
    gn_cfg = GNConfig(max_iters=10, damping=5e-3, max_step_norm=0.5)

    sigma = jnp.array([args.sigma_trans] * 3 + [args.sigma_rot] * 3,
                       dtype=jnp.float32)

    print("=" * 70)
    print("  exp27 -- KITTI Trajectory Measurement Denoising")
    print("=" * 70)

    # ---- Load or generate trajectory ----
    gt_poses, data_info = None, None
    if args.kitti_root:
        print(f"Loading KITTI sequence {args.seq}...", flush=True)
        gt_poses, data_info = load_kitti_poses(
            args.kitti_root, args.seq, args.max_frames)

    if gt_poses is None:
        n_synth = args.max_frames or 500
        print(f"Generating KITTI-like synthetic trajectory ({n_synth} poses)...",
              flush=True)
        gt_poses, data_info = generate_kitti_like_trajectory(n_synth)

    n_poses = gt_poses.shape[0]
    gt_np = np.array(gt_poses)
    n_measurements = n_poses - 1

    # ---- Compute GT relative measurements and add noise ----
    print("Computing relative measurements and adding noise...", flush=True)
    gt_measurements = jnp.stack([
        relative_pose_se3(gt_poses[i], gt_poses[i + 1])
        for i in range(n_measurements)
    ])

    key = jax.random.PRNGKey(args.seed)
    noise = jax.random.normal(key, shape=gt_measurements.shape) * sigma
    noisy_measurements = gt_measurements + noise

    # ---- Define anchor positions ----
    anchor_indices = list(range(0, n_poses, args.anchor_spacing))
    if anchor_indices[-1] != n_poses - 1:
        anchor_indices.append(n_poses - 1)

    # Segment-level anchors: for denoising, we need GT at every segment
    # boundary. Segments span `segment_stride` measurements.
    seg_anchor_spacing = segment_stride
    seg_anchor_indices = list(range(0, n_poses, seg_anchor_spacing))
    if seg_anchor_indices[-1] != n_poses - 1:
        seg_anchor_indices.append(n_poses - 1)

    n_segments = n_measurements // segment_stride

    # Trajectory length.
    traj_len = float(np.sum(np.linalg.norm(
        np.diff(gt_np[:, :3], axis=0), axis=1)))

    print(f"  Data source:    {data_info['source']}")
    print(f"  Poses:          {n_poses}")
    print(f"  Measurements:   {n_measurements}")
    print(f"  Trajectory len: {traj_len:.1f} m")
    print(f"  Noise:          sigma_t={args.sigma_trans}m, sigma_r={args.sigma_rot}rad")
    print(f"  PGO anchors:    {len(anchor_indices)} (every {args.anchor_spacing} poses)")
    print(f"  Denoising segs: {n_segments} x {segment_size}-pose")
    print(f"  Seg anchors:    {len(seg_anchor_indices)} (every {seg_anchor_spacing} poses)")
    print()

    # ---- Baseline: noisy trajectory ----
    noisy_poses_np = reconstruct_trajectory(gt_poses[0], noisy_measurements)
    ate_noisy = compute_ate(noisy_poses_np, gt_np)
    rpe_noisy = compute_rpe(noisy_poses_np, gt_np)
    kitti_noisy = compute_kitti_metric(noisy_poses_np, gt_np)
    meas_err_before = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))

    print(f"Baseline (noisy, no PGO):")
    print(f"  ATE trans RMSE:  {ate_noisy['ate_trans_rmse']:.4f} m")
    print(f"  KITTI trans err: {kitti_noisy['kitti_trans_err_pct']:.2f}%")
    print(f"  Meas error:      {meas_err_before:.4f}")
    print()

    # ---- Build JIT-compiled PGO solver (shared by noisy and denoised) ----
    print("Building JIT-compiled PGO solver...", flush=True)
    pgo_solve = build_pgo_solver(
        n_poses, anchor_indices, sigma,
        gn_iters=20, damping=1e-3)

    # Warm-up PGO JIT.
    t_pgo_jit_start = time.perf_counter()
    pgo_noisy_poses, _ = run_pgo_with_measurements(
        noisy_measurements, gt_poses[0], gt_poses, anchor_indices,
        sigma, pgo_solve_fn=pgo_solve)
    t_pgo_jit = time.perf_counter() - t_pgo_jit_start
    print(f"PGO JIT compile + first solve: {t_pgo_jit:.1f}s")

    # ---- Step 1: PGO with NOISY measurements (downstream baseline) ----
    print("Running PGO with noisy measurements...", flush=True)
    pgo_noisy_poses, t_pgo_noisy = run_pgo_with_measurements(
        noisy_measurements, gt_poses[0], gt_poses, anchor_indices,
        sigma, pgo_solve_fn=pgo_solve)

    ate_pgo_noisy = compute_ate(pgo_noisy_poses, gt_np)
    rpe_pgo_noisy = compute_rpe(pgo_noisy_poses, gt_np)
    kitti_pgo_noisy = compute_kitti_metric(pgo_noisy_poses, gt_np)
    print(f"PGO (noisy meas, {len(anchor_indices)} anchors): {t_pgo_noisy:.1f}s")
    print(f"  ATE trans RMSE:  {ate_pgo_noisy['ate_trans_rmse']:.4f} m")
    print(f"  KITTI trans err: {kitti_pgo_noisy['kitti_trans_err_pct']:.2f}%")
    print()

    # ---- Step 2: Denoise measurements ----
    print("Building canonical segment and JIT compiling...", flush=True)
    t_jit_start = time.perf_counter()

    (bs_seq, mt_seq, pose_slices, factor_info,
     residual_fns, state_dim) = build_canonical_segment(sigma)

    solve_and_loss = build_segment_loss_fn(
        bs_seq, mt_seq, pose_slices, factor_info, residual_fns,
        state_dim, gn_cfg, sigma)

    grad_fn = jax.jit(jax.grad(solve_and_loss))

    # Warm-up.
    dummy_theta = jnp.zeros((segment_stride, 6), dtype=jnp.float32)
    dummy_x = jnp.zeros(state_dim, dtype=jnp.float32)
    dummy_a = jnp.zeros(6, dtype=jnp.float32)
    _aw = jnp.float32(anchor_weight)
    _rw = jnp.float32(reg_weight)
    _sw = jnp.float32(smooth_weight)
    _ = grad_fn(dummy_theta, dummy_x, dummy_a, dummy_a, dummy_a,
                dummy_theta, _aw, _rw, _sw).block_until_ready()

    t_jit = time.perf_counter() - t_jit_start
    print(f"JIT compilation: {t_jit:.1f}s")
    print()

    print("Denoising measurements...", flush=True)
    t_denoise_start = time.perf_counter()
    denoised_measurements = np.array(noisy_measurements).copy()

    for seg in range(n_segments):
        meas_start = seg * segment_stride
        meas_end = meas_start + segment_stride
        pose_start = meas_start

        seg_meas = jnp.array(denoised_measurements[meas_start:meas_end])

        # Local coordinates: centre segment at origin.
        gt_first = gt_poses[pose_start]
        anchor_first = jnp.zeros(6, dtype=jnp.float32)
        anchor_mid = relative_pose_se3(gt_first, gt_poses[pose_start + 2])
        anchor_last = relative_pose_se3(gt_first, gt_poses[pose_start + 4])

        # x_init: forward-compose from origin.
        init_poses_list = [anchor_first]
        for k in range(segment_stride):
            init_poses_list.append(
                compose_pose_se3(init_poses_list[-1], seg_meas[k]))
        x_init = jnp.concatenate(init_poses_list)

        # Outer gradient descent.
        theta = seg_meas.copy()
        for _ in range(n_outer_iters):
            g = grad_fn(theta, x_init, anchor_first, anchor_mid, anchor_last,
                        seg_meas, _aw, _rw, _sw)
            g.block_until_ready()
            theta = theta - lr * g

        for i in range(segment_stride):
            denoised_measurements[meas_start + i] = np.array(theta[i])

        if (seg + 1) % 25 == 0 or seg == 0 or seg == n_segments - 1:
            elapsed = time.perf_counter() - t_denoise_start
            print(f"  Segment {seg+1}/{n_segments} elapsed={elapsed:.1f}s",
                  flush=True)

    t_denoise = time.perf_counter() - t_denoise_start
    denoised_meas_jnp = jnp.array(denoised_measurements)

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

    print(f"\nDenoising complete: {n_segments} segments, {t_denoise:.1f}s")
    print(f"  Meas error: {meas_err_before:.4f} -> {meas_err_after:.4f} "
          f"({(1 - meas_err_after/meas_err_before)*100:.1f}%)")
    print(f"  Trans:      {meas_trans_before:.4f} -> {meas_trans_after:.4f} "
          f"({(1 - meas_trans_after/meas_trans_before)*100:.1f}%)")
    print(f"  Rot:        {meas_rot_before:.4f} -> {meas_rot_after:.4f} "
          f"({(1 - meas_rot_after/meas_rot_before)*100:.1f}%)")
    print()

    # ---- Step 3: PGO with DENOISED measurements (downstream benefit) ----
    print("Running PGO with denoised measurements...", flush=True)
    pgo_denoised_poses, t_pgo_denoised = run_pgo_with_measurements(
        denoised_meas_jnp, gt_poses[0], gt_poses, anchor_indices,
        sigma, pgo_solve_fn=pgo_solve)

    ate_pgo_denoised = compute_ate(pgo_denoised_poses, gt_np)
    rpe_pgo_denoised = compute_rpe(pgo_denoised_poses, gt_np)
    kitti_pgo_denoised = compute_kitti_metric(pgo_denoised_poses, gt_np)

    # ---- Also reconstruct denoised trajectory without PGO ----
    denoised_poses_np = reconstruct_trajectory(gt_poses[0], denoised_meas_jnp)
    ate_denoised = compute_ate(denoised_poses_np, gt_np)
    rpe_denoised = compute_rpe(denoised_poses_np, gt_np)
    kitti_denoised = compute_kitti_metric(denoised_poses_np, gt_np)

    # ---- Summary ----
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
    for label, key_name in [
        ("ATE trans RMSE [m]", "ate_trans_rmse"),
        ("ATE rot RMSE [rad]", "ate_rot_rmse"),
    ]:
        v1 = ate_pgo_noisy[key_name]
        v2 = ate_pgo_denoised[key_name]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")

    for label, key_name in [
        ("RPE trans RMSE [m]", "rpe_trans_rmse"),
        ("RPE rot RMSE [rad]", "rpe_rot_rmse"),
    ]:
        v1 = rpe_pgo_noisy[key_name]
        v2 = rpe_pgo_denoised[key_name]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")

    for label, key_name in [
        ("KITTI trans [%]", "kitti_trans_err_pct"),
        ("KITTI rot [deg/m]", "kitti_rot_err_degm"),
    ]:
        v1 = kitti_pgo_noisy[key_name]
        v2 = kitti_pgo_denoised[key_name]
        imp = (1 - v2 / v1) * 100 if v1 > 0 else 0
        print(f"  {label:20s} {v1:12.4f} {v2:12.4f} {imp:9.1f}%")

    print()
    print("--- Timing ---")
    print(f"  Denoise JIT:         {t_jit:8.1f} s")
    print(f"  PGO JIT:             {t_pgo_jit:8.1f} s")
    print(f"  Denoising:           {t_denoise:8.1f} s  ({n_segments} segments)")
    print(f"  PGO (noisy):         {t_pgo_noisy:8.1f} s")
    print(f"  PGO (denoised):      {t_pgo_denoised:8.1f} s")
    print()

    # ---- Save results ----
    results = {
        "config": {
            "data_source": data_info["source"],
            "n_poses": n_poses,
            "trajectory_length_m": round(traj_len, 1),
            "sigma_trans": args.sigma_trans,
            "sigma_rot": args.sigma_rot,
            "anchor_spacing": args.anchor_spacing,
            "n_pgo_anchors": len(anchor_indices),
            "n_segments": n_segments,
            "segment_size": segment_size,
            "n_seg_anchors": len(seg_anchor_indices),
            "anchor_weight": anchor_weight,
            "reg_weight": reg_weight,
            "smooth_weight": smooth_weight,
            "lr": lr,
            "n_outer_iters": n_outer_iters,
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
            **{k: round(v, 6) for k, v in ate_pgo_denoised.items()},
            **{k: round(v, 6) for k, v in rpe_pgo_denoised.items()},
            **{k: round(v, 4) for k, v in kitti_pgo_denoised.items()},
        },
        "timing": {
            "denoise_jit_s": round(t_jit, 2),
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
