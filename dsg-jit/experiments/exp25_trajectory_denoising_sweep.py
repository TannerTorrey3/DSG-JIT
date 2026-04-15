# experiments/exp25_trajectory_denoising_sweep.py
"""
Scaled trajectory denoising with hyperparameter sweep.

Extends exp24 to a 30-pose SE(3) trajectory and sweeps over:
  - number of anchor poses (and uniform placement)
  - anchor_weight, reg_weight, smooth_weight
  - outer learning rate

The sweep is organised so that configs sharing the same anchor count
are grouped together — this avoids redundant JIT recompilation since
different weight/lr values do not change the JAX trace.

Each config runs the full inner-manifold-GN + outer-gradient-descent
pipeline with early stopping, then records ATE/RPE/measurement metrics.
The best configuration (by ATE translational RMSE) is re-evaluated and
printed in detail.
"""

from __future__ import annotations

import itertools
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
from dsg_jit.telemetry import telemetry_span


# ---------------------------------------------------------------------------
# Helpers (shared with exp24)
# ---------------------------------------------------------------------------

def _to_slice(idx):
    if isinstance(idx, slice):
        return idx
    start, length = idx
    return slice(start, start + length)


def generate_ground_truth_trajectory(n_poses: int, step: float = 1.0) -> jnp.ndarray:
    poses = []
    for i in range(n_poses):
        angle = 0.05 * i
        tx = step * i * jnp.cos(angle)
        ty = step * i * jnp.sin(angle)
        tz = 0.0
        poses.append(jnp.array([tx, ty, tz, 0.0, 0.0, angle], dtype=jnp.float32))
    return jnp.stack(poses)


def relative_measurements_from_poses(gt_poses: jnp.ndarray) -> jnp.ndarray:
    from dsg_jit.core.math3d import relative_pose_se3
    n = gt_poses.shape[0]
    meas = []
    for i in range(n - 1):
        meas.append(relative_pose_se3(gt_poses[i], gt_poses[i + 1]))
    return jnp.stack(meas)


def add_gaussian_noise(
    measurements: jnp.ndarray,
    sigma: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    noise = jax.random.normal(key, shape=measurements.shape) * sigma
    return measurements + noise


# ---------------------------------------------------------------------------
# Factor graph construction
# ---------------------------------------------------------------------------

def build_trajectory_graph(n_poses, noisy_measurements, gt_poses, sigma):
    wm = WorldModel()
    from dsg_jit.core.math3d import compose_pose_se3

    init_poses = [gt_poses[0]]
    for k in range(n_poses - 1):
        next_pose = compose_pose_se3(init_poses[-1], noisy_measurements[k])
        init_poses.append(next_pose)

    pose_ids = []
    for i in range(n_poses):
        pid = wm.add_variable(var_type="pose_se3", value=init_poses[i])
        pose_ids.append(pid)

    odom_weight = sigma_to_weight(sigma)
    wm.add_factor(
        f_type="prior",
        var_ids=(pose_ids[0],),
        params={
            "target": gt_poses[0],
            "weight": sigma_to_weight(jnp.full(6, 0.01)),
        },
    )

    odom_factor_ids = []
    for k in range(n_poses - 1):
        fid = wm.add_factor(
            f_type="odom_se3_geodesic",
            var_ids=(pose_ids[k], pose_ids[k + 1]),
            params={"measurement": noisy_measurements[k], "weight": odom_weight},
        )
        odom_factor_ids.append(fid)

    wm.register_residual("prior", prior_residual)
    wm.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)

    x_init, index = wm.pack_state()
    return wm, x_init, index, pose_ids, odom_factor_ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
# Anchor placement
# ---------------------------------------------------------------------------

def uniform_anchors(n_poses: int, n_anchors: int) -> list[int]:
    """Return uniformly spaced anchor indices, always including first and last."""
    if n_anchors <= 1:
        return [0]
    if n_anchors == 2:
        return [0, n_poses - 1]
    indices = [int(round(i * (n_poses - 1) / (n_anchors - 1)))
               for i in range(n_anchors)]
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


# ---------------------------------------------------------------------------
# Single sweep configuration runner
# ---------------------------------------------------------------------------

def run_config(
    *,
    # Shared pre-built data (avoid rebuilding per config).
    gt_poses, gt_measurements, noisy_measurements, gt_np,
    wm, x_init, index, pose_ids, odom_fids,
    bs_seq, mt_seq, pose_slices, sigma,
    # Config-specific.
    anchor_indices, anchor_weight, reg_weight, smooth_weight,
    lr, n_outer_iters, gn_cfg,
    # Pre-compiled functions (may be None on first call per anchor group).
    compiled_loss_fn=None, compiled_grad_fn=None,
):
    """Run one sweep config and return metrics dict.

    If compiled_loss_fn / compiled_grad_fn are provided they are reused
    (only valid when anchor_indices match the previous call).
    """
    info_weight = sigma_to_weight(sigma)
    anchor_gt = jnp.stack([gt_poses[i] for i in anchor_indices])
    anchor_slices = [pose_slices[i] for i in anchor_indices]

    factors = list(wm.fg.factors.values())
    residual_fns = wm._residual_registry

    def residual_param_fn(x, theta):
        var_values = wm.unpack_state(x, index)
        res_list = []
        odom_idx = 0
        for f in factors:
            res_fn = residual_fns[f.type]
            stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])
            if f.type == "odom_se3_geodesic":
                params = dict(f.params)
                params["measurement"] = theta[odom_idx]
                odom_idx += 1
            else:
                params = f.params
            r = res_fn(stacked, params)
            res_list.append(r)
        return jnp.concatenate(res_list)

    def solve_and_loss(theta, aw, rw, sw):
        def residual_fn(x):
            return residual_param_fn(x, theta)
        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)

        anchor_loss = 0.0
        for i, sl in enumerate(anchor_slices):
            diff = x_opt[sl] - anchor_gt[i]
            anchor_loss = anchor_loss + jnp.sum(diff ** 2)

        deviation = theta - noisy_measurements
        reg_loss = jnp.sum(info_weight * deviation ** 2)

        diffs = theta[1:] - theta[:-1]
        smooth_loss = jnp.sum(diffs ** 2)

        return aw * anchor_loss + rw * reg_loss + sw * smooth_loss

    need_compile = compiled_loss_fn is None

    if need_compile:
        loss_fn = jax.jit(solve_and_loss)
        grad_fn = jax.jit(jax.grad(solve_and_loss))
        # Warm-up.
        theta = noisy_measurements.copy()
        _aw = jnp.float32(anchor_weight)
        _rw = jnp.float32(reg_weight)
        _sw = jnp.float32(smooth_weight)
        t_jit_start = time.perf_counter()
        _ = loss_fn(theta, _aw, _rw, _sw).block_until_ready()
        _ = grad_fn(theta, _aw, _rw, _sw).block_until_ready()
        t_jit = time.perf_counter() - t_jit_start
    else:
        loss_fn = compiled_loss_fn
        grad_fn = compiled_grad_fn
        t_jit = 0.0

    theta = noisy_measurements.copy()
    _aw = jnp.float32(anchor_weight)
    _rw = jnp.float32(reg_weight)
    _sw = jnp.float32(smooth_weight)

    # Outer optimisation with early stopping.
    loss_history = []
    t_opt_start = time.perf_counter()
    for step_i in range(n_outer_iters):
        g = grad_fn(theta, _aw, _rw, _sw)
        g.block_until_ready()
        theta = theta - lr * g

        loss_val = float(loss_fn(theta, _aw, _rw, _sw))
        loss_history.append(loss_val)

        # Early stopping: loss change < 1e-6 for 5 consecutive iters.
        if len(loss_history) >= 6:
            recent = loss_history[-6:]
            if all(abs(recent[i+1] - recent[i]) < 1e-6 for i in range(5)):
                break

    t_opt = time.perf_counter() - t_opt_start

    # Extract optimised poses.
    def get_optimised_poses(theta_val):
        def residual_fn(x):
            return residual_param_fn(x, theta_val)
        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)
        return jnp.stack([x_opt[sl] for sl in pose_slices])

    poses_after_np = np.array(get_optimised_poses(theta))
    ate = compute_ate(poses_after_np, gt_np)
    rpe = compute_rpe(poses_after_np, gt_np)

    meas_error = float(jnp.mean(jnp.linalg.norm(theta - gt_measurements, axis=1)))

    return {
        "anchor_indices": anchor_indices,
        "n_anchors": len(anchor_indices),
        "anchor_weight": anchor_weight,
        "reg_weight": reg_weight,
        "smooth_weight": smooth_weight,
        "lr": lr,
        "converged_iter": len(loss_history),
        "final_loss": round(loss_history[-1], 6) if loss_history else None,
        "jit_compile_s": round(t_jit, 2),
        "opt_time_s": round(t_opt, 2),
        **{k: round(v, 6) for k, v in ate.items()},
        **{k: round(v, 6) for k, v in rpe.items()},
        "meas_error": round(meas_error, 6),
    }, loss_fn, grad_fn


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@telemetry_span(component="experiment", op="exp25_trajectory_denoising_sweep")
def main():
    n_poses = 5
    step = 1.0
    sigma = jnp.array([0.15, 0.15, 0.15, 0.08, 0.08, 0.08], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    gn_cfg = GNConfig(max_iters=10, damping=5e-3, max_step_norm=0.5)
    n_outer_iters = 400

    # ---- Sweep grid for 5-pose window ----
    # With 5 poses, 2 anchors (first + last) is the natural sliding-window
    # setup; 3 anchors adds a midpoint.
    sweep = {
        "n_anchors":     [2, 3],
        "anchor_weight": [0.5, 1.0, 2.0, 5.0],
        "reg_weight":    [0.1, 0.25, 0.5, 1.0],
        "smooth_weight": [0.25, 0.5, 1.0, 2.0],
        "lr":            [0.0005, 0.001, 0.002],
    }

    total_configs = 1
    for v in sweep.values():
        total_configs *= len(v)

    print("=" * 70)
    print("  exp25 — 5-pose Window Denoising Sweep")
    print("=" * 70)
    print(f"  Trajectory: {n_poses} SE(3) poses")
    print(f"  Noise: sigma_trans={sigma[0]:.2f}m, sigma_rot={sigma[3]:.2f}rad")
    print(f"  Inner GN: {gn_cfg.max_iters} iters, damping={gn_cfg.damping}")
    print(f"  Outer: up to {n_outer_iters} iters (early stopping)")
    print(f"  Sweep: {total_configs} configs "
          f"({len(sweep['n_anchors'])} anchor groups)")
    print()

    # ---- Generate shared data ----
    print("Generating trajectory and building factor graph...", flush=True)
    gt_poses = generate_ground_truth_trajectory(n_poses, step)
    gt_measurements = relative_measurements_from_poses(gt_poses)
    noisy_measurements = add_gaussian_noise(gt_measurements, sigma, key)

    wm, x_init, index, pose_ids, odom_fids = build_trajectory_graph(
        n_poses, noisy_measurements, gt_poses, sigma)

    packed = (x_init, index)
    block_slices_dict, manifold_types_dict = build_manifold_metadata(packed, wm.fg)
    bs_seq = list(block_slices_dict.items())
    mt_seq = list(manifold_types_dict.items())
    pose_slices = [_to_slice(index[pid]) for pid in pose_ids]
    gt_np = np.array(gt_poses)

    # Before-denoising baseline.
    def get_poses_from_theta(theta_val):
        factors = list(wm.fg.factors.values())
        residual_fns = wm._residual_registry
        def residual_param_fn(x, theta):
            var_values = wm.unpack_state(x, index)
            res_list = []
            odom_idx = 0
            for f in factors:
                res_fn = residual_fns[f.type]
                stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])
                if f.type == "odom_se3_geodesic":
                    params = dict(f.params)
                    params["measurement"] = theta[odom_idx]
                    odom_idx += 1
                else:
                    params = f.params
                r = res_fn(stacked, params)
                res_list.append(r)
            return jnp.concatenate(res_list)
        def residual_fn(x):
            return residual_param_fn(x, theta_val)
        x_opt = gauss_newton_manifold(residual_fn, x_init, bs_seq, mt_seq, gn_cfg)
        return jnp.stack([x_opt[sl] for sl in pose_slices])

    poses_before_np = np.array(get_poses_from_theta(noisy_measurements))
    baseline_ate = compute_ate(poses_before_np, gt_np)
    baseline_rpe = compute_rpe(poses_before_np, gt_np)
    baseline_meas = float(jnp.mean(jnp.linalg.norm(
        noisy_measurements - gt_measurements, axis=1)))

    print(f"Baseline ATE trans RMSE: {baseline_ate['ate_trans_rmse']:.4f} m")
    print(f"Baseline ATE rot RMSE:   {baseline_ate['ate_rot_rmse']:.4f} rad")
    print(f"Baseline meas error:     {baseline_meas:.4f}")
    print()

    # ---- Run sweep ----
    all_results = []
    best_result = None
    best_ate_trans = float("inf")

    # Group by n_anchors to reuse JIT compilation.
    weight_combos = list(itertools.product(
        sweep["anchor_weight"], sweep["reg_weight"],
        sweep["smooth_weight"], sweep["lr"],
    ))

    config_num = 0
    t_total_start = time.perf_counter()

    for n_anchors in sweep["n_anchors"]:
        anchor_indices = uniform_anchors(n_poses, n_anchors)
        actual_n_anchors = len(anchor_indices)

        print(f"--- Anchor group: {actual_n_anchors} anchors "
              f"({len(weight_combos)} weight combos) ---", flush=True)

        compiled_loss = None
        compiled_grad = None

        for aw, rw, sw, lr in weight_combos:
            config_num += 1
            label = (f"[{config_num}/{total_configs}] "
                     f"a={actual_n_anchors} aw={aw} rw={rw} sw={sw} lr={lr}")

            try:
                result, compiled_loss, compiled_grad = run_config(
                    gt_poses=gt_poses, gt_measurements=gt_measurements,
                    noisy_measurements=noisy_measurements, gt_np=gt_np,
                    wm=wm, x_init=x_init, index=index,
                    pose_ids=pose_ids, odom_fids=odom_fids,
                    bs_seq=bs_seq, mt_seq=mt_seq,
                    pose_slices=pose_slices, sigma=sigma,
                    anchor_indices=anchor_indices,
                    anchor_weight=aw, reg_weight=rw,
                    smooth_weight=sw, lr=lr,
                    n_outer_iters=n_outer_iters, gn_cfg=gn_cfg,
                    compiled_loss_fn=compiled_loss,
                    compiled_grad_fn=compiled_grad,
                )
            except Exception as e:
                print(f"  {label} — FAILED: {e}")
                continue

            all_results.append(result)
            ate_t = result["ate_trans_rmse"]
            improved = ate_t < best_ate_trans

            if improved:
                best_ate_trans = ate_t
                best_result = result

            marker = " *BEST*" if improved else ""
            print(f"  {label} — ATE_t={ate_t:.4f}m "
                  f"iter={result['converged_iter']}{marker}", flush=True)

    t_total = time.perf_counter() - t_total_start

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)
    print(f"  Total time:       {t_total:.1f} s")
    print(f"  Configs run:      {len(all_results)} / {total_configs}")
    print()

    print("--- Baseline (noisy, no denoising) ---")
    print(f"  ATE trans RMSE:   {baseline_ate['ate_trans_rmse']:.4f} m")
    print(f"  ATE rot RMSE:     {baseline_ate['ate_rot_rmse']:.4f} rad")
    print(f"  RPE trans RMSE:   {baseline_rpe['rpe_trans_rmse']:.4f} m")
    print(f"  RPE rot RMSE:     {baseline_rpe['rpe_rot_rmse']:.4f} rad")
    print(f"  Meas error:       {baseline_meas:.4f}")
    print()

    if best_result:
        b = best_result
        print("--- Best configuration ---")
        print(f"  n_anchors:        {b['n_anchors']}")
        print(f"  anchor_weight:    {b['anchor_weight']}")
        print(f"  reg_weight:       {b['reg_weight']}")
        print(f"  smooth_weight:    {b['smooth_weight']}")
        print(f"  lr:               {b['lr']}")
        print(f"  Converged at:     iter {b['converged_iter']}")
        print()
        print(f"  ATE trans RMSE:   {b['ate_trans_rmse']:.4f} m  "
              f"({(1 - b['ate_trans_rmse']/baseline_ate['ate_trans_rmse'])*100:.1f}% improvement)")
        print(f"  ATE rot RMSE:     {b['ate_rot_rmse']:.4f} rad  "
              f"({(1 - b['ate_rot_rmse']/baseline_ate['ate_rot_rmse'])*100:.1f}% improvement)")
        print(f"  RPE trans RMSE:   {b['rpe_trans_rmse']:.4f} m  "
              f"({(1 - b['rpe_trans_rmse']/baseline_rpe['rpe_trans_rmse'])*100:.1f}% improvement)")
        print(f"  RPE rot RMSE:     {b['rpe_rot_rmse']:.4f} rad  "
              f"({(1 - b['rpe_rot_rmse']/baseline_rpe['rpe_rot_rmse'])*100:.1f}% improvement)")
        print(f"  Meas error:       {b['meas_error']:.4f}  "
              f"({(1 - b['meas_error']/baseline_meas)*100:.1f}% improvement)")
        print()

    # ---- Top 10 ----
    sorted_results = sorted(all_results, key=lambda r: r["ate_trans_rmse"])
    print("--- Top 10 configs (by ATE trans RMSE) ---")
    print(f"  {'#':>3s}  {'anchors':>7s}  {'aw':>5s}  {'rw':>5s}  "
          f"{'sw':>5s}  {'lr':>6s}  {'ATE_t':>8s}  {'ATE_r':>8s}  "
          f"{'RPE_t':>8s}  {'iters':>5s}")
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1:3d}  {r['n_anchors']:7d}  {r['anchor_weight']:5.1f}  "
              f"{r['reg_weight']:5.1f}  {r['smooth_weight']:5.1f}  "
              f"{r['lr']:6.4f}  {r['ate_trans_rmse']:8.4f}  "
              f"{r['ate_rot_rmse']:8.4f}  {r['rpe_trans_rmse']:8.4f}  "
              f"{r['converged_iter']:5d}")
    print()

    # ---- Save full results ----
    output = {
        "config": {
            "n_poses": n_poses,
            "sigma_trans": float(sigma[0]),
            "sigma_rot": float(sigma[3]),
            "inner_gn_iters": gn_cfg.max_iters,
            "inner_damping": gn_cfg.damping,
            "inner_max_step_norm": gn_cfg.max_step_norm,
            "n_outer_iters_max": n_outer_iters,
            "sweep_grid": {k: v for k, v in sweep.items()},
            "total_configs": total_configs,
            "configs_completed": len(all_results),
        },
        "baseline": {
            **{k: round(v, 6) for k, v in baseline_ate.items()},
            **{k: round(v, 6) for k, v in baseline_rpe.items()},
            "meas_error": round(baseline_meas, 6),
        },
        "best": best_result,
        "top_10": sorted_results[:10],
        "all_results": sorted_results,
        "timing_total_s": round(t_total, 2),
    }

    out_path = "exp25_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results written to {out_path}")

    return output


if __name__ == "__main__":
    main()
