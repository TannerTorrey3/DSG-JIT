"""
diag_exp44_stitching.py

Follow-up to diag_exp44_translation_phase.py: that script confirmed every
window's OWN cumulative loss_t improves under the outer loop (13/13 for
seq01/seed9 at sigma_t=0.01, several dropping to <1% of their initial
value) -- ruling out a per-window optimization bug. So if a seed's
aggregate dT is still catastrophically bad (seq01/seed9: dT=-195.8% at
sigma_t=0.01), the problem must be in how per-window corrections get
STITCHED TOGETHER into the final global trajectory, not in any single
window's own solve.

This script runs the ACTUAL production reconstruction (build_denoiser +
denoise_sequence_pooled, unmodified) for one seed, then walks the
resulting GLOBAL trajectory pose-by-pose comparing denoised vs ground
truth position, to find exactly where the error suddenly jumps -- and
cross-references that pose index against the window boundaries
(lo/hi/write_lo/write_hi) actually used by denoise_sequence_pooled, to
identify which window (or window-boundary handoff) is responsible.

Run (on the machine with real KITTI data):
  python -m experiments.diag_exp44_stitching --kitti-root /path/to/kitti \
      --seq 01 --seed 9 --sigma-t 0.01 --sigma-r 0.005 --output-dir ~/exp_res
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp44_anchored_sesync_gn import (
    ExpCfg,
    so3_log,
    build_denoiser,
    denoise_sequence_pooled,
    integrate_poses,
    _add_kitti_noise,
    _sigma_and_precision,
    noise_adaptive_inner_outer_cfg,
)
from experiments.diag_exp44_outer_regression import load_sequence


def compute_windows(window: int, overlap: int, n_poses: int):
    """Mirrors denoise_sequence_pooled's own windowing loop exactly (same
    stride/write_lo/write_hi formula), so window boundaries here line up
    with what actually produced the trajectory being inspected."""
    stride = window - overlap
    windows = []
    pos = 0
    last_written = 0
    while pos + window <= n_poses - 1:
        lo, hi = pos, pos + window
        n_edges = hi - 1 - lo
        write_lo = overlap // 2 if pos > 0 else 0
        write_hi = min(n_edges, stride + overlap // 2) if hi < n_poses - 1 else n_edges
        windows.append({"lo": lo, "hi": hi, "write_lo": write_lo, "write_hi": write_hi,
                         "global_write_lo": lo + write_lo, "global_write_hi": lo + write_hi})
        last_written = lo + write_hi
        pos += stride
    if last_written < n_poses - 1:
        hi = n_poses
        lo = max(0, hi - window)
        n_edges = hi - 1 - lo
        write_lo = last_written - lo
        windows.append({"lo": lo, "hi": hi, "write_lo": write_lo, "write_hi": n_edges,
                         "global_write_lo": lo + write_lo, "global_write_hi": lo + n_edges})
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--seq", default="01")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--sigma-t", type=float, default=0.01)
    ap.add_argument("--sigma-r", type=float, default=0.005)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--pool-seeds", type=int, default=22)
    ap.add_argument("--anchor-spacing", type=int, default=50)
    ap.add_argument("--kappa-anchor", type=float, default=100.0)
    ap.add_argument("--adaptive-reference-sigma-t", type=float, default=0.03)
    ap.add_argument("--top-k", type=int, default=10,
                     help="how many worst single-step error jumps to report")
    ap.add_argument("--output-dir", default=os.path.expanduser("~/exp_res"))
    args = ap.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"diag_exp44_stitching_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    seq_hash = int(args.seq)
    print(f"Loading seq {args.seq} ...")
    gt_mats, gt_global, gt_R_mats, gt_rel = load_sequence(args.kitti_root, args.seq)
    n_poses = gt_global.shape[0]
    print(f"  n_poses={n_poses}")

    noisy_rels = np.stack([
        _add_kitti_noise(gt_rel, s, seq_hash, args.sigma_t, args.sigma_r)
        for s in range(args.pool_seeds)
    ])

    base_inner_kwargs = {"n_iters_rot": 15, "damping_init": 1e-4, "damping_min": 1e-6,
                          "damping_max": 1e2, "damping_down": 0.5, "damping_up": 4.0,
                          "anchor_spacing": args.anchor_spacing, "kappa_anchor": args.kappa_anchor}
    base_outer_kwargs = {"n_trans1": 30, "n_rot": 20, "n_trans2": 20,
                          "lr_trans": 1e-3, "lr_rot": 1e-3, "warmup_steps": 5}
    inner_cfg, outer_cfg = noise_adaptive_inner_outer_cfg(
        sigma_t=args.sigma_t, base_inner_kwargs=base_inner_kwargs,
        base_outer_kwargs=base_outer_kwargs, reference_sigma_t=args.adaptive_reference_sigma_t,
    )
    exp_cfg = ExpCfg(window=args.window, overlap=args.overlap, sigma_t=args.sigma_t,
                      sigma_r=args.sigma_r, seeds=args.pool_seeds)

    print("Building denoiser (actual production build_denoiser, unmodified)...")
    denoiser_fn = build_denoiser(args.window, inner_cfg, outer_cfg, exp_cfg)
    lk, mr = exp_cfg.local_k, exp_cfg.max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision, local_k=lk, max_kappa_ratio=mr))

    print(f"Running denoise_sequence_pooled on {args.pool_seeds} pooled seeds "
          f"(target seed={args.seed}) ...")
    denoised_globals, denoised_rels = denoise_sequence_pooled(
        noisy_rels, gt_global, gt_R_mats, denoiser_fn, prec_pooled_fn, exp_cfg
    )
    denoised_global = denoised_globals[args.seed]
    noisy_global = integrate_poses(noisy_rels[args.seed])

    # Per-pose squared position error, denoised vs ground truth.
    err_denoised = np.sum((denoised_global[:, :3] - gt_global[:, :3]) ** 2, axis=-1)
    err_noisy = np.sum((noisy_global[:, :3] - gt_global[:, :3]) ** 2, axis=-1)

    windows = compute_windows(args.window, args.overlap, n_poses)
    print(f"\n{len(windows)} windows in this sequence (stitching boundaries):")
    for wi, w in enumerate(windows):
        print(f"  win {wi:2d}  local=[{w['lo']:4d}:{w['hi']:4d}]  "
              f"writes global poses [{w['global_write_lo']:4d}:{w['global_write_hi']:4d}]")

    # Find the largest single-step JUMPS in per-pose error (not just the
    # largest absolute error, which could be a slow drift -- a jump right at
    # a window's write boundary is the stitching-discontinuity signature).
    step_jump = np.abs(np.diff(err_denoised))
    top_idx = np.argsort(step_jump)[::-1][:args.top_k]

    print(f"\nTop {args.top_k} largest single-pose-step jumps in squared position error "
          f"(denoised vs GT):")
    print(f"{'pose':>6} {'err[i]':>12} {'err[i+1]':>12} {'jump':>12} {'nearest window boundary':>30}")
    for idx in sorted(top_idx):
        pose_i = int(idx)
        # Find which window boundary (if any) this pose sits at/near.
        nearest = None
        for w in windows:
            if abs(pose_i - w["global_write_lo"]) <= 1 or abs(pose_i - w["global_write_hi"]) <= 1:
                nearest = f"win boundary @ {w['global_write_lo']} or {w['global_write_hi']}"
                break
        nearest = nearest or "(mid-window, not a boundary)"
        print(f"{pose_i:6d} {err_denoised[pose_i]:12.4f} {err_denoised[pose_i+1]:12.4f} "
              f"{step_jump[pose_i]:12.4f}  {nearest:>30}")

    result = {
        "seq": args.seq, "seed": args.seed, "sigma_t": args.sigma_t, "n_poses": n_poses,
        "windows": windows,
        "err_denoised": err_denoised.tolist(),
        "err_noisy": err_noisy.tolist(),
        "top_jump_poses": [int(i) for i in sorted(top_idx)],
    }
    with open(os.path.join(out_dir, "stitching_error_trace.json"), "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"\nSaved full per-pose error trace and window boundaries to {out_dir}")

    final_dT_style = float(np.sqrt(np.mean(err_denoised)))
    print(f"\nFinal ATE-style position RMSE (denoised vs GT, whole trajectory): "
          f"{final_dT_style:.3f} m")


if __name__ == "__main__":
    main()
