"""
diag_exp44_rotation_bias.py

Follow-up to diag_exp44_stitching.py: that script found seq01/seed9's
global position error grows completely SMOOTHLY (no discontinuities,
ruling out a window-stitching bug) from 0 at pose 0 to ~154m around pose
960, then smoothly decreases -- the classic signature of a small,
persistent ROTATION bias compounding across a long chain (dead-reckoning
drift), not a stitching artifact.

The hypothesis: the translation phase's loss_t (confirmed to improve
beautifully in EVERY window, diag_exp44_translation_phase.py) only
measures translation fit. t_star is recovered from a FIXED R_star via a
closed-form solve, so the translation phase can locally "absorb" a
rotation bias and still hit a low window-relative loss_t, without the
underlying rotation direction ever being corrected. That bias wouldn't
show up in any translation-based check -- but chained across ~1000 poses,
an uncorrected heading error compounds geometrically into exactly the
smooth drift observed.

This script checks rotation accuracy directly: for each window, computes
the per-pose rotation error vector (axis-angle residual between R_star and
ground truth, in the window-relative frame) across the window's WRITE
region (the poses that actually get stitched into the final trajectory).
Reports both:
  - mean error MAGNITUDE (degrees) -- how big the rotation error is
  - "bias ratio" = |mean error VECTOR| / mean(|per-pose error vectors|) --
    close to 0 for zero-mean random noise (individual errors cancel out),
    close to 1 for a consistent-direction systematic bias (they don't)
  - the mean bias vector's direction, to check whether it stays consistent
    ACROSS windows (via cosine similarity to the previous window's bias
    vector) -- a persistently-aligned direction across many windows is
    what would actually explain compounding global drift.

Run (on the machine with real KITTI data):
  python -m experiments.diag_exp44_rotation_bias --kitti-root /path/to/kitti \
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
    sesync_inner_solve,
    outer_adam_loop,
    _add_kitti_noise,
    _sigma_and_precision,
    noise_adaptive_inner_outer_cfg,
)
from experiments.diag_exp44_outer_regression import load_sequence
from experiments.diag_exp44_stitching import compute_windows


def per_window_rotation_bias(win_odom, win_gt, win_gt_R, kappa, omega, n, inner_cfg, outer_cfg):
    """Runs the ACTUAL production reconstruction (outer_adam_loop for
    theta_opt, then sesync_inner_solve(theta_opt, ...) for the final
    R_star -- exactly matching build_denoiser) and returns the per-pose
    rotation error vectors (axis-angle, window-relative frame).
    """
    theta_init = jnp.zeros_like(win_odom)
    gt_R0 = win_gt_R[0]
    gt_t0 = win_gt[0, :3]
    gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(win_gt_R)

    theta_opt = outer_adam_loop(theta_init, win_odom, win_gt, win_gt_R, kappa, omega, n, inner_cfg, outer_cfg)
    R_star, _ = sesync_inner_solve(theta_opt, win_odom, kappa, omega, n, inner_cfg, gt_R_rel)

    # Per-pose rotation error vector: so3_log(gt^T @ est) -- axis-angle
    # residual, window-relative frame, sign/direction preserved (NOT just
    # magnitude) so we can check for a consistent bias direction.
    err_vec = jax.vmap(lambda Rg, Re: so3_log(Rg.T @ Re))(gt_R_rel, R_star)
    return err_vec  # (n, 3)


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
    ap.add_argument("--output-dir", default=os.path.expanduser("~/exp_res"))
    args = ap.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"diag_exp44_rotation_bias_{run_ts}")
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
    print(f"Inner cfg: n_iters_rot={inner_cfg.n_iters_rot} damping_up={inner_cfg.damping_up:.2f}")
    print(f"Outer cfg: rot_loss_boost={outer_cfg.rot_loss_boost:.2f}")

    n = args.window
    bias_fn = jax.jit(functools.partial(per_window_rotation_bias, n=n,
                                         inner_cfg=inner_cfg, outer_cfg=outer_cfg))

    lk, mr = ExpCfg().local_k, ExpCfg().max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision, local_k=lk, max_kappa_ratio=mr))

    noisy_rels_j = jnp.array(noisy_rels)
    gt_global_j = jnp.array(gt_global)
    gt_R_mats_j = jnp.array(gt_R_mats)

    windows = compute_windows(args.window, args.overlap, n_poses)
    print(f"\n{len(windows)} windows, seq={args.seq} seed={args.seed} sigma_t={args.sigma_t} ...\n")
    print(f"{'win':>4} {'mean|err| (deg)':>16} {'bias_ratio':>11} {'mean_vec (axis-angle)':>28} "
          f"{'cos_sim(prev)':>14}")

    results = []
    prev_mean_vec = None
    for wi, w in enumerate(windows):
        lo, hi, write_lo, write_hi = w["lo"], w["hi"], w["write_lo"], w["write_hi"]
        win_omds = noisy_rels_j[:, lo:hi - 1, :]
        win_gt = gt_global_j[lo:hi]
        win_gt_R = gt_R_mats_j[lo:hi]
        kappa = prec_pooled_fn(win_omds[:, :, 3:])
        omega = prec_pooled_fn(win_omds[:, :, :3])
        win_odom = win_omds[args.seed]

        err_vec = np.array(bias_fn(win_odom, win_gt, win_gt_R, kappa, omega))
        # Restrict to the write region (+1 since err_vec is per-POSE, write
        # region is defined in EDGE indices -- poses write_lo..write_hi cover it).
        seg = err_vec[write_lo:write_hi + 1]

        mags = np.linalg.norm(seg, axis=-1)
        mean_mag_deg = float(np.degrees(np.mean(mags)))
        mean_vec = seg.mean(axis=0)
        mean_vec_mag = float(np.linalg.norm(mean_vec))
        bias_ratio = mean_vec_mag / (float(np.mean(mags)) + 1e-12)

        cos_sim = None
        if prev_mean_vec is not None:
            denom = (np.linalg.norm(mean_vec) * np.linalg.norm(prev_mean_vec)) + 1e-12
            cos_sim = float(np.dot(mean_vec, prev_mean_vec) / denom)
        prev_mean_vec = mean_vec

        results.append({"window": wi, "lo": lo, "hi": hi,
                         "mean_err_deg": mean_mag_deg, "bias_ratio": bias_ratio,
                         "mean_vec": mean_vec.tolist(), "cos_sim_prev": cos_sim})
        cos_str = f"{cos_sim:+.3f}" if cos_sim is not None else "   n/a"
        vec_str = f"[{mean_vec[0]:+.4f},{mean_vec[1]:+.4f},{mean_vec[2]:+.4f}]"
        print(f"{wi:4d} {mean_mag_deg:>16.3f} {bias_ratio:>11.3f} {vec_str:>28} {cos_str:>14}")

    with open(os.path.join(out_dir, "rotation_bias_trace.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    mean_bias_ratio = float(np.mean([r["bias_ratio"] for r in results]))
    cos_sims = [r["cos_sim_prev"] for r in results if r["cos_sim_prev"] is not None]
    mean_cos_sim = float(np.mean(cos_sims)) if cos_sims else float("nan")
    print(f"\n=== Summary over {len(windows)} windows ===")
    print(f"  Mean bias_ratio across windows: {mean_bias_ratio:.3f}  "
          f"(near 0 = random noise, near 1 = consistent systematic bias)")
    print(f"  Mean cosine similarity between consecutive windows' bias vectors: {mean_cos_sim:+.3f}  "
          f"(near 0 = unrelated directions, near +1 = persistently aligned direction across windows)")
    print(f"\nSaved per-window results to {out_dir}")


if __name__ == "__main__":
    main()
