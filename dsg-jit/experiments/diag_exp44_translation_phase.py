"""
diag_exp44_translation_phase.py

Root-causes the translation-phase regression flagged in
project_exp44_outer_loop_regression memory but never actually resolved:
a PER-EDGE local metric showed translation getting worse under the outer
loop in nearly every window, but that metric is NOT the quantity the outer
loop actually optimizes (loss_t is a windowed, CUMULATIVE ATE-style
position error over the whole window, not a per-edge difference) -- so
the earlier finding might just reflect that cumulative losses don't
require per-edge fidelity, not a genuine optimization failure.

Confirmed today (2026-07-17) on real full-pipeline data: seq01/seed9's dT
gets WORSE, not better, when rot_loss_boost is increased (-195.8% ->
-287.9%), even though theta_trans has zero gradient contribution from
loss_r (rotation phases mask the gradient to rotation-only components) --
ruling out rot_loss_boost as the driver and pointing back at this
unresolved translation-phase issue as the likely root cause of seed9 (and
now seed2/3/10)'s badness.

This script directly checks, per window, whether the CUMULATIVE position
metric the outer loop actually optimizes (loss_t: mean squared error of
t_star vs ground truth, over the whole window) genuinely improves from
theta=0 to the final theta_opt -- not a proxy, the literal training
objective. If loss_t gets WORSE in some windows, that's a real, direct
optimization failure (not a metric-mismatch artifact) and points at a bug
in the translation phase itself. If loss_t always improves, the seed's bad
aggregate dT must come from window-stitching / cross-window effects
instead, not from any single window's own optimization.

Run (on the machine with real KITTI data):
  python -m experiments.diag_exp44_translation_phase --kitti-root /path/to/kitti \
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
    InnerCfg,
    OuterCfg,
    ExpCfg,
    so3_exp,
    so3_log,
    sesync_inner_solve,
    _add_kitti_noise,
    _sigma_and_precision,
    noise_adaptive_inner_outer_cfg,
)
from experiments.diag_exp44_outer_regression import instrumented_outer_adam_loop, load_sequence


def per_window_translation_check(win_odom, win_gt, win_gt_R, kappa, omega, n, inner_cfg, outer_cfg):
    """Runs the ACTUAL outer_adam_loop (unmodified optimization) and directly
    compares loss_t (the literal cumulative position objective it optimizes)
    at theta=0 vs at the final theta_opt -- not a proxy metric.
    """
    theta_init = jnp.zeros_like(win_odom)
    theta_opt, trace = instrumented_outer_adam_loop(
        theta_init, win_odom, win_gt, win_gt_R, kappa, omega, n, inner_cfg, outer_cfg
    )
    loss_t_trace, loss_r_trace, _, _, _ = trace
    loss_t_init = loss_t_trace[0]  # recorded BEFORE the first Adam step -- theta still ~0 here

    # Recompute the FINAL loss cleanly at theta_opt (trace's last entry is
    # recorded before the LAST step, i.e. one step behind theta_opt).
    gt_R0 = win_gt_R[0]
    gt_t0 = win_gt[0, :3]
    gt_t_rel = jax.vmap(lambda t: gt_R0.T @ (t - gt_t0))(win_gt[:, :3])
    gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(win_gt_R)
    _, t_star_final = sesync_inner_solve(theta_opt, win_odom, kappa, omega, n, inner_cfg, gt_R_rel)
    loss_t_final = jnp.mean(jnp.sum((t_star_final - gt_t_rel) ** 2, axis=-1))

    return loss_t_init, loss_t_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--seq", default="01")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--sigma-t", type=float, default=0.01)
    ap.add_argument("--sigma-r", type=float, default=0.005)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--pool-seeds", type=int, default=22,
                     help="how many seeds to pool precision over, matching the real sweep")
    ap.add_argument("--anchor-spacing", type=int, default=50)
    ap.add_argument("--kappa-anchor", type=float, default=100.0)
    ap.add_argument("--adaptive-reference-sigma-t", type=float, default=0.03)
    ap.add_argument("--rot-loss-boost-override", type=float, default=None)
    ap.add_argument("--output-dir", default=os.path.expanduser("~/exp_res"))
    args = ap.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"diag_exp44_translation_phase_{run_ts}")
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
    if args.rot_loss_boost_override is not None:
        from dataclasses import replace
        outer_cfg = replace(outer_cfg, rot_loss_boost=args.rot_loss_boost_override)
    print(f"Inner cfg: n_iters_rot={inner_cfg.n_iters_rot} damping_up={inner_cfg.damping_up:.2f}")
    print(f"Outer cfg: rot_loss_boost={outer_cfg.rot_loss_boost:.2f} "
          f"n_trans1={outer_cfg.n_trans1} n_rot={outer_cfg.n_rot} n_trans2={outer_cfg.n_trans2}")

    n = args.window
    check_fn = jax.jit(functools.partial(per_window_translation_check, n=n,
                                          inner_cfg=inner_cfg, outer_cfg=outer_cfg))

    lk, mr = ExpCfg().local_k, ExpCfg().max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision, local_k=lk, max_kappa_ratio=mr))

    noisy_rels_j = jnp.array(noisy_rels)
    gt_global_j = jnp.array(gt_global)
    gt_R_mats_j = jnp.array(gt_R_mats)

    stride = args.window - args.overlap
    windows = []
    pos = 0
    while pos + args.window <= n_poses - 1:
        windows.append((pos, pos + args.window))
        pos += stride
    if not windows or windows[-1][1] < n_poses:
        hi = n_poses
        lo = max(0, hi - args.window)
        windows.append((lo, hi))

    print(f"\n{len(windows)} windows, seq={args.seq} seed={args.seed} sigma_t={args.sigma_t} ...\n")
    print(f"{'win':>4} {'loss_t_init':>14} {'loss_t_final':>14} {'ratio(final/init)':>18} {'result':>10}")

    results = []
    n_improved = 0
    n_worsened = 0
    for wi, (lo, hi) in enumerate(windows):
        win_omds = noisy_rels_j[:, lo:hi - 1, :]
        win_gt = gt_global_j[lo:hi]
        win_gt_R = gt_R_mats_j[lo:hi]
        kappa = prec_pooled_fn(win_omds[:, :, 3:])
        omega = prec_pooled_fn(win_omds[:, :, :3])
        win_odom = win_omds[args.seed]

        loss_t_init, loss_t_final = check_fn(win_odom, win_gt, win_gt_R, kappa, omega)
        loss_t_init, loss_t_final = float(loss_t_init), float(loss_t_final)
        ratio = loss_t_final / (loss_t_init + 1e-12)
        improved = loss_t_final < loss_t_init
        if improved:
            n_improved += 1
        else:
            n_worsened += 1

        results.append({"window": wi, "lo": lo, "hi": hi,
                         "loss_t_init": loss_t_init, "loss_t_final": loss_t_final,
                         "ratio": ratio, "improved": improved})
        flag = "" if improved else "  <-- WORSENED"
        print(f"{wi:4d} {loss_t_init:>14.4f} {loss_t_final:>14.4f} {ratio:>18.3f} "
              f"{'improved' if improved else 'WORSE':>10}{flag}")

    with open(os.path.join(out_dir, "per_window_translation_phase.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n=== Summary over {len(windows)} windows (seq={args.seq}, seed={args.seed}, "
          f"sigma_t={args.sigma_t}) ===")
    print(f"  Windows where the ACTUAL cumulative loss_t improved: {n_improved}/{len(windows)}")
    print(f"  Windows where the ACTUAL cumulative loss_t got WORSE: {n_worsened}/{len(windows)}")
    if n_worsened > 0:
        print(f"  -> Real optimization failure confirmed in {n_worsened} window(s): the outer loop "
              f"is making its OWN objective worse, not just a per-edge proxy. This is a genuine bug "
              f"in the translation phase, not a metric-mismatch artifact.")
    else:
        print(f"  -> loss_t never gets worse per-window -- if this seed's aggregate dT is still bad, "
              f"the cause is likely window-stitching / cross-window discontinuities, not any single "
              f"window's own optimization.")
    print(f"\nSaved per-window results to {out_dir}")


if __name__ == "__main__":
    main()
