"""
diag_exp44_multistart_oracle.py

Establishes the CEILING on what multi-start GN (InnerCfg.n_starts=2, see
exp44_anchored_sesync_gn._build_anchor_interpolated_R_init) could achieve,
before investing further in redesigning its selection criterion.

diag_exp44_multistart.py already showed the real n_starts=2 selection
(lower _rotation_cost) is structurally biased toward the chain-composed
candidate: on seq13/seed12 it picks the anchor candidate in only 6/37
windows by cost, but the anchor candidate actually has lower TRANSLATION
RMSE (the metric that drives dT/dC for this seed) in 13/37 windows, with
several large misses (e.g. window 35: chain=2.171m vs anchor=0.349m, a 6.2x
difference). seq01/seed9 shows a similar pattern concentrated in fewer,
earlier windows (window 0: chain=2.163m vs anchor=0.887m).

This script is NOT a proposed fix -- it builds an ORACLE denoiser that picks
between the two candidates using the REAL translation RMSE against dense
ground truth at every window, which is not available at inference (anchors
are meant to be sparse; this uses dense GT). It exists purely to answer:
if selection were "perfect" in this narrow sense, does the actual stitched
dT/dC for seq01/seed9 and seq13/seed12 flip positive? If yes, the multi-start
idea is worth pursuing with a legitimate (non-GT) selection criterion. If
the oracle still doesn't fix it, something else is going on and multi-start
GN isn't the answer for these two seeds.

Runs BOTH the baseline (n_starts=1 equivalent, build_inner_solver_only_denoiser)
and the oracle side by side on the same pooled-precision seed batch, so the
target seed's dT/dR/dC is directly comparable to the known baseline numbers
(exp44inner/INNER *.json).

Run (on the machine with real KITTI data):
  python -m experiments.diag_exp44_multistart_oracle --kitti-root /path/to/kitti \
      --seq 13 --seed 12 --sigma-t 0.01 --sigma-r 0.005 --output-dir ~/exp_res
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp44_anchored_sesync_gn import (
    InnerCfg,
    ExpCfg,
    so3_exp,
    so3_log,
    rotation_gn_ift,
    recover_translations,
    _build_anchor_interpolated_R_init,
    _add_kitti_noise,
    _sigma_and_precision,
    integrate_poses,
    integrate_rotations,
    delta_metric,
    denoise_sequence_pooled,
    noise_adaptive_inner_outer_cfg,
)
from experiments.exp44_inner_solver_only import build_inner_solver_only_denoiser
from experiments.diag_exp44_outer_regression import load_sequence


def build_oracle_multistart_denoiser(n: int, inner_cfg: InnerCfg):
    """ORACLE multi-start denoiser -- diagnostic only, NOT deployable: picks
    between the chain-composed and anchor-interpolated R_init candidates
    using REAL translation RMSE against dense ground truth, instead of the
    _rotation_cost proxy the real InnerCfg.n_starts=2 selection uses. Exists
    to establish the ceiling on multi-start GN's potential (see module
    docstring), not as a proposed fix.

    Same (noisy_odom, gt_poses, gt_R_world, kappa, omega) -> theta_opt
    signature as build_inner_solver_only_denoiser, so it drops into
    denoise_sequence_pooled unchanged.
    """
    def denoise(noisy_odom, gt_poses, gt_R_world, kappa, omega):
        theta = jnp.zeros_like(noisy_odom)
        corrected = noisy_odom + theta
        t_meas = corrected[:, :3]
        w_meas = corrected[:, 3:]
        R_meas = jax.vmap(so3_exp)(w_meas)

        gt_R0 = gt_R_world[0]
        gt_t0 = gt_poses[0, :3]
        gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(gt_R_world)
        gt_t_rel = jax.vmap(lambda t: gt_R0.T @ (t - gt_t0))(gt_poses[:, :3])

        _, R_traj = jax.lax.scan(
            lambda R, Rm: (R @ Rm, R @ Rm), jnp.eye(3), R_meas
        )
        R_init_chain = jnp.concatenate([jnp.eye(3)[None], R_traj], axis=0)

        anchor_idx = jnp.concatenate([
            jnp.arange(inner_cfg.anchor_spacing, n - 1, inner_cfg.anchor_spacing, dtype=jnp.int32),
            jnp.array([n - 1], dtype=jnp.int32),
        ])
        anchor_targets = gt_R_rel[anchor_idx]
        R_init_anchor = _build_anchor_interpolated_R_init(n, inner_cfg.anchor_spacing, anchor_targets)

        def solve(R_init):
            return rotation_gn_ift(R_init, R_meas, kappa, anchor_idx, anchor_targets, inner_cfg.kappa_anchor,
                                    inner_cfg.n_iters_rot,
                                    inner_cfg.damping_init, inner_cfg.damping_min, inner_cfg.damping_max,
                                    inner_cfg.damping_down, inner_cfg.damping_up)

        R_star_chain = solve(R_init_chain)
        R_star_anchor = solve(R_init_anchor)
        t_star_chain = recover_translations(R_star_chain, t_meas, omega, n)
        t_star_anchor = recover_translations(R_star_anchor, t_meas, omega, n)

        def rmse_t(t_est):
            return jnp.sqrt(jnp.mean(jnp.sum((t_est - gt_t_rel) ** 2, axis=-1)))

        use_anchor = rmse_t(t_star_anchor) < rmse_t(t_star_chain)
        R_star = jnp.where(use_anchor, R_star_anchor, R_star_chain)
        t_star = jnp.where(use_anchor, t_star_anchor, t_star_chain)

        dR = jax.vmap(lambda Ri, Rj: Ri.T @ Rj)(R_star[:-1], R_star[1:])
        dw = jax.vmap(so3_log)(dR)
        dt = jax.vmap(lambda Ri, ti, tj: Ri.T @ (tj - ti))(R_star[:-1], t_star[:-1], t_star[1:])
        corrected_out = jnp.concatenate([dt, dw], axis=-1)
        return corrected_out - noisy_odom

    return jax.jit(denoise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--seq", default="13")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--sigma-t", type=float, default=0.01)
    ap.add_argument("--sigma-r", type=float, default=0.005)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--pool-seeds", type=int, default=22,
                     help="how many seeds to pool precision over, matching the real sweep")
    ap.add_argument("--anchor-spacing", type=int, default=50)
    ap.add_argument("--kappa-anchor", type=float, default=100.0)
    ap.add_argument("--adaptive-reference-sigma-t", type=float, default=0.03)
    ap.add_argument("--output-dir", default=os.path.expanduser("~/exp_res"))
    args = ap.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"diag_exp44_multistart_oracle_{run_ts}")
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
    base_outer_kwargs = {"n_trans1": 0, "n_rot": 0, "n_trans2": 0,
                          "lr_trans": 0.0, "lr_rot": 0.0, "warmup_steps": 0}
    inner_cfg, _ = noise_adaptive_inner_outer_cfg(
        sigma_t=args.sigma_t, base_inner_kwargs=base_inner_kwargs,
        base_outer_kwargs=base_outer_kwargs, reference_sigma_t=args.adaptive_reference_sigma_t,
    )
    print(f"Inner cfg: n_iters_rot={inner_cfg.n_iters_rot} damping_up={inner_cfg.damping_up:.2f} "
          f"anchor_spacing={inner_cfg.anchor_spacing} kappa_anchor={inner_cfg.kappa_anchor:.2f}")

    n = args.window
    baseline_fn = build_inner_solver_only_denoiser(n, inner_cfg)
    oracle_fn = build_oracle_multistart_denoiser(n, inner_cfg)

    lk, mr = ExpCfg().local_k, ExpCfg().max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision, local_k=lk, max_kappa_ratio=mr))
    exp_cfg = ExpCfg(window=args.window, overlap=args.overlap, sigma_t=args.sigma_t,
                      sigma_r=args.sigma_r, seeds=args.pool_seeds)

    print(f"\nRunning baseline (n_starts=1 equivalent) ...")
    t0 = time.time()
    baseline_globals, baseline_rels = denoise_sequence_pooled(
        noisy_rels, gt_global, gt_R_mats, baseline_fn, prec_pooled_fn, exp_cfg
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"Running ORACLE multi-start (picks by real translation RMSE, not deployable) ...")
    t0 = time.time()
    oracle_globals, oracle_rels = denoise_sequence_pooled(
        noisy_rels, gt_global, gt_R_mats, oracle_fn, prec_pooled_fn, exp_cfg
    )
    print(f"  done in {time.time() - t0:.1f}s")

    gt_R_int = integrate_rotations(gt_rel)
    results = []
    for s in range(args.pool_seeds):
        noisy_global_s = integrate_poses(noisy_rels[s])
        noisy_R_s = integrate_rotations(noisy_rels[s])

        baseline_R_s = integrate_rotations(baseline_rels[s])
        dT_b, dR_b, dC_b = delta_metric(noisy_global_s, baseline_globals[s], gt_global,
                                         noisy_R_s, baseline_R_s, gt_R_int)

        oracle_R_s = integrate_rotations(oracle_rels[s])
        dT_o, dR_o, dC_o = delta_metric(noisy_global_s, oracle_globals[s], gt_global,
                                         noisy_R_s, oracle_R_s, gt_R_int)

        marker = "  <-- TARGET SEED" if s == args.seed else ""
        print(f"  [seed={s:2d}]  baseline: dT={dT_b:+7.1f} dR={dR_b:+7.1f} dC={dC_b:+7.1f}  |  "
              f"oracle: dT={dT_o:+7.1f} dR={dR_o:+7.1f} dC={dC_o:+7.1f}{marker}")
        results.append({"seed": s, "dT_baseline": dT_b, "dR_baseline": dR_b, "dC_baseline": dC_b,
                         "dT_oracle": dT_o, "dR_oracle": dR_o, "dC_oracle": dC_o})

    target = results[args.seed]
    print(f"\n=== TARGET seq={args.seq} seed={args.seed} sigma_t={args.sigma_t} ===")
    print(f"  Baseline (n_starts=1 equivalent):  dT={target['dT_baseline']:+.1f}%  "
          f"dR={target['dR_baseline']:+.1f}%  dC={target['dC_baseline']:+.1f}%")
    print(f"  Oracle multi-start (ceiling):       dT={target['dT_oracle']:+.1f}%  "
          f"dR={target['dR_oracle']:+.1f}%  dC={target['dC_oracle']:+.1f}%")
    flipped = target['dC_baseline'] < 0 and target['dC_oracle'] > 0
    print(f"  Flips from negative to positive dC: {flipped}")

    with open(os.path.join(out_dir, "oracle_vs_baseline.json"), "w") as fp:
        json.dump({"seq": args.seq, "target_seed": args.seed, "sigma_t": args.sigma_t,
                    "results": results}, fp, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
