"""
diag_exp44_multistart.py

Diagnostic for the finding that multi-start GN (InnerCfg.n_starts=2, see
exp44_anchored_sesync_gn._build_anchor_interpolated_R_init) does NOT fix
exp44_inner_solver_only's deterministic bad seeds: a real sweep on
seq=01/seed=9 and seq=13/seed=12 at sigma_t=0.01 came back essentially
IDENTICAL to the n_starts=1 baseline (dC within 0.03 of baseline for every
seed checked, not just the two targets) -- meaning the anchor-interpolated
candidate essentially never won.

Leading hypothesis: _rotation_cost sums a chain term (many edges, weighted
by kappa) plus an anchor term (1-2 sparse anchors, weighted by kappa_anchor).
The chain-composed R_init starts with the chain term at ~zero BY
CONSTRUCTION (it's literally built by composing R_meas), leaving only the
small anchor misfit to fix -- so it likely always starts (and stays, after
a fixed small iteration budget) at a lower total cost than the
anchor-interpolated candidate, regardless of which one is actually closer to
the truth geometrically. If true, the SELECTION criterion (final
_rotation_cost) is structurally biased toward the chain-composed candidate,
independent of whether the anchor-interpolated one would actually help.

A first pass (rotation-only) on real seq13/seed12 data confirmed a severe
disagreement: the anchor candidate wins by real rotation error in 23/37
windows but by cost_final in only 6/37 -- the cost proxy is structurally
biased toward the chain-composed candidate regardless of merit. BUT
seq01/seed9 and seq13/seed12's actual badness (dT=-89.5%/-33.7%) is almost
entirely a TRANSLATION collapse, not a rotation one (dR=+6.1%/+0.4%, near
baseline) -- so this script also reports each candidate's effect on the
TRANSLATION recovery (recover_translations), which is the metric that
actually drives dT/dC for these two seeds, not rotation accuracy alone.

This script isolates a single (seq, seed) pair and, for every window,
reports:
  - cost_final for BOTH candidates (chain-composed, anchor-interpolated)
    under the self-checking GN solve, and which one n_starts=2's real
    selection logic would pick (lower cost_final)
  - the REAL local geodesic rotation error AND translation RMSE (vs ground
    truth, not a cost proxy) for BOTH candidates' R_star/t_star
  - an ORACLE comparison for each: what if selection used real rotation
    error, or real translation RMSE, instead of _rotation_cost? Since
    translation is what actually drives these two seeds' badness, the
    cost/TRANS disagreement count matters more here than the cost/rotation
    one.

Run (on the machine with real KITTI data):
  python -m experiments.diag_exp44_multistart --kitti-root /path/to/kitti \
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
    ExpCfg,
    so3_exp,
    so3_log,
    rotation_gn_ift,
    recover_translations,
    _rotation_cost,
    _anchor_residual,
    _build_anchor_interpolated_R_init,
    _add_kitti_noise,
    _sigma_and_precision,
    noise_adaptive_inner_outer_cfg,
)
from experiments.diag_exp44_outer_regression import load_sequence


def per_window_multistart_diagnostics(win_odom, win_gt_R, win_gt_t, kappa, omega, n, inner_cfg):
    """Reimplements sesync_inner_solve's math (theta=0, matching
    exp44_inner_solver_only's usage), but returns BOTH candidates and their
    costs instead of collapsing to a single selected R_star.

    win_gt_t: (n, 3) absolute ground-truth translations for this window
    (world frame) -- needed to check whether a candidate's ROTATION choice
    actually matters for the TRANSLATION recovery, since seq01/seed9 and
    seq13/seed12's real badness (dT=-89.5%/-33.7%) is almost entirely a
    translation collapse, not a rotation one (dR=+6.1%/+0.4%, near baseline)
    -- comparing candidates by rotation error alone risks measuring a
    dimension that isn't what's actually broken for these seeds.
    """
    theta = jnp.zeros_like(win_odom)
    corrected = win_odom + theta
    w_meas = corrected[:, 3:]
    R_meas = jax.vmap(so3_exp)(w_meas)

    _, R_traj = jax.lax.scan(
        lambda R, Rm: (R @ Rm, R @ Rm), jnp.eye(3), R_meas
    )
    R_init_chain = jnp.concatenate([jnp.eye(3)[None], R_traj], axis=0)

    gt_R0 = win_gt_R[0]
    gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(win_gt_R)

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

    cost_init_chain = _rotation_cost(R_init_chain, R_meas, kappa, anchor_idx, anchor_targets, inner_cfg.kappa_anchor)
    cost_init_anchor = _rotation_cost(R_init_anchor, R_meas, kappa, anchor_idx, anchor_targets, inner_cfg.kappa_anchor)
    cost_final_chain = _rotation_cost(R_star_chain, R_meas, kappa, anchor_idx, anchor_targets, inner_cfg.kappa_anchor)
    cost_final_anchor = _rotation_cost(R_star_anchor, R_meas, kappa, anchor_idx, anchor_targets, inner_cfg.kappa_anchor)

    # Anchor-term-ONLY cost (no GT beyond what the anchors already legitimately
    # use, unlike the translation-RMSE oracle) -- tests whether ignoring the
    # chain term (which the anchor-interpolated candidate can't converge in a
    # fixed small iteration budget, biasing total cost toward chain-composed
    # regardless of merit) gives a selection signal that tracks real
    # translation error better than total _rotation_cost does.
    def anchor_only_cost(R_star):
        r_anchor = _anchor_residual(R_star, anchor_idx, anchor_targets)
        return inner_cfg.kappa_anchor * jnp.sum(r_anchor ** 2)

    cost_anchor_only_chain = anchor_only_cost(R_star_chain)
    cost_anchor_only_anchor = anchor_only_cost(R_star_anchor)

    def mean_geodesic_err_deg(R_est, R_gt):
        rel = jax.vmap(lambda Ra, Rb: Ra.T @ Rb)(R_est, R_gt)
        tr = jnp.einsum('nii->n', rel)
        ang = jnp.arccos(jnp.clip((tr - 1.0) / 2.0, -1.0, 1.0))
        return jnp.degrees(jnp.mean(ang))

    err_chain = mean_geodesic_err_deg(R_star_chain, gt_R_rel)
    err_anchor = mean_geodesic_err_deg(R_star_anchor, gt_R_rel)

    # Translation recovery, same as sesync_inner_solve, driven by each
    # candidate's R_star -- this is the metric that actually matters for
    # seq01/seed9 and seq13/seed12, whose badness is almost entirely a
    # translation collapse (see docstring above).
    t_meas = corrected[:, :3]
    gt_t0 = win_gt_t[0]
    gt_t_rel = jax.vmap(lambda t: gt_R0.T @ (t - gt_t0))(win_gt_t)

    t_star_chain = recover_translations(R_star_chain, t_meas, omega, n)
    t_star_anchor = recover_translations(R_star_anchor, t_meas, omega, n)

    def rmse_t(t_est, t_gt):
        return jnp.sqrt(jnp.mean(jnp.sum((t_est - t_gt) ** 2, axis=-1)))

    t_err_chain = rmse_t(t_star_chain, gt_t_rel)
    t_err_anchor = rmse_t(t_star_anchor, gt_t_rel)

    # Return raw jnp scalars, NOT python floats -- this function is called
    # through jax.jit below, and float() on a traced value inside a jitted
    # function raises ConcretizationTypeError. Callers convert to float
    # themselves once the jit call has returned concrete values.
    return {
        "cost_init_chain": cost_init_chain, "cost_init_anchor": cost_init_anchor,
        "cost_final_chain": cost_final_chain, "cost_final_anchor": cost_final_anchor,
        "cost_anchor_only_chain": cost_anchor_only_chain, "cost_anchor_only_anchor": cost_anchor_only_anchor,
        "real_err_deg_chain": err_chain, "real_err_deg_anchor": err_anchor,
        "t_rmse_chain": t_err_chain, "t_rmse_anchor": t_err_anchor,
    }


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
    ap.add_argument("--output-dir", default=os.path.expanduser("~/exp_res"))
    args = ap.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"diag_exp44_multistart_{run_ts}")
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
                          "anchor_spacing": args.anchor_spacing, "kappa_anchor": args.kappa_anchor,
                          "n_starts": 2}
    base_outer_kwargs = {"n_trans1": 0, "n_rot": 0, "n_trans2": 0,
                          "lr_trans": 0.0, "lr_rot": 0.0, "warmup_steps": 0}
    inner_cfg, _ = noise_adaptive_inner_outer_cfg(
        sigma_t=args.sigma_t, base_inner_kwargs=base_inner_kwargs,
        base_outer_kwargs=base_outer_kwargs, reference_sigma_t=args.adaptive_reference_sigma_t,
    )
    print(f"Inner cfg: n_iters_rot={inner_cfg.n_iters_rot} damping_up={inner_cfg.damping_up:.2f} "
          f"anchor_spacing={inner_cfg.anchor_spacing} kappa_anchor={inner_cfg.kappa_anchor:.2f}")

    n = args.window
    diag_fn = jax.jit(functools.partial(per_window_multistart_diagnostics, n=n, inner_cfg=inner_cfg))

    lk, mr = ExpCfg().local_k, ExpCfg().max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision, local_k=lk, max_kappa_ratio=mr))

    noisy_rels_j = jnp.array(noisy_rels)
    gt_R_mats_j = jnp.array(gt_R_mats)
    gt_t_j = jnp.array(gt_global[:, :3])

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
    print(f"{'win':>4} {'cost(chain/anchor)':>20} {'anchor_only(chain/anchor)':>26} "
          f"{'winner(cost)':>12} {'winner(anc-only)':>16} "
          f"{'t_rmse_m(chain/anchor)':>22} {'winner(trans)':>13}")

    results = []
    n_anchor_wins_cost = 0
    n_anchor_wins_rot = 0
    n_anchor_wins_trans = 0
    n_anchor_wins_anchoronly = 0
    n_disagree_rot = 0
    n_disagree_trans = 0
    n_anchoronly_disagree_trans = 0
    for wi, (lo, hi) in enumerate(windows):
        win_omds = noisy_rels_j[:, lo:hi - 1, :]
        win_gt_R = gt_R_mats_j[lo:hi]
        win_gt_t = gt_t_j[lo:hi]
        kappa = prec_pooled_fn(win_omds[:, :, 3:])
        omega = prec_pooled_fn(win_omds[:, :, :3])
        win_odom = win_omds[args.seed]

        r = {k: float(v) for k, v in diag_fn(win_odom, win_gt_R, win_gt_t, kappa, omega).items()}
        winner_cost = "anchor" if r["cost_final_anchor"] < r["cost_final_chain"] else "chain"
        winner_anchoronly = "anchor" if r["cost_anchor_only_anchor"] < r["cost_anchor_only_chain"] else "chain"
        winner_rot = "anchor" if r["real_err_deg_anchor"] < r["real_err_deg_chain"] else "chain"
        winner_trans = "anchor" if r["t_rmse_anchor"] < r["t_rmse_chain"] else "chain"
        if winner_cost == "anchor":
            n_anchor_wins_cost += 1
        if winner_anchoronly == "anchor":
            n_anchor_wins_anchoronly += 1
        if winner_rot == "anchor":
            n_anchor_wins_rot += 1
        if winner_trans == "anchor":
            n_anchor_wins_trans += 1
        if winner_cost != winner_rot:
            n_disagree_rot += 1
        if winner_cost != winner_trans:
            n_disagree_trans += 1
        if winner_anchoronly != winner_trans:
            n_anchoronly_disagree_trans += 1

        row = {"window": wi, "lo": lo, "hi": hi, **r,
               "winner_cost": winner_cost, "winner_anchoronly": winner_anchoronly,
               "winner_rot": winner_rot, "winner_trans": winner_trans}
        results.append(row)
        flag = ""
        if winner_anchoronly != winner_trans:
            flag = "  <-- anchor-only/TRANS DISAGREE"
        print(f"{wi:4d} {r['cost_final_chain']:>9.2f}/{r['cost_final_anchor']:<9.2f} "
              f"{r['cost_anchor_only_chain']:>12.3f}/{r['cost_anchor_only_anchor']:<12.3f} "
              f"{winner_cost:>12} {winner_anchoronly:>16} "
              f"{r['t_rmse_chain']:>10.3f}/{r['t_rmse_anchor']:<10.3f} "
              f"{winner_trans:>13}{flag}")

    with open(os.path.join(out_dir, "per_window_multistart.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n=== Summary over {len(windows)} windows (seq={args.seq}, seed={args.seed}, "
          f"sigma_t={args.sigma_t}) ===")
    print(f"  Anchor candidate wins by cost_final (what n_starts=2 actually selects): "
          f"{n_anchor_wins_cost}/{len(windows)}")
    print(f"  Anchor candidate wins by rotation geodesic error (oracle, not used by selection): "
          f"{n_anchor_wins_rot}/{len(windows)}")
    print(f"  Anchor candidate wins by TRANSLATION RMSE (oracle, the metric that actually drives "
          f"dT/dC for seq01/seed9 and seq13/seed12): {n_anchor_wins_trans}/{len(windows)}")
    print(f"  Windows where cost-based selection disagrees with the rotation-optimal pick: "
          f"{n_disagree_rot}/{len(windows)}")
    print(f"  Windows where cost-based selection disagrees with the TRANSLATION-optimal pick: "
          f"{n_disagree_trans}/{len(windows)}")
    print(f"  --- candidate selection criterion: ANCHOR-TERM-ONLY cost (no GT beyond what anchors "
          f"already use) ---")
    print(f"  Anchor candidate wins by anchor-only cost: {n_anchor_wins_anchoronly}/{len(windows)}")
    print(f"  Windows where anchor-only-cost selection disagrees with the TRANSLATION-optimal pick: "
          f"{n_anchoronly_disagree_trans}/{len(windows)}  "
          f"(lower than the {n_disagree_trans}/{len(windows)} total-cost disagreement above = "
          f"anchor-only cost is a better proxy for what actually matters)")
    print(f"\nSaved per-window results to {out_dir}")


if __name__ == "__main__":
    main()
