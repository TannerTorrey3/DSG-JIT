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

This script isolates a single (seq, seed) pair and, for every window,
reports:
  - cost_init / cost_final for BOTH candidates (chain-composed, anchor-
    interpolated) under the self-checking GN solve
  - which candidate the real n_starts=2 selection logic would pick (lower
    cost_final)
  - the REAL local geodesic rotation error (vs ground truth, not a cost
    proxy) for BOTH candidates' R_star
  - an ORACLE comparison: what if selection used real error instead of
    _rotation_cost? If the anchor candidate has lower real error more often
    than it has lower cost, the candidate design might still be useful --
    the SELECTION criterion would need fixing, not the candidate itself.

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
    _rotation_cost,
    _build_anchor_interpolated_R_init,
    _add_kitti_noise,
    _sigma_and_precision,
    noise_adaptive_inner_outer_cfg,
)
from experiments.diag_exp44_outer_regression import load_sequence


def per_window_multistart_diagnostics(win_odom, win_gt_R, kappa, omega, n, inner_cfg):
    """Reimplements sesync_inner_solve's math (theta=0, matching
    exp44_inner_solver_only's usage), but returns BOTH candidates and their
    costs instead of collapsing to a single selected R_star.
    """
    del omega  # translation recovery isn't needed for this diagnostic
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

    def mean_geodesic_err_deg(R_est, R_gt):
        rel = jax.vmap(lambda Ra, Rb: Ra.T @ Rb)(R_est, R_gt)
        tr = jnp.einsum('nii->n', rel)
        ang = jnp.arccos(jnp.clip((tr - 1.0) / 2.0, -1.0, 1.0))
        return jnp.degrees(jnp.mean(ang))

    err_chain = mean_geodesic_err_deg(R_star_chain, gt_R_rel)
    err_anchor = mean_geodesic_err_deg(R_star_anchor, gt_R_rel)

    return {
        "cost_init_chain": float(cost_init_chain), "cost_init_anchor": float(cost_init_anchor),
        "cost_final_chain": float(cost_final_chain), "cost_final_anchor": float(cost_final_anchor),
        "real_err_deg_chain": float(err_chain), "real_err_deg_anchor": float(err_anchor),
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
    print(f"{'win':>4} {'cost_init(chain/anchor)':>26} {'cost_final(chain/anchor)':>26} "
          f"{'winner(cost)':>12} {'real_err_deg(chain/anchor)':>28} {'winner(real)':>12}")

    results = []
    n_anchor_wins_cost = 0
    n_anchor_wins_real = 0
    n_disagree = 0
    for wi, (lo, hi) in enumerate(windows):
        win_omds = noisy_rels_j[:, lo:hi - 1, :]
        win_gt_R = gt_R_mats_j[lo:hi]
        kappa = prec_pooled_fn(win_omds[:, :, 3:])
        omega = prec_pooled_fn(win_omds[:, :, :3])
        win_odom = win_omds[args.seed]

        r = diag_fn(win_odom, win_gt_R, kappa, omega)
        winner_cost = "anchor" if r["cost_final_anchor"] < r["cost_final_chain"] else "chain"
        winner_real = "anchor" if r["real_err_deg_anchor"] < r["real_err_deg_chain"] else "chain"
        if winner_cost == "anchor":
            n_anchor_wins_cost += 1
        if winner_real == "anchor":
            n_anchor_wins_real += 1
        if winner_cost != winner_real:
            n_disagree += 1

        row = {"window": wi, "lo": lo, "hi": hi, **r,
               "winner_cost": winner_cost, "winner_real": winner_real}
        results.append(row)
        flag = "  <-- cost/real DISAGREE" if winner_cost != winner_real else ""
        print(f"{wi:4d} {r['cost_init_chain']:>11.2f}/{r['cost_init_anchor']:<11.2f} "
              f"{r['cost_final_chain']:>11.2f}/{r['cost_final_anchor']:<11.2f} "
              f"{winner_cost:>12} {r['real_err_deg_chain']:>12.2f}/{r['real_err_deg_anchor']:<12.2f} "
              f"{winner_real:>12}{flag}")

    with open(os.path.join(out_dir, "per_window_multistart.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n=== Summary over {len(windows)} windows (seq={args.seq}, seed={args.seed}, "
          f"sigma_t={args.sigma_t}) ===")
    print(f"  Anchor candidate wins by cost_final (what n_starts=2 actually selects): "
          f"{n_anchor_wins_cost}/{len(windows)}")
    print(f"  Anchor candidate wins by REAL geodesic error (oracle, not used by selection): "
          f"{n_anchor_wins_real}/{len(windows)}")
    print(f"  Windows where cost-based selection disagrees with the real-error-optimal pick: "
          f"{n_disagree}/{len(windows)}")
    print(f"\nSaved per-window results to {out_dir}")


if __name__ == "__main__":
    main()
