"""
diag_exp44_loss_mode_sweep.py

Sweeps OuterCfg's loss_mode="anchor_only" weights (anchor_trans_weight,
fidelity_weight, smoothness_weight) against real KITTI sequences.

Why this exists: a manual anchor_spacing=50 comparison on seq 01,13
(sigma_t=0.03, 3 seeds) found the untuned anchor_only defaults
(anchor_trans_weight=100, fidelity_weight=1.0, smoothness_weight=1.0)
scoring mean ΔC=+11.4% with 2/6 negative seeds -- WORSE than simply running
no outer loop at all (exp44_inner_solver_only: +32.1%, 0/6 negative) at the
same anchor_spacing, and far below loss_mode="dense_gt"'s oracle/ceiling
+60.9%. This sweeps the three new weights to find a combination that at
least beats the plain-anchors baseline, before concluding whether a GT-free
outer loop can add value at all. See OuterCfg.loss_mode's docstring in
exp44_anchored_sesync_gn.py for the full anchor_only design.

Run:
  python -m experiments.diag_exp44_loss_mode_sweep \
    --kitti-root /path/to/kitti/sequences --seqs 01,13 --seeds 3 \
    --sigma-t 0.03 --anchor-spacing 50 --output-dir ~/exp_res

Compare the winning row's mean_dC/n_negative against a separate
exp44_inner_solver_only run at the same --anchor-spacing/--sigma-t to see
whether ANY weight combination actually beats the plain-anchors baseline.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.exp44_anchored_sesync_gn import (
    InnerCfg, OuterCfg, ExpCfg,
    build_denoiser, denoise_sequence_pooled, _add_kitti_noise,
    relative_poses_from_mats, delta_metric, integrate_poses, integrate_rotations,
    _sigma_and_precision, so3_log,
)


def load_kitti_sequence(kitti_root: str, seq_id: str, max_poses):
    root_p = Path(kitti_root)
    seq_str = f"{int(seq_id):02d}"
    candidates = [
        root_p / seq_str / "poses.txt",
        root_p / "sequences" / seq_str / "poses.txt",
        root_p / "poses" / f"{seq_str}.txt",
    ]
    poses_path = next((p for p in candidates if p.exists()), None)
    if poses_path is None:
        raise FileNotFoundError(f"poses.txt not found for seq {seq_id} (tried {candidates})")

    raw_mats = []
    with poses_path.open() as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float32)
            T[:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
            raw_mats.append(T)
    if not raw_mats:
        raise ValueError(f"Empty poses.txt for seq {seq_id}")

    gt_mats = np.stack(raw_mats, axis=0)
    if max_poses is not None:
        gt_mats = gt_mats[:max_poses]
    gt_global = np.zeros((len(gt_mats), 6), dtype=np.float32)
    gt_global[:, :3] = gt_mats[:, :3, 3]
    gt_global[:, 3:] = np.array(
        jax.vmap(so3_log)(jnp.array(gt_mats[:, :3, :3], dtype=jnp.float32))
    )
    gt_R_mats = gt_mats[:, :3, :3]
    gt_rel = relative_poses_from_mats(gt_mats)
    return gt_global, gt_R_mats, gt_rel


def evaluate_config(seq_data: dict, inner_cfg: InnerCfg, outer_cfg: OuterCfg,
                     exp_cfg: ExpCfg, window: int) -> dict:
    """seq_data: seq_id -> (gt_global, gt_R_mats, gt_rel, noisy_rels)."""
    denoiser_fn = build_denoiser(window, inner_cfg, outer_cfg, exp_cfg)
    prec_pooled_fn = jax.jit(functools.partial(
        _sigma_and_precision, local_k=exp_cfg.local_k, max_kappa_ratio=exp_cfg.max_kappa_ratio
    ))

    # Warm up both jitted functions once per config (shapes only, no retrace after).
    S = next(iter(seq_data.values()))[3].shape[0]
    dummy_odom = jnp.zeros((window - 1, 6))
    dummy_gt = jnp.zeros((window, 6))
    dummy_gR = jnp.zeros((window, 3, 3))
    dummy_k = jnp.ones(window - 1)
    dummy_w = jnp.ones(window - 1)
    _ = denoiser_fn(dummy_odom, dummy_gt, dummy_gR, dummy_k, dummy_w).block_until_ready()
    dummy_omds = jnp.zeros((S, window - 1, 6))
    _ = prec_pooled_fn(dummy_omds[:, :, :3]).block_until_ready()

    all_dC = []
    per_seq = {}
    for seq_id, (gt_global, gt_R_mats, gt_rel, noisy_rels) in seq_data.items():
        denoised_globals, denoised_rels = denoise_sequence_pooled(
            noisy_rels, gt_global, gt_R_mats, denoiser_fn, prec_pooled_fn, exp_cfg
        )
        seq_dC = []
        for s in range(noisy_rels.shape[0]):
            noisy_global_s = integrate_poses(noisy_rels[s])
            noisy_R_s = integrate_rotations(noisy_rels[s])
            denoised_R_s = integrate_rotations(denoised_rels[s])
            _, _, dC = delta_metric(noisy_global_s, denoised_globals[s], gt_global,
                                     noisy_R_s, denoised_R_s, gt_R_mats)
            seq_dC.append(float(dC))
        all_dC.extend(seq_dC)
        per_seq[seq_id] = {
            "mean_dC": float(np.mean(seq_dC)),
            "seed_dC": seq_dC,
            "n_negative": int(sum(1 for d in seq_dC if d < 0)),
        }

    return {
        "mean_dC": float(np.mean(all_dC)),
        "n_negative": int(sum(1 for d in all_dC if d < 0)),
        "per_seq": per_seq,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep exp44's loss_mode=anchor_only weights against real KITTI data"
    )
    parser.add_argument("--kitti-root", type=str, required=True)
    parser.add_argument("--seqs", type=str, required=True, help="Comma-separated seq IDs")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sigma-t", type=float, default=0.03)
    parser.add_argument("--sigma-r", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--anchor-spacing", type=int, default=50)
    parser.add_argument("--kappa-anchor", type=float, default=100.0)
    parser.add_argument("--max-poses", type=int, default=None)
    parser.add_argument("--anchor-trans-weights", type=str, default="50,100,200",
                        help="Comma-separated grid values for OuterCfg.anchor_trans_weight")
    parser.add_argument("--fidelity-weights", type=str, default="0.01,0.1,1.0",
                        help="Comma-separated grid values for OuterCfg.fidelity_weight")
    parser.add_argument("--smoothness-weights", type=str, default="0.01,0.1,1.0",
                        help="Comma-separated grid values for OuterCfg.smoothness_weight")
    parser.add_argument("--output-dir", type=str, default=os.path.expanduser("~/exp_res"))
    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp44_loss_mode_sweep_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    atw = [float(x) for x in args.anchor_trans_weights.split(",")]
    fw  = [float(x) for x in args.fidelity_weights.split(",")]
    sw  = [float(x) for x in args.smoothness_weights.split(",")]
    grid = list(itertools.product(atw, fw, sw))
    print(f"Sweeping {len(grid)} combinations: "
          f"anchor_trans_weight in {atw}, fidelity_weight in {fw}, smoothness_weight in {sw}")

    seq_ids = [s.strip() for s in args.seqs.split(",")]
    exp_cfg = ExpCfg(window=args.window, overlap=args.overlap,
                      sigma_t=args.sigma_t, sigma_r=args.sigma_r, seeds=args.seeds)

    print("Loading KITTI sequences and generating noise (shared across the whole sweep)...")
    seq_data = {}
    for seq_id in seq_ids:
        gt_global, gt_R_mats, gt_rel = load_kitti_sequence(args.kitti_root, seq_id, args.max_poses)
        seq_hash = int(seq_id) if seq_id.isdigit() else 0
        noisy_rels = np.stack([
            _add_kitti_noise(gt_rel, seed, seq_hash, args.sigma_t, args.sigma_r)
            for seed in range(args.seeds)
        ])
        seq_data[seq_id] = (gt_global, gt_R_mats, gt_rel, noisy_rels)
        print(f"  loaded seq {seq_id}: {len(gt_global)} poses")

    inner_cfg = InnerCfg(
        n_iters_rot=15, damping_init=1e-4, damping_min=1e-6, damping_max=1e2,
        damping_down=0.5, damping_up=4.0,
        anchor_spacing=args.anchor_spacing, kappa_anchor=args.kappa_anchor,
        n_starts=1, multistart_criterion="anchor_only", anchor_n_iters_rot=60,
    )

    results = []
    for i, (a_w, f_w, s_w) in enumerate(grid):
        outer_cfg = OuterCfg(
            n_trans1=30, n_rot=20, n_trans2=20, lr_trans=1e-3, lr_rot=1e-3,
            warmup_steps=5, rot_loss_boost=1.0,
            loss_mode="anchor_only",
            anchor_trans_weight=a_w, fidelity_weight=f_w, smoothness_weight=s_w,
        )
        t0 = time.time()
        result = evaluate_config(seq_data, inner_cfg, outer_cfg, exp_cfg, args.window)
        elapsed = time.time() - t0
        result.update({"anchor_trans_weight": a_w, "fidelity_weight": f_w,
                       "smoothness_weight": s_w, "elapsed_s": elapsed})
        results.append(result)
        print(f"[{i + 1}/{len(grid)}] atw={a_w} fw={f_w} sw={s_w}  "
              f"mean_dC={result['mean_dC']:+.1f}%  n_negative={result['n_negative']}  "
              f"({elapsed:.1f}s)")

    results.sort(key=lambda r: (-r["mean_dC"]))
    print("\n=== Top 10 by mean ΔC ===")
    for r in results[:10]:
        print(f"  atw={r['anchor_trans_weight']} fw={r['fidelity_weight']} "
              f"sw={r['smoothness_weight']}  mean_dC={r['mean_dC']:+.1f}%  "
              f"n_negative={r['n_negative']}")

    out_path = os.path.join(run_dir, "sweep_results.json")
    with open(out_path, "w") as fp:
        json.dump({"config": vars(args), "results": results}, fp, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
