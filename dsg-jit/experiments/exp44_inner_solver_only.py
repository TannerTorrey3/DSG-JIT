"""
exp44_inner_solver_only.py

Experiment 44, inner-solver-only variant: isolates exp44's anchored SE-Sync
rotation GN solve from the outer Adam loop that learns per-edge corrections
(theta).

Why this exists: exp44 (exp44_anchored_sesync_gn.py) reintroduced sparse GT
rotation anchors into the inner solve so it's no longer mathematically inert
on this chain-graph topology (see that file's module docstring for the full
root-cause background). But exp44's reported ΔT/ΔR/ΔC numbers reflect the
COMBINED effect of (a) the anchor-augmented inner solve and (b) ~50-70 outer
Adam steps optimizing theta against that inner solve's IFT-shortcut gradient.
This script strips out (b) entirely: theta is fixed at zero, and the
"denoised" trajectory is read directly off a SINGLE call to the same
sesync_inner_solve used inside exp44 -- no gradient descent, no outer loop,
one dispatch per window per seed instead of ~50-70.

This answers a different question than exp44 does: how much of the
denoising comes from the anchor-constrained self-consistency projection
alone, vs. how much needs the outer loop's learned theta on top of it? A
result close to exp44's full numbers would say the anchors are already doing
most of the work; a result far below would say the outer loop's theta
optimization is still carrying most of the correction even with anchors
present.

Everything else (data loading, noise injection, precision pooling, window
stitching, ATE/ARE/ΔC metrics) is imported unchanged from
exp44_anchored_sesync_gn.py -- this file only replaces build_denoiser's role.

Run:
  python -m experiments.exp44_inner_solver_only --kitti-root /path/to/kitti --sigma-t 0.03
  python -m experiments.exp44_inner_solver_only --synthetic  # no KITTI needed
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
    sesync_inner_solve,
    denoise_sequence,
    denoise_sequence_pooled,
    relative_poses_from_mats,
    _add_kitti_noise,
    _sigma_and_precision,
    integrate_poses,
    integrate_rotations,
    delta_metric,
    make_synthetic_sequence,
    noise_adaptive_inner_outer_cfg,
)


def build_inner_solver_only_denoiser(n: int, inner_cfg: InnerCfg):
    """One-shot anchored SE-Sync inner solve, theta fixed at zero -- no outer
    Adam loop.

    Returns a denoiser_fn with the same (win_odom, win_gt, win_gt_R, kappa,
    omega) -> theta_opt signature build_denoiser's output has, so it drops
    into denoise_sequence/denoise_sequence_pooled unchanged. Internally:
    theta=0 is fed to sesync_inner_solve, and the resulting window-relative
    absolute poses (R_star, t_star) are converted back into the (n-1, 6)
    relative-pose format expected by the harness, then expressed as a "theta"
    correction (corrected = noisy_odom + theta_opt) purely so the write-back
    arithmetic in denoise_sequence[_pooled] needs no changes.
    """
    def denoise(noisy_odom, gt_poses, gt_R_world, kappa, omega):
        theta = jnp.zeros_like(noisy_odom)

        # Window-relative GT rotations AND translations, anchored at pose 0 --
        # same construction outer_adam_loop uses to build gt_R_rel/gt_t_rel for
        # anchor targets. Previously gt_poses was discarded here (translation
        # had no anchor mechanism at all in the inner solve); now both rotation
        # and translation get the same sparse-GT-anchor treatment.
        gt_R0 = gt_R_world[0]
        gt_t0 = gt_poses[0, :3]
        gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(gt_R_world)
        gt_t_rel = jax.vmap(lambda t: gt_R0.T @ (t - gt_t0))(gt_poses[:, :3])

        R_star, t_star = sesync_inner_solve(
            theta, noisy_odom, kappa, omega, n, inner_cfg, gt_R_rel, gt_t_rel
        )

        # R_star/t_star are absolute window-relative poses (R_star[0]=I,
        # t_star[0]=0), matching gt_R_rel's convention -- convert back to
        # per-edge relative poses in pose i's local frame, matching
        # relative_poses_from_mats' convention (dt = Ri^T(tj-ti),
        # dw = so3_log(Ri^T Rj)).
        dR = jax.vmap(lambda Ri, Rj: Ri.T @ Rj)(R_star[:-1], R_star[1:])
        dw = jax.vmap(so3_log)(dR)
        dt = jax.vmap(lambda Ri, ti, tj: Ri.T @ (tj - ti))(
            R_star[:-1], t_star[:-1], t_star[1:]
        )
        corrected = jnp.concatenate([dt, dw], axis=-1)  # (n-1, 6)
        return corrected - noisy_odom

    return jax.jit(denoise)


def main():
    parser = argparse.ArgumentParser(
        description="Exp44 inner-solver-only: anchored SE-Sync GN alone, no outer Adam loop"
    )
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--seqs",       type=str, default=None,
                        help="Comma-separated sequence IDs (default: all sequences found in kitti-root)")
    parser.add_argument("--sigma-t",    type=float, default=0.03)
    parser.add_argument("--sigma-r",    type=float, default=0.01)
    parser.add_argument("--window",     type=int,   default=50)
    parser.add_argument("--overlap",    type=int,   default=10)
    parser.add_argument("--seeds",      type=int,   default=5)
    parser.add_argument("--synthetic",  action="store_true",
                        help="Run on synthetic data (no KITTI needed)")
    parser.add_argument("--n-poses-synth", type=int, default=200)
    parser.add_argument("--max-poses",  type=int, default=None,
                        help="Truncate each sequence to this many poses (for quick local tests)")
    parser.add_argument("--anchor-spacing", type=int, default=50,
                        help="Sparse GT rotation anchors every this-many poses (+ last pose). "
                             "Without these the inner rotation GN solve is provably a no-op on "
                             "this chain graph (see exp44's InnerCfg.anchor_spacing docstring).")
    parser.add_argument("--kappa-anchor", type=float, default=100.0,
                        help="Fixed anchor precision. Needs empirical tuning: too small and "
                             "anchors don't matter in practice, too large and every window just "
                             "clamps to GT (which, with NO outer loop here, is the entire output).")
    parser.add_argument("--kappa-t-anchor", type=float, default=100.0,
                        help="Sparse GT TRANSLATION anchor precision, same treatment as "
                             "--kappa-anchor but for recover_translations (see exp44's "
                             "InnerCfg.kappa_t_anchor docstring). Unvalidated starting guess.")
    parser.add_argument("--n-starts", type=int, default=1,
                        help="Multi-start GN: 1 = today's single chain-composed R_init (default, "
                             "unchanged behavior). 2 = also try an anchor-interpolated R_init and "
                             "keep whichever the self-checking GN solve reaches lower cost from -- "
                             "targets the deterministic bad-basin seeds (see InnerCfg.n_starts).")
    parser.add_argument("--multistart-criterion", type=str, default="anchor_only",
                        choices=["anchor_only", "matched_total", "anchor_only_matched"],
                        help="Selection criterion when --n-starts=2 (see InnerCfg.multistart_criterion "
                             "for the evidence behind each -- this is a live A/B, not settled).")
    parser.add_argument("--anchor-n-iters-rot", type=int, default=60,
                        help="GN iteration budget for the anchor candidate when "
                             "--multistart-criterion=matched_total (ignored otherwise).")
    parser.add_argument("--adaptive-solver", action=argparse.BooleanOptionalAction, default=True,
                        help="Scale n_iters_rot/damping_up with sigma_t above the reference noise "
                             "level (default: on) -- same scaling exp44 uses for its inner solve.")
    parser.add_argument("--adaptive-reference-sigma-t", type=float, default=0.03)
    parser.add_argument("--output-dir", type=str,   default=os.path.expanduser("~/exp_res"),
                        help="Directory to write run results (timestamped sub-dir created automatically)")
    args = parser.parse_args()

    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp44_inner_only_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    base_inner_kwargs = {"n_iters_rot": 15, "damping_init": 1e-4, "damping_min": 1e-6,
                         "damping_max": 1e2, "damping_down": 0.5, "damping_up": 4.0,
                         "anchor_spacing": args.anchor_spacing, "kappa_anchor": args.kappa_anchor,
                         "kappa_t_anchor": args.kappa_t_anchor,
                         "n_starts": args.n_starts, "multistart_criterion": args.multistart_criterion,
                         "anchor_n_iters_rot": args.anchor_n_iters_rot}
    # noise_adaptive_inner_outer_cfg returns (InnerCfg, OuterCfg) -- OuterCfg is
    # irrelevant here (no outer loop), only the InnerCfg's noise-scaled
    # n_iters_rot/damping_up matter, kept consistent with what exp44 itself uses.
    base_outer_kwargs = {"n_trans1": 0, "n_rot": 0, "n_trans2": 0,
                         "lr_trans": 0.0, "lr_rot": 0.0, "warmup_steps": 0}

    if args.adaptive_solver:
        inner_cfg, _ = noise_adaptive_inner_outer_cfg(
            sigma_t=args.sigma_t,
            base_inner_kwargs=base_inner_kwargs,
            base_outer_kwargs=base_outer_kwargs,
            reference_sigma_t=args.adaptive_reference_sigma_t,
        )
        print(f"Noise-adaptive inner-solve settings (sigma_t={args.sigma_t}, "
              f"reference={args.adaptive_reference_sigma_t}): "
              f"n_iters_rot={inner_cfg.n_iters_rot}, damping_up={inner_cfg.damping_up:.2f}, "
              f"anchor_spacing={inner_cfg.anchor_spacing}, kappa_anchor={inner_cfg.kappa_anchor:.2f}, "
              f"kappa_t_anchor={inner_cfg.kappa_t_anchor:.2f}, "
              f"n_starts={inner_cfg.n_starts}, multistart_criterion={inner_cfg.multistart_criterion}, "
              f"anchor_n_iters_rot={inner_cfg.anchor_n_iters_rot}")
    else:
        inner_cfg = InnerCfg(n_iters_rot=base_inner_kwargs["n_iters_rot"],
                              damping_init=base_inner_kwargs["damping_init"],
                              damping_min=base_inner_kwargs["damping_min"],
                              damping_max=base_inner_kwargs["damping_max"],
                              damping_down=base_inner_kwargs["damping_down"],
                              damping_up=base_inner_kwargs["damping_up"],
                              anchor_spacing=base_inner_kwargs["anchor_spacing"],
                              kappa_anchor=base_inner_kwargs["kappa_anchor"],
                              kappa_t_anchor=base_inner_kwargs["kappa_t_anchor"],
                              n_starts=base_inner_kwargs["n_starts"],
                              multistart_criterion=base_inner_kwargs["multistart_criterion"],
                              anchor_n_iters_rot=base_inner_kwargs["anchor_n_iters_rot"])
        print(f"Fixed (non-adaptive) inner-solve settings: n_iters_rot={inner_cfg.n_iters_rot}, "
              f"damping_up={inner_cfg.damping_up:.2f}, "
              f"anchor_spacing={inner_cfg.anchor_spacing}, kappa_anchor={inner_cfg.kappa_anchor:.2f}, "
              f"kappa_t_anchor={inner_cfg.kappa_t_anchor:.2f}, "
              f"n_starts={inner_cfg.n_starts}, multistart_criterion={inner_cfg.multistart_criterion}, "
              f"anchor_n_iters_rot={inner_cfg.anchor_n_iters_rot}")

    exp_cfg = ExpCfg(
        window=args.window,
        overlap=args.overlap,
        sigma_t=args.sigma_t,
        sigma_r=args.sigma_r,
        seeds=args.seeds,
    )

    print(f"Compiling inner-solver-only denoiser (window={args.window}, seeds={args.seeds})...")
    t0 = time.time()
    denoiser_fn = build_inner_solver_only_denoiser(args.window, inner_cfg)

    lk, mr = exp_cfg.local_k, exp_cfg.max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision,
                                                local_k=lk,
                                                max_kappa_ratio=mr))

    dummy_odom = jnp.zeros((args.window - 1, 6))
    dummy_gt   = jnp.zeros((args.window, 6))
    dummy_gR   = jnp.zeros((args.window, 3, 3))
    dummy_k    = jnp.ones(args.window - 1)
    dummy_w    = jnp.ones(args.window - 1)
    _ = denoiser_fn(dummy_odom, dummy_gt, dummy_gR, dummy_k, dummy_w).block_until_ready()
    dummy_omds = jnp.zeros((args.seeds, args.window - 1, 6))
    _ = prec_pooled_fn(dummy_omds[:, :, :3]).block_until_ready()
    print(f"  Compiled in {time.time() - t0:.1f}s  (no retrace after this)")

    all_dT, all_dR, all_dC = [], [], []
    all_seq_results = {}

    if args.synthetic:
        sequences = [("synth", args.n_poses_synth)]
    else:
        if args.kitti_root is None:
            raise ValueError("Provide --kitti-root or use --synthetic")
        if args.seqs is not None:
            seq_ids = [s.strip() for s in args.seqs.split(",")]
        else:
            seq_ids = sorted(
                d for d in os.listdir(args.kitti_root)
                if os.path.isfile(os.path.join(args.kitti_root, d, "poses.txt"))
            )
        sequences = [(s, None) for s in seq_ids]

    for seq_id, n_synth in sequences:
        seq_dT, seq_dR, seq_dC = [], [], []
        seq_hash = int(seq_id) if seq_id.isdigit() else 0

        seq_seed_results = []
        if seq_id == "synth":
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed * 1000 + seq_hash)
                gt_rel, noisy_rel, gt_global, noisy_global = make_synthetic_sequence(
                    n_synth, args.sigma_t, args.sigma_r, rng
                )
                gt_R_mats = np.array(jax.vmap(so3_exp)(jnp.array(gt_global[:, 3:])))
                t_start = time.time()
                denoised_global, denoised_rel = denoise_sequence(
                    noisy_rel, gt_global, gt_R_mats, denoiser_fn, exp_cfg
                )
                elapsed = time.time() - t_start
                poses_per_sec = len(gt_global) / (elapsed + 1e-9)
                gt_R      = integrate_rotations(gt_rel)
                noisy_R   = integrate_rotations(noisy_rel)
                denoised_R = integrate_rotations(denoised_rel)
                dT, dR, dC = delta_metric(noisy_global, denoised_global, gt_global,
                                           noisy_R, denoised_R, gt_R)
                seq_dT.append(dT); seq_dR.append(dR); seq_dC.append(dC)
                print(f"  [{seq_id}|seed={seed}]  ΔT={dT:+.1f}%  ΔR={dR:+.1f}%  ΔC={dC:+.1f}%  "
                      f"({poses_per_sec:.0f} poses/s)")
                seed_out = {"seq": seq_id, "seed": seed, "n_poses": int(len(gt_global)),
                            "dT": float(dT), "dR": float(dR), "dC": float(dC),
                            "elapsed_s": float(elapsed), "poses_per_s": float(poses_per_sec)}
                seq_seed_results.append(seed_out)
                with open(os.path.join(run_dir, f"{seq_id}_seed_{seed:04d}.json"), "w") as fp:
                    json.dump(seed_out, fp, indent=2)
        else:
            root_p  = Path(args.kitti_root)
            seq_str = f"{int(seq_id):02d}"
            candidates = [
                root_p / seq_str / "poses.txt",
                root_p / "sequences" / seq_str / "poses.txt",
                root_p / "poses" / f"{seq_str}.txt",
            ]
            poses_path = next((p for p in candidates if p.exists()), None)
            if poses_path is None:
                print(f"  [{seq_id}] poses.txt not found (tried {candidates}), skipping.")
                continue

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
                print(f"  [{seq_id}] Empty poses.txt, skipping.")
                continue

            gt_mats = np.stack(raw_mats, axis=0)
            if args.max_poses is not None:
                gt_mats = gt_mats[:args.max_poses]
            gt_global = np.zeros((len(gt_mats), 6), dtype=np.float32)
            gt_global[:, :3] = gt_mats[:, :3, 3]
            gt_global[:, 3:] = np.array(
                jax.vmap(so3_log)(jnp.array(gt_mats[:, :3, :3], dtype=jnp.float32))
            )
            gt_R_mats = gt_mats[:, :3, :3]
            gt_rel    = relative_poses_from_mats(gt_mats)

            noisy_rels = np.stack([
                _add_kitti_noise(gt_rel, seed, seq_hash, args.sigma_t, args.sigma_r)
                for seed in range(args.seeds)
            ])

            t_start = time.time()
            denoised_globals, denoised_rels = denoise_sequence_pooled(
                noisy_rels, gt_global, gt_R_mats, denoiser_fn, prec_pooled_fn, exp_cfg
            )
            elapsed = time.time() - t_start
            poses_per_sec = len(gt_global) * args.seeds / (elapsed + 1e-9)

            per_seed_sec = elapsed / args.seeds
            pps = len(gt_global) / (per_seed_sec + 1e-9)
            for s in range(args.seeds):
                noisy_global_s = integrate_poses(noisy_rels[s])
                noisy_R_s = integrate_rotations(noisy_rels[s])
                denoised_R_s = integrate_rotations(denoised_rels[s])
                dT, dR, dC = delta_metric(noisy_global_s, denoised_globals[s], gt_global,
                                           noisy_R_s, denoised_R_s, gt_R_mats)
                seq_dT.append(dT); seq_dR.append(dR); seq_dC.append(dC)
                print(f"  [{seq_id}|seed={s}]  ΔT={dT:+.1f}%  ΔR={dR:+.1f}%  ΔC={dC:+.1f}%  "
                      f"({pps:.0f} poses/s·seed)")
                seed_out = {"seq": seq_id, "seed": s, "n_poses": int(len(gt_global)),
                            "dT": float(dT), "dR": float(dR), "dC": float(dC),
                            "elapsed_s": float(per_seed_sec), "poses_per_s": float(pps)}
                seq_seed_results.append(seed_out)
                with open(os.path.join(run_dir, f"{seq_id}_seed_{s:04d}.json"), "w") as fp:
                    json.dump(seed_out, fp, indent=2)
            print(f"  [{seq_id}] {args.seeds} seeds (inner-solver-only, pooled precision)  "
                  f"total {poses_per_sec:.0f} poses/s·seed")

        if seq_dT:
            m_dT = float(np.mean(seq_dT))
            m_dR = float(np.mean(seq_dR))
            m_dC = float(np.mean(seq_dC))
            all_dT.append(m_dT)
            all_dR.append(m_dR)
            all_dC.append(m_dC)
            all_seq_results[seq_id] = {
                "seeds": seq_seed_results,
                "mean_dT": m_dT, "mean_dR": m_dR, "mean_dC": m_dC,
            }
            print(f"  [{seq_id}] mean  ΔT={m_dT:+.1f}%  ΔR={m_dR:+.1f}%  ΔC={m_dC:+.1f}%")

    if all_dC:
        print("\n=== Exp44 Inner-Solver-Only Summary ===")
        print(f"  σ_t={args.sigma_t}  window={args.window}  overlap={args.overlap}")
        print(f"  Mean across seqs:  ΔT={np.mean(all_dT):+.1f}%  "
              f"ΔR={np.mean(all_dR):+.1f}%  ΔC={np.mean(all_dC):+.1f}%")

        config = {
            "exp": "exp44_inner_solver_only",
            "kitti_root": args.kitti_root,
            "seqs": args.seqs,
            "sigma_t": args.sigma_t,
            "sigma_r": args.sigma_r,
            "window": args.window,
            "overlap": args.overlap,
            "seeds": args.seeds,
            "synthetic": args.synthetic,
            "run_ts": run_ts,
            "adaptive_solver": args.adaptive_solver,
            "adaptive_reference_sigma_t": args.adaptive_reference_sigma_t,
            "n_iters_rot_used": inner_cfg.n_iters_rot,
            "damping_up_used": inner_cfg.damping_up,
            "anchor_spacing": inner_cfg.anchor_spacing,
            "kappa_anchor": inner_cfg.kappa_anchor,
            "kappa_t_anchor": inner_cfg.kappa_t_anchor,
            "n_starts": inner_cfg.n_starts,
            "multistart_criterion": inner_cfg.multistart_criterion,
            "anchor_n_iters_rot": inner_cfg.anchor_n_iters_rot,
        }
        aggregate = {
            "config": config,
            "sequences": all_seq_results,
            "overall_mean_dT": float(np.mean(all_dT)),
            "overall_mean_dR": float(np.mean(all_dR)),
            "overall_mean_dC": float(np.mean(all_dC)),
        }
        agg_path = os.path.join(run_dir, "aggregate.json")
        with open(agg_path, "w") as fp:
            json.dump(aggregate, fp, indent=2)

        txt_lines = [
            f"Exp44 Inner-Solver-Only Run  {run_ts}",
            f"  kitti_root={args.kitti_root}  seqs={args.seqs}",
            f"  sigma_t={args.sigma_t}  sigma_r={args.sigma_r}",
            f"  window={args.window}  overlap={args.overlap}  seeds={args.seeds}",
            f"  anchor_spacing={inner_cfg.anchor_spacing}  kappa_anchor={inner_cfg.kappa_anchor}  "
            f"kappa_t_anchor={inner_cfg.kappa_t_anchor}  "
            f"n_starts={inner_cfg.n_starts}",
            "",
        ]
        for sid, sr in all_seq_results.items():
            txt_lines.append(f"[{sid}]  mean  ΔT={sr['mean_dT']:+.1f}%  "
                             f"ΔR={sr['mean_dR']:+.1f}%  ΔC={sr['mean_dC']:+.1f}%")
            for sd in sr["seeds"]:
                txt_lines.append(f"  seed={sd['seed']:04d}  ΔT={sd['dT']:+.1f}%  "
                                 f"ΔR={sd['dR']:+.1f}%  ΔC={sd['dC']:+.1f}%  "
                                 f"({sd['poses_per_s']:.0f} poses/s)")
        txt_lines += [
            "",
            f"Overall mean:  ΔT={np.mean(all_dT):+.1f}%  "
            f"ΔR={np.mean(all_dR):+.1f}%  ΔC={np.mean(all_dC):+.1f}%",
        ]
        res_path = os.path.join(run_dir, "results.txt")
        with open(res_path, "w") as fp:
            fp.write("\n".join(txt_lines) + "\n")

        print(f"\nSaved: {agg_path}")
        print(f"Saved: {res_path}")


if __name__ == "__main__":
    main()
