"""
diag_exp44_outer_regression.py

Diagnostic for the finding that exp44's full pipeline (anchored inner solve +
outer Adam loop over theta) scores WORSE than exp44_inner_solver_only (anchors
alone, theta=0, no outer loop) at every tested noise level, and increasingly
so as noise grows:

  sigma_t   inner-only meanDC   full-exp44 meanDC   full negative-dC count
  0.01      +23.6%              +9.9%               154/484
  0.03      +40.5%              +16.4%               118/484
  0.05      +46.1%              -8.7%                258/484
  0.10      +36.8%              -5.0%                260/484

Leading hypothesis: noise_adaptive_inner_outer_cfg's rot_loss_boost scaling
(up to 30x, validated against exp43 -- BEFORE anchors were reintroduced into
the inner solve in exp44) was never re-tuned for the anchor-augmented solver,
so at higher noise the outer loop's rotation-phase gradient gets massively
overweighted relative to what's actually well-conditioned now, and Adam takes
oversized steps that push theta away from, not toward, the optimum.

This script isolates a single window inside a known-bad (seq, seed) pair
(default seq=13, seed=12 @ sigma_t=0.05, dC=-49.3% inner-only / -91.6% full)
and, for every window in that sequence, compares three outer-loop configs:
  A) actual   -- the exact adaptive cfg the real sweep used
  B) noboost  -- same adaptive cfg, rot_loss_boost forced to 1.0
  C) fixed    -- non-adaptive base cfg (n_iters_rot=15, damping_up=4.0,
                 warmup_steps=5, rot_loss_boost=1.0) -- exp43-style knobs

For each window, reports a LOCAL edge-space metric independent of downstream
propagation: does this window's theta correction move its own write-region
edges closer to or further from ground truth than doing nothing (theta=0)?
This isolates each window's own outer-loop behavior from the (separate,
already-understood) fact that one bad window's error propagates to every
pose after it once relative poses are chain-integrated.

The worst window under config A gets a full per-step trace (loss_t, loss_r,
theta_norm, grad_norm) dumped to JSON for configs A and B, to see exactly
where in the ~70-step trajectory the boosted-rotation config diverges.

Run (on the machine with real KITTI data):
  python -m experiments.diag_exp44_outer_regression --kitti-root /path/to/kitti \
      --seq 13 --seed 12 --sigma-t 0.05 --output-dir ~/exp_res
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
    OuterCfg,
    ExpCfg,
    so3_exp,
    so3_log,
    sesync_inner_solve,
    relative_poses_from_mats,
    _add_kitti_noise,
    _sigma_and_precision,
    integrate_poses,
    noise_adaptive_inner_outer_cfg,
)


# ---------------------------------------------------------------------------
# Instrumented copy of outer_adam_loop -- identical math, but the scan also
# threads out (loss_t, loss_r, theta_norm, grad_norm) at every step instead
# of discarding them, so the trajectory can be inspected after the fact.
# ---------------------------------------------------------------------------

def instrumented_outer_adam_loop(theta_init, noisy_odom, gt_poses, gt_R_direct,
                                  kappa, omega, n, inner_cfg, outer_cfg):
    N_T1 = outer_cfg.n_trans1
    N_R = outer_cfg.n_rot
    total = N_T1 + N_R + outer_cfg.n_trans2

    gt_R_world = gt_R_direct
    gt_R0 = gt_R_world[0]
    gt_t0 = gt_poses[0, :3]
    gt_t_rel = jax.vmap(lambda t: gt_R0.T @ (t - gt_t0))(gt_poses[:, :3])
    gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(gt_R_world)

    def loss_components(theta):
        R_star, t_star = sesync_inner_solve(
            theta, noisy_odom, kappa, omega, n, inner_cfg, gt_R_rel
        )
        loss_t = jnp.mean(jnp.sum((t_star - gt_t_rel) ** 2, axis=-1))
        loss_r = jnp.mean(jax.vmap(
            lambda Ra, Rb: jnp.sum((Ra - Rb) ** 2)
        )(R_star, gt_R_rel))
        return loss_t, loss_r

    def loss_fn(theta):
        loss_t, loss_r = loss_components(theta)
        return loss_t + outer_cfg.rot_loss_boost * loss_r

    grad_fn = jax.grad(loss_fn)

    m_init = jnp.zeros_like(theta_init)
    v_init = jnp.zeros_like(theta_init)

    def adam_step(carry, step_idx):
        theta, m, v = carry

        in_rot = (step_idx >= N_T1) & (step_idx < N_T1 + N_R)
        at_bound = (step_idx == N_T1) | (step_idx == N_T1 + N_R)

        m = jnp.where(at_bound, jnp.zeros_like(m), m)
        v = jnp.where(at_bound, jnp.zeros_like(v), v)

        local_t = jnp.where(in_rot,
                             step_idx - N_T1 + 1,
                             jnp.where(step_idx >= N_T1 + N_R,
                                       step_idx - N_T1 - N_R + 1,
                                       step_idx + 1))

        lr_base = jnp.where(in_rot, outer_cfg.lr_rot, outer_cfg.lr_trans)
        lr_scale = jnp.minimum(1.0, local_t.astype(jnp.float32) / float(outer_cfg.warmup_steps))
        lr = lr_base * lr_scale

        loss_t_val, loss_r_val = loss_components(theta)
        g = grad_fn(theta)

        mask_t = jnp.concatenate([jnp.ones((theta.shape[0], 3)),
                                   jnp.zeros((theta.shape[0], 3))], axis=-1)
        mask_r = 1.0 - mask_t
        g_masked = jnp.where(in_rot, g * mask_r, g * mask_t)

        m_new = outer_cfg.beta1 * m + (1.0 - outer_cfg.beta1) * g_masked
        v_new = outer_cfg.beta2 * v + (1.0 - outer_cfg.beta2) * g_masked ** 2

        m_hat = m_new / (1.0 - outer_cfg.beta1 ** local_t)
        v_hat = v_new / (1.0 - outer_cfg.beta2 ** local_t)

        theta_new = theta - lr * m_hat / (jnp.sqrt(v_hat) + outer_cfg.eps)

        diag = (loss_t_val, loss_r_val, jnp.linalg.norm(theta_new), jnp.linalg.norm(g_masked), lr)
        return (theta_new, m_new, v_new), diag

    (theta_opt, _, _), trace = jax.lax.scan(
        adam_step, (theta_init, m_init, v_init), jnp.arange(total)
    )
    return theta_opt, trace


def theta0_denoise(win_odom, win_gt_R, kappa, omega, n, inner_cfg):
    """Anchor-only baseline (theta fixed at 0, no outer loop) -- same math as
    exp44_inner_solver_only.build_inner_solver_only_denoiser, reimplemented
    here to avoid importing a second jit-wrapped denoiser closed over a
    different n/inner_cfg. Isolates what the anchored inner solve alone does
    to this window, so per-window badness can be attributed to the anchors
    themselves vs. the outer loop layered on top."""
    gt_R0 = win_gt_R[0]
    gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(win_gt_R)
    theta = jnp.zeros_like(win_odom)
    R_star, t_star = sesync_inner_solve(theta, win_odom, kappa, omega, n, inner_cfg, gt_R_rel)
    dR = jax.vmap(lambda Ri, Rj: Ri.T @ Rj)(R_star[:-1], R_star[1:])
    dw = jax.vmap(so3_log)(dR)
    dt = jax.vmap(lambda Ri, ti, tj: Ri.T @ (tj - ti))(R_star[:-1], t_star[:-1], t_star[1:])
    return jnp.concatenate([dt, dw], axis=-1)


def load_sequence(kitti_root, seq_id):
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
    gt_mats = np.stack(raw_mats, axis=0)
    gt_global = np.zeros((len(gt_mats), 6), dtype=np.float32)
    gt_global[:, :3] = gt_mats[:, :3, 3]
    gt_global[:, 3:] = np.array(jax.vmap(so3_log)(jnp.array(gt_mats[:, :3, :3], dtype=jnp.float32)))
    gt_R_mats = gt_mats[:, :3, :3]
    gt_rel = relative_poses_from_mats(gt_mats)
    return gt_mats, gt_global, gt_R_mats, gt_rel


def local_edge_error(rel_a, rel_b, lo, hi):
    """Mean squared trans/rot deviation between two (n-1,6) relative-pose
    arrays over edges [lo, hi) -- edge-space, no integration, no propagation."""
    seg_a = rel_a[lo:hi]
    seg_b = rel_b[lo:hi]
    err_t = float(np.mean(np.sum((seg_a[:, :3] - seg_b[:, :3]) ** 2, axis=-1)))
    err_r = float(np.mean(np.sum((seg_a[:, 3:] - seg_b[:, 3:]) ** 2, axis=-1)))
    return err_t, err_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--seq", default="13")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--sigma-t", type=float, default=0.05)
    ap.add_argument("--sigma-r", type=float, default=0.03)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--pool-seeds", type=int, default=22, help="how many seeds to pool precision over, matching the real sweep")
    ap.add_argument("--anchor-spacing", type=int, default=50)
    ap.add_argument("--kappa-anchor", type=float, default=100.0)
    ap.add_argument("--adaptive-reference-sigma-t", type=float, default=0.03)
    ap.add_argument("--output-dir", default=os.path.expanduser("~/exp_res"))
    args = ap.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"diag_exp44_outer_regression_{run_ts}")
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
    target_noisy_rel = noisy_rels[args.seed]

    base_inner_kwargs = {"n_iters_rot": 15, "damping_init": 1e-4, "damping_min": 1e-6,
                          "damping_max": 1e2, "damping_down": 0.5, "damping_up": 4.0,
                          "anchor_spacing": args.anchor_spacing, "kappa_anchor": args.kappa_anchor}
    base_outer_kwargs = {"n_trans1": 30, "n_rot": 20, "n_trans2": 20,
                          "lr_trans": 1e-3, "lr_rot": 1e-3, "warmup_steps": 5}

    inner_cfg_A, outer_cfg_A = noise_adaptive_inner_outer_cfg(
        sigma_t=args.sigma_t, base_inner_kwargs=base_inner_kwargs,
        base_outer_kwargs=base_outer_kwargs, reference_sigma_t=args.adaptive_reference_sigma_t,
    )
    print(f"Config A (actual adaptive): n_iters_rot={inner_cfg_A.n_iters_rot} "
          f"damping_up={inner_cfg_A.damping_up:.2f} warmup_steps={outer_cfg_A.warmup_steps} "
          f"rot_loss_boost={outer_cfg_A.rot_loss_boost:.3f}")

    outer_cfg_B = OuterCfg(n_trans1=outer_cfg_A.n_trans1, n_rot=outer_cfg_A.n_rot,
                            n_trans2=outer_cfg_A.n_trans2, lr_trans=outer_cfg_A.lr_trans,
                            lr_rot=outer_cfg_A.lr_rot, warmup_steps=outer_cfg_A.warmup_steps,
                            rot_loss_boost=1.0)
    inner_cfg_B = inner_cfg_A
    print(f"Config B (adaptive, boost pinned to 1.0): rot_loss_boost=1.0, "
          f"n_iters_rot={inner_cfg_B.n_iters_rot} damping_up={inner_cfg_B.damping_up:.2f}")

    inner_cfg_C = InnerCfg(n_iters_rot=base_inner_kwargs["n_iters_rot"],
                            damping_init=base_inner_kwargs["damping_init"],
                            damping_min=base_inner_kwargs["damping_min"],
                            damping_max=base_inner_kwargs["damping_max"],
                            damping_down=base_inner_kwargs["damping_down"],
                            damping_up=base_inner_kwargs["damping_up"],
                            anchor_spacing=base_inner_kwargs["anchor_spacing"],
                            kappa_anchor=base_inner_kwargs["kappa_anchor"])
    outer_cfg_C = OuterCfg(n_trans1=base_outer_kwargs["n_trans1"], n_rot=base_outer_kwargs["n_rot"],
                            n_trans2=base_outer_kwargs["n_trans2"], lr_trans=base_outer_kwargs["lr_trans"],
                            lr_rot=base_outer_kwargs["lr_rot"], warmup_steps=base_outer_kwargs["warmup_steps"],
                            rot_loss_boost=1.0)
    print(f"Config C (fixed, non-adaptive, exp43-style): n_iters_rot={inner_cfg_C.n_iters_rot} "
          f"damping_up={inner_cfg_C.damping_up:.2f} warmup_steps={outer_cfg_C.warmup_steps} rot_loss_boost=1.0")

    n = args.window
    denoiser_A = jax.jit(functools.partial(instrumented_outer_adam_loop, n=n, inner_cfg=inner_cfg_A, outer_cfg=outer_cfg_A))
    denoiser_B = jax.jit(functools.partial(instrumented_outer_adam_loop, n=n, inner_cfg=inner_cfg_B, outer_cfg=outer_cfg_B))
    denoiser_C = jax.jit(functools.partial(instrumented_outer_adam_loop, n=n, inner_cfg=inner_cfg_C, outer_cfg=outer_cfg_C))
    denoiser_D = jax.jit(functools.partial(theta0_denoise, n=n, inner_cfg=inner_cfg_A))  # anchors alone, no outer loop

    lk, mr = ExpCfg().local_k, ExpCfg().max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision, local_k=lk, max_kappa_ratio=mr))

    noisy_rels_j = jnp.array(noisy_rels)
    gt_global_j = jnp.array(gt_global)
    gt_R_mats_j = jnp.array(gt_R_mats)

    stride = args.window - args.overlap
    windows = []
    pos = 0
    while pos + args.window <= n_poses - 1:
        lo, hi = pos, pos + args.window
        n_edges = hi - 1 - lo
        write_lo = args.overlap // 2 if pos > 0 else 0
        write_hi = min(n_edges, stride + args.overlap // 2) if hi < n_poses - 1 else n_edges
        windows.append((lo, hi, write_lo, write_hi))
        pos += stride
    if windows:
        last_hi_edge = windows[-1][0] + windows[-1][3]
    else:
        last_hi_edge = 0
    if last_hi_edge < n_poses - 1:
        hi = n_poses
        lo = max(0, hi - args.window)
        n_edges = hi - 1 - lo
        write_lo = last_hi_edge - lo
        windows.append((lo, hi, write_lo, n_edges))

    print(f"\n{len(windows)} windows, evaluating seed={args.seed} on each ...\n")

    results = []
    for wi, (lo, hi, write_lo, write_hi) in enumerate(windows):
        win_omds = noisy_rels_j[:, lo:hi - 1, :]
        win_gt = gt_global_j[lo:hi]
        win_gt_R = gt_R_mats_j[lo:hi]
        kappa = prec_pooled_fn(win_omds[:, :, 3:])
        omega = prec_pooled_fn(win_omds[:, :, :3])

        win_odom = win_omds[args.seed]
        theta_init = jnp.zeros_like(win_odom)

        theta_A, _ = denoiser_A(theta_init, win_odom, win_gt, win_gt_R, kappa, omega)
        theta_B, _ = denoiser_B(theta_init, win_odom, win_gt, win_gt_R, kappa, omega)
        theta_C, _ = denoiser_C(theta_init, win_odom, win_gt, win_gt_R, kappa, omega)
        corrected_D = np.array(denoiser_D(win_odom, win_gt_R, kappa, omega))  # theta=0, anchors only

        corrected_A = np.array(win_odom + theta_A)
        corrected_B = np.array(win_odom + theta_B)
        corrected_C = np.array(win_odom + theta_C)
        noisy_np = np.array(win_odom)
        gt_seg = gt_rel[lo:hi - 1]

        err_t_before, err_r_before = local_edge_error(noisy_np, gt_seg, write_lo, write_hi)
        err_t_A, err_r_A = local_edge_error(corrected_A, gt_seg, write_lo, write_hi)
        err_t_B, err_r_B = local_edge_error(corrected_B, gt_seg, write_lo, write_hi)
        err_t_C, err_r_C = local_edge_error(corrected_C, gt_seg, write_lo, write_hi)
        err_t_D, err_r_D = local_edge_error(corrected_D, gt_seg, write_lo, write_hi)

        def pct(before, after):
            return 100.0 * (before - after) / (before + 1e-12)

        row = {
            "window": wi, "lo": lo, "hi": hi, "write_lo": write_lo, "write_hi": write_hi,
            "err_t_before": err_t_before, "err_r_before": err_r_before,
            "dT_A": pct(err_t_before, err_t_A), "dR_A": pct(err_r_before, err_r_A),
            "dT_B": pct(err_t_before, err_t_B), "dR_B": pct(err_r_before, err_r_B),
            "dT_C": pct(err_t_before, err_t_C), "dR_C": pct(err_r_before, err_r_C),
            "dT_D": pct(err_t_before, err_t_D), "dR_D": pct(err_r_before, err_r_D),
        }
        results.append(row)
        print(f"  win {wi:2d} [{lo:4d}:{hi:4d}]  "
              f"D(anchors-only): dT={row['dT_D']:+7.1f} dR={row['dR_D']:+7.1f}  |  "
              f"A(actual): dT={row['dT_A']:+7.1f} dR={row['dR_A']:+7.1f}  |  "
              f"B(noboost): dT={row['dT_B']:+7.1f} dR={row['dR_B']:+7.1f}  |  "
              f"C(fixed): dT={row['dT_C']:+7.1f} dR={row['dR_C']:+7.1f}")

    with open(os.path.join(out_dir, "per_window_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    worst = min(results, key=lambda r: r["dR_A"] + r["dT_A"])
    print(f"\nWorst window under config A: win {worst['window']} [{worst['lo']}:{worst['hi']}]  "
          f"dT_A={worst['dT_A']:+.1f} dR_A={worst['dR_A']:+.1f}  "
          f"(vs D anchors-only dT={worst['dT_D']:+.1f} dR={worst['dR_D']:+.1f}, "
          f"vs B dT={worst['dT_B']:+.1f} dR={worst['dR_B']:+.1f}, "
          f"vs C dT={worst['dT_C']:+.1f} dR={worst['dR_C']:+.1f})")

    mean_D_dR = sum(r["dR_D"] for r in results) / len(results)
    mean_A_dR = sum(r["dR_A"] for r in results) / len(results)
    mean_D_dT = sum(r["dT_D"] for r in results) / len(results)
    mean_A_dT = sum(r["dT_A"] for r in results) / len(results)
    print(f"\nMean over all {len(results)} windows: "
          f"D(anchors-only) dT={mean_D_dT:+.1f} dR={mean_D_dR:+.1f}  |  "
          f"A(actual outer loop) dT={mean_A_dT:+.1f} dR={mean_A_dR:+.1f}")

    lo, hi, write_lo, write_hi = worst["lo"], worst["hi"], worst["write_lo"], worst["write_hi"]
    win_omds = noisy_rels_j[:, lo:hi - 1, :]
    win_gt = gt_global_j[lo:hi]
    win_gt_R = gt_R_mats_j[lo:hi]
    kappa = prec_pooled_fn(win_omds[:, :, 3:])
    omega = prec_pooled_fn(win_omds[:, :, :3])
    win_odom = win_omds[args.seed]
    theta_init = jnp.zeros_like(win_odom)

    _, trace_A = denoiser_A(theta_init, win_odom, win_gt, win_gt_R, kappa, omega)
    _, trace_B = denoiser_B(theta_init, win_odom, win_gt, win_gt_R, kappa, omega)

    def trace_to_list(trace):
        loss_t, loss_r, theta_norm, grad_norm, lr = trace
        return [
            {"step": i, "loss_t": float(loss_t[i]), "loss_r": float(loss_r[i]),
             "theta_norm": float(theta_norm[i]), "grad_norm": float(grad_norm[i]), "lr": float(lr[i])}
            for i in range(len(loss_t))
        ]

    trace_dump = {
        "window": worst, "seq": args.seq, "seed": args.seed, "sigma_t": args.sigma_t,
        "outer_cfg_A_rot_loss_boost": outer_cfg_A.rot_loss_boost,
        "phase_boundaries": [outer_cfg_A.n_trans1, outer_cfg_A.n_trans1 + outer_cfg_A.n_rot],
        "trace_A_actual": trace_to_list(trace_A),
        "trace_B_noboost": trace_to_list(trace_B),
    }
    with open(os.path.join(out_dir, "worst_window_trace.json"), "w") as fp:
        json.dump(trace_dump, fp, indent=2)

    print(f"\nSaved per-window results and worst-window trace to {out_dir}")
    print(f"\n=== Worst window trace summary (phase boundaries at steps "
          f"{outer_cfg_A.n_trans1}, {outer_cfg_A.n_trans1 + outer_cfg_A.n_rot}) ===")
    tA = trace_to_list(trace_A)
    tB = trace_to_list(trace_B)
    for i in range(0, len(tA), max(1, len(tA) // 20)):
        print(f"  step {i:3d}  A: loss_t={tA[i]['loss_t']:.4f} loss_r={tA[i]['loss_r']:.6f} "
              f"|theta|={tA[i]['theta_norm']:.4f}  |  B: loss_t={tB[i]['loss_t']:.4f} "
              f"loss_r={tB[i]['loss_r']:.6f} |theta|={tB[i]['theta_norm']:.4f}")


if __name__ == "__main__":
    main()
