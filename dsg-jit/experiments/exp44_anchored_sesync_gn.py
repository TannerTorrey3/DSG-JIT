"""
exp44_anchored_sesync_gn.py

Experiment 44: exp43 + sparse ground-truth rotation anchors in the inner solve.

Root cause addressed: exp43's inner rotation GN solve is mathematically inert
on this chain-graph topology. R_init is built by chain-composing the same
R_meas the cost function measures against, so the residual (and gradient) is
exactly zero at every edge, for any input, always — confirmed directly on
real KITTI data (cost_init=0.000000, every candidate step rejected, R_star
always exactly equal to R_init). The self-checking accept/reject + adaptive
damping machinery contributes nothing; all rotation correction in exp43
comes entirely from the outer Adam loop changing theta, which changes
R_meas, which changes what the (otherwise-inert) inner solve reconstructs.

This was a regression, not an inherent limit: exp43's own ancestors (exp36,
exp38, exp40, exp41 — all still in this directory) all wove sparse
ground-truth anchors directly into their inner solve's residual, giving it
a real, non-trivial, external objective. exp42 replaced that with the
SE-Sync connection/rotation-Laplacian self-consistency formulation and
dropped anchors from the inner solve entirely; exp43 inherited that.

Changes from Exp43:
  - Sparse GT rotation anchors reintroduced into the inner rotation GN solve
    (build_rotation_laplacian, _rotation_cost, _rotation_gn_candidate_step,
    rotation_gn_ift's custom_vjp) — see InnerCfg.anchor_spacing/kappa_anchor
    and build_rotation_laplacian's docstring for the full derivation. Anchor
    targets come from ground truth already passed into the pipeline for the
    outer loss (gt_R_rel), just a new consumer of already-available data, not
    a new dependency. kappa_anchor is a fixed hyperparameter in this first
    pass, not noise-adaptive or learned.

Changes from Exp42 (inherited from exp43, unchanged here):
  - Seed-axis GPU batching removed: each seed is solved with its own
    dispatch via the single-seed (non-vmapped) denoiser, not one combined
    vmap'd call across all S seeds. Simpler dispatch pattern, no batched
    Hessian solve — at the cost of the throughput win batching provided.
  - Cross-seed precision pooling is KEPT despite removing batching: kappa/
    omega are still computed once per window from all S seeds' raw data
    (a plain vectorized reduction, not a GPU-dispatch batch), then reused
    for every seed's independent single-seed solve. See denoise_sequence_pooled.
  - integrate_poses split into a jit-compiled pure-JAX core plus a thin
    numpy wrapper, so each per-seed call benefits from JIT (no vmap, since
    there's no seed axis to batch over without the GPU batching above).
  - Inner GN is now self-checking: each candidate step is accepted only if
    it reduces the GN cost, with Levenberg-Marquardt-style adaptive damping
    (relax on accept, tighten on reject) instead of a fixed damping constant
    and blind step acceptance. The IFT backward pass uses damping_used (the
    damping active at the last ACCEPTED step), not the scan's raw final
    damping state, which can be arbitrarily inflated by trailing rejected
    steps after convergence — see _rotation_gn_raw's docstring.
  - Outer Adam ramps its learning rate over the first few steps of each
    phase instead of taking an unconditional full-magnitude step at
    local_t=1, where Adam's bias correction otherwise collapses
    m_hat/sqrt(v_hat) to exactly sign(gradient) regardless of whether that
    gradient is trustworthy.

Run:
  python exp44_anchored_sesync_gn.py --kitti-root /path/to/kitti --sigma-t 0.03
  python exp44_anchored_sesync_gn.py --synthetic  # no KITTI needed
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dsg_jit.core.math3d import so3_exp, so3_log

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InnerCfg:
    n_iters_rot:  int   = 15     # bumped from 10 — rejected LM steps spend a scan slot without progress
    damping_init: float = 1e-4
    damping_min:  float = 1e-6
    damping_max:  float = 1e2
    damping_down: float = 0.5    # relax factor on accept
    damping_up:   float = 4.0    # tighten factor on reject
    anchor_spacing: int = 50     # sparse GT rotation anchors every this-many poses (+ last pose).
                                  # Without these the rotation GN solve is provably a no-op on this
                                  # chain graph: R_init is built by chain-composing the same R_meas
                                  # the cost measures against, so residual/gradient are exactly zero
                                  # everywhere, always (confirmed on real data: cost_init=0.000000,
                                  # every step rejected) -- see build_rotation_laplacian's docstring.
                                  # Anchor index 0 is always a no-op (pose 0 is hard-fixed already),
                                  # so spacing is chosen relative to poses 1..n-1.
    kappa_anchor: float = 100.0  # fixed anchor precision -- NOT noise-adaptive or learned in this
                                  # first pass (unlike kappa/omega). Needs its own empirical tuning
                                  # sweep: too small and anchors don't break the trivial-optimum trap
                                  # in practice; too large and every window just clamps to GT,
                                  # defeating the point of learning theta at all.
    n_starts: int = 1            # multi-start GN: try this many candidate R_inits per window,
                                  # keep whichever the self-checking GN solve reaches the lowest
                                  # _rotation_cost from. 1 = today's single-start behavior
                                  # (chain-composed R_init only), byte-identical. 2 adds one
                                  # anchor-interpolated candidate (see
                                  # _build_anchor_interpolated_R_init) -- targets the deterministic
                                  # bad-basin seeds where the chain-composed R_init is biased in an
                                  # adversarial direction by that seed's specific noise-draw shape
                                  # (see project_exp44_bad_seeds_deterministic memory): the
                                  # anchor-interpolated candidate isn't derived from noisy
                                  # measurements at all, so it can't inherit that same bias. Values
                                  # other than 1 or 2 are not currently meaningful.
    multistart_criterion: str = "anchor_only"  # selection criterion when n_starts>1:
                                  # "anchor_only" (default) -- pick by anchor-term-only cost
                                  # (_anchor_residual), ignoring the chain-term residual the
                                  # anchor-interpolated candidate can't converge within n_iters_rot.
                                  # Fixed seq01/seed9 and seq13/seed12 (dC -41.7%->+5.5%,
                                  # -16.9%->+21.3%) but is unreliable elsewhere: introduced new
                                  # regressions on seq01/seed4,11,13 that per-window diagnosis
                                  # (diag_exp44_multistart.py) traced to anchor-only cost being
                                  # blind to interior (non-anchored) edges.
                                  # "matched_total" -- give the anchor candidate anchor_n_iters_rot
                                  # iterations to better converge, then pick by TOTAL
                                  # _rotation_cost. Tested as an alternative on the same regressed
                                  # seeds; per-window disagreement with real translation RMSE was
                                  # NOT better (8/13->10/13 on seed4, 4/13->7/13 on seed13) by raw
                                  # count, though it did fix the specific high-leverage window 0
                                  # miss anchor_only made on all three -- real aggregate dT/dC
                                  # comparison (not just per-window proxy agreement) needed before
                                  # trusting either fully. Kept as a config option for that A/B test.
    anchor_n_iters_rot: int = 60  # GN iteration budget for the anchor-interpolated candidate when
                                  # multistart_criterion="matched_total" (ignored otherwise).

@dataclass(frozen=True)
class OuterCfg:
    n_trans1:  int   = 30    # Phase 1: translation
    n_rot:     int   = 20    # Phase 2: rotation
    n_trans2:  int   = 20    # Phase 3: translation refinement (30+20+20=70, matches exp41)
    lr_trans:  float = 1e-3  # exp41 used a single lr=1e-3 for all phases
    lr_rot:    float = 1e-3
    beta1:     float = 0.9
    beta2:     float = 0.999
    eps:       float = 1e-8
    warmup_steps: int = 5    # ramp lr up over this many steps after each phase-boundary
                              # reset — at local_t=1, Adam's bias correction makes
                              # m_hat/sqrt(v_hat) collapse to exactly sign(gradient),
                              # i.e. a maximum-magnitude step regardless of gradient
                              # trustworthiness; this softens that blind first step
    rot_loss_boost: float = 1.0  # multiplies loss_r in outer_adam_loop's loss_fn.
                              # loss_t is ~180-200x larger than loss_r at every tested
                              # noise level (unweighted sum), so the rotation phase's
                              # masked gradient is dominated by rotation's effect on
                              # the much bigger translation loss rather than its own
                              # objective. 1.0 = today's unweighted behavior.

@dataclass(frozen=True)
class ExpCfg:
    window:         int   = 100  # matches exp41 default
    overlap:        int   = 10
    sigma_t:        float = 0.03
    sigma_r:        float = 0.01
    local_k:        int   = 3     # neighborhood radius for per-edge precision
    max_kappa_ratio: float = 50.0  # cap on kappa/omega ratio (prevents collapse)
    seeds:          int   = 5

# ---------------------------------------------------------------------------
# Dense Laplacian builders  (no Python loops inside JIT)
# ---------------------------------------------------------------------------

def build_rotation_laplacian(R_meas: jnp.ndarray,
                              kappa: jnp.ndarray,
                              n: int,
                              anchor_idx: jnp.ndarray,
                              kappa_anchor: float) -> jnp.ndarray:
    """Build dense (3n x 3n) GN Hessian for the rotation synchronization subproblem.

    R_meas: (n-1, 3, 3)  per-edge rotation measurements (unused — kept for API)
    kappa:  (n-1,)       per-edge rotation precision
    anchor_idx:   (m,) pose indices with a sparse ground-truth anchor. Index 0
                  is always a no-op here (pose 0 is already hard-fixed, not
                  soft-anchored — see sesync_inner_solve/rotation_gn_ift, which
                  drop it from the free system entirely). Pass e.g. [0] with
                  kappa_anchor=0.0 for "no anchors".
    kappa_anchor: scalar anchor precision. Anchors connect one free pose to a
                  FIXED external target (not another free pose), so — unlike a
                  real chain edge, which connects two free poses — an anchor
                  contributes a diagonal-only term, no off-diagonal block.

    SE-Sync (IJRR §4.1, Eq. 14) derives the connection Laplacian L(G̃ρ) with
    off-diagonal blocks −κᵢⱼ R̃ᵢⱼ from the Frobenius cost ‖Rⱼ − Rᵢ R̃ᵢⱼ‖²_F
    with RIGHT retraction.  This implementation uses the geodesic cost
    ‖so3_log(R̃ᵢⱼᵀ Rᵢᵀ Rⱼ)‖² with LEFT retraction R ← so3_exp(u) @ R.
    Under left retraction the GN Hessian approximation has off-diagonal blocks
    −κᵢⱼ I₃ (standard graph Laplacian).  Both formulations are self-consistent
    GN variants that converge to the same rotation MLE fixed point; for KITTI's
    small per-frame rotations (< 0.1 rad) R̃ᵢⱼ ≈ I₃ so the two Hessians are
    numerically near-identical.

    NOTE: on a pure chain (no anchors, no loop closures) this Hessian's fixed
    point is trivially R_init itself: R_init is built by chain-composing the
    same R_meas this Laplacian measures against, so the chain-only residual
    (and gradient) is exactly zero everywhere, always — confirmed directly on
    real KITTI data (cost_init=0.000000, every GN step rejected). The sparse
    anchor terms below exist specifically to break that: they're external
    information, not derived from R_meas, so they give the solve a real,
    non-trivial objective for the first time since exp42 dropped exp36/38/
    40/41's anchor-constrained inner solve in favor of this pure self-
    consistency formulation.

    Off-diagonal blocks: L[i, i+1] = -kappa[i] * I_3
    Diagonal blocks:     L[i, i]   =  sum of incident kappa * I3, plus
                                      kappa_anchor at anchored poses
    """
    kappa_left  = jnp.concatenate([jnp.zeros(1), kappa])
    kappa_right = jnp.concatenate([kappa, jnp.zeros(1)])
    kappa_diag  = kappa_left + kappa_right
    kappa_diag  = kappa_diag.at[anchor_idx].add(kappa_anchor)
    L_diag = jnp.diag(jnp.repeat(kappa_diag, 3))

    i_idx = jnp.arange(n - 1)
    j_idx = jnp.arange(1, n)
    scaled_I = kappa[:, None, None] * jnp.eye(3)[None]      # (n-1, 3, 3)
    upper_blocks = jnp.zeros((n, n, 3, 3))
    upper_blocks = upper_blocks.at[i_idx, j_idx].set(scaled_I)

    L_upper = upper_blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    return L_diag - L_upper - L_upper.T


def build_translation_laplacian(omega: jnp.ndarray, n: int) -> jnp.ndarray:
    """Build dense (3n x 3n) translation Laplacian.

    omega: (n-1,)  per-edge translation precision
    Off-diagonal blocks are -omega[i] * I3 (no rotation — translations are Euclidean).
    """
    omega_left  = jnp.concatenate([jnp.zeros(1), omega])
    omega_right = jnp.concatenate([omega, jnp.zeros(1)])
    omega_diag  = omega_left + omega_right
    L_diag = jnp.diag(jnp.repeat(omega_diag, 3))

    i_idx = jnp.arange(n - 1)
    j_idx = jnp.arange(1, n)
    scaled_I = omega[:, None, None] * jnp.eye(3)[None]      # (n-1, 3, 3)
    upper_blocks = jnp.zeros((n, n, 3, 3))
    upper_blocks = upper_blocks.at[i_idx, j_idx].set(scaled_I)
    L_upper = upper_blocks.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    return L_diag - L_upper - L_upper.T


# ---------------------------------------------------------------------------
# Rotation GN with IFT custom_vjp
# ---------------------------------------------------------------------------

def _anchor_residual(R: jnp.ndarray, anchor_idx: jnp.ndarray,
                      anchor_targets: jnp.ndarray) -> jnp.ndarray:
    """r_anchor[a] = so3_log(anchor_targets[a]^T @ R[anchor_idx[a]]).

    Same so3_log(X^T Y) shape as a chain edge's residual, with X=anchor_targets[a]
    FIXED (external ground truth, not derived from R_meas) and Y=R[anchor_idx[a]]
    the only perturbable quantity — structurally identical to a real edge's
    free/"second-pose" side. See build_rotation_laplacian's docstring for why
    this is what breaks the chain-only formulation's trivial zero-residual trap.
    """
    R_anchored = R[anchor_idx]                                # (m, 3, 3)
    return jax.vmap(
        lambda Rt, Ra: so3_log(Rt.T @ Ra)
    )(anchor_targets, R_anchored)                              # (m, 3)


def _rotation_cost(R: jnp.ndarray, R_meas: jnp.ndarray, kappa: jnp.ndarray,
                    anchor_idx: jnp.ndarray, anchor_targets: jnp.ndarray,
                    kappa_anchor: float) -> jnp.ndarray:
    """Scalar GN objective: sum_i kappa_i * ||so3_log(R_meas_i^T R_i^T R_{i+1})||^2
    + kappa_anchor * sum_a ||so3_log(anchor_targets[a]^T R[anchor_idx[a]])||^2.

    Used by the accept/reject logic in _rotation_gn_raw — a candidate step is
    only accepted if it actually reduces this cost (Levenberg-Marquardt style).
    The anchor term must be included here (not just in the gradient/Hessian):
    otherwise accept/reject would decouple from the objective the gradient and
    Hessian are actually solving.
    """
    r = jax.vmap(
        lambda Ri, Rj, Rij: so3_log(Rij.T @ Ri.T @ Rj)
    )(R[:-1], R[1:], R_meas)
    cost_chain = jnp.sum(kappa * jnp.sum(r ** 2, axis=-1))
    r_anchor = _anchor_residual(R, anchor_idx, anchor_targets)
    cost_anchor = kappa_anchor * jnp.sum(r_anchor ** 2)
    return cost_chain + cost_anchor


def _rotation_gn_candidate_step(R: jnp.ndarray,
                                 R_meas: jnp.ndarray,
                                 kappa: jnp.ndarray,
                                 anchor_idx: jnp.ndarray,
                                 anchor_targets: jnp.ndarray,
                                 kappa_anchor: float,
                                 n: int,
                                 damping: float) -> jnp.ndarray:
    """One candidate GN step on SO(3)^n for rotation synchronization.

    Solves: (L_free + damping*I) delta = -g_free
    then retracts: R[j] <- so3_exp(delta[j]) @ R[j]

    All edge residuals computed in parallel via vmap.
    Gradient accumulated via vectorized scatter (.at[].add).
    System solved via jnp.linalg.solve on the dense (3(n-1) x 3(n-1)) matrix.

    Called "candidate" because _rotation_gn_raw accepts/rejects the result
    based on whether it actually reduces _rotation_cost (see below) — this
    function itself is unconditional, matching the pre-accept/reject behavior.
    """
    # Rotation residuals for all edges simultaneously — vmap, no loop
    r = jax.vmap(
        lambda Ri, Rj, Rij: so3_log(Rij.T @ Ri.T @ Rj)
    )(R[:-1], R[1:], R_meas)                                 # (n-1, 3)

    # Gradient: g[j] = kappa[j-1]*r[j-1] - kappa[j]*r[j]
    # Accumulated via scatter — no Python loop
    kappa_r = kappa[:, None] * r                             # (n-1, 3)
    g_all = jnp.zeros((n, 3))
    g_all = g_all.at[1:].add(kappa_r)                       # pose j+1 from edge j
    g_all = g_all.at[:-1].add(-kappa_r)                     # pose j from edge j

    # Anchor gradient contribution: +kappa_anchor * r_anchor[a] at pose anchor_idx[a]
    # (same sign as a chain edge's free/"second-pose" side — see _anchor_residual).
    r_anchor = _anchor_residual(R, anchor_idx, anchor_targets)   # (m, 3)
    g_all = g_all.at[anchor_idx].add(kappa_anchor * r_anchor)

    g_free = g_all[1:].reshape(-1)                          # (3(n-1),)

    # Rotation GN Hessian (dense) — standard graph Laplacian for left-retraction GN,
    # plus sparse anchor terms (see build_rotation_laplacian's docstring)
    L = build_rotation_laplacian(R_meas, kappa, n, anchor_idx, kappa_anchor)
    L_free = L[3:, 3:] + damping * jnp.eye(3 * (n - 1))   # (3(n-1), 3(n-1))

    # Dense solve — cuBLAS LU on GPU
    delta_free = jnp.linalg.solve(L_free, -g_free)          # (3(n-1),)
    delta_all  = jnp.concatenate([jnp.zeros(3), delta_free]).reshape(n, 3)  # (n, 3)

    # Retract on SO(3) for all poses — vmap, no loop
    R_new = jax.vmap(lambda Ri, di: so3_exp(di) @ Ri)(R, delta_all)
    return R_new


def _rotation_gn_raw(R_init: jnp.ndarray,
                      R_meas: jnp.ndarray,
                      kappa: jnp.ndarray,
                      anchor_idx: jnp.ndarray,
                      anchor_targets: jnp.ndarray,
                      kappa_anchor: float,
                      n_iters: int,
                      damping_init: float,
                      damping_min: float,
                      damping_max: float,
                      damping_down: float,
                      damping_up: float):
    """Self-checking (Levenberg-Marquardt style) GN via jax.lax.scan.

    Fixed iteration count — JIT-compilable, no true early exit. Instead, each
    candidate step is accepted only if it reduces the GN cost; on accept,
    damping relaxes (bigger, more confident steps next time); on reject, the
    pose stays put and damping tightens (smaller, more cautious steps).

    Returns (R_star, damping_used) where damping_used is the damping value
    that actually produced R_star (i.e. the damping active at the LAST
    ACCEPTED step), NOT whatever damping the scan's running trust-region
    state ends up at. These differ whenever R_star converges before n_iters
    is exhausted: subsequent steps keep getting rejected (no more room to
    improve) and ratchet the running damping up every remaining iteration,
    with no bound tied to what actually shaped R_star. Using that inflated,
    post-convergence value in the IFT backward pass over-regularizes the
    backward solve and silently corrupts the gradient — confirmed via a
    finite-difference check (cosine similarity vs analytic dropped to ~0.55
    when using the raw scan-final damping, vs ~0.94 for the original
    fixed-damping code). damping_used tracks the damping at the moment of
    the last accept, which is the value IFT actually needs.
    """
    n = R_init.shape[0]
    cost_init = _rotation_cost(R_init, R_meas, kappa, anchor_idx, anchor_targets, kappa_anchor)

    def step(carry, _):
        R, damping, cost, damping_used = carry
        R_cand = _rotation_gn_candidate_step(R, R_meas, kappa, anchor_idx, anchor_targets,
                                              kappa_anchor, n, damping)
        cost_cand = _rotation_cost(R_cand, R_meas, kappa, anchor_idx, anchor_targets, kappa_anchor)
        accept = cost_cand < cost

        R_new = jnp.where(accept, R_cand, R)
        # Record the damping that produced R_new whenever a step is accepted —
        # this is what the backward pass must use, not the post-update damping.
        damping_used_new = jnp.where(accept, damping, damping_used)
        damping_new = jnp.where(
            accept,
            jnp.maximum(damping * damping_down, damping_min),
            jnp.minimum(damping * damping_up, damping_max),
        )
        cost_new = jnp.where(accept, cost_cand, cost)
        return (R_new, damping_new, cost_new, damping_used_new), None

    (R_star, _, _, damping_used), _ = jax.lax.scan(
        step, (R_init, damping_init, cost_init, damping_init), None, length=n_iters
    )
    return R_star, damping_used


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def rotation_gn_ift(R_init: jnp.ndarray,
                    R_meas: jnp.ndarray,
                    kappa: jnp.ndarray,
                    anchor_idx: jnp.ndarray,
                    anchor_targets: jnp.ndarray,
                    kappa_anchor: float,
                    n_iters: int,
                    damping_init: float,
                    damping_min: float,
                    damping_max: float,
                    damping_down: float,
                    damping_up: float) -> jnp.ndarray:
    """Rotation GN with IFT shortcut backward.

    Forward: run fixed-count self-checking GN, return R_star.
    Backward: IFT — one 3(n-1) x 3(n-1) solve instead of unrolling N iters.

    anchor_idx/anchor_targets/kappa_anchor: sparse ground-truth rotation
    anchors (see build_rotation_laplacian's docstring for why these exist —
    without them this solve is provably a no-op on a pure chain graph).

    n_iters is nondiff_argnums=(6,): jax.lax.scan's `length` inside
    _rotation_gn_raw requires a concrete Python int. Without declaring this,
    custom_vjp's abstract-eval path (used whenever this is traced under jit
    without an enclosing jax.grad -- e.g. a bare forward-only inner-solve
    call) abstracts EVERY positional arg uniformly, turning n_iters into a
    traced array and crashing scan's length check. This previously worked
    only by the accident that every existing call site routes through
    jax.grad nested inside jax.lax.scan (outer_adam_loop's adam_step) --
    confirmed by direct repro: identical calls fail under plain jit (no
    grad) or jit+grad without a scan, and succeed only in that one exact
    combination. nondiff_argnums makes it work unconditionally.
    """
    R_star, _ = _rotation_gn_raw(R_init, R_meas, kappa, anchor_idx, anchor_targets,
                                  kappa_anchor, n_iters,
                                  damping_init, damping_min, damping_max,
                                  damping_down, damping_up)
    return R_star


def _rotation_gn_ift_fwd(R_init, R_meas, kappa, anchor_idx, anchor_targets, kappa_anchor,
                          n_iters, damping_init, damping_min, damping_max, damping_down, damping_up):
    R_star, damping_used = _rotation_gn_raw(R_init, R_meas, kappa, anchor_idx, anchor_targets,
                                             kappa_anchor, n_iters,
                                             damping_init, damping_min, damping_max,
                                             damping_down, damping_up)
    # NOTE: pack damping_used (the damping active at the LAST ACCEPTED step),
    # not damping_init and not whatever the scan's running trust-region state
    # ends at — see _rotation_gn_raw's docstring for why these three differ
    # and why using the wrong one silently corrupts the backward gradient.
    return R_star, (R_star, R_meas, kappa, anchor_idx, anchor_targets, kappa_anchor, damping_used)


def _rotation_gn_ift_bwd(n_iters, res, g_R_star):
    """IFT backward pass for rotation GN.

    n_iters is the nondiff_argnums=(6,) value (unused here -- the forward
    solve already ran with it; it's only in this signature because
    nondiff_argnums passes nondiff values as bwd's leading positional args).

    Given upstream gradient g_R_star (n, 3, 3) w.r.t. R_star:
    1. Project to tangent space: g_free (3(n-1),) — skip anchor
    2. IFT: solve L_free @ v = g_free  (one dense solve, not N unrolled steps)
    3. Propagate v back through the gradient function to get dL/d(R_meas, kappa)

    Uses damping_used (the damping active at the LAST ACCEPTED GN step —
    i.e. the damping that actually produced R_star), NOT the solver's initial
    damping and NOT the scan's final running trust-region state (which can be
    arbitrarily inflated by trailing rejected steps after convergence — see
    _rotation_gn_raw's docstring). This backward pass already linearizes
    around the damped Hessian used in the step that produced R_star (that's
    why damping*eye appears here at all), so using any other damping value
    would linearize around a Hessian inconsistent with the one that actually
    produced R_star: a silent, non-crashing, wrong-gradient bug — confirmed
    via finite-difference check (see _rotation_gn_raw docstring for numbers).

    L is rebuilt here WITH the same anchor_idx/kappa_anchor the forward solve
    used — the backward solve must linearize around the exact Hessian that
    produced R_star, same principle as damping_used above. gradient_fn (used
    for the R_meas/kappa VJP below) needs no anchor term: the anchor residual
    has zero dependence on R_meas or kappa (only on R_star, the fixed anchor
    target, and the fixed kappa_anchor hyperparameter), so d(g_total)/d(R_meas,
    kappa) = d(g_chain)/d(R_meas, kappa) exactly — the anchor's effect on this
    backward pass flows entirely through v_free (solved against the anchor-
    augmented L_free), not through any extra term in gradient_fn.
    """
    R_star, R_meas, kappa, anchor_idx, anchor_targets, kappa_anchor, damping_used = res
    n = R_star.shape[0]

    # Project upstream gradient to axis-angle increments on SO(3).
    # Under left retraction R ← exp(u) @ R, the tangent component is:
    #   g_u = vee( (R^T g - g^T R) / 2 )  =  vee(skew_part(R^T g))
    def project_grad(Ri, gi):
        skew = Ri.T @ gi - gi.T @ Ri          # = 2 * skew_part(R^T g)
        return jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2.0

    g_tangent = jax.vmap(project_grad)(R_star, g_R_star)    # (n, 3)
    g_free = g_tangent[1:].reshape(-1)                       # (3(n-1),) — drop anchor

    # IFT: solve L_free @ v = g_free  (same anchor-augmented Hessian as forward GN, damping_used)
    L = build_rotation_laplacian(R_meas, kappa, n, anchor_idx, kappa_anchor)
    L_free = L[3:, 3:] + damping_used * jnp.eye(3 * (n - 1))
    v_free = jnp.linalg.solve(L_free, g_free)               # (3(n-1),)
    v_all = jnp.concatenate([jnp.zeros(3), v_free]).reshape(n, 3)  # (n, 3)

    # Propagate v through gradient function: dL/d(R_meas, kappa)
    # The GN gradient g(R_meas, kappa) = Jt(R_star) @ r(R_star; R_meas, kappa)
    # We need: v^T @ d_g/d(R_meas, kappa) evaluated at R_star
    # (no anchor term here — see this function's docstring for why)
    def gradient_fn(R_meas_, kappa_):
        r = jax.vmap(
            lambda Ri, Rj, Rij: so3_log(Rij.T @ Ri.T @ Rj)
        )(R_star[:-1], R_star[1:], R_meas_)
        kappa_r = kappa_[:, None] * r
        g = jnp.zeros((n, 3))
        g = g.at[1:].add(kappa_r)
        g = g.at[:-1].add(-kappa_r)
        return g[1:].reshape(-1)

    # vjp: dL/d(R_meas, kappa) = -v^T @ Jacobian of gradient_fn
    # (negative because IFT: delta = solve(H, -g) → dL/dtheta = -v^T @ dg/dtheta)
    _, grad_params = jax.vjp(gradient_fn, R_meas, kappa)
    g_R_meas, g_kappa = grad_params(-v_free)

    # Sensitivity to R_init decays exponentially with n_iters — zero is standard
    g_R_init = jnp.zeros_like(R_star)
    # anchor_targets: fixed GT-derived values, same treatment as R_init above.
    # anchor_idx (index array) and kappa_anchor (fixed hyperparameter, not
    # learned in this pass — see build_rotation_laplacian's docstring) get None.
    g_anchor_targets = jnp.zeros_like(anchor_targets)
    # 12 primal args, one (n_iters, index 6) is nondiff_argnums -> 11-tuple
    # cotangents for the remaining 11: 4 real grads + 7 None (anchor_idx,
    # kappa_anchor, 5 damping-bound scalars). n_iters gets NO entry at all
    # (not even None) since nondiff_argnums positions are excluded entirely.
    return (g_R_init, g_R_meas, g_kappa, None, g_anchor_targets, None,
            None, None, None, None, None)


rotation_gn_ift.defvjp(_rotation_gn_ift_fwd, _rotation_gn_ift_bwd)


# ---------------------------------------------------------------------------
# Analytic translation recovery
# ---------------------------------------------------------------------------

def recover_translations(R_star: jnp.ndarray,
                          t_meas: jnp.ndarray,
                          omega: jnp.ndarray,
                          n: int) -> jnp.ndarray:
    """Recover translations analytically given converged R_star.

    Solves: L_t_free @ t_free = b_free
    where b_free accumulates omega[i] * R_star[i] @ t_meas[i] per edge.

    Uses jnp.linalg.solve (dense, cuBLAS on GPU).
    JAX's built-in VJP for linalg.solve handles the backward pass automatically.

    Anchor: t[0] = 0  (window anchor pose fixed at origin)
    """
    # RHS contributions per free pose (j=1..n-1)
    # b[j] = omega[j-1] * R*[j-1] @ t_meas[j-1]  (from left edge)
    #      - omega[j]   * R*[j]   @ t_meas[j]     (from right edge, j < n-1)
    rhs_left = jax.vmap(
        lambda R, t, w: w * R @ t
    )(R_star[:-1], t_meas, omega)                            # (n-1, 3)

    rhs_right_inner = jax.vmap(
        lambda R, t, w: w * R @ t
    )(R_star[1:-1], t_meas[1:], omega[1:])                  # (n-2, 3)
    rhs_right = jnp.concatenate(
        [rhs_right_inner, jnp.zeros((1, 3))], axis=0
    )                                                         # (n-1, 3)

    b_free = (rhs_left - rhs_right).reshape(-1)             # (3(n-1),)

    # Dense translation Laplacian (free block)
    L_t = build_translation_laplacian(omega, n)
    L_t_free = L_t[3:, 3:]                                  # (3(n-1), 3(n-1))

    t_free = jnp.linalg.solve(L_t_free, b_free)             # (3(n-1),)
    t_star = jnp.concatenate([jnp.zeros(3), t_free]).reshape(n, 3)
    return t_star


# ---------------------------------------------------------------------------
# Multi-start GN: anchor-interpolated alternate R_init
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _anchor_interp_tables(n: int, anchor_spacing: int):
    """Static left/right anchor-slot index + SLERP alpha tables, one entry per
    pose 0..n-1. Slot 0 is the implicit identity anchor at pose 0; slots
    1..m are anchor_idx's targets in order (see sesync_inner_solve's
    anchor_idx construction, which this mirrors exactly in plain
    numpy/Python instead of jnp -- n and anchor_spacing are always concrete
    Python ints closed over by build_denoiser's jax.jit, never traced, so
    this table is a trace-time constant, cached per (n, anchor_spacing).
    """
    # Returns plain NumPy arrays, NOT jnp arrays: this is cached across calls
    # (and across many independent JAX traces -- one per window/seed dispatch,
    # plus retraces inside outer_adam_loop's jax.grad/lax.scan). A jnp array
    # created the first time this runs while already inside an active trace
    # would get bound to that trace's now-dead context and leak as a stale
    # DynamicJaxprTracer on every later call -- confirmed directly: caching
    # jnp.array(...) here crashed outer_adam_loop with UnexpectedTracerError
    # ("intermediate value... allowed to escape the scope of the
    # transformation"). NumPy arrays are host-side/trace-agnostic, so they're
    # safe to cache and reuse in any trace; _build_anchor_interpolated_R_init
    # converts them to jnp fresh on every call instead.
    anchor_positions = list(range(anchor_spacing, n - 1, anchor_spacing)) + [n - 1]
    positions = np.array([0] + anchor_positions, dtype=np.int64)  # strictly ascending by construction

    js = np.arange(n)
    seg = np.clip(np.searchsorted(positions, js, side="right") - 1, 0, len(positions) - 1)
    seg_right = np.clip(seg + 1, 0, len(positions) - 1)
    lo, hi = positions[seg], positions[seg_right]
    denom = np.where(hi == lo, 1, hi - lo)
    alpha = np.where(hi == lo, 0.0, (js - lo) / denom).astype(np.float32)

    return seg.astype(np.int32), seg_right.astype(np.int32), alpha


def _build_anchor_interpolated_R_init(n: int, anchor_spacing: int,
                                       anchor_targets: jnp.ndarray) -> jnp.ndarray:
    """Alternate R_init candidate for multi-start GN (see InnerCfg.n_starts):
    piecewise-geodesic SLERP through the sparse GT anchors (pose 0 implicitly
    anchored at identity, plus anchor_idx/anchor_targets), instead of
    chain-composing the raw noisy R_meas.

    Unlike the chain-composed R_init, this candidate has zero dependence on
    noisy measurements, so it cannot inherit whatever adversarial noise-draw
    direction biases the chain-composed candidate toward a bad GN basin for a
    given (seq, seed) pair.
    """
    left_idx_np, right_idx_np, alpha_np = _anchor_interp_tables(n, anchor_spacing)
    left_idx, right_idx, alpha = jnp.array(left_idx_np), jnp.array(right_idx_np), jnp.array(alpha_np)
    anchor_targets_full = jnp.concatenate([jnp.eye(3)[None], anchor_targets], axis=0)  # (m+1, 3, 3)
    R_lo = anchor_targets_full[left_idx]   # (n, 3, 3)
    R_hi = anchor_targets_full[right_idx]  # (n, 3, 3)

    def slerp_one(Rlo, Rhi, a):
        rel = so3_log(Rlo.T @ Rhi)
        return Rlo @ so3_exp(a * rel)

    return jax.vmap(slerp_one)(R_lo, R_hi, alpha)


# ---------------------------------------------------------------------------
# Full inner solver
# ---------------------------------------------------------------------------

def sesync_inner_solve(theta: jnp.ndarray,
                        noisy_odom: jnp.ndarray,
                        kappa: jnp.ndarray,
                        omega: jnp.ndarray,
                        n: int,
                        cfg: InnerCfg,
                        gt_R_rel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SE-Sync structured inner solve.

    theta:      (n-1, 6) learned corrections to odometry
    noisy_odom: (n-1, 6) noisy odometry measurements [t(3) | w(3)]
    kappa:      (n-1,) per-edge rotation precision
    omega:      (n-1,) per-edge translation precision
    gt_R_rel:   (n, 3, 3) window-relative ground-truth rotations, used ONLY to
                slice out sparse anchor targets (see cfg.anchor_spacing) —
                this is the same gt_R_rel outer_adam_loop already computes for
                the outer loss; anchors are just a new consumer of already-
                available data, not a new dependency.
    Returns: (R_star (n,3,3), t_star (n,3))
    """
    corrected = noisy_odom + theta                           # (n-1, 6)
    t_meas = corrected[:, :3]                               # (n-1, 3)
    w_meas = corrected[:, 3:]                               # (n-1, 3) axis-angle

    # Rotation measurements: so3_exp of corrected rotation part
    R_meas = jax.vmap(so3_exp)(w_meas)                      # (n-1, 3, 3)

    # Build R_init by integrating the current corrected rotation chain.
    # Recomputing at every outer step keeps R_init consistent with R_meas —
    # critical for n=100 where 10 GN iterations are insufficient to recover
    # from a stale initialization built from the original noisy odometry.
    # The custom_vjp zeros g_R_init, so this scan is never differentiated.
    _, R_traj = jax.lax.scan(
        lambda R, Rm: (R @ Rm, R @ Rm), jnp.eye(3), R_meas
    )                                                        # (n-1, 3, 3)
    R_init = jnp.concatenate([jnp.eye(3)[None], R_traj], axis=0)  # (n, 3, 3)

    # Sparse GT rotation anchors — see InnerCfg.anchor_spacing's docstring for
    # why these exist (without them this solve is provably a no-op). Index 0
    # is always a no-op (pose 0 already hard-fixed), so spacing starts at
    # cfg.anchor_spacing, not 0; the final pose is always included too.
    anchor_idx = jnp.concatenate([
        jnp.arange(cfg.anchor_spacing, n - 1, cfg.anchor_spacing, dtype=jnp.int32),
        jnp.array([n - 1], dtype=jnp.int32),
    ])
    anchor_targets = gt_R_rel[anchor_idx]                    # (m, 3, 3)

    # Rotation GN with IFT backward (3(n-1) x 3(n-1) system)
    R_star_chain = rotation_gn_ift(R_init, R_meas, kappa, anchor_idx, anchor_targets, cfg.kappa_anchor,
                                    cfg.n_iters_rot,
                                    cfg.damping_init, cfg.damping_min, cfg.damping_max,
                                    cfg.damping_down, cfg.damping_up)

    if cfg.n_starts <= 1:
        R_star = R_star_chain
    else:
        # Multi-start GN (see InnerCfg.n_starts): also solve from an
        # anchor-interpolated R_init and keep whichever candidate wins under
        # cfg.multistart_criterion (see InnerCfg's docstring for the
        # evidence behind each option and why neither is unconditionally
        # better -- this is a live A/B config, not a settled choice yet).
        # cfg.n_starts and cfg.multistart_criterion are static Python values
        # (InnerCfg fields are closed over at jax.jit trace time, never
        # traced), so these branches are resolved once at trace time, same
        # as every other cfg-based branch in this module -- n_starts=1
        # recompiles to the exact single-start code path above,
        # byte-identical.
        R_init_anchor = _build_anchor_interpolated_R_init(n, cfg.anchor_spacing, anchor_targets)

        if cfg.multistart_criterion == "matched_total":
            R_star_anchor = rotation_gn_ift(R_init_anchor, R_meas, kappa, anchor_idx, anchor_targets,
                                             cfg.kappa_anchor, cfg.anchor_n_iters_rot,
                                             cfg.damping_init, cfg.damping_min, cfg.damping_max,
                                             cfg.damping_down, cfg.damping_up)
            cost_chain = _rotation_cost(R_star_chain, R_meas, kappa, anchor_idx, anchor_targets, cfg.kappa_anchor)
            cost_anchor = _rotation_cost(R_star_anchor, R_meas, kappa, anchor_idx, anchor_targets, cfg.kappa_anchor)
        else:  # "anchor_only" (default)
            R_star_anchor = rotation_gn_ift(R_init_anchor, R_meas, kappa, anchor_idx, anchor_targets,
                                             cfg.kappa_anchor, cfg.n_iters_rot,
                                             cfg.damping_init, cfg.damping_min, cfg.damping_max,
                                             cfg.damping_down, cfg.damping_up)

            def anchor_only_cost(R):
                r_anchor = _anchor_residual(R, anchor_idx, anchor_targets)
                return cfg.kappa_anchor * jnp.sum(r_anchor ** 2)

            cost_chain = anchor_only_cost(R_star_chain)
            cost_anchor = anchor_only_cost(R_star_anchor)

        R_star = jnp.where(cost_anchor < cost_chain, R_star_anchor, R_star_chain)

    # Analytic translation recovery (one dense solve, auto-diff backward)
    t_star = recover_translations(R_star, t_meas, omega, n)

    return R_star, t_star


# ---------------------------------------------------------------------------
# Per-edge precision from local windowed variance
# ---------------------------------------------------------------------------

def compute_per_edge_precision(diffs: jnp.ndarray,
                                sigma_noise: float,
                                k: int,
                                max_ratio: float) -> jnp.ndarray:
    """Compute per-edge precision from local windowed variance.

    diffs: (n_edges, d) finite differences of measurements
    sigma_noise: scalar MAD-based noise floor
    k: neighborhood radius (window = 2k+1 edges)

    For each edge i, variance estimated from edges i-k..i+k.
    sigma_process[i]^2 = max(0, local_var[i] - 2*sigma_noise^2)
    precision[i] = 1 / (sigma_noise^2 + sigma_process[i]^2)

    Uses masked vmap — O(n_edges^2) but n_edges <= 49 for window=50, negligible.
    """
    n_edges = diffs.shape[0]
    diffs_sq = jnp.sum(diffs ** 2, axis=-1)                  # (n_edges,) scalar per edge
    indices = jnp.arange(n_edges)

    def local_var_at(i):
        mask = (indices >= i - k) & (indices <= i + k)
        count = jnp.sum(mask).astype(jnp.float32) + 1e-8
        masked = jnp.where(mask, diffs_sq, 0.0)
        mean = jnp.sum(masked) / count
        var  = jnp.sum(jnp.where(mask, (diffs_sq - mean) ** 2, 0.0)) / count
        return var

    local_vars = jax.vmap(local_var_at)(indices)             # (n_edges,)
    sigma_process_sq = jnp.maximum(0.0, local_vars - 2.0 * sigma_noise ** 2)
    precision = 1.0 / (sigma_noise ** 2 + sigma_process_sq + 1e-8)

    # Cap ratio to prevent collapse (replaces global max_sw_rw_ratio from exp41)
    min_prec = jnp.min(precision)
    precision = jnp.minimum(precision, min_prec * max_ratio)
    return precision


# Pure-JAX pooled sigma+precision — no numpy, no D2H copy.
# Replaces: estimate_noise_mad(np.array(...)) + compute_per_edge_precision(...)
#
# Pools diffs ACROSS ALL S SEEDS before estimating per-edge precision, instead
# of computing an independent (and much noisier) estimate per seed. Valid only
# because all S seeds in this evaluation harness are independent noise draws
# on the SAME ground-truth trajectory (confirmed via _add_kitti_noise's
# seed*1000+seq_hash RNG scheme) — this does NOT apply to the synthetic path
# (make_synthetic_sequence draws a different GT per seed), so this function
# must only be called from the KITTI/denoise_sequence_pooled path, never from
# compute_per_edge_precision/denoise_sequence.
def _sigma_and_precision(diffs, local_k, max_kappa_ratio):
    """(S, n_edges, d) -> (n_edges,) precision, POOLED and SHARED across all S seeds."""
    S = diffs.shape[0]
    n_edges = diffs.shape[1]

    flat = diffs.reshape(-1, diffs.shape[-1])                      # (S*n_edges, d)
    med = jnp.median(flat, axis=0)
    sigma_noise = jnp.median(jnp.abs(flat - med)) * 1.4826 + 1e-8  # scalar, pooled

    diffs_sq = jnp.sum(diffs ** 2, axis=-1)                         # (S, n_edges)
    indices  = jnp.arange(n_edges)

    def local_var_at(i):
        mask    = (indices >= i - local_k) & (indices <= i + local_k)   # (n_edges,)
        n_mask  = jnp.sum(mask).astype(jnp.float32)
        count   = n_mask * S + 1e-8
        masked  = jnp.where(mask[None, :], diffs_sq, 0.0)               # (S, n_edges)
        mean    = jnp.sum(masked) / count
        return jnp.sum(jnp.where(mask[None, :], (diffs_sq - mean) ** 2, 0.0)) / count

    local_vars       = jax.vmap(local_var_at)(indices)              # (n_edges,)
    sigma_process_sq = jnp.maximum(0.0, local_vars - 2.0 * sigma_noise ** 2)
    precision        = 1.0 / (sigma_noise ** 2 + sigma_process_sq + 1e-8)
    min_prec         = jnp.min(precision)
    return jnp.minimum(precision, min_prec * max_kappa_ratio)


# ---------------------------------------------------------------------------
# Outer 3-phase Adam loop  (fused single fori_loop with phase masks)
# ---------------------------------------------------------------------------

def outer_adam_loop(theta_init: jnp.ndarray,
                    noisy_odom: jnp.ndarray,
                    gt_poses: jnp.ndarray,
                    gt_R_direct: jnp.ndarray,
                    kappa: jnp.ndarray,
                    omega: jnp.ndarray,
                    n: int,
                    inner_cfg: InnerCfg,
                    outer_cfg: OuterCfg) -> jnp.ndarray:
    """3-phase outer Adam loop over theta.

    Phase 1 (steps 0..N_T1-1):       update theta[:, :3] (translation corrections)
    Phase 2 (steps N_T1..N_T1+NR-1): update theta[:, 3:] (rotation corrections)
    Phase 3 (steps N_T1+NR..end):    update theta[:, :3] (translation refinement)

    All phases fused into one fori_loop via jnp.where masks.
    Adam state (m, v) reset at phase boundaries via jnp.where.
    Phase-local iteration count for bias correction.
    """
    N_T1  = outer_cfg.n_trans1
    N_R   = outer_cfg.n_rot
    total = N_T1 + N_R + outer_cfg.n_trans2

    # GT poses in window-relative frame (anchor = first pose of window).
    # t_star is anchored at origin with identity first rotation, so gt must match.
    gt_R_world = gt_R_direct                                    # (n, 3, 3) world frame — raw, bypasses so3_log singularity at θ→π
    gt_R0      = gt_R_world[0]                                  # first pose rotation
    gt_t0      = gt_poses[0, :3]                                # first pose translation
    # Rotate and translate GT into window-local frame
    gt_t_rel   = jax.vmap(lambda t: gt_R0.T @ (t - gt_t0))(gt_poses[:, :3])   # (n, 3)
    gt_R_rel   = jax.vmap(lambda R: gt_R0.T @ R)(gt_R_world)                   # (n, 3, 3)

    # Loss: ATE on translations + Frobenius rotation error.
    # Frobenius ||Ra - Rb||_F^2 has no singularity at any rotation angle — safe
    # for urban sequences where the window-accumulated rotation can exceed π/2,
    # where so3_log(Ra.T @ Rb) would become ill-conditioned if Adam drifts Ra
    # away from Rb. Both Ra (R_star) and Rb (gt_R_rel) are in window-relative
    # frame, so the Frobenius gradient 2*(Ra - Rb) gives the correct denoising
    # direction when Ra ≈ Rb (within noise level).
    #
    # loss_r is scaled by outer_cfg.rot_loss_boost before summing with loss_t.
    # Measured loss_t is ~180-200x larger than loss_r at every tested noise
    # level, so an unweighted sum lets the rotation phase's masked gradient be
    # dominated by rotation's effect on the (much bigger) translation loss
    # rather than its own objective — rot_loss_boost=1.0 reproduces that
    # original unweighted behavior exactly.
    def loss_fn(theta):
        R_star, t_star = sesync_inner_solve(
            theta, noisy_odom, kappa, omega, n, inner_cfg, gt_R_rel
        )
        loss_t = jnp.mean(jnp.sum((t_star - gt_t_rel) ** 2, axis=-1))
        loss_r = jnp.mean(jax.vmap(
            lambda Ra, Rb: jnp.sum((Ra - Rb) ** 2)
        )(R_star, gt_R_rel))
        return loss_t + outer_cfg.rot_loss_boost * loss_r

    grad_fn = jax.grad(loss_fn)

    # Adam state
    m_init = jnp.zeros_like(theta_init)
    v_init = jnp.zeros_like(theta_init)

    def adam_step(carry, step_idx):
        theta, m, v = carry

        # Phase detection
        in_rot   = (step_idx >= N_T1) & (step_idx < N_T1 + N_R)
        at_bound = (step_idx == N_T1) | (step_idx == N_T1 + N_R)

        # Reset Adam state at phase boundaries
        m = jnp.where(at_bound, jnp.zeros_like(m), m)
        v = jnp.where(at_bound, jnp.zeros_like(v), v)

        # Phase-local iteration for bias correction
        local_t = jnp.where(in_rot,
                             step_idx - N_T1 + 1,
                             jnp.where(step_idx >= N_T1 + N_R,
                                       step_idx - N_T1 - N_R + 1,
                                       step_idx + 1))

        # Learning rate per phase, ramped up over warmup_steps at the start of
        # each phase — local_t already resets to 1 at every phase boundary, so
        # the ramp re-triggers automatically at all three transitions. Softens
        # the maximum-magnitude blind first step described in OuterCfg above.
        lr_base = jnp.where(in_rot, outer_cfg.lr_rot, outer_cfg.lr_trans)
        lr_scale = jnp.minimum(1.0, local_t.astype(jnp.float32) / float(outer_cfg.warmup_steps))
        lr = lr_base * lr_scale

        g = grad_fn(theta)

        # Mask gradient to active component
        mask_t = jnp.concatenate([jnp.ones((theta.shape[0], 3)),
                                   jnp.zeros((theta.shape[0], 3))], axis=-1)
        mask_r = 1.0 - mask_t
        g_masked = jnp.where(in_rot, g * mask_r, g * mask_t)

        # Adam update
        m_new = outer_cfg.beta1 * m + (1.0 - outer_cfg.beta1) * g_masked
        v_new = outer_cfg.beta2 * v + (1.0 - outer_cfg.beta2) * g_masked ** 2

        m_hat = m_new / (1.0 - outer_cfg.beta1 ** local_t)
        v_hat = v_new / (1.0 - outer_cfg.beta2 ** local_t)

        theta_new = theta - lr * m_hat / (jnp.sqrt(v_hat) + outer_cfg.eps)
        return (theta_new, m_new, v_new), None

    (theta_opt, _, _), _ = jax.lax.scan(
        adam_step,
        (theta_init, m_init, v_init),
        jnp.arange(total)
    )
    return theta_opt


# ---------------------------------------------------------------------------
# Build the JIT-compiled denoiser for a fixed window size
# ---------------------------------------------------------------------------

def build_denoiser(n: int,
                   inner_cfg: InnerCfg,
                   outer_cfg: OuterCfg,
                   exp_cfg: ExpCfg):
    """Build and JIT-compile the full denoiser for window size n.

    Compiled once, reused across all windows and sequences.
    Dynamic arguments (noisy_odom, gt_poses, kappa, omega) passed at call time.
    Static arguments (n, cfgs) closed over — no retrace across sequences.
    """
    def denoise(noisy_odom, gt_poses, gt_R_world, kappa, omega):
        theta_init = jnp.zeros_like(noisy_odom)
        theta_opt = outer_adam_loop(
            theta_init, noisy_odom, gt_poses, gt_R_world,
            kappa, omega, n, inner_cfg, outer_cfg
        )

        # outer_adam_loop's loss is computed on R_star/t_star (the anchored,
        # GN-refined solve), NOT on theta_opt itself -- theta is one of many
        # additive corrections that can produce a similar R_star/t_star once
        # passed back through that non-injective solve (anchor-averaging,
        # global translation recovery), so returning theta_opt as-is discards
        # the very quantity the optimization targeted and lets noisy_odom +
        # theta_opt diverge arbitrarily far from a sane relative pose even
        # while R_star/t_star (and the loss) look fine. Reconstruct the
        # corrected relative pose FROM the final R_star/t_star instead --
        # same reconstruction exp44_inner_solver_only.py's
        # build_inner_solver_only_denoiser already uses.
        gt_R0 = gt_R_world[0]
        gt_R_rel = jax.vmap(lambda R: gt_R0.T @ R)(gt_R_world)
        R_star, t_star = sesync_inner_solve(theta_opt, noisy_odom, kappa, omega, n, inner_cfg, gt_R_rel)
        dR = jax.vmap(lambda Ri, Rj: Ri.T @ Rj)(R_star[:-1], R_star[1:])
        dw = jax.vmap(so3_log)(dR)
        dt = jax.vmap(lambda Ri, ti, tj: Ri.T @ (tj - ti))(R_star[:-1], t_star[:-1], t_star[1:])
        corrected = jnp.concatenate([dt, dw], axis=-1)
        return corrected - noisy_odom

    return jax.jit(denoise)


# ---------------------------------------------------------------------------
# Noise estimation utilities  (NumPy, called outside JIT)
# ---------------------------------------------------------------------------

def estimate_noise_mad(diffs: np.ndarray) -> float:
    """Online MAD estimate of sigma_noise."""
    d = np.abs(diffs - np.median(diffs, axis=0))
    return float(np.median(d)) * 1.4826 + 1e-8


def relative_poses_from_mats(gt_mats: np.ndarray) -> np.ndarray:
    """Compute (n-1, 6) relative poses from (n, 4, 4) SE(3) matrices.

    Uses raw rotation matrices directly — avoids so3_log on absolute world-frame
    rotations, which is numerically unstable when any pose has θ near π (e.g.
    U-turns in urban sequences). Consecutive-frame relative rotations are always
    small (< 0.15 rad at 10 Hz), so so3_log on dR is safe.
    """
    Ri   = jnp.array(gt_mats[:-1, :3, :3], dtype=jnp.float32)
    Rj   = jnp.array(gt_mats[1:,  :3, :3], dtype=jnp.float32)
    diff = jnp.array(gt_mats[1:, :3, 3] - gt_mats[:-1, :3, 3], dtype=jnp.float32)
    dt   = jnp.einsum('nji,nj->ni', Ri, diff)
    dR   = jnp.einsum('nji,njk->nik', Ri, Rj)
    dw   = jax.vmap(so3_log)(dR)
    return np.concatenate([np.array(dt), np.array(dw)], axis=-1)  # (n-1, 6)


@functools.lru_cache(maxsize=None)
def _get_jitted_integrator():
    """jit-compile the pure-JAX integration core once, reused across all calls.

    No vmap here — batching was removed, so each seed's integrate_poses call
    is its own dispatch. jit alone still avoids the eager/un-jitted overhead
    the un-batched original paid on every one of those per-seed calls.
    Retraces once per distinct n_poses (scan length) it sees, same as before.
    """
    def _integrate_poses_core(rel_poses: jnp.ndarray) -> jnp.ndarray:
        def step(carry, rel):
            R, t = carry
            dt, dw = rel[:3], rel[3:]
            dR    = so3_exp(dw)
            t_new = t + R @ dt
            R_new = R @ dR
            return (R_new, t_new), jnp.concatenate([t_new, so3_log(R_new)])

        init = (jnp.eye(3, dtype=rel_poses.dtype), jnp.zeros(3, dtype=rel_poses.dtype))
        _, poses_1_to_n = jax.lax.scan(step, init, rel_poses)
        pose0 = jnp.zeros((1, 6), dtype=rel_poses.dtype)
        return jnp.concatenate([pose0, poses_1_to_n], axis=0)

    return jax.jit(_integrate_poses_core)


def integrate_poses(rel_poses: np.ndarray) -> np.ndarray:
    """Integrate (n-1, 6) relative poses to (n, 6) global poses."""
    integrator = _get_jitted_integrator()
    return np.array(integrator(jnp.array(rel_poses, dtype=jnp.float32)))


@functools.lru_cache(maxsize=None)
def _get_jitted_rotation_integrator():
    """Same accumulation as _get_jitted_integrator, but returns the accumulated
    ROTATION MATRICES directly instead of an so3_log-encoded vector.

    _integrate_poses_core's per-step output calls so3_log(R_new) on the
    ACCUMULATED (world-frame) rotation — exactly the operation the seq06 fix
    (relative_poses_from_mats, commit c915da3) was written to avoid, because
    so3_log is numerically singular at theta=pi (division by sin(theta)->0,
    no large-angle safe branch — see dsg_jit.core.math3d.so3_log). Real KITTI
    sequences do reach theta≈180° (confirmed directly: 15/22 sequences peak
    at 178.6-180.0°), so integrate_poses' (n,6) rotation column is corrupted
    by up to ~178° at that exact pose for those sequences. This function
    exists so callers that only need rotation (the ATE/ARE metrics) can get
    the correct matrices without ever routing through so3_log.
    """
    def _integrate_rotations_core(rel_poses: jnp.ndarray) -> jnp.ndarray:
        def step(R, rel):
            dR = so3_exp(rel[3:])
            R_new = R @ dR
            return R_new, R_new

        init = jnp.eye(3, dtype=rel_poses.dtype)
        _, Rs_1_to_n = jax.lax.scan(step, init, rel_poses)
        R0 = jnp.eye(3, dtype=rel_poses.dtype)[None]
        return jnp.concatenate([R0, Rs_1_to_n], axis=0)

    return jax.jit(_integrate_rotations_core)


def integrate_rotations(rel_poses: np.ndarray) -> np.ndarray:
    """Integrate (n-1, 6) relative poses' rotation part into (n, 3, 3)
    accumulated rotation matrices — direct matrix composition, no so3_log
    anywhere, so no singularity at theta -> pi (unlike integrate_poses'
    (n, 6) vector output). Use this (not integrate_poses[:, 3:]) whenever
    the accumulated rotation itself is needed, e.g. for ATE/ARE metrics.
    """
    integrator = _get_jitted_rotation_integrator()
    return np.array(integrator(jnp.array(rel_poses, dtype=jnp.float32)))


# ---------------------------------------------------------------------------
# KITTI-style metrics: delta_T, delta_R, delta_C
# ---------------------------------------------------------------------------

def compute_ate(poses_est: np.ndarray, poses_gt: np.ndarray) -> float:
    """Absolute trajectory error (translation RMSE)."""
    return float(np.sqrt(np.mean(np.sum((poses_est[:, :3] - poses_gt[:, :3]) ** 2, axis=-1))))


def compute_are(Rs_est: np.ndarray, Rs_gt: np.ndarray) -> float:
    """Absolute rotation error (mean geodesic, degrees).

    Takes accumulated ROTATION MATRICES directly (n, 3, 3), not an so3_log-
    encoded pose vector — the geodesic angle is computed via the trace
    formula (arccos((trace(Ra^T@Rb)-1)/2)), which has no singularity at any
    angle, instead of reconstructing via so3_exp(stored vector) and calling
    so3_log again. so3_log itself has no large-angle-safe branch (only a
    documented small-angle one — see dsg_jit.core.math3d.so3_log), so it
    blows up near theta=pi: confirmed directly on real KITTI data, e.g. seq00
    pose 3128 (true angle 180.00°) round-tripped through so3_log/so3_exp as
    1.76° — a 178° error at that single pose. 15 of 22 KITTI sequences peak
    at 178.6-180.0°, so this isn't a rare edge case for this benchmark.
    """
    Rs_e = jnp.array(Rs_est, dtype=jnp.float32)
    Rs_g = jnp.array(Rs_gt,  dtype=jnp.float32)
    dRs  = jnp.einsum('nij,nik->njk', Rs_e, Rs_g)          # Rs_e^T @ Rs_g per pose
    tr   = jnp.trace(dRs, axis1=-2, axis2=-1)
    cos_theta  = jnp.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angles_rad = jnp.arccos(cos_theta)
    return float(jnp.mean(jnp.degrees(angles_rad)))


def delta_metric(noisy_poses, denoised_poses, gt_poses, noisy_R, denoised_R, gt_R):
    """Percent improvement: (noisy_err - denoised_err) / noisy_err * 100.

    noisy_R/denoised_R/gt_R are (n, 3, 3) accumulated rotation matrices from
    integrate_rotations()/raw GT matrices — see compute_are for why these,
    not the (n, 6) pose arrays' rotation column, are used for ARE.
    """
    ate_noisy    = compute_ate(noisy_poses, gt_poses)
    ate_denoised = compute_ate(denoised_poses, gt_poses)
    are_noisy    = compute_are(noisy_R, gt_R)
    are_denoised = compute_are(denoised_R, gt_R)

    dT = (ate_noisy - ate_denoised) / (ate_noisy + 1e-10) * 100.0
    dR = (are_noisy - are_denoised) / (are_noisy + 1e-10) * 100.0
    dC = 0.5 * (dT + dR)
    return dT, dR, dC


# ---------------------------------------------------------------------------
# Synthetic data  (no KITTI needed for sanity check)
# ---------------------------------------------------------------------------

def make_synthetic_sequence(n_poses: int, sigma_t: float, sigma_r: float,
                              rng: np.random.Generator) -> Tuple:
    """Generate a random walk trajectory with Gaussian noise."""
    gt_rel = np.zeros((n_poses - 1, 6), dtype=np.float32)
    gt_rel[:, 0] = 0.5                                       # constant forward motion
    gt_rel[:, 1] = 0.02 * rng.standard_normal(n_poses - 1)  # slight lateral drift
    gt_rel[:, 5] = 0.01 * rng.standard_normal(n_poses - 1)  # slight yaw

    noisy_rel = gt_rel.copy()
    noisy_rel[:, :3] += sigma_t * rng.standard_normal((n_poses - 1, 3))
    noisy_rel[:, 3:] += sigma_r * rng.standard_normal((n_poses - 1, 3))

    gt_global    = integrate_poses(gt_rel)
    noisy_global = integrate_poses(noisy_rel)
    return gt_rel, noisy_rel, gt_global, noisy_global


# ---------------------------------------------------------------------------
# Window denoising loop
# ---------------------------------------------------------------------------

def denoise_sequence(noisy_rel: np.ndarray,
                      gt_global: np.ndarray,
                      gt_R_mats: np.ndarray,
                      denoiser_fn,
                      exp_cfg: ExpCfg) -> np.ndarray:
    """Slide a window over the sequence, denoise each window, stitch output.

    Returns (denoised global trajectory (n_poses, 6), denoised relative poses
    (n_poses-1, 6)) — callers needing the accumulated rotation (e.g. ARE)
    should compute integrate_rotations(denoised_rel), not use the global
    trajectory's so3_log-encoded rotation column (singular at theta=pi).
    """
    n_poses  = gt_global.shape[0]
    stride   = exp_cfg.window - exp_cfg.overlap
    denoised_rel = noisy_rel.copy()

    def solve_and_write(lo, hi, write_lo, write_hi):
        win_odom = jnp.array(noisy_rel[lo:hi - 1])          # (W-1, 6)
        win_gt   = jnp.array(gt_global[lo:hi])               # (W, 6)
        win_gt_R = jnp.array(gt_R_mats[lo:hi])              # (W, 3, 3) raw rotations

        # Per-edge precision from local windowed variance (eager — small computation)
        sigma_t_est = estimate_noise_mad(np.array(win_odom[:, :3]))
        sigma_r_est = estimate_noise_mad(np.array(win_odom[:, 3:]))

        kappa = np.array(compute_per_edge_precision(
            jnp.array(win_odom[:, 3:]),
            sigma_r_est,
            exp_cfg.local_k,
            exp_cfg.max_kappa_ratio,
        ))
        omega = np.array(compute_per_edge_precision(
            jnp.array(win_odom[:, :3]),
            sigma_t_est,
            exp_cfg.local_k,
            exp_cfg.max_kappa_ratio,
        ))

        theta_opt = denoiser_fn(
            win_odom,
            win_gt,
            win_gt_R,
            jnp.array(kappa),
            jnp.array(omega)
        )

        # Apply corrections — write back to non-overlapping region
        corrected = np.array(win_odom) + np.array(theta_opt)
        denoised_rel[lo + write_lo: lo + write_hi] = corrected[write_lo:write_hi]
        return lo + write_hi   # global edge index written up to (exclusive)

    pos = 0
    last_written = 0
    while pos + exp_cfg.window <= n_poses - 1:
        lo, hi = pos, pos + exp_cfg.window
        n_edges = hi - 1 - lo
        write_lo = exp_cfg.overlap // 2 if pos > 0 else 0
        # Core write width is `stride` edges (+ the front half-overlap for window 0),
        # NOT `n_edges - overlap//2` — n_edges (=window-1) and stride (=window-overlap)
        # differ by exactly 1, so subtracting overlap//2 from n_edges left a 1-edge gap
        # between every consecutive pair of windows that never got denoised at all
        # (confirmed by simulation: exactly `n_windows-1` permanently-raw edges, one at
        # every window boundary). Writing `stride + overlap//2` instead makes this
        # window's write region end exactly where the next window's (write_lo=overlap//2)
        # begins, whenever there IS a next window (hi < n_poses - 1).
        write_hi = min(n_edges, stride + exp_cfg.overlap // 2) if hi < n_poses - 1 else n_edges
        last_written = solve_and_write(lo, hi, write_lo, write_hi)
        pos += stride

    # Tail: the fixed-stride loop above stops once no further full-size window fits,
    # which can leave up to `stride-1` trailing edges never covered by any window
    # (independent of the gap fix above — this is the loop simply running out before
    # reaching the sequence end). Shift one more same-sized window backward so it ends
    # exactly at the last edge, and start its write region exactly where the previous
    # window left off — closes any remaining gap (1 edge or many) with no overlap.
    if last_written < n_poses - 1:
        hi = n_poses   # exclusive pose bound -> reaches the true last edge (n_poses-2)
        lo = max(0, hi - exp_cfg.window)
        n_edges = hi - 1 - lo
        write_lo = last_written - lo
        solve_and_write(lo, hi, write_lo, n_edges)

    return integrate_poses(denoised_rel), denoised_rel


def _add_kitti_noise(gt_rel: np.ndarray, seed: int, seq_hash: int,
                     sigma_t: float, sigma_r: float) -> np.ndarray:
    """Return a noisy copy of gt_rel using the standard seed scheme."""
    rng = np.random.default_rng(seed * 1000 + seq_hash)
    noisy = gt_rel.copy()
    noisy[:, :3] += sigma_t * rng.standard_normal(gt_rel[:, :3].shape).astype(np.float32)
    noisy[:, 3:] += sigma_r * rng.standard_normal(gt_rel[:, 3:].shape).astype(np.float32)
    return noisy


def denoise_sequence_pooled(noisy_rels: np.ndarray,
                             gt_global: np.ndarray,
                             gt_R_mats: np.ndarray,
                             denoiser_fn,
                             prec_pooled_fn,
                             exp_cfg: ExpCfg) -> list:
    """Slide a window over the sequence, denoising each seed independently —
    no seed-axis batching/vmap, one dispatch per seed per window — while
    still pooling per-edge precision (kappa/omega) across all seeds for a
    given window before solving.

    This keeps Fix 4's statistical benefit (a ~154-sample pooled estimate
    instead of a 7-sample per-seed estimate) while removing the single-GPU-
    dispatch batching over seeds: the pooling step is a plain vectorized
    reduction over already-loaded data, not a GPU-dispatch batch, so it
    doesn't require vmap'ing the (expensive) solver itself.

    noisy_rels: (S, N-1, 6) — one noisy trajectory per seed
    gt_global:  (N, 6)      — shared GT global poses
    gt_R_mats:  (N, 3, 3)   — shared GT rotation matrices

    Returns (list of S denoised global trajectories each (N, 6), denoised
    relative poses (S, N-1, 6)) — callers needing the accumulated rotation
    (e.g. ARE) should compute integrate_rotations(denoised_rels[s]), not use
    the global trajectory's so3_log-encoded rotation column (singular at
    theta=pi).
    """
    S = noisy_rels.shape[0]
    n_poses = gt_global.shape[0]
    stride = exp_cfg.window - exp_cfg.overlap
    denoised_rels = noisy_rels.copy()   # (S, N-1, 6)

    # Convert entire arrays to JAX once — avoids per-window numpy→JAX copies
    noisy_rels_j = jnp.array(noisy_rels)
    gt_global_j  = jnp.array(gt_global)
    gt_R_mats_j  = jnp.array(gt_R_mats)

    def solve_and_write_all_seeds(lo, hi, write_lo, write_hi):
        win_omds = noisy_rels_j[:, lo:hi - 1, :]   # (S, n-1, 6) — JAX slice, no copy
        win_gt   = gt_global_j[lo:hi]               # (n, 6)
        win_gt_R = gt_R_mats_j[lo:hi]               # (n, 3, 3)

        # Precision pooled across all S seeds for this window (Fix 4, retained).
        # One reduction over already-loaded data — not a GPU-dispatch batch.
        kappa = prec_pooled_fn(win_omds[:, :, 3:])   # (n-1,)
        omega = prec_pooled_fn(win_omds[:, :, :3])    # (n-1,)

        # No seed-axis batching: one dispatch per seed, reusing the shared
        # kappa/omega. Fix 1 (jit-compiled integrate_poses) and Fix 2
        # (self-checking GN + Adam warmup) apply identically per seed here.
        for s in range(S):
            theta_opt = denoiser_fn(win_omds[s], win_gt, win_gt_R, kappa, omega)
            corrected = np.array(win_omds[s] + theta_opt)   # (n-1, 6)
            denoised_rels[s, lo + write_lo:lo + write_hi, :] = corrected[write_lo:write_hi]
        return lo + write_hi   # global edge index written up to (exclusive)

    pos = 0
    last_written = 0
    while pos + exp_cfg.window <= n_poses - 1:
        lo, hi = pos, pos + exp_cfg.window
        n_edges = hi - 1 - lo
        write_lo = exp_cfg.overlap // 2 if pos > 0 else 0
        # Core write width is `stride` edges (+ front half-overlap for window 0), NOT
        # `n_edges - overlap//2` — n_edges (=window-1) and stride (=window-overlap) differ
        # by exactly 1, so subtracting overlap//2 from n_edges left a 1-edge gap between
        # every consecutive pair of windows that never got denoised at all (confirmed by
        # simulation: exactly `n_windows-1` permanently-raw edges, one per window boundary).
        # Writing `stride + overlap//2` instead makes this window's write region end exactly
        # where the next window's (write_lo=overlap//2) begins, whenever there IS a next
        # window (hi < n_poses - 1).
        write_hi = min(n_edges, stride + exp_cfg.overlap // 2) if hi < n_poses - 1 else n_edges
        last_written = solve_and_write_all_seeds(lo, hi, write_lo, write_hi)
        pos += stride

    # Tail: the fixed-stride loop above stops once no further full-size window fits,
    # which can leave up to `stride-1` trailing edges never covered by any window
    # (independent of the gap fix above — this is the loop simply running out before
    # reaching the sequence end). Shift one more same-sized window backward so it ends
    # exactly at the last edge, and start its write region exactly where the previous
    # window left off — closes any remaining gap (1 edge or many) with no overlap.
    if last_written < n_poses - 1:
        hi = n_poses   # exclusive pose bound -> reaches the true last edge (n_poses-2)
        lo = max(0, hi - exp_cfg.window)
        n_edges = hi - 1 - lo
        write_lo = last_written - lo
        solve_and_write_all_seeds(lo, hi, write_lo, n_edges)

    return [integrate_poses(denoised_rels[s]) for s in range(S)], denoised_rels


# ---------------------------------------------------------------------------
# Noise-adaptive solver settings
# ---------------------------------------------------------------------------

def noise_adaptive_inner_outer_cfg(sigma_t: float,
                                    base_inner_kwargs: dict,
                                    base_outer_kwargs: dict,
                                    reference_sigma_t: float = 0.03) -> Tuple[InnerCfg, OuterCfg]:
    """Scale a few self-checking-solver settings based on noise level, relative
    to a reference noise level where the fixed settings are already well-tuned.

    Background: comparing exp43 (self-checking GN + adaptive damping + Adam
    warmup) against the pre-self-checking baseline across sigma_t = 0.01,
    0.03, 0.05, 0.10 showed a noise-dependent pattern -- exp43 clearly helps
    at low noise (0.01: +29.2% vs +21.5% baseline), is roughly a wash at
    sigma_t=0.03 (+38.6% vs +39.1%), but underperforms the baseline at higher
    noise (0.05: +18.9% vs +31.9%; 0.10: +7.4% vs +12.4%). Hypothesis: the
    solver's caution (rejecting steps, escalating damping, ramping Adam's LR)
    costs little when the needed correction is small, but at higher noise the
    correction needed is larger, and the same FIXED iteration/step budget
    used at every noise level doesn't leave enough room to both be cautious
    and fully reach the correction.

    This function keeps settings IDENTICAL to the fixed defaults at or below
    reference_sigma_t (scale == 1.0 there), and only adjusts four settings
    as noise grows past that reference:
      - n_iters_rot:    more attempts for the inner GN solve
      - damping_up:     escalate caution more gently after a rejected step
      - warmup_steps:   reach full-strength outer Adam steps sooner
      - rot_loss_boost: weight the outer loss's rotation term more heavily

    Comparing sigma_t=0.03 (roughly a wash) against 0.05/0.10 (regressions)
    motivates using 0.03 as the reference point below which nothing changes.

    rot_loss_boost separately validated across all four sweep noise levels:
    a FIXED boost (e.g. 30.0 at every noise level) reliably helped rotation
    everywhere but hurt translation more often than not at sigma_t <=
    reference_sigma_t (where exp43 was already the stronger regime and this
    rebalancing overcorrects a problem that barely exists yet). Scaling it
    the same way as the other three settings -- off (1.0) at/below the
    reference, ramping up above it -- kept the high-noise win (rotation and
    translation both improved) without the low-noise regression.
    """
    scale = max(1.0, sigma_t / reference_sigma_t)

    base_n_iters = base_inner_kwargs.get("n_iters_rot", 15)
    n_iters_rot = int(min(40, round(base_n_iters * scale)))

    base_damping_up = base_inner_kwargs.get("damping_up", 4.0)
    damping_up = max(1.5, base_damping_up / scale)

    base_warmup = base_outer_kwargs.get("warmup_steps", 5)
    warmup_steps = int(max(1, round(base_warmup / scale)))

    max_rot_loss_boost = base_outer_kwargs.get("max_rot_loss_boost", 30.0)
    rot_loss_boost = min(max_rot_loss_boost, 1.0 + (max_rot_loss_boost - 1.0) * (scale - 1.0))

    inner_cfg = InnerCfg(
        n_iters_rot=n_iters_rot,
        damping_init=base_inner_kwargs.get("damping_init", 1e-4),
        damping_min=base_inner_kwargs.get("damping_min", 1e-6),
        damping_max=base_inner_kwargs.get("damping_max", 1e2),
        damping_down=base_inner_kwargs.get("damping_down", 0.5),
        damping_up=damping_up,
        # anchor_spacing/kappa_anchor/n_starts/multistart_criterion/
        # anchor_n_iters_rot are fixed hyperparameters, not noise-adaptive
        # (see InnerCfg's docstring) -- pass through unchanged at every
        # noise level.
        anchor_spacing=base_inner_kwargs.get("anchor_spacing", 50),
        kappa_anchor=base_inner_kwargs.get("kappa_anchor", 100.0),
        n_starts=base_inner_kwargs.get("n_starts", 1),
        multistart_criterion=base_inner_kwargs.get("multistart_criterion", "anchor_only"),
        anchor_n_iters_rot=base_inner_kwargs.get("anchor_n_iters_rot", 60),
    )
    outer_cfg = OuterCfg(
        n_trans1=base_outer_kwargs["n_trans1"],
        n_rot=base_outer_kwargs["n_rot"],
        n_trans2=base_outer_kwargs["n_trans2"],
        lr_trans=base_outer_kwargs["lr_trans"],
        lr_rot=base_outer_kwargs["lr_rot"],
        warmup_steps=warmup_steps,
        rot_loss_boost=rot_loss_boost,
    )
    return inner_cfg, outer_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp43: Unbatched SE-Sync Structured GN with Self-Checking Inner Solver"
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
    parser.add_argument("--n-trans1",   type=int,   default=30)
    parser.add_argument("--n-rot",      type=int,   default=20)
    parser.add_argument("--n-trans2",   type=int,   default=20)
    parser.add_argument("--lr-trans",   type=float, default=1e-3)
    parser.add_argument("--lr-rot",     type=float, default=1e-3)
    parser.add_argument("--anchor-spacing", type=int, default=50,
                        help="Sparse GT rotation anchors every this-many poses (+ last pose). "
                             "Without these the inner rotation GN solve is provably a no-op on "
                             "this chain graph (see InnerCfg.anchor_spacing's docstring).")
    parser.add_argument("--kappa-anchor", type=float, default=100.0,
                        help="Fixed anchor precision (not noise-adaptive or learned). Needs "
                             "empirical tuning: too small and anchors don't matter in practice, "
                             "too large and every window just clamps to GT.")
    parser.add_argument("--n-starts", type=int, default=1,
                        help="Multi-start GN: 1 = today's single chain-composed R_init (default, "
                             "unchanged behavior). 2 = also try an anchor-interpolated R_init and "
                             "keep whichever the self-checking GN solve reaches lower cost from -- "
                             "targets the deterministic bad-basin seeds (see InnerCfg.n_starts).")
    parser.add_argument("--multistart-criterion", type=str, default="anchor_only",
                        choices=["anchor_only", "matched_total"],
                        help="Selection criterion when --n-starts=2 (see InnerCfg.multistart_criterion "
                             "for the evidence behind each -- this is a live A/B, not settled).")
    parser.add_argument("--anchor-n-iters-rot", type=int, default=60,
                        help="GN iteration budget for the anchor candidate when "
                             "--multistart-criterion=matched_total (ignored otherwise).")
    parser.add_argument("--adaptive-solver", action=argparse.BooleanOptionalAction, default=True,
                        help="Scale n_iters_rot/damping_up/warmup_steps with sigma_t above the "
                             "reference noise level (default: on). Use --no-adaptive-solver to "
                             "restore the original fixed settings at every noise level.")
    parser.add_argument("--adaptive-reference-sigma-t", type=float, default=0.03,
                        help="Noise level at/below which adaptive settings are identical to the "
                             "fixed defaults; settings only change above this level.")
    parser.add_argument("--output-dir", type=str,   default=os.path.expanduser("~/exp_res"),
                        help="Directory to write run results (timestamped sub-dir created automatically)")
    args = parser.parse_args()

    # Create timestamped output directory
    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp44_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    base_inner_kwargs = {"n_iters_rot": 15, "damping_init": 1e-4, "damping_min": 1e-6,
                         "damping_max": 1e2, "damping_down": 0.5, "damping_up": 4.0,
                         "anchor_spacing": args.anchor_spacing, "kappa_anchor": args.kappa_anchor,
                         "n_starts": args.n_starts, "multistart_criterion": args.multistart_criterion,
                         "anchor_n_iters_rot": args.anchor_n_iters_rot}
    base_outer_kwargs = {"n_trans1": args.n_trans1, "n_rot": args.n_rot, "n_trans2": args.n_trans2,
                         "lr_trans": args.lr_trans, "lr_rot": args.lr_rot, "warmup_steps": 5}

    if args.adaptive_solver:
        inner_cfg, outer_cfg = noise_adaptive_inner_outer_cfg(
            sigma_t=args.sigma_t,
            base_inner_kwargs=base_inner_kwargs,
            base_outer_kwargs=base_outer_kwargs,
            reference_sigma_t=args.adaptive_reference_sigma_t,
        )
        print(f"Noise-adaptive solver settings (sigma_t={args.sigma_t}, "
              f"reference={args.adaptive_reference_sigma_t}): "
              f"n_iters_rot={inner_cfg.n_iters_rot}, damping_up={inner_cfg.damping_up:.2f}, "
              f"warmup_steps={outer_cfg.warmup_steps}, rot_loss_boost={outer_cfg.rot_loss_boost:.2f}, "
              f"anchor_spacing={inner_cfg.anchor_spacing}, kappa_anchor={inner_cfg.kappa_anchor:.2f}, "
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
                              n_starts=base_inner_kwargs["n_starts"],
                              multistart_criterion=base_inner_kwargs["multistart_criterion"],
                              anchor_n_iters_rot=base_inner_kwargs["anchor_n_iters_rot"])
        outer_cfg = OuterCfg(n_trans1=base_outer_kwargs["n_trans1"],
                              n_rot=base_outer_kwargs["n_rot"],
                              n_trans2=base_outer_kwargs["n_trans2"],
                              lr_trans=base_outer_kwargs["lr_trans"],
                              lr_rot=base_outer_kwargs["lr_rot"],
                              warmup_steps=base_outer_kwargs["warmup_steps"],
                              rot_loss_boost=1.0)
        print(f"Fixed (non-adaptive) solver settings: n_iters_rot={inner_cfg.n_iters_rot}, "
              f"damping_up={inner_cfg.damping_up:.2f}, warmup_steps={outer_cfg.warmup_steps}, "
              f"rot_loss_boost={outer_cfg.rot_loss_boost:.2f}, "
              f"anchor_spacing={inner_cfg.anchor_spacing}, kappa_anchor={inner_cfg.kappa_anchor:.2f}, "
              f"n_starts={inner_cfg.n_starts}, multistart_criterion={inner_cfg.multistart_criterion}, "
              f"anchor_n_iters_rot={inner_cfg.anchor_n_iters_rot}")

    exp_cfg   = ExpCfg(
        window=args.window,
        overlap=args.overlap,
        sigma_t=args.sigma_t,
        sigma_r=args.sigma_r,
        seeds=args.seeds,
    )

    # Build the single-seed JIT-compiled denoiser once for this window size.
    # No batched/vmap'd variant — batching was removed, every seed (synthetic
    # or KITTI) is dispatched through this same function, one call at a time.
    print(f"Compiling denoiser (window={args.window}, seeds={args.seeds})...")
    t0 = time.time()
    denoiser_fn = build_denoiser(args.window, inner_cfg, outer_cfg, exp_cfg)

    # Build prec_pooled_fn once using functools.partial — stable reference, no recompile
    # per sequence. No vmap here: _sigma_and_precision pools across the seed axis
    # internally and returns one (n_edges,) precision shared by all seeds (Fix 4,
    # retained despite batching removal — see denoise_sequence_pooled).
    lk, mr = exp_cfg.local_k, exp_cfg.max_kappa_ratio
    prec_pooled_fn = jax.jit(functools.partial(_sigma_and_precision,
                                                local_k=lk,
                                                max_kappa_ratio=mr))

    dummy_odom = jnp.zeros((args.window - 1, 6))
    dummy_gt   = jnp.zeros((args.window, 6))
    dummy_gR   = jnp.zeros((args.window, 3, 3))
    dummy_k    = jnp.ones(args.window - 1)
    dummy_w    = jnp.ones(args.window - 1)
    # Warm up the single-seed denoiser — same shapes serve synthetic and KITTI,
    # since kappa/omega are (window-1,) either way (per-seed or pooled-and-shared).
    _ = denoiser_fn(dummy_odom, dummy_gt, dummy_gR, dummy_k, dummy_w).block_until_ready()
    # Warm up the pooled-precision function (used for KITTI); input keeps the
    # (S, n_edges, d) shape since pooling still needs all seeds' raw data.
    dummy_omds = jnp.zeros((args.seeds, args.window - 1, 6))
    _ = prec_pooled_fn(dummy_omds[:, :, :3]).block_until_ready()
    print(f"  Compiled in {time.time() - t0:.1f}s  (no retrace after this)")

    # Collect results
    all_dT, all_dR, all_dC = [], [], []
    all_seq_results = {}  # seq_id -> list of per-seed dicts

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
            # Synthetic: GT differs per seed (random lateral/yaw), so run sequentially.
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
                # ARE metric uses accumulated rotation matrices directly (integrate_rotations),
                # not gt_global/denoised_global's so3_log-encoded rotation column — see
                # compute_are's docstring for why (singular at theta=pi, confirmed on real
                # KITTI sequences; harmless here since synthetic rotations stay tiny, but kept
                # uniform with the KITTI path so the metric never depends on that assumption).
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
            # KITTI: GT is fixed per sequence — load once, batch all seeds.
            from pathlib import Path
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

            # Generate all S noisy trajectories upfront
            noisy_rels = np.stack([
                _add_kitti_noise(gt_rel, seed, seq_hash, args.sigma_t, args.sigma_r)
                for seed in range(args.seeds)
            ])   # (S, N-1, 6)

            # Denoise each seed independently (no batching), reusing pooled precision
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
                # ARE metric uses accumulated rotation matrices directly, not the (n,6)
                # trajectories' so3_log-encoded rotation column — see compute_are's
                # docstring. gt_R_mats is already the raw (safe) matrices from the KITTI
                # file; noisy/denoised need integrate_rotations for the same reason.
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
            print(f"  [{seq_id}] {args.seeds} seeds (unbatched, pooled precision)  total {poses_per_sec:.0f} poses/s·seed")

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
        print("\n=== Exp43 Summary ===")
        print(f"  σ_t={args.sigma_t}  window={args.window}  overlap={args.overlap}")
        print(f"  Mean across seqs:  ΔT={np.mean(all_dT):+.1f}%  "
              f"ΔR={np.mean(all_dR):+.1f}%  ΔC={np.mean(all_dC):+.1f}%")

        config = {
            "exp": "exp44",
            "kitti_root": args.kitti_root,
            "seqs": args.seqs,
            "sigma_t": args.sigma_t,
            "sigma_r": args.sigma_r,
            "window": args.window,
            "overlap": args.overlap,
            "seeds": args.seeds,
            "n_trans1": args.n_trans1,
            "n_rot": args.n_rot,
            "n_trans2": args.n_trans2,
            "lr_trans": args.lr_trans,
            "lr_rot": args.lr_rot,
            "synthetic": args.synthetic,
            "run_ts": run_ts,
            "adaptive_solver": args.adaptive_solver,
            "adaptive_reference_sigma_t": args.adaptive_reference_sigma_t,
            "n_iters_rot_used": inner_cfg.n_iters_rot,
            "damping_up_used": inner_cfg.damping_up,
            "warmup_steps_used": outer_cfg.warmup_steps,
            "rot_loss_boost_used": outer_cfg.rot_loss_boost,
            "anchor_spacing": inner_cfg.anchor_spacing,
            "kappa_anchor": inner_cfg.kappa_anchor,
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
            f"Exp43 Run  {run_ts}",
            f"  kitti_root={args.kitti_root}  seqs={args.seqs}",
            f"  sigma_t={args.sigma_t}  sigma_r={args.sigma_r}",
            f"  window={args.window}  overlap={args.overlap}  seeds={args.seeds}",
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
