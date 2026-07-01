"""
exp42_sesync_structured_gn.py

Experiment 42: SE-Sync Structured GN Inner Solver for Odometry Denoising.

Changes from Exp41:
  - Inner solver restructured as SE-Sync Path B:
      * Rotation-only GN on SO(3)^n (3n x 3n system)
      * Analytic translation recovery via one dense linear solve
  - Dense (3n x 3n) Laplacians for GPU parallelism (cuBLAS path)
      * Built via vectorized scatter — no Python loops inside JIT
  - Per-edge precision (kappa_ij, omega_ij) from local windowed variance
      * Replaces global sw/rw scalar — collapses of one edge don't drag all edges
  - jax.vmap over all edges for rotation residuals (parallel, no scan per-edge)
  - jax.lax.scan for GN iteration loop (fixed count, JIT-compilable)
  - IFT custom_vjp on rotation GN: 3n x 3n backward solve
  - Translation recovery: jnp.linalg.solve — JAX auto-diff handles backward
  - Outer 3-phase Adam unchanged: 30T + 20R + 20T = 70 iters, fused fori_loop

Run:
  python exp42_sesync_structured_gn.py --kitti-root /path/to/kitti --sigma-t 0.03
  python exp42_sesync_structured_gn.py --synthetic  # no KITTI needed
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
    n_iters_rot: int = 10
    damping:     float = 1e-4

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
                              n: int) -> jnp.ndarray:
    """Build dense (3n x 3n) GN Hessian for the rotation synchronization subproblem.

    R_meas: (n-1, 3, 3)  per-edge rotation measurements (unused — kept for API)
    kappa:  (n-1,)       per-edge rotation precision

    SE-Sync (IJRR §4.1, Eq. 14) derives the connection Laplacian L(G̃ρ) with
    off-diagonal blocks −κᵢⱼ R̃ᵢⱼ from the Frobenius cost ‖Rⱼ − Rᵢ R̃ᵢⱼ‖²_F
    with RIGHT retraction.  This implementation uses the geodesic cost
    ‖so3_log(R̃ᵢⱼᵀ Rᵢᵀ Rⱼ)‖² with LEFT retraction R ← so3_exp(u) @ R.
    Under left retraction the GN Hessian approximation has off-diagonal blocks
    −κᵢⱼ I₃ (standard graph Laplacian).  Both formulations are self-consistent
    GN variants that converge to the same rotation MLE fixed point; for KITTI's
    small per-frame rotations (< 0.1 rad) R̃ᵢⱼ ≈ I₃ so the two Hessians are
    numerically near-identical.

    NOTE: for an anchored chain graph the Schur complement Q̃τ (SE-Sync Eq. 24c)
    is zero because the anchored translation Laplacian has no null space, so
    Problem 4 reduces to pure rotation synchronization — exactly what this solves.

    Off-diagonal blocks: L[i, i+1] = -kappa[i] * I_3
    Diagonal blocks:     L[i, i]   =  sum of incident kappa * I3
    """
    kappa_left  = jnp.concatenate([jnp.zeros(1), kappa])
    kappa_right = jnp.concatenate([kappa, jnp.zeros(1)])
    kappa_diag  = kappa_left + kappa_right
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

def _rotation_gn_step(R: jnp.ndarray,
                       R_meas: jnp.ndarray,
                       kappa: jnp.ndarray,
                       n: int,
                       damping: float) -> jnp.ndarray:
    """One GN step on SO(3)^n for rotation synchronization.

    Solves: (L_free + damping*I) delta = -g_free
    then retracts: R[j] <- so3_exp(delta[j]) @ R[j]

    All edge residuals computed in parallel via vmap.
    Gradient accumulated via vectorized scatter (.at[].add).
    System solved via jnp.linalg.solve on the dense (3(n-1) x 3(n-1)) matrix.
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

    g_free = g_all[1:].reshape(-1)                          # (3(n-1),)

    # Rotation GN Hessian (dense) — standard graph Laplacian for left-retraction GN
    L = build_rotation_laplacian(R_meas, kappa, n)
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
                      n_iters: int,
                      damping: float) -> jnp.ndarray:
    """Fixed-count GN via jax.lax.scan — JIT-compilable, no early exit."""
    n = R_init.shape[0]
    def step(R, _):
        return _rotation_gn_step(R, R_meas, kappa, n, damping), None
    R_star, _ = jax.lax.scan(step, R_init, None, length=n_iters)
    return R_star


@jax.custom_vjp
def rotation_gn_ift(R_init: jnp.ndarray,
                    R_meas: jnp.ndarray,
                    kappa: jnp.ndarray,
                    n_iters: int,
                    damping: float) -> jnp.ndarray:
    """Rotation GN with IFT shortcut backward.

    Forward: run fixed-count GN, return R_star.
    Backward: IFT — one 3(n-1) x 3(n-1) solve instead of unrolling N iters.
    """
    return _rotation_gn_raw(R_init, R_meas, kappa, n_iters, damping)


def _rotation_gn_ift_fwd(R_init, R_meas, kappa, n_iters, damping):
    R_star = _rotation_gn_raw(R_init, R_meas, kappa, n_iters, damping)
    return R_star, (R_star, R_meas, kappa, n_iters, damping)


def _rotation_gn_ift_bwd(res, g_R_star):
    """IFT backward pass for rotation GN.

    Given upstream gradient g_R_star (n, 3, 3) w.r.t. R_star:
    1. Project to tangent space: g_free (3(n-1),) — skip anchor
    2. IFT: solve L_free @ v = g_free  (one dense solve, not N unrolled steps)
    3. Propagate v back through the gradient function to get dL/d(R_meas, kappa)
    """
    R_star, R_meas, kappa, n_iters, damping = res
    n = R_star.shape[0]

    # Project upstream gradient to axis-angle increments on SO(3).
    # Under left retraction R ← exp(u) @ R, the tangent component is:
    #   g_u = vee( (R^T g - g^T R) / 2 )  =  vee(skew_part(R^T g))
    def project_grad(Ri, gi):
        skew = Ri.T @ gi - gi.T @ Ri          # = 2 * skew_part(R^T g)
        return jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / 2.0

    g_tangent = jax.vmap(project_grad)(R_star, g_R_star)    # (n, 3)
    g_free = g_tangent[1:].reshape(-1)                       # (3(n-1),) — drop anchor

    # IFT: solve L_free @ v = g_free  (same Hessian as forward GN)
    L = build_rotation_laplacian(R_meas, kappa, n)
    L_free = L[3:, 3:] + damping * jnp.eye(3 * (n - 1))
    v_free = jnp.linalg.solve(L_free, g_free)               # (3(n-1),)
    v_all = jnp.concatenate([jnp.zeros(3), v_free]).reshape(n, 3)  # (n, 3)

    # Propagate v through gradient function: dL/d(R_meas, kappa)
    # The GN gradient g(R_meas, kappa) = Jt(R_star) @ r(R_star; R_meas, kappa)
    # We need: v^T @ d_g/d(R_meas, kappa) evaluated at R_star
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
    return g_R_init, g_R_meas, g_kappa, None, None


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
# Full inner solver
# ---------------------------------------------------------------------------

def sesync_inner_solve(theta: jnp.ndarray,
                        noisy_odom: jnp.ndarray,
                        kappa: jnp.ndarray,
                        omega: jnp.ndarray,
                        n: int,
                        cfg: InnerCfg) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SE-Sync structured inner solve.

    theta:      (n-1, 6) learned corrections to odometry
    noisy_odom: (n-1, 6) noisy odometry measurements [t(3) | w(3)]
    kappa:      (n-1,) per-edge rotation precision
    omega:      (n-1,) per-edge translation precision
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

    # Rotation GN with IFT backward (3(n-1) x 3(n-1) system)
    R_star = rotation_gn_ift(R_init, R_meas, kappa,
                              cfg.n_iters_rot, cfg.damping)

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


# Pure-JAX combined sigma+precision — no numpy, no D2H copy.
# Replaces: estimate_noise_mad(np.array(...)) + compute_per_edge_precision(...)
# Called via vmap over seeds so the whole batch stays on GPU.
def _sigma_and_precision(diffs, local_k, max_kappa_ratio):
    """(n_edges, d) -> (n_edges,) precision. Entirely in JAX, JIT-safe."""
    med = jnp.median(diffs, axis=0)
    sigma_noise = jnp.median(jnp.abs(diffs - med)) * 1.4826 + 1e-8

    n_edges = diffs.shape[0]
    diffs_sq = jnp.sum(diffs ** 2, axis=-1)
    indices  = jnp.arange(n_edges)

    def local_var_at(i):
        mask  = (indices >= i - local_k) & (indices <= i + local_k)
        count = jnp.sum(mask).astype(jnp.float32) + 1e-8
        mean  = jnp.sum(jnp.where(mask, diffs_sq, 0.0)) / count
        return jnp.sum(jnp.where(mask, (diffs_sq - mean) ** 2, 0.0)) / count

    local_vars       = jax.vmap(local_var_at)(indices)
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
    def loss_fn(theta):
        R_star, t_star = sesync_inner_solve(
            theta, noisy_odom, kappa, omega, n, inner_cfg
        )
        loss_t = jnp.mean(jnp.sum((t_star - gt_t_rel) ** 2, axis=-1))
        loss_r = jnp.mean(jax.vmap(
            lambda Ra, Rb: jnp.sum((Ra - Rb) ** 2)
        )(R_star, gt_R_rel))
        return loss_t + loss_r

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

        # Learning rate per phase
        lr = jnp.where(in_rot, outer_cfg.lr_rot, outer_cfg.lr_trans)

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
        return theta_opt

    return jax.jit(denoise)


def build_batched_denoiser(n: int,
                            inner_cfg: InnerCfg,
                            outer_cfg: OuterCfg,
                            exp_cfg: ExpCfg):
    """Build a seed-batched denoiser via vmap.

    Processes S seeds in a single GPU dispatch per window instead of S
    sequential dispatches. win_odom/kappa/omega vary per seed (in_axes=0);
    gt_poses/gt_R_world are shared across seeds (in_axes=None).
    """
    def denoise(noisy_odom, gt_poses, gt_R_world, kappa, omega):
        theta_init = jnp.zeros_like(noisy_odom)
        theta_opt = outer_adam_loop(
            theta_init, noisy_odom, gt_poses, gt_R_world,
            kappa, omega, n, inner_cfg, outer_cfg
        )
        return theta_opt

    batched = jax.vmap(denoise, in_axes=(0, None, None, 0, 0))
    return jax.jit(batched)


# ---------------------------------------------------------------------------
# Noise estimation utilities  (NumPy, called outside JIT)
# ---------------------------------------------------------------------------

def estimate_noise_mad(diffs: np.ndarray) -> float:
    """Online MAD estimate of sigma_noise."""
    d = np.abs(diffs - np.median(diffs, axis=0))
    return float(np.median(d)) * 1.4826 + 1e-8


def relative_poses_from_global(global_poses: np.ndarray) -> np.ndarray:
    """Compute (n-1, 6) relative poses from (n, 6) global poses."""
    from dsg_jit.core.math3d import so3_exp as so3_exp_np, so3_log as so3_log_np
    n = global_poses.shape[0]
    rel = []
    for i in range(n - 1):
        ti, wi = global_poses[i, :3], global_poses[i, 3:]
        tj, wj = global_poses[i + 1, :3], global_poses[i + 1, 3:]
        Ri = np.array(so3_exp_np(jnp.array(wi)))
        Rj = np.array(so3_exp_np(jnp.array(wj)))
        dt = Ri.T @ (tj - ti)
        dR = Ri.T @ Rj
        dw = np.array(so3_log_np(jnp.array(dR)))
        rel.append(np.concatenate([dt, dw]))
    return np.stack(rel)                                     # (n-1, 6)


def relative_poses_from_mats(gt_mats: np.ndarray) -> np.ndarray:
    """Compute (n-1, 6) relative poses from (n, 4, 4) SE(3) matrices.

    Uses raw rotation matrices directly — avoids so3_log on absolute world-frame
    rotations, which is numerically unstable when any pose has θ near π (e.g.
    U-turns in urban sequences). Consecutive-frame relative rotations are always
    small (< 0.15 rad at 10 Hz), so so3_log on dR is safe.
    """
    from dsg_jit.core.math3d import so3_log as so3_log_np
    n = gt_mats.shape[0]
    rel = []
    for i in range(n - 1):
        Ri = gt_mats[i,   :3, :3]
        Rj = gt_mats[i+1, :3, :3]
        ti = gt_mats[i,   :3,  3]
        tj = gt_mats[i+1, :3,  3]
        dt = Ri.T @ (tj - ti)
        dR = Ri.T @ Rj
        dw = np.array(so3_log_np(jnp.array(dR)))
        rel.append(np.concatenate([dt, dw]))
    return np.stack(rel)                                     # (n-1, 6)


def integrate_poses(rel_poses: np.ndarray) -> np.ndarray:
    """Integrate (n-1, 6) relative poses to (n, 6) global poses."""
    from dsg_jit.core.math3d import so3_exp as so3_exp_np, so3_log as so3_log_np
    n = rel_poses.shape[0] + 1
    poses = np.zeros((n, 6), dtype=np.float32)
    R = np.eye(3)
    t = np.zeros(3)
    for i, rel in enumerate(rel_poses):
        dt, dw = rel[:3], rel[3:]
        dR = np.array(so3_exp_np(jnp.array(dw)))
        t = t + R @ dt
        R = R @ dR
        w = np.array(so3_log_np(jnp.array(R)))
        poses[i + 1] = np.concatenate([t, w])
    return poses


# ---------------------------------------------------------------------------
# KITTI-style metrics: delta_T, delta_R, delta_C
# ---------------------------------------------------------------------------

def compute_ate(poses_est: np.ndarray, poses_gt: np.ndarray) -> float:
    """Absolute trajectory error (translation RMSE)."""
    return float(np.sqrt(np.mean(np.sum((poses_est[:, :3] - poses_gt[:, :3]) ** 2, axis=-1))))


def compute_are(poses_est: np.ndarray, poses_gt: np.ndarray) -> float:
    """Absolute rotation error (mean geodesic, degrees)."""
    from dsg_jit.core.math3d import so3_exp as so3e, so3_log as so3l
    errs = []
    for i in range(len(poses_est)):
        Re = np.array(so3e(jnp.array(poses_est[i, 3:])))
        Rg = np.array(so3e(jnp.array(poses_gt[i, 3:])))
        dR = Re.T @ Rg
        angle = float(np.linalg.norm(np.array(so3l(jnp.array(dR)))))
        errs.append(np.degrees(angle))
    return float(np.mean(errs))


def delta_metric(noisy_poses, denoised_poses, gt_poses):
    """Percent improvement: (noisy_err - denoised_err) / noisy_err * 100."""
    ate_noisy    = compute_ate(noisy_poses, gt_poses)
    ate_denoised = compute_ate(denoised_poses, gt_poses)
    are_noisy    = compute_are(noisy_poses, gt_poses)
    are_denoised = compute_are(denoised_poses, gt_poses)

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

    Returns denoised global trajectory (n_poses, 6).
    """
    n_poses  = gt_global.shape[0]
    stride   = exp_cfg.window - exp_cfg.overlap
    denoised_rel = noisy_rel.copy()

    pos = 0
    while pos + exp_cfg.window <= n_poses - 1:
        # Extract window
        lo, hi = pos, pos + exp_cfg.window
        win_odom = jnp.array(noisy_rel[lo:hi - 1])          # (W-1, 6)
        win_gt   = jnp.array(gt_global[lo:hi])               # (W, 6)
        win_gt_R = jnp.array(gt_R_mats[lo:hi])              # (W, 3, 3) raw rotations

        n_edges = win_odom.shape[0]

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
        write_lo = exp_cfg.overlap // 2 if pos > 0 else 0
        write_hi = n_edges - exp_cfg.overlap // 2 if hi < n_poses - 1 else n_edges
        denoised_rel[lo + write_lo: lo + write_hi] = corrected[write_lo:write_hi]

        pos += stride

    return integrate_poses(denoised_rel)


def _add_kitti_noise(gt_rel: np.ndarray, seed: int, seq_hash: int,
                     sigma_t: float, sigma_r: float) -> np.ndarray:
    """Return a noisy copy of gt_rel using the standard seed scheme."""
    rng = np.random.default_rng(seed * 1000 + seq_hash)
    noisy = gt_rel.copy()
    noisy[:, :3] += sigma_t * rng.standard_normal(gt_rel[:, :3].shape).astype(np.float32)
    noisy[:, 3:] += sigma_r * rng.standard_normal(gt_rel[:, 3:].shape).astype(np.float32)
    return noisy


def denoise_sequence_batched(noisy_rels: np.ndarray,
                              gt_global: np.ndarray,
                              gt_R_mats: np.ndarray,
                              batched_denoiser_fn,
                              prec_batched_fn,
                              exp_cfg: ExpCfg) -> list:
    """Slide a window over the sequence, denoising all S seeds simultaneously.

    noisy_rels: (S, N-1, 6) — one noisy trajectory per seed
    gt_global:  (N, 6)      — shared GT global poses
    gt_R_mats:  (N, 3, 3)   — shared GT rotation matrices

    Returns list of S denoised global trajectories, each (N, 6).

    Per window: one vmap'd GPU call over S seeds instead of S sequential calls.
    The GPU sees a batched (S, n-1, 6) solve rather than S × (n-1, 6) solves,
    improving utilisation for small n where dispatch overhead dominates.
    """
    S = noisy_rels.shape[0]
    n_poses = gt_global.shape[0]
    stride = exp_cfg.window - exp_cfg.overlap
    denoised_rels = noisy_rels.copy()   # (S, N-1, 6)

    # Convert entire arrays to JAX once — avoids per-window numpy→JAX copies
    noisy_rels_j = jnp.array(noisy_rels)
    gt_global_j  = jnp.array(gt_global)
    gt_R_mats_j  = jnp.array(gt_R_mats)

    pos = 0
    while pos + exp_cfg.window <= n_poses - 1:
        lo, hi = pos, pos + exp_cfg.window
        n_edges = hi - 1 - lo

        win_omds = noisy_rels_j[:, lo:hi - 1, :]   # (S, n-1, 6) — JAX slice, no copy
        win_gt   = gt_global_j[lo:hi]               # (n, 6)
        win_gt_R = gt_R_mats_j[lo:hi]               # (n, 3, 3)

        # Two batched GPU calls — prec_batched_fn compiled once in main(), stable reference
        kappas = prec_batched_fn(win_omds[:, :, 3:])   # (S, n-1)
        omegas = prec_batched_fn(win_omds[:, :, :3])    # (S, n-1)

        # One batched GPU call
        theta_opts = batched_denoiser_fn(win_omds, win_gt, win_gt_R,
                                         kappas, omegas)      # (S, n-1, 6)

        corrected = np.array(win_omds) + np.array(theta_opts)  # (S, n-1, 6)
        write_lo = exp_cfg.overlap // 2 if pos > 0 else 0
        write_hi = n_edges - exp_cfg.overlap // 2 if hi < n_poses - 1 else n_edges
        denoised_rels[:, lo + write_lo:lo + write_hi, :] = corrected[:, write_lo:write_hi, :]

        pos += stride

    return [integrate_poses(denoised_rels[s]) for s in range(S)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp42: SE-Sync Structured GN Inner Solver for Odometry Denoising"
    )
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--seqs",       type=str, default="00,01,02,05,06,07,08,09,10")
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
    parser.add_argument("--output-dir", type=str,   default=os.path.expanduser("~/exp_res"),
                        help="Directory to write run results (timestamped sub-dir created automatically)")
    args = parser.parse_args()

    # Create timestamped output directory
    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"exp42_{run_ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    inner_cfg = InnerCfg(n_iters_rot=10, damping=1e-4)
    outer_cfg = OuterCfg(
        n_trans1=args.n_trans1,
        n_rot=args.n_rot,
        n_trans2=args.n_trans2,
        lr_trans=args.lr_trans,
        lr_rot=args.lr_rot,
    )
    exp_cfg   = ExpCfg(
        window=args.window,
        overlap=args.overlap,
        sigma_t=args.sigma_t,
        sigma_r=args.sigma_r,
        seeds=args.seeds,
    )

    # Build JIT-compiled denoisers once for this window size
    print(f"Compiling denoisers (window={args.window}, seeds={args.seeds})...")
    t0 = time.time()
    denoiser_fn         = build_denoiser(args.window, inner_cfg, outer_cfg, exp_cfg)
    batched_denoiser_fn = build_batched_denoiser(args.window, inner_cfg, outer_cfg, exp_cfg)

    # Build prec_batched_fn once using functools.partial — stable reference, no recompile per sequence
    lk, mr = exp_cfg.local_k, exp_cfg.max_kappa_ratio
    prec_batched_fn = jax.jit(jax.vmap(functools.partial(_sigma_and_precision,
                                                          local_k=lk,
                                                          max_kappa_ratio=mr)))

    dummy_odom = jnp.zeros((args.window - 1, 6))
    dummy_gt   = jnp.zeros((args.window, 6))
    dummy_gR   = jnp.zeros((args.window, 3, 3))
    dummy_k    = jnp.ones(args.window - 1)
    dummy_w    = jnp.ones(args.window - 1)
    # Warm up single denoiser (used for synthetic)
    _ = denoiser_fn(dummy_odom, dummy_gt, dummy_gR, dummy_k, dummy_w).block_until_ready()
    # Warm up batched denoiser and precision function (used for KITTI)
    dummy_omds = jnp.zeros((args.seeds, args.window - 1, 6))
    dummy_ks   = jnp.ones((args.seeds, args.window - 1))
    dummy_ws   = jnp.ones((args.seeds, args.window - 1))
    _ = batched_denoiser_fn(dummy_omds, dummy_gt, dummy_gR, dummy_ks, dummy_ws).block_until_ready()
    _ = prec_batched_fn(dummy_omds[:, :, :3]).block_until_ready()
    print(f"  Compiled in {time.time() - t0:.1f}s  (no retrace after this)")

    # Collect results
    all_dT, all_dR, all_dC = [], [], []
    all_seq_results = {}  # seq_id -> list of per-seed dicts

    if args.synthetic:
        sequences = [("synth", args.n_poses_synth)]
    else:
        if args.kitti_root is None:
            raise ValueError("Provide --kitti-root or use --synthetic")
        sequences = [(s.strip(), None) for s in args.seqs.split(",")]

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
                denoised_global = denoise_sequence(
                    noisy_rel, gt_global, gt_R_mats, denoiser_fn, exp_cfg
                )
                elapsed = time.time() - t_start
                poses_per_sec = len(gt_global) / (elapsed + 1e-9)
                dT, dR, dC = delta_metric(noisy_global, denoised_global, gt_global)
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
            from dsg_jit.core.math3d import so3_log as so3l
            gt_global = np.zeros((len(gt_mats), 6), dtype=np.float32)
            for i, T in enumerate(gt_mats):
                gt_global[i, :3] = T[:3, 3]
                gt_global[i, 3:] = np.array(so3l(jnp.array(T[:3, :3])))
            gt_R_mats = gt_mats[:, :3, :3]
            gt_rel    = relative_poses_from_mats(gt_mats)

            # Generate all S noisy trajectories upfront
            noisy_rels = np.stack([
                _add_kitti_noise(gt_rel, seed, seq_hash, args.sigma_t, args.sigma_r)
                for seed in range(args.seeds)
            ])   # (S, N-1, 6)

            # Denoise all seeds in one batched pass
            t_start = time.time()
            denoised_globals = denoise_sequence_batched(
                noisy_rels, gt_global, gt_R_mats, batched_denoiser_fn, prec_batched_fn, exp_cfg
            )
            elapsed = time.time() - t_start
            poses_per_sec = len(gt_global) * args.seeds / (elapsed + 1e-9)

            per_seed_sec = elapsed / args.seeds
            pps = len(gt_global) / (per_seed_sec + 1e-9)
            for s in range(args.seeds):
                noisy_global_s = integrate_poses(noisy_rels[s])
                dT, dR, dC = delta_metric(noisy_global_s, denoised_globals[s], gt_global)
                seq_dT.append(dT); seq_dR.append(dR); seq_dC.append(dC)
                print(f"  [{seq_id}|seed={s}]  ΔT={dT:+.1f}%  ΔR={dR:+.1f}%  ΔC={dC:+.1f}%  "
                      f"({pps:.0f} poses/s·seed)")
                seed_out = {"seq": seq_id, "seed": s, "n_poses": int(len(gt_global)),
                            "dT": float(dT), "dR": float(dR), "dC": float(dC),
                            "elapsed_s": float(per_seed_sec), "poses_per_s": float(pps)}
                seq_seed_results.append(seed_out)
                with open(os.path.join(run_dir, f"{seq_id}_seed_{s:04d}.json"), "w") as fp:
                    json.dump(seed_out, fp, indent=2)
            print(f"  [{seq_id}] {args.seeds} seeds batched  total {poses_per_sec:.0f} poses/s·seed")

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
        print("\n=== Exp42 Summary ===")
        print(f"  σ_t={args.sigma_t}  window={args.window}  overlap={args.overlap}")
        print(f"  Mean across seqs:  ΔT={np.mean(all_dT):+.1f}%  "
              f"ΔR={np.mean(all_dR):+.1f}%  ΔC={np.mean(all_dC):+.1f}%")

        config = {
            "exp": "exp42",
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
            f"Exp42 Run  {run_ts}",
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
