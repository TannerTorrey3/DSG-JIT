# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Nonlinear optimization solvers for DSG-JIT.

This module implements the core iterative solvers used throughout the system,
with a focus on JAX-friendly, JIT-compilable routines that operate on flat
state vectors and manifold-aware blocks (e.g., SE(3) poses).

The solvers are designed to work with residual functions produced by
`core.factor_graph.FactorGraph`, and are used in:

    • Pure SE3 SLAM chains
    • Voxel grid smoothness / observation problems
    • Hybrid SE3 + voxel joint optimization
    • Differentiable experiments where measurements or weights are learned

Key Concepts
------------
GNConfig
    Dataclass holding configuration for Gauss–Newton:
    - max_iters: maximum number of GN iterations
    - damping: Levenberg–Marquardt-style damping
    - max_step_norm: optional clamp on update step size
    - verbose / debug flags (if enabled)

gauss_newton(residual_fn, x0, cfg)
    Classic Gauss–Newton on a flat Euclidean state:
    - residual_fn: r(x) -> (m,) JAX array
    - x0: initial state
    - cfg: GNConfig

    Computes updates using normal equations:
        Jᵀ J Δx = -Jᵀ r
    and returns the optimized state.

gauss_newton_manifold(residual_fn, x0, block_slices, manifold_types, cfg)
    Manifold-aware Gauss–Newton:
    - residual_fn: r(x) -> (m,)
    - x0: initial flat state vector
    - block_slices: NodeId -> slice in x
    - manifold_types: NodeId -> {"se3", "euclidean", ...}
    - cfg: GNConfig

    For SE3 blocks:
        • The update is computed in the tangent space (se(3))
        • Applied via retract / exponential map
        • Ensures updates stay on the manifold

    For Euclidean blocks:
        • Updates are applied additively.

Design Goals
------------
• Fully JAX-compatible:
    All heavy operations are written in terms of JAX primitives so that
    solvers can be JIT-compiled and differentiated through when needed.

• Stable and controlled:
    Optional damping and step-norm clamping help avoid NaNs and divergence
    in difficult configurations (e.g., bad initialization or large residuals).

• Reusable:
    Experiments and higher-level training loops (e.g., in `experiments/`
    and `optimization/jit_wrappers.py`) call into these solvers as the
    core iterative engine for DSG-JIT.

Notes
-----
These solvers are intentionally minimal and generic. They do not know
anything about SE3 or voxels directly; instead, they rely on the factor
graph and manifold metadata to interpret the state vector correctly.

If you add new manifold types (e.g., quaternions or higher-dimensional
poses), extend the manifold handling logic in the manifold-aware solver.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from core.math3d import se3_retract_left

ObjectiveFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass
class GDConfig:
    learning_rate: float = 1e-1
    max_iters: int = 200


def gradient_descent(objective: ObjectiveFn, x0: jnp.ndarray, cfg: GDConfig) -> jnp.ndarray:
    """
    Very simple gradient descent loop, enough for tests.

    Args:
        objective: f(x) -> scalar loss
        x0: initial state vector
        cfg: hyperparameters

    Returns:
        x_opt: optimized state vector
    """
    grad_fn = jax.grad(objective)

    x = x0
    for _ in range(cfg.max_iters):
        g = grad_fn(x)
        x = x - cfg.learning_rate * g
    return x

@dataclass
class NewtonConfig:
    max_iters: int = 30
    damping: float = 1e-3  # LM-style diagonal damping


def damped_newton(objective: ObjectiveFn, x0: jnp.ndarray, cfg: NewtonConfig) -> jnp.ndarray:
    """
    Simple damped Newton / Gauss-Newton-like optimizer.

    For small problems, we can afford full Hessian:
       delta = (H + λ I)^-1 g
       x_new = x - delta
    """
    grad_fn = jax.grad(objective)
    hess_fn = jax.hessian(objective)

    x = x0
    for _ in range(cfg.max_iters):
        g = grad_fn(x)
        H = hess_fn(x)

        n = x.shape[0]
        H_damped = H + cfg.damping * jnp.eye(n)

        # Solve H_damped * delta = g
        delta = jnp.linalg.solve(H_damped, g)

        x = x - delta

    return x

@dataclass
class GNConfig:
    max_iters: int = 20
    damping: float = 1e-3       # LM-style diagonal damping
    max_step_norm: float = 1.0  # clamp step size for stability


def gauss_newton(residual_fn: ObjectiveFn, x0: jnp.ndarray, cfg: GNConfig) -> jnp.ndarray:
    """
    Gauss-Newton on residual function r(x): R^n -> R^m.

    residual_fn: x -> r, with shapes:
        x.shape == (n,)
        r.shape == (m,)

    J = dr/dx has shape (m, n), matching math convention.
    """
    J_fn = jax.jacobian(residual_fn)  # J: (m, n)

    def step(x: jnp.ndarray) -> jnp.ndarray:
        r = residual_fn(x)    # (m,)
        J = J_fn(x)           # (m, n)

        H = J.T @ J           # (n, n)
        g = J.T @ r           # (n,)

        n = x.shape[0]
        H_damped = H + cfg.damping * jnp.eye(n)

        delta = jnp.linalg.solve(H_damped, g)  # (n,)

        # Optional step-size clamp to avoid huge jumps
        step_norm = jnp.linalg.norm(delta)
        scale = jnp.minimum(1.0, cfg.max_step_norm / (step_norm + 1e-9))

        return x - scale * delta

    x = x0
    for _ in range(cfg.max_iters):
        x = step(x)
    return x

def gauss_newton_manifold(
    residual_fn: ObjectiveFn,
    x0: jnp.ndarray,
    block_slices: Dict,     # NodeId -> slice
    manifold_types: Dict,   # NodeId -> "se3" / "euclidean"
    cfg: GNConfig,
) -> jnp.ndarray:
    """
    Manifold-aware Gauss-Newton:

      - residual_fn: x -> r(x), with x in R^n, r in R^m
      - block_slices: maps each NodeId to a slice in x
      - manifold_types: maps each NodeId to a manifold label, currently:
            - "se3": updated via SE(3) retraction
            - "euclidean": updated via simple subtraction

    Still solve in the global flat space, but apply updates per-variable
    using the appropriate retraction.
    """
    # J: (m, n), r: (m,)
    J_fn = jax.jacobian(residual_fn)

    x = x0

    for _ in range(cfg.max_iters):
        r = residual_fn(x)       # (m,)
        J = J_fn(x)              # (m, n)

        H = J.T @ J              # (n, n)
        g = J.T @ r              # (n,)

        n = x.shape[0]
        H_damped = H + cfg.damping * jnp.eye(n)

        delta = jnp.linalg.solve(H_damped, g)  # (n,)

        # Step size clamp
        step_norm = jnp.linalg.norm(delta)
        scale = jnp.minimum(1.0, cfg.max_step_norm / (step_norm + 1e-9))
        delta_scaled = scale * delta

        # Apply updates per variable block using the right manifold
        x_new = x
        for nid, sl in block_slices.items():
            d_i = delta_scaled[sl]
            x_i = x[sl]
            mtype = manifold_types[nid]

            if mtype == "se3":
                # Interpret d_i as a twist in se(3) and apply left retraction
                x_i_new = se3_retract_left(x_i, -d_i)
            else:
                # Euclidean update
                x_i_new = x_i - d_i

            x_new = x_new.at[sl].set(x_i_new)

        x = x_new

    return x