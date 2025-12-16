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
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp

from dsg_jit.core.math3d import se3_retract_left

ObjectiveFn = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass
class GDConfig:
    learning_rate: float = 1e-1
    max_iters: int = 200


def gradient_descent(objective: ObjectiveFn, x0: jnp.ndarray, cfg: GDConfig) -> jnp.ndarray:
    """Simple gradient descent optimizer.

    Performs iterative updates of the form::

        x_{k+1} = x_k - learning_rate * 
abla f(x_k)

    until ``max_iters`` is reached.

    :param objective: Objective function ``f(x)`` that maps a state vector to a scalar loss.
    :type objective: Callable[[jnp.ndarray], jnp.ndarray]
    :param x0: Initial state vector.
    :type x0: jnp.ndarray
    :param cfg: Gradient-descent configuration (learning rate and number of iterations).
    :type cfg: GDConfig
    :return: Optimized state vector after gradient descent.
    :rtype: jnp.ndarray
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
    """Damped Newton optimizer for small problems.

    Uses a Levenberg–Marquardt-style update::

        (H + λ I) \\delta = abla f(x)
        x_{k+1} = x_k - \\delta

    where ``H`` is the Hessian of the objective and ``λ`` is a damping factor.

    :param objective: Objective function ``f(x)`` that maps a state vector to a scalar loss.
    :type objective: Callable[[jnp.ndarray], jnp.ndarray]
    :param x0: Initial state vector.
    :type x0: jnp.ndarray
    :param cfg: Newton solver configuration (number of iterations and damping).
    :type cfg: NewtonConfig
    :return: Optimized state vector after damped Newton iterations.
    :rtype: jnp.ndarray
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
    """Gauss–Newton on a residual function ``r(x): R^n -> R^m``.

    The algorithm forms the normal equations::

        J^T J \\delta = J^T r
        x_{k+1} = x_k - \\delta

    with optional diagonal damping and step-size clamping for stability.

    :param residual_fn: Residual function ``r(x)`` returning a 1D array of shape ``(m,)``.
    :type residual_fn: Callable[[jnp.ndarray], jnp.ndarray]
    :param x0: Initial state vector of shape ``(n,)``.
    :type x0: jnp.ndarray
    :param cfg: Gauss–Newton configuration (iterations, damping, step-norm clamp).
    :type cfg: GNConfig
    :return: Optimized state vector after Gauss–Newton iterations.
    :rtype: jnp.ndarray
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
    block_slices: Union[Mapping[Any, slice], Sequence[Tuple[Any, slice]]],
    manifold_types: Union[Mapping[Any, str], Sequence[Tuple[Any, str]]],
    cfg: GNConfig,
) -> jnp.ndarray:
    """Manifold-aware Gauss–Newton solver.

    This variant still solves in a flat parameter space, but applies updates
    block-wise using the appropriate manifold retraction. In particular:

    * Blocks marked as ``"se3"`` are updated via ``se3_retract_left`` in the
      Lie algebra ``se(3)``.
    * Blocks marked as ``"euclidean"`` are updated additively.

    :param residual_fn: Residual function ``r(x)`` returning a 1D array of shape ``(m,)``.
    :type residual_fn: Callable[[jnp.ndarray], jnp.ndarray]
    :param x0: Initial flat state vector of shape ``(n,)``.
    :type x0: jnp.ndarray
    :param block_slices: Mapping from node identifier to slice in ``x`` defining that variable's block.
                         May be a dict or a sequence of (node_id, slice) pairs.
    :type block_slices: Union[Mapping[Any, slice], Sequence[Tuple[Any, slice]]]
    :param manifold_types: Mapping from node identifier to manifold label (e.g. ``"se3"`` or ``"euclidean"``).
                           May be a dict or a sequence of (node_id, manifold_type) pairs.
    :type manifold_types: Union[Mapping[Any, str], Sequence[Tuple[Any, str]]]
    :param cfg: Gauss–Newton configuration (iterations, damping, step-norm clamp).
    :type cfg: GNConfig
    :return: Optimized state vector after manifold-aware Gauss–Newton iterations.
    :rtype: jnp.ndarray
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
        
        x_new = x

        if isinstance(block_slices, Mapping) and isinstance(manifold_types, Mapping):
            # Legacy path: dict lookups.
            for nid, sl in block_slices.items():
                d_i = delta_scaled[sl]
                x_i = x[sl]
                mtype = manifold_types.get(nid, "euclidean")

                if mtype == "se3":
                    # Interpret d_i as a twist in se(3) and apply left retraction
                    x_i_new = se3_retract_left(x_i, -d_i)
                else:
                    # Euclidean update
                    x_i_new = x_i - d_i

                x_new = x_new.at[sl].set(x_i_new)
        else:
            # JAX-friendly path: iterate in lockstep over (nid, slice) and (nid, mtype)
            # without dict indexing. `block_slices` and `manifold_types` are expected
            # to be aligned and ordered consistently.
            for (nid, sl), (_, mtype) in zip(block_slices, manifold_types):
                d_i = delta_scaled[sl]
                x_i = x[sl]

                if mtype == "se3":
                    x_i_new = se3_retract_left(x_i, -d_i)
                else:
                    x_i_new = x_i - d_i

                x_new = x_new.at[sl].set(x_i_new)

        x = x_new

    return x