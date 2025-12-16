# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
JIT-friendly optimization wrappers and training utilities for DSG-JIT.

This module provides higher-level utilities that sit on top of the core
solvers in `optimization.solvers`. They are responsible for:

    • Building JIT-compiled solve functions for a fixed world model-backed
      factor graph
    • Wrapping Gauss–Newton in a functional interface (solve(x0) -> x_opt)
    • Supporting differentiable inner loops for meta-learning experiments
    • Implementing simple trainer-style loops used in Phase 4 experiments

Typical Usage
-------------
The experiments in `experiments/` use this module to:

    • Construct a `WorldModel`-backed factor graph (SE3, voxels, hybrid)
    • Get a JIT-compiled residual or objective from the world model
      (e.g., via :meth:`WorldModel.build_residual`, which internally groups
      factors by type and shape and uses :func:`jax.vmap` for efficiency)
    • Build a `solve_once(x0)` function using Gauss–Newton
    • Use `jax.grad` or `jax.value_and_grad` over an outer loss that depends
      on the optimized state

Example patterns include:

    • Learning SE3 odometry measurements by backpropagating through the
      inner Gauss–Newton solve
    • Learning voxel observation points that make a grid consistent with
      known ground-truth centers
    • Learning factor-type weights (log-scales) for odometry vs. observations
      via supervised losses on final poses/voxels

Key Utilities (typical contents)
--------------------------------
build_jit_gauss_newton(...)
    Given a WorldModel and a GNConfig, returns a JIT-compiled function:
        solve_once(x0) -> x_opt

build_param_residual(...)
    Wraps a residual function so that it depends both on the state `x` and
    on learnable parameters `theta` (e.g., measurements, observation points).

DSGTrainer (if present)
    A lightweight helper class implementing:
        - inner_solve(theta): run Gauss–Newton or GD on the graph
        - loss(theta): compute a supervised loss on the optimized state
        - step(theta): one gradient step on theta

Design Goals
------------
• Separate concerns:
    The low-level solver logic lives in `solvers.py`, while experiment-
    specific JIT wiring and training loops live here.

• Encourage functional patterns:
    All wrappers aim to expose pure functions that JAX can JIT and
    differentiate, avoiding hidden state and side effects.

• Make research experiments easy:
    This is the layer where new meta-learning or differentiable-graph
    experiments should be prototyped before they are promoted into a
    more general API.

Notes
-----
Because these wrappers are tailored to DSG-JIT’s factor graph structure,
they assume:

    • Residual functions derived from :class:`WorldModel`, e.g.
      :meth:`WorldModel.build_residual` and its hyper-parameterized
      variants
    • State vectors packed/unpacked via the world model’s core graph
      machinery (``WorldModel.pack_state`` / ``WorldModel.unpack_state``)

When modifying or extending this module, take care to preserve JIT
and grad-friendliness: avoid Python-side mutation inside jitted
functions and keep logic purely functional wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from dsg_jit.optimization.solvers import gauss_newton, gauss_newton_manifold, GNConfig

if TYPE_CHECKING:
    # Imported only for static type checking to avoid circular import
    from dsg_jit.world.model import WorldModel


@dataclass
class JittedGN:
    """JIT-compiled Gauss–Newton solver for a fixed world model-backed factor graph.

    Note
    ----
    This wrapper targets the Euclidean solver :func:`gauss_newton`. For
    SE(3)/manifold problems use :class:`JittedGNManifold` instead.

    This lightweight wrapper stores a jitted solve function and the
    configuration used to build it. Typical usage:

        residual_fn = wm.build_residual()  # vmap-optimized residual
        cfg = GNConfig(...)
        jgn = JittedGN.from_residual(residual_fn, cfg)
        x_opt = jgn(x0)

    :param fn: JIT-compiled function that maps an initial state
               vector ``x0`` to an optimized state ``x_opt``.
    :param cfg: Gauss–Newton configuration used when building
                the jitted solver.
    """
    fn: Callable[[jnp.ndarray], jnp.ndarray]
    cfg: GNConfig

    def __call__(self, x0: jnp.ndarray) -> jnp.ndarray:
        """Run the jitted Gauss–Newton solve on an initial state.

        :param x0: Initial flat state vector to optimize.
        :return: Optimized state vector after running Gauss–Newton.
        """
        return self.fn(x0)

    @staticmethod
    def from_residual(
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
        cfg: GNConfig,
    ) -> "JittedGN":
        """Construct a :class:`JittedGN` from a residual function.

        This wraps :func:`gauss_newton` with the provided configuration
        and JIT-compiles the resulting ``solve(x0)`` function.

        :param residual_fn: Residual function ``r(x)`` returning the stacked
                            residual vector for a fixed factor graph.
        :param cfg: Gauss–Newton configuration (step limits, damping, etc.).
        :return: A :class:`JittedGN` instance whose ``__call__`` method
                 runs the jitted Gauss–Newton solve.
        """
        # Wrap existing gauss_newton. cfg is closed over and treated as static.
        def solve(x0: jnp.ndarray) -> jnp.ndarray:
            return gauss_newton(residual_fn, x0, cfg)

        # jit the whole solve for this graph
        jitted = jax.jit(solve)
        return JittedGN(fn=jitted, cfg=cfg)

    @staticmethod
    def from_world_model(
        wm: "WorldModel",
        cfg: GNConfig,
    ) -> "JittedGN":
        """Construct a :class:`JittedGN` directly from a :class:`WorldModel`.

        This helper calls :meth:`WorldModel.build_residual` to obtain the
        vmap-optimized residual function for the current world, and then
        wraps it in a jitted Gauss–Newton solve.

        Typical usage::

            wm = WorldModel()
            # ... add variables, factors, register residuals ...
            jgn = JittedGN.from_world_model(wm, GNConfig(max_iters=20))
            x0, _ = wm.pack_state()
            x_opt = jgn(x0)

        :param wm: World model whose factor graph defines the optimization
                   problem. Its :meth:`build_residual` method is used to
                   obtain the residual function.
        :param cfg: Gauss–Newton configuration (step limits, damping, etc.).
        :return: A :class:`JittedGN` instance whose ``__call__`` method
                 runs the jitted Gauss–Newton solve using the world model’s
                 vmap-optimized residual.
        """
        residual_fn = wm.build_residual()
        return JittedGN.from_residual(residual_fn, cfg)


@dataclass
class JittedGNManifold:
    """JIT-compiled manifold Gauss–Newton solver for a fixed graph.

    This wrapper is intended for SLAM-style problems where the packed state
    vector is a concatenation of manifold variables (e.g., SE(3) poses and
    R^3 landmarks). It closes over the residual function and manifold
    metadata and returns a single jitted solve function.

    Typical usage::

        residual_fn = wm.build_residual()
        manifold_types, block_slices = build_manifold_metadata(...)
        cfg = GNConfig(max_iters=1)
        jgn = JittedGNManifold.from_residual(residual_fn, manifold_types, block_slices, cfg)
        x_opt = jgn(x0)

    IMPORTANT:
        To avoid repeated compilation, construct this once and reuse it for
        every incremental step. Ensure the shapes/dtypes of ``x0`` and the
        residual output remain constant across steps (template mode).

    :param fn: JIT-compiled function mapping ``x0`` -> ``x_opt``.
    :param cfg: Gauss–Newton configuration.
    """

    fn: Callable[[jnp.ndarray], jnp.ndarray]
    cfg: GNConfig

    def __call__(self, x0: jnp.ndarray) -> jnp.ndarray:
        """Run the jitted manifold Gauss–Newton solve."""
        return self.fn(x0)

    @staticmethod
    def from_residual(
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
        manifold_types: Any,
        block_slices: Any,
        cfg: GNConfig,
    ) -> "JittedGNManifold":
        """Construct a :class:`JittedGNManifold` from residual + metadata.

        :param residual_fn: Residual function ``r(x)``.
        :param manifold_types: Per-block manifold type strings.
        :param block_slices: Per-block slices into the packed vector.
        :param cfg: Solver configuration.
        :return: A reusable, jitted solver.
        """

        def solve(x0: jnp.ndarray) -> jnp.ndarray:
            return gauss_newton_manifold(
                residual_fn=residual_fn,
                x0=x0,
                manifold_types=manifold_types,
                block_slices=block_slices,
                cfg=cfg,
            )

        # JIT the whole solve; cfg/manifold metadata are closed over.
        jitted = jax.jit(solve)
        return JittedGNManifold(fn=jitted, cfg=cfg)

    @staticmethod
    def from_world_model(
        wm: "WorldModel",
        manifold_types: list[str],
        block_slices: list[slice],
        cfg: GNConfig,
    ) -> "JittedGNManifold":
        """Construct a manifold GN solver directly from a :class:`WorldModel`.

        This helper obtains the residual via :meth:`WorldModel.build_residual`.

        :param wm: World model.
        :param manifold_types: Per-block manifold types.
        :param block_slices: Per-block slices.
        :param cfg: Solver configuration.
        :return: A reusable, jitted solver.
        """
        residual_fn = wm.build_residual()
        return JittedGNManifold.from_residual(residual_fn=residual_fn, manifold_types=manifold_types, block_slices=block_slices, cfg=cfg)