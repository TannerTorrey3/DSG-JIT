# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
JIT-friendly optimization wrappers and training utilities for DSG-JIT.

This module provides higher-level utilities that sit on top of the core
solvers in `optimization.solvers`. They are responsible for:

    • Building JIT-compiled solve functions for a fixed factor graph
    • Wrapping Gauss–Newton in a functional interface (solve(x0) -> x_opt)
    • Supporting differentiable inner loops for meta-learning experiments
    • Implementing simple trainer-style loops used in Phase 4 experiments

Typical Usage
-------------
The experiments in `experiments/` use this module to:

    • Construct a `FactorGraph` (SE3, voxels, hybrid)
    • Get a JIT-compiled residual or objective from the graph
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
    Given a FactorGraph and a GNConfig, returns a JIT-compiled function:
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

    • A fixed graph topology during the inner optimization loop
    • Residual functions derived from `FactorGraph.build_*`
    • State vectors packed/unpacked via the core graph machinery

When modifying or extending this module, take care to preserve JIT
and grad-friendliness: avoid Python-side mutation inside jitted
functions and keep logic purely functional wherever possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from dsg_jit.optimization.solvers import gauss_newton, GNConfig


@dataclass
class JittedGN:
    """JIT-compiled Gauss–Newton solver for a fixed factor graph.

    This lightweight wrapper stores a jitted solve function and the
    configuration used to build it. Typical usage:

        residual_fn = fg.build_residual_function()
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