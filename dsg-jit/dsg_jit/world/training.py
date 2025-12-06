# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
High-level training utilities for differentiable scene graph experiments.

This module provides a small training harness that sits on top of:

    • `world.model.WorldModel` / `world.scene_graph.SceneGraphWorld`
    • JAX-based optimizers and Gauss–Newton solvers
    • Residual functions from `slam.measurements`

Its main role is to support *meta-learning* and *hyperparameter learning*
over the differentiable DSG-JIT engine. Examples include:

    • Learning factor-type weights (e.g. odometry vs. observation).
    • Learning measurement parameters (odom SE3 chains, voxel obs).
    • Running outer-loop gradient descent over:
        - log-scale weights,
        - observation locations,
        - or other “theta” parameters that influence the inner solve.

Typical structure
-----------------
A typical training loop implemented here follows this pattern:

    1. Build a world / scene graph for a given scenario.
    2. Build a residual function that depends both on:
           - the state x (poses, voxels, etc.), and
           - learnable parameters θ (e.g. measurements, log-scales).
    3. Run an inner optimization (Gauss–Newton or gradient descent)
       to obtain x*(θ).
    4. Compute a supervised loss L(x*(θ), target).
    5. Differentiate L w.r.t. θ using JAX (`jax.grad` or `jax.value_and_grad`).
    6. Update θ with an outer optimizer step.

The `DSGTrainer` (or equivalent helper) encapsulates this pattern,
exposing `step` / `train_step`–style methods that return both the new
parameters and useful diagnostics (loss, gradient norms, etc.).

Design goals
------------
- **Keep experiments small**:
    Training logic lives here so individual experiments can focus on
    constructing the world and defining the supervision signal.
- **JAX-first design**:
    Training functions are written to be JIT-able and differentiable,
    allowing seamless scaling from toy experiments to larger graphs.
- **Research-friendly**:
    The code is intentionally lightweight and easy to modify for new
    research ideas around learnable costs, priors, and structure.
"""

from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp

from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.world.model import WorldModel


@dataclass
class InnerGDConfig:
    """
    Configuration for the inner gradient–descent solver.

    :param learning_rate: Step size used for each inner GD update on the state.
    :param max_iters: Maximum number of inner GD iterations.
    :param max_step_norm: Maximum allowed L2 norm of a single GD step; used to clamp overly large updates for numerical stability.
    """
    learning_rate: float = 1e-2
    max_iters: int = 40
    max_step_norm: float = 1.0  # simple safety clamp


@dataclass
class DSGTrainer:
    """
    High-level trainer for differentiable DSG experiments.

    This class encapsulates a simple bi-level optimization pattern where:
    an inner loop solves for the scene graph state x, and an outer loop
    optimizes meta-parameters such as factor-type weights.

    :param wm: World model containing the factor graph and scene graph.
    :param factor_type_order: Ordered list of factor type names; each entry corresponds to a log-scale entry in the weight vector.
    :param inner_cfg: Configuration for the inner gradient–descent solver applied to the state.
    """
    wm: WorldModel
    factor_type_order: List[str]
    inner_cfg: InnerGDConfig

    def __post_init__(self):
        """
        Post-initialization hook.

        This method caches the underlying factor graph from the world model
        and builds a residual function that accepts per-factor-type log-scales.
        """
        self.fg: FactorGraph = self.wm.fg
        self.residual_w = self.fg.build_residual_function_with_type_weights(
            self.factor_type_order
        )

    def solve_state(self, log_scales: jnp.ndarray) -> jnp.ndarray:
        """
        Run the inner optimization to solve for the state vector.

        Given a vector of log-scales for factor types, this method performs
        explicit gradient descent on the objective

            0.5 * || r(x, log_scales) ||^2,

        where r is the weighted residual function built from the factor graph.

        :param log_scales: Array of shape ``(T,)`` containing per-factor-type log-scale weights.
        :return: Optimized flat state vector ``x`` after the inner GD loop.
        """
        x0, _ = self.wm.pack_state()

        def loss_x(x, log_scales):
            r = self.residual_w(x, log_scales)
            return 0.5 * jnp.sum(r * r)

        grad_loss_x = jax.grad(loss_x)

        # plain Python loop (no jax.lax.while_loop, easier to debug)
        x = x0
        for _ in range(self.inner_cfg.max_iters):
            g = grad_loss_x(x, log_scales)

            # gradient may be large; clamp the step
            step = -self.inner_cfg.learning_rate * g
            step_norm = jnp.linalg.norm(step)
            max_norm = self.inner_cfg.max_step_norm

            def clamp_step(step, step_norm, max_norm):
                scale = max_norm / (step_norm + 1e-8)
                return step * scale

            step = jax.lax.cond(
                step_norm > max_norm,
                lambda _: clamp_step(step, step_norm, max_norm),
                lambda _: step,
                operand=None,
            )

            x = x + step

        return x

    def unpack_state(self, x: jnp.ndarray):
        """
        Unpack a flat state vector into a NodeId-keyed dictionary.

        This is a thin wrapper around the factor graph's ``unpack_state``
        that uses the index structure implied by the current world model.

        :param x: Flat state vector to be unpacked.
        :return: Mapping from ``NodeId`` to the corresponding slice of ``x`` as a JAX array.
        """
        _, index = self.wm.pack_state()
        return self.wm.unpack_state(x, index)