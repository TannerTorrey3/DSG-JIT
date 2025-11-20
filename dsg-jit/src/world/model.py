"""
World-level wrapper around the core factor graph.

This module defines the *world model* abstraction: a thin, typed layer on
top of `core.factor_graph.FactorGraph` that knows about high-level
entities (poses, places, rooms, voxels, objects, agents) but still
remains generic enough to be reused across experiments.

Key responsibilities
--------------------
- Manage the underlying `FactorGraph` instance.
- Provide ergonomic helpers to:
    • Add variables with automatically assigned `NodeId`s.
    • Add typed factors (e.g. priors, odometry, attachments).
    • Pack / unpack state vectors for optimization.
- Maintain simple bookkeeping structures (e.g. maps from user-facing
  handles / indices back to `NodeId`s) so that experiments and higher-
  level layers do not need to manipulate `NodeId` directly.

Role in DSG-JIT
---------------
The world model is the bridge between:

    • Low-level optimization (factor graph, residual functions, manifolds)
    • High-level scene graph abstractions (poses, agents, rooms, voxels)

Experiments typically:

    1. Construct a `WorldModel`.
    2. Add variables & factors according to a scenario.
    3. Call into `optimization.solvers` to run Gauss–Newton or a
       manifold-aware variant using the world model’s factor graph.
    4. Decode and interpret the optimized state via the world model’s
       convenience accessors.

Design goals
------------
- **Thin wrapper**: keep most of the complexity in `FactorGraph`,
  `slam.manifold`, and residuals, so `WorldModel` stays small and easy to
  reason about.
- **Scene-friendly**: provide just enough structure that scene graphs and
  voxel modules can build on top of it without duplicating graph logic.
"""

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp

from core.factor_graph import FactorGraph
from core.types import Variable, Factor, NodeId, FactorId
from optimization.solvers import (
    gradient_descent, GDConfig,
    damped_newton, NewtonConfig,
    gauss_newton, GNConfig,
    gauss_newton_manifold,
)
from slam.manifold import build_manifold_metadata
from optimization.jit_wrappers import JittedGN


@dataclass
class WorldModel:
    fg: FactorGraph

    def __init__(self) -> None:
        self.fg = FactorGraph()

    def add_variable(self, var_type: str, value: jnp.ndarray) -> NodeId:
        """
        Allocate a new variable id, create the Variable, add it to the graph,
        and return its NodeId.
        """
        nid_int = len(self.fg.variables)
        nid = NodeId(nid_int)
        v = Variable(id=nid, type=var_type, value=value)
        self.fg.add_variable(v)
        return nid

    def add_factor(self, f_type: str, var_ids, params: Dict) -> FactorId:
        """
        Allocate a new factor id, create the Factor, add it to the graph,
        and return its FactorId.
        """
        fid_int = len(self.fg.factors)
        fid = FactorId(fid_int)

        # Normalize everything to NodeId
        node_ids = tuple(NodeId(int(vid)) for vid in var_ids)

        f = Factor(
            id=fid,
            type=f_type,
            var_ids=node_ids,
            params=params,
        )
        self.fg.add_factor(f)
        return fid

    def optimize(
        self,
        lr: float = 0.1,
        iters: int = 300,
        method: str = "gd",
        damping: float = 1e-3,
        max_step_norm: float = 1.0,
    ) -> None:
        """
        Optimize the current factor graph in-place.

        method:
          - "gd"          : gradient descent
          - "newton"      : damped Newton
          - "gn"          : Gauss-Newton on stacked state
          - "manifold_gn" : manifold-aware GN (SE(3) poses etc.)
        """
        x_init, index = self.fg.pack_state()
        residual_fn = self.fg.build_residual_function()

        if method == "gd":
            obj = self.fg.build_objective()
            cfg = GDConfig(learning_rate=lr, max_iters=iters)
            x_opt = gradient_descent(obj, x_init, cfg)

        elif method == "newton":
            obj = self.fg.build_objective()
            cfg = NewtonConfig(max_iters=iters, damping=damping)
            x_opt = damped_newton(obj, x_init, cfg)

        elif method == "gn":
            cfg = GNConfig(max_iters=iters, damping=damping, max_step_norm=max_step_norm)
            x_opt = gauss_newton(residual_fn, x_init, cfg)

        elif method == "manifold_gn":
            block_slices, manifold_types = build_manifold_metadata(self.fg)
            cfg = GNConfig(max_iters=iters, damping=damping, max_step_norm=max_step_norm)
            x_opt = gauss_newton_manifold(
                residual_fn, x_init, block_slices, manifold_types, cfg
            )

        elif method == "gn_jit":
            cfg = GNConfig(max_iters=iters, damping=damping, max_step_norm=max_step_norm)
            jgn = JittedGN.from_residual(residual_fn, cfg)
            x_opt = jgn(x_init)
        else:
            raise ValueError(f"Unknown optimization method '{method}'")

        # Write back
        values = self.fg.unpack_state(x_opt, index)
        for nid, val in values.items():
            self.fg.variables[nid].value = val