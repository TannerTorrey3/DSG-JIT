# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Core factor graph data structure for DSG-JIT.

This module implements a minimal, backend-agnostic factor graph. It stores
variables and factors and provides basic helpers for managing the graph
structure, but it does **not** contain any JAX, residual, or JIT-specific
logic. All residual construction, vmap batching, and solver orchestration
are handled by higher-level components such as :class:`dsg_jit.world.model.WorldModel`.

The FactorGraph stores:
    - Variables (nodes in the optimization graph)
    - Factors (constraints between variables)

Design Philosophy
-----------------
- Keep this layer small and generic so it can serve as a stable backend
  for multiple front-ends (WorldModel, alternative world models, or
  external bindings).
- Do not depend on JAX or expose residual-building APIs here; instead,
  expose only structural information (variables, factors, IDs).
- Allow higher layers to interpret variables and factors however they
  like (e.g., as poses, voxels, semantic objects) without baking that
  interpretation into the core graph.

Typical Usage
-------------
- ``add_variable`` / ``add_factor``: Build up the factor graph structure.
- Higher-level code (e.g., :mod:`dsg_jit.world.model`) inspects
  :attr:`variables` and :attr:`factors` to construct residual functions
  and objectives suitable for Gauss–Newton or gradient-based solvers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor


@dataclass
class FactorGraph:
    """Abstract factor graph for DSG-JIT.

    This class stores variables and factors and exposes simple helpers
    to register them. All residual-building and optimization logic is
    delegated to higher-level components such as :class:`WorldModel`.
    """
    variables: Dict[NodeId, Variable] = field(default_factory=dict)
    factors: Dict[FactorId, Factor] = field(default_factory=dict)
    # TODO: add NeRF/radiance-related metadata as needed in the future.

    def add_variable(self, var: Variable) -> None:
        """Register a new variable in the factor graph.

        This does *not* modify any existing factors; it simply makes the
        variable available to be referenced by factors.

        :param var: Variable to add to the graph. Its ``id`` must be unique.
        :type var: Variable
        """
        assert var.id not in self.variables
        self.variables[var.id] = var

    def add_factor(self, factor: Factor) -> None:
        """Register a new factor in the factor graph.

        The factor must only reference variables that already exist in
        :attr:`variables`.

        :param factor: Factor to add to the graph. Its ``id`` must be unique.
        :type factor: Factor
        """
        assert factor.id not in self.factors
        self.factors[factor.id] = factor
