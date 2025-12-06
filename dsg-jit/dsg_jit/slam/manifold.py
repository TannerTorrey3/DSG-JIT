# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Manifold utilities for SE(3) and Euclidean variables in DSG-JIT.

This module centralizes the *geometric* logic needed by manifold-aware
optimization routines, in particular:

    • SE(3) exponential / logarithm maps
    • Retraction and local parameterization for poses
    • Jacobian-friendly helpers for composing / inverting SE(3)
    • Metadata that maps variable types to their manifold model
      (e.g. "pose_se3" → "se3", "place1d" → "euclidean")

The core idea is that the optimizer (e.g. Gauss–Newton) should work in
a local tangent space while the *state* lives on a manifold (SE(3) for
poses, ℝⁿ for Euclidean variables). This module provides:

    • Primitive SE(3) operations:
        - `se3_exp`, `se3_log`         (tangent ↔ group)
        - `so3_exp`, `so3_log`         (rotation-only)
        - `relative_pose_se3`          (pose_a⁻¹ ∘ pose_b)
        - `se3_retract`                (pose ⊕ δξ update rule)

    • Manifold metadata helpers:
        - `TYPE_TO_MANIFOLD`           (str → {"se3", "euclidean", ...})
        - `get_manifold_for_var_type`
        - `build_manifold_metadata`    (NodeId → slice, manifold type)

Integration with the Optimizer
------------------------------
`optimization.solvers.gauss_newton_manifold` uses this module to:

    1. Split the global state vector into blocks per variable.
    2. Decide which update rule to apply:
        - SE(3) retraction for "pose_se3" blocks
        - Plain addition for Euclidean blocks
    3. Keep the core solver logic generic while remaining
       numerically stable on curved manifolds.

Design Goals
------------
• **Numerical stability**:
    Use small-angle fallbacks and well-conditioned SE(3) operations to
    avoid NaNs in optimization and differentiation.

• **Separation of concerns**:
    The factor graph and residuals should not hard-code SE(3) math; all
    manifold operations live here, behind a clean API.

• **JAX-friendly**:
    All functions are written in a way that is compatible with JIT
    compilation, `jax.grad`, and `jax.jvp` / `vmap`.

Notes
-----
This module currently focuses on SE(3) + Euclidean manifolds, but the
design allows extending to other manifolds (e.g. SO(2), quaternions,
Lie groups for velocities) by:

    • Adding new entries to `TYPE_TO_MANIFOLD`
    • Implementing the corresponding retract / exp / log primitives
    • Extending the manifold-aware solver dispatch if needed
"""

from __future__ import annotations

from typing import Dict, Tuple
import jax.numpy as jnp

from dsg_jit.core.types import NodeId
from dsg_jit.core.factor_graph import FactorGraph



TYPE_TO_MANIFOLD: Dict[str, str] = {
    "pose_se3": "se3",
    "place1d": "euclidean",
    "room1d": "euclidean",
    "landmark3d": "euclidean",  
    "voxel_cell": "euclidean", 
}


def get_manifold_for_var_type(var_type: str) -> str:
    """Return the manifold model name for a given variable type.

    This is a thin helper around :data:`TYPE_TO_MANIFOLD` that maps a
    high-level variable type tag (e.g. ``"pose_se3"``, ``"place1d"``)
    to the underlying manifold model used by manifold-aware solvers.

    :param var_type: Variable type string, such as ``"pose_se3"``,
        ``"place1d"``, ``"room1d"``, ``"landmark3d"`` or
        ``"voxel_cell"``.
    :return: The manifold model name (for example ``"se3"`` or
        ``"euclidean"``). If the type is unknown, ``"euclidean"`` is
        returned by default.
    """
    return TYPE_TO_MANIFOLD.get(var_type, "euclidean")


def build_manifold_metadata(
    packed_state: jnp.ndarray,
    fg: FactorGraph
) -> Tuple[Dict[NodeId, slice], Dict[NodeId, str]]:
    """Build manifold metadata for a factor graph.

    This function inspects the variables in a :class:`~core.factor_graph.FactorGraph`
    and constructs two lookup tables that are consumed by
    manifold-aware solvers such as
    :func:`optimization.solvers.gauss_newton_manifold`:

    * ``block_slices`` maps each :class:`~core.types.NodeId` to a
      :class:`slice` in the packed state vector.
    * ``manifold_types`` maps each :class:`~core.types.NodeId` to a
      short string describing the manifold model (e.g. ``"se3"`` or
      ``"euclidean"``).

    The indices produced by :meth:`core.factor_graph.FactorGraph.pack_state`
    may be stored either as slices or as ``(start, length)`` tuples;
    this helper normalizes everything to proper Python ``slice``
    objects so the solver does not need to handle multiple formats.

    :param fg: The factor graph whose variables should be analyzed to
        construct manifold metadata.
    :return: A tuple ``(block_slices, manifold_types)`` where
        ``block_slices`` is a mapping from :class:`~core.types.NodeId`
        to :class:`slice` in the flat state vector, and
        ``manifold_types`` is a mapping from :class:`~core.types.NodeId`
        to a manifold model name string.
    """
    _, index = packed_state

    block_slices: Dict[NodeId, slice] = {}
    manifold_types: Dict[NodeId, str] = {}

    for nid, var in fg.variables.items():
        idx = index[nid]

        # Normalize to a slice
        if isinstance(idx, slice):
            sl = idx
        else:
            # assume (start, length)
            start, length = idx
            sl = slice(start, start + length)

        block_slices[nid] = sl

        manifold = get_manifold_for_var_type(var.type)
        manifold_types[nid] = manifold

    return block_slices, manifold_types