# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Core typed data structures for DSG-JIT.

This module defines the lightweight container classes used throughout the
differentiable factor graph system. These types are intentionally minimal:
they store only structural information and initial values, while all numerical
operations are performed by JAX-compiled functions in the optimization layer.

Classes
-------
Variable
    Represents a node in the factor graph. A variable contains:
    - id: Unique identifier (string or int)
    - value: Initial numeric state, typically a 1-D JAX array
    - metadata: Optional dictionary for semantic/scene-graph information

Factor
    Represents a constraint between one or more variables. A factor contains:
    - id: Unique identifier
    - type: String key selecting a residual function
    - var_ids: Ordered list of variable ids used by the residual
    - params: Dictionary of parameters passed into the residual function
              (e.g., weights, measurements, priors)

Notes
-----
These objects are deliberately simple and mutable; they are not meant to be
used directly inside JAX-compiled functions. During optimization, the
FactorGraph packs variable values into a flat JAX array `x`, ensuring that
JIT-compiled solvers operate on purely functional data.

This module forms the backbone of DSG-JIT's dynamic scene graph architecture,
enabling hybrid SE3, voxel, and semantic structures to be represented in a
unified factor graph.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import NewType, Dict, Any

NodeId = NewType("NodeId", int)
FactorId = NewType("FactorId", int)


@dataclass(frozen=True)
class Pose3:
    """Minimal 3D pose holder: we will later switch to proper SE(3) utilities."""
    # For now: (x, y, z, roll, pitch, yaw)
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class Variable:
    """Generic optimization variable node in the factor graph."""
    id: NodeId
    type: str          # e.g. "pose3", "landmark3d", "object_pose", etc.
    value: Any         # JAX array or simple tuple; we will standardize later.


@dataclass
class Factor:
    """Abstract factor connecting variables."""
    id: FactorId
    type: str          # e.g. "odom", "loop_closure", "object_prior"
    var_ids: tuple[NodeId, ...]
    params: Dict[str, Any]  # Measurement, noise, etc.