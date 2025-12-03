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
    """Minimal 3D pose holder.

    This is a lightweight container for a 3D pose parameterized as
    (x, y, z, roll, pitch, yaw).

    :param x: Position along the x-axis in meters.
    :type x: float
    :param y: Position along the y-axis in meters.
    :type y: float
    :param z: Position along the z-axis in meters.
    :type z: float
    :param roll: Rotation around the x-axis in radians.
    :type roll: float
    :param pitch: Rotation around the y-axis in radians.
    :type pitch: float
    :param yaw: Rotation around the z-axis in radians.
    :type yaw: float
    """
    # For now: (x, y, z, roll, pitch, yaw)
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class Variable:
    """Generic optimization variable node in the factor graph.

    Each variable represents a node in the factor graph and stores its
    identifier, semantic type string, and current numeric value.

    :param id: Unique identifier for the variable node.
    :type id: NodeId
    :param type: Semantic type of the variable (e.g., "pose3", "landmark3d").
    :type type: str
    :param value: Initial or current numeric value for this variable, typically
        a 1-D JAX array or other array-like object.
    :type value: Any
    """
    id: NodeId
    type: str          # e.g. "pose3", "landmark3d", "object_pose", etc.
    value: Any         # JAX array or simple tuple; we will standardize later.


@dataclass
class Factor:
    """Abstract factor connecting one or more variables.

    A factor encodes a residual term in the overall objective, defined over
    an ordered tuple of variable ids and parameterized by a small dictionary
    of measurements, noise models, or other hyperparameters.

    :param id: Unique identifier for the factor.
    :type id: FactorId
    :param type: String key indicating the factor/residual type
        (e.g., "odom", "loop_closure", "object_prior").
    :type type: str
    :param var_ids: Ordered tuple of NodeIds that this factor connects.
    :type var_ids: tuple[NodeId, ...]
    :param params: Dictionary of parameters passed into the residual function,
        such as measurements, noise weights, or prior means.
    :type params: Dict[str, Any]
    """
    id: FactorId
    type: str          # e.g. "odom", "loop_closure", "object_prior"
    var_ids: tuple[NodeId, ...]
    params: Dict[str, Any]  # Measurement, noise, etc.