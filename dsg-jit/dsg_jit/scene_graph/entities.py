# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Entity definitions for the Dynamic Scene Graph.

This module defines the core node types stored in the DSG-JIT scene graph,
such as:

    • Poses / agents (SE3 trajectories)
    • Places (topological nodes)
    • Rooms / regions
    • Objects
    • Voxel cells or volumetric primitives (if represented as entities)

Each entity is typically a lightweight data structure that carries:

    • A unique identifier
    • A semantic type (e.g., "agent", "object", "place", "room")
    • Optional metric information (pose, position, bounding volume)
    • Optional attributes (class labels, instance ids, timestamps, etc.)

These entities form the *nodes* of the scene graph; edges and semantic
relations between them are defined in `scene_graph.relations`.

Typical Contents
----------------
Common patterns in this module include:

    • Data classes for:
        - BaseEntity: minimal base type for all nodes
        - AgentEntity: for robots / people with SE3 pose trajectories
        - ObjectEntity: for movable / manipulable objects
        - PlaceEntity: for nodes in the topological graph
        - RoomEntity: for higher-level spatial regions

    • Utility functions for:
        - Creating entities from raw SLAM / perception outputs
        - Updating metric fields (e.g., pose) after optimization
        - Serializing / deserializing entity metadata

Integration with DSG-JIT
------------------------
The entity layer is tightly coupled to:

    • `world.scene_graph.SceneGraphWorld`:
        which stores entities, relations, and their mapping to variables
        in the factor graph.

    • `core.factor_graph.FactorGraph`:
        entities that carry metric state (e.g., poses, voxel centers)
        are typically tied to variables in the factor graph so they can
        be optimized along with the rest of the world model.

    • `slam.measurements`:
        certain measurements (e.g., pose attachments, voxel observations)
        operate directly on entity-linked variables.

Design Goals
------------
• **Thin, explicit data model**:
    Entities should be simple, well-typed containers that are easy to
    reason about and manipulate in experiments.

• **Bridging symbolic and geometric worlds**:
    Entities provide the anchor points where semantic structure (object,
    room, agent, place) meets metric state (pose, voxel position, etc.).

• **Extendable for research**:
    New entity types (e.g., semantic regions, affordance nodes, learned
    latent entities) can be added without changing the core optimization
    backend, as long as they are mapped to appropriate variable types.
"""

from __future__ import annotations
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class RoomNode:
    """Scene graph room centroid represented as a 3D point.

    :param id: Integer identifier for this room node.
    :param position: 3D position of the room centroid as a JAX array of shape (3,).
    """
    id: int
    position: jnp.ndarray  # shape (3,)


@dataclass
class ObjectNode:
    """Scene graph object node with a 3D position.

    :param id: Integer identifier for this object node.
    :param position: 3D position of the object as a JAX array of shape (3,).
    """
    id: int
    position: jnp.ndarray  # shape (3,)