"""
Semantic and topological relations for the Dynamic Scene Graph.

This module defines the core relation types and utilities used to connect
entities in the DSG-JIT scene graph, such as:

    • Place ↔ Place (topological connectivity)
    • Room ↔ Room (adjacency, containment)
    • Object ↔ Place / Room (support, inside, on, near)
    • Agent ↔ Object / Place (interaction, visibility, reachability)

The goal is to provide a lightweight, JAX-friendly representation of edges
and relation labels that can be used both for:

    • Pure graph reasoning (e.g., "what objects are on this table?")
    • Differentiable optimization (e.g., factors that enforce relational
      consistency between metric poses and symbolic structure)

Typical Contents
----------------
Although the exact API may evolve, this module usually contains:

    • Enumerations or string constants for relation types
      (e.g., "on", "inside", "adjacent", "connected_to", "observes")

    • Simple data classes / containers for relations:
        - relation id
        - source entity id
        - target entity id
        - relation type
        - optional attributes (weights, confidences, timestamps)

    • Helper functions for:
        - Adding / removing relations in a `SceneGraphWorld`
        - Querying neighbors by relation type
        - Converting relations into factor-graph constraints when needed

Design Goals
------------
• **Separation of concerns**:
    Geometry (poses, voxels) is stored elsewhere; this module only
    cares about *relationships* between entities.

• **Compatibility with optimization**:
    When relations induce constraints (e.g., "object is on a surface"),
    these can be translated into factors in the world model or SLAM layer.

• **Extensibility**:
    New relation types or attributes should be easy to add without
    breaking the core graph structure.

Notes
-----
The scene graph can be used in both non-differentiable and differentiable
modes. In the differentiable setting, certain relations may correspond
to factors whose residuals live in `slam.measurements`. This module
provides the symbolic layer that those factors are grounded in.
"""

from __future__ import annotations
from typing import Dict

import jax.numpy as jnp


def room_centroid_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Residual enforcing that a room's position matches
    the centroid of its member places.

    x is a flat vector containing:
        [room_pos, place0_pos, place1_pos, ..., placeN_pos]

    All positions are in R^d. We pass `dim` in params to know d.

    residual = room_pos - mean(place_positions)
    """

    dim = int(params["dim"])  # e.g. 1 or 3

    # room position is first `dim` entries
    room = x[:dim]

    # remaining entries are stacked member positions
    members_flat = x[dim:]
    if members_flat.size == 0:
        # No members -> no constraint, residual 0
        return jnp.zeros_like(room)

    members = members_flat.reshape(-1, dim)  # shape (num_members, dim)
    centroid = jnp.mean(members, axis=0)

    return room - centroid

def pose_place_attachment_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Tie a place's position to a pose's translation component.

    x contains: [pose_vec, place_vec]
    - pose_vec is length pose_dim (6 for SE(3))
    - place_vec is length place_dim (1 for now)

    params:
      - "pose_dim": int, dimension of pose vector (default 6)
      - "place_dim": int, dimension of place vector (default 1)
      - "pose_coord_index": int, which component of pose to use (default 0: tx)
      - "offset": shape (place_dim,), optional, default 0

    We compute:
      target_place = pose[pose_coord_index] + offset
      residual = place - target_place
    """
    pose_dim = int(params.get("pose_dim", 6))
    place_dim = int(params.get("place_dim", 1))
    coord_idx = int(params.get("pose_coord_index", 0))

    pose_vec = x[:pose_dim]
    place_vec = x[pose_dim:pose_dim + place_dim]

    offset = params.get("offset", jnp.zeros(place_dim))

    # For 1D, pose_coord_index picks one scalar from pose_vec
    pose_val = pose_vec[coord_idx]
    target = pose_val + offset  # broadcast if place_dim > 1
    # Make target same shape as place_vec
    target_vec = jnp.ones_like(place_vec) * target

    return place_vec - target_vec