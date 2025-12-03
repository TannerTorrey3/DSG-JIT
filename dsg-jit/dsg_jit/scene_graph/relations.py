# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
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
    Compute a residual enforcing that a room's position matches the centroid of its member places.

    The input state vector ``x`` is assumed to contain a concatenation of the room
    position followed by the positions of its member places::

        x = [room_pos, place0_pos, place1_pos, ..., placeN_pos]

    All positions live in :math:`\\mathbb{R}^d`, and ``d`` is provided via
    ``params["dim"]``.

    The residual is defined as::

        r = room_pos - mean(place_positions)

    If no member places are provided, the residual is a zero vector of the same
    shape as ``room_pos``.

    :param x: Flat state vector containing the room position followed by the
              positions of its member places. Shape ``(dim * (N + 1),)`` where
              ``dim`` is the position dimension and ``N`` is the number of
              member places.
    :param params: Dictionary of parameters. Must contain:
                   ``"dim"`` (int or scalar array) indicating the dimension
                   of each position vector.
    :return: Residual vector ``r`` with shape ``(dim,)`` enforcing that the room
             lies at the centroid of its member places.
    :rtype: jax.numpy.ndarray
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
    Residual tying a place's position to a pose's translation component.

    The input state vector ``x`` is assumed to be the concatenation of a pose
    vector and a place vector::

        x = [pose_vec, place_vec]

    By default, the pose is represented in ``se(3)`` as a 6D vector
    ``[tx, ty, tz, rx, ry, rz]`` and the place is a 1D scalar.

    The constraint enforces that the place coordinate tracks one component of the
    pose translation plus an optional offset::

        pose_val     = pose_vec[pose_coord_index]
        target_place = pose_val + offset
        r            = place_vec - target_place

    :param x: Flat state vector containing the pose and place variables,
              ``[pose_vec, place_vec]``. The first ``pose_dim`` entries are the
              pose, followed by ``place_dim`` entries for the place.
    :param params: Dictionary of parameters controlling the attachment:
                   ``"pose_dim"`` (int, default ``6``), the dimension of the
                   pose vector; ``"place_dim"`` (int, default ``1``), the
                   dimension of the place vector; ``"pose_coord_index"`` (int,
                   default ``0``), index of the pose component used to attach
                   the place; ``"offset"`` (array-like of shape ``(place_dim,)``,
                   optional), additive offset applied to the selected pose
                   component before comparison.
    :return: Residual vector ``r`` with shape ``(place_dim,)`` enforcing the
             attachment between pose and place.
    :rtype: jax.numpy.ndarray
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