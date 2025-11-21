# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
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
from typing import Dict, Optional

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
    """High-level world model built on top of :class:`FactorGraph`.

    In addition to wrapping the core factor graph, this class keeps simple
    bookkeeping dictionaries that make it easier to build static and dynamic
    scene graphs on top of DSG-JIT. These maps are deliberately lightweight
    and optional: if you never pass a name when adding variables, the
    underlying optimization behavior is unchanged.
    """

    fg: FactorGraph
    # Optional semantic maps for higher-level layers (scene graphs, DSG, etc.).
    pose_ids: Dict[str, NodeId]
    room_ids: Dict[str, NodeId]
    place_ids: Dict[str, NodeId]
    object_ids: Dict[str, NodeId]
    agent_pose_ids: Dict[str, Dict[int, NodeId]]

    def __init__(self) -> None:
        # Core factor graph
        self.fg = FactorGraph()
        # Semantic maps; these are purely for convenience and do not affect
        # the underlying optimization.
        self.pose_ids = {}
        self.room_ids = {}
        self.place_ids = {}
        self.object_ids = {}
        # Mapping: agent_id -> {timestep -> NodeId}
        self.agent_pose_ids = {}

    def add_variable(self, var_type: str, value: jnp.ndarray) -> NodeId:
        """
        Allocate a new variable id, create the Variable, add it to the graph,
        and return its NodeId.

        Higher-level helpers may register semantic names in bookkeeping maps.
        """
        nid_int = len(self.fg.variables)
        nid = NodeId(nid_int)
        v = Variable(id=nid, type=var_type, value=value)
        self.fg.add_variable(v)
        return nid

    def add_pose(self, value: jnp.ndarray, name: Optional[str] = None) -> NodeId:
        """Add an SE(3) pose variable.

        This is a thin wrapper around :meth:`add_variable`. If ``name`` is
        provided, the pose is also registered in :attr:`pose_ids`, which can
        be useful for scene-graph style code that wants stable, human-readable
        handles.

        :param value: Initial pose value, typically a 6D se(3) vector.
        :param name: Optional semantic name used as a key in :attr:`pose_ids`.
        :returns: The :class:`NodeId` of the newly created pose variable.
        """
        nid = self.add_variable("pose", value)
        if name is not None:
            self.pose_ids[name] = nid
        return nid

    def add_room(self, center: jnp.ndarray, name: Optional[str] = None) -> NodeId:
        """Add a room center variable (3D point).

        :param center: 3D position of the room center.
        :param name: Optional semantic name to register in :attr:`room_ids`.
        :returns: The :class:`NodeId` of the new room variable.
        """
        nid = self.add_variable("room", center)
        if name is not None:
            self.room_ids[name] = nid
        return nid

    def add_place(self, center: jnp.ndarray, name: Optional[str] = None) -> NodeId:
        """Add a place / waypoint variable (3D point).

        :param center: 3D position of the place/waypoint.
        :param name: Optional semantic name to register in :attr:`place_ids`.
        :returns: The :class:`NodeId` of the new place variable.
        """
        nid = self.add_variable("place", center)
        if name is not None:
            self.place_ids[name] = nid
        return nid

    def add_object(self, center: jnp.ndarray, name: Optional[str] = None) -> NodeId:
        """Add an object centroid variable (3D point).

        :param center: 3D position of the object centroid.
        :param name: Optional semantic name to register in :attr:`object_ids`.
        :returns: The :class:`NodeId` of the new object variable.
        """
        nid = self.add_variable("object", center)
        if name is not None:
            self.object_ids[name] = nid
        return nid

    def add_agent_pose(
        self,
        agent_id: str,
        t: int,
        value: jnp.ndarray,
        var_type: str = "pose",
    ) -> NodeId:
        """Add (and register) a pose for a particular agent at a timestep.

        This convenience helper is meant for dynamic scene graphs where you
        track multiple agents over time. It simply delegates to
        :meth:`add_variable` and then records the mapping ``(agent_id, t)``.

        :param agent_id: String identifier for the agent (e.g. ``"robot_0"``).
        :param t: Discrete timestep index.
        :param value: Initial pose value for this agent at time ``t``.
        :param var_type: Underlying variable type to use (defaults to
            ``"pose"``; you can change this to ``"pose_se3"`` in advanced
            use-cases).
        :returns: The :class:`NodeId` of the new agent pose variable.
        """
        nid = self.add_variable(var_type, value)
        if agent_id not in self.agent_pose_ids:
            self.agent_pose_ids[agent_id] = {}
        self.agent_pose_ids[agent_id][t] = nid
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

    def get_variable_value(self, nid: NodeId) -> jnp.ndarray:
        """Return the current value of a variable.

        This is a thin convenience wrapper over the underlying
        :class:`FactorGraph` variable storage and is useful when building
        dynamic scene graphs that want to query individual nodes.

        :param nid: Identifier of the variable.
        :returns: A JAX array holding the variable's current value.
        """
        return self.fg.variables[nid].value

    def snapshot_state(self) -> Dict[int, jnp.ndarray]:
        """Capture a shallow snapshot of the current world state.

        The snapshot maps integer node ids to their current values. This is
        intentionally simple and serialization-friendly, and is meant to be
        consumed by higher-level dynamic scene graph structures that want to
        record the evolution of the world over time.

        :returns: A dictionary mapping ``int(NodeId)`` to JAX arrays.
        """
        return {int(nid): jnp.array(var.value) for nid, var in self.fg.variables.items()}