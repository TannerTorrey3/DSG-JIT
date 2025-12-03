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

from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.core.types import Variable, Factor, NodeId, FactorId
from dsg_jit.optimization.solvers import (
    gradient_descent, GDConfig,
    damped_newton, NewtonConfig,
    gauss_newton, GNConfig,
    gauss_newton_manifold,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.jit_wrappers import JittedGN


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
        """Add a new variable to the underlying factor graph.

        This allocates a fresh :class:`NodeId`, constructs a
        :class:`core.types.Variable` with the given type and initial value,
        registers it in :attr:`fg`, and returns the newly created id.

        :param var_type: String describing the variable type (e.g. ``"pose"``,
            ``"room"``, ``"place"``, ``"object"``). This is used by
            residual functions and manifold metadata to interpret the state.
        :param value: Initial value for the variable, represented as a
            1D JAX array. The dimensionality is inferred from
            ``value.shape[0]``.
        :returns: The :class:`NodeId` of the newly added variable.
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
        """Add a new factor to the underlying factor graph.

        This allocates a fresh :class:`FactorId`, normalizes the input
        variable identifiers to :class:`NodeId` instances, constructs a
        :class:`core.types.Factor`, and registers it in :attr:`fg`.

        :param f_type: String identifying the factor type. This must match a
            key in :attr:`FactorGraph.residual_fns` so that the appropriate
            residual function can be looked up during optimization.
        :param var_ids: Iterable of variable identifiers (ints or
            :class:`NodeId` instances) that this factor connects.
        :param params: Dictionary of factor parameters passed through to the
            residual function (e.g. measurements, noise models, weights).
        :returns: The :class:`FactorId` of the newly added factor.
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

    def add_camera_bearings(
        self,
        pose_id: NodeId,
        landmark_ids: list[NodeId],
        bearings: jnp.ndarray,
        weight: float | None = None,
        factor_type: str = "pose_landmark_bearing",
    ) -> FactorId:
        """Add one or more camera bearing factors for a single pose.

        This is a thin convenience wrapper for camera-like measurements that
        observe known landmarks via bearing (direction) only. It assumes that
        the underlying factor type is implemented by a residual such as
        :func:`slam.measurements.pose_landmark_bearing_residual`.

        Each row of :param:`bearings` is expected to correspond to one
        landmark in :param:`landmark_ids`. The dimensionality (e.g. 2D angle
        or 3D unit vector) is left to the residual function.

        :param pose_id: Identifier of the pose variable from which all
            bearings are taken.
        :param landmark_ids: List of landmark node identifiers, one per row
            in ``bearings``.
        :param bearings: Array of shape ``(N, D)`` containing bearing
            measurements in the sensor or camera frame.
        :param weight: Optional scalar weight or inverse noise level applied
            uniformly to all bearings in this call. If ``None``, the default
            inside the residual is used.
        :param factor_type: Factor type string to register in the underlying
            :class:`FactorGraph`. Defaults to ``"pose_landmark_bearing"``.
        :returns: The :class:`FactorId` of the last factor added. One factor
            is added per (pose, landmark) pair.
        """
        if bearings.shape[0] != len(landmark_ids):
            raise ValueError(
                "add_camera_bearings expected len(landmark_ids) == bearings.shape[0], "
                f"got {len(landmark_ids)} vs {bearings.shape[0]}"
            )

        last_fid: FactorId | None = None
        for lm_id, b in zip(landmark_ids, bearings):
            params: Dict[str, object] = {"bearing": jnp.asarray(b)}
            if weight is not None:
                params["weight"] = float(weight)
            last_fid = self.add_factor(factor_type, [pose_id, lm_id], params)

        # mypy/linters: last_fid will never be None if bearings is non-empty.
        if last_fid is None:
            raise ValueError("add_camera_bearings called with empty bearings array.")
        return last_fid


    def add_lidar_ranges(
        self,
        pose_id: NodeId,
        landmark_ids: list[NodeId],
        ranges: jnp.ndarray,
        directions: Optional[jnp.ndarray] = None,
        weight: float | None = None,
        factor_type: str = "pose_lidar_range",
    ) -> FactorId:
        """Add LiDAR-style range factors for a single pose.

        This helper is intended for simple range-only or range-with-direction
        measurements to known landmarks, coming from a LiDAR or depth sensor.

        The interpretation of ``directions`` depends on the chosen residual
        implementation, but a common convention is that each row is a unit
        vector in the sensor frame pointing toward the target.

        :param pose_id: Identifier of the pose variable from which ranges
            are measured.
        :param landmark_ids: List of landmark node identifiers, one per range
            sample.
        :param ranges: Array of shape ``(N,)`` holding range values in meters.
        :param directions: Optional array of shape ``(N, 3)`` with unit
            direction vectors associated with each range measurement.
        :param weight: Optional scalar weight applied to all range factors.
        :param factor_type: Factor type string to register; by default this is
            ``"pose_lidar_range"``. The residual function for this type is
            expected to consume ``"range"`` and optionally ``"direction"`` in
            ``params``.
        :returns: The :class:`FactorId` of the last factor added.
        """
        if ranges.shape[0] != len(landmark_ids):
            raise ValueError(
                "add_lidar_ranges expected len(landmark_ids) == ranges.shape[0], "
                f"got {len(landmark_ids)} vs {ranges.shape[0]}"
            )
        if directions is not None and directions.shape[0] != ranges.shape[0]:
            raise ValueError(
                "add_lidar_ranges expected directions.shape[0] == ranges.shape[0], "
                f"got {directions.shape[0]} vs {ranges.shape[0]}"
            )

        last_fid: FactorId | None = None
        for i, lm_id in enumerate(landmark_ids):
            params: Dict[str, object] = {"range": float(ranges[i])}
            if directions is not None:
                params["direction"] = jnp.asarray(directions[i])
            if weight is not None:
                params["weight"] = float(weight)
            last_fid = self.add_factor(factor_type, [pose_id, lm_id], params)

        if last_fid is None:
            raise ValueError("add_lidar_ranges called with empty ranges array.")
        return last_fid


    def add_imu_preintegration_factor(
        self,
        pose_i: NodeId,
        pose_j: NodeId,
        delta: Dict[str, jnp.ndarray],
        weight: float | None = None,
        factor_type: str = "pose_imu_preintegration",
    ) -> FactorId:
        """Add an IMU preintegration-style factor between two poses.

        This is intended to work with a preintegrated IMU summary (e.g. as
        produced by :mod:`sensors.imu`), where ``delta`` contains fields such
        as ``"dR"``, ``"dv"``, ``"dp"``, and corresponding covariance or
        information terms.

        The exact keys expected in ``delta`` are left to the residual
        implementation for ``factor_type``, but by storing the dictionary
        unchanged in ``params["delta"]`` we keep this interface flexible.

        :param pose_i: NodeId of the starting pose (time :math:`t_k`).
        :param pose_j: NodeId of the ending pose (time :math:`t_{k+1}`).
        :param delta: Dictionary describing the preintegrated IMU increment
            between ``pose_i`` and ``pose_j``. All arrays should be JAX
            arrays or types convertible via :func:`jax.numpy.asarray`.
        :param weight: Optional scalar weight / scaling to apply to the IMU
            factor inside the residual.
        :param factor_type: Factor type string to register; by default this is
            ``"pose_imu_preintegration"``.
        :returns: The :class:`FactorId` of the created IMU factor.
        """
        params: Dict[str, object] = {"delta": {k: jnp.asarray(v) for k, v in delta.items()}}
        if weight is not None:
            params["weight"] = float(weight)
        return self.add_factor(factor_type, [pose_i, pose_j], params)

    def optimize(
        self,
        lr: float = 0.1,
        iters: int = 300,
        method: str = "gd",
        damping: float = 1e-3,
        max_step_norm: float = 1.0,
    ) -> None:
        """Run a local optimizer on the current world state.

        This method packs the current variables into a flat state vector,
        constructs an appropriate objective or residual function, runs one
        of the supported optimizers, and writes the optimized state back
        into :attr:`fg.variables`.

        Supported methods:

        - ``"gd"``: vanilla gradient descent on the scalar objective
          :math:`\\|r(x)\\|^2`.
        - ``"newton"``: damped Newton on the same scalar objective.
        - ``"gn"``: Gauss--Newton on the stacked residual vector assuming
          Euclidean variables.
        - ``"manifold_gn"``: manifold-aware Gauss--Newton that uses
          :func:`slam.manifold.build_manifold_metadata` to handle SE(3)
          and Euclidean blocks differently.
        - ``"gn_jit"``: JIT-compiled Gauss--Newton using
          :class:`optimization.jit_wrappers.JittedGN`.

        :param lr: Learning rate for gradient-descent-based methods
            (currently used when ``method == "gd"``).
        :param iters: Maximum number of iterations for the chosen optimizer.
        :param method: Name of the optimization method to use. See the list
            above for supported values.
        :param damping: Damping / regularization parameter used by the
            Newton and Gauss--Newton variants.
        :param max_step_norm: Maximum allowed step norm for Gauss--Newton
            methods; steps larger than this are clamped to improve stability.
        :returns: ``None``. The world model is updated in place.
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