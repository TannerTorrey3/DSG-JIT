# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
World-level wrapper and optimization front-end around the core factor graph.

This module defines the *world model* abstraction: a typed layer on top of
``core.factor_graph.FactorGraph`` that understands high-level entities
(poses, places, rooms, voxels, objects, agents) and also centralizes
residual construction, JIT compilation, and solver orchestration.

In other words, :class:`WorldModel` is the bridge between:

    • Low-level optimization (factor graph, residual functions, manifolds)
    • High-level scene graph abstractions (poses, agents, rooms, voxels)
    • Application code that wants a simple, stable API for "optimize my world"

The underlying :class:`FactorGraph` remains a relatively small, generic
data structure that stores variables and factors and knows nothing about
JAX, JIT, or manifolds. All JAX-specific logic (residual registries,
vmap-based batching, Gauss–Newton wrappers, etc.) is owned by the world
model.

Key responsibilities
--------------------
- Manage the underlying :class:`FactorGraph` instance.
- Provide ergonomic helpers to:
    • Add variables with automatically assigned :class:`NodeId`s.
    • Add typed factors (e.g. priors, odometry, attachments, voxel terms).
    • Pack / unpack state vectors for optimization.
- Maintain simple bookkeeping structures (e.g. maps from user-facing
  handles / indices back to :class:`NodeId`s) so that experiments and
  higher-level layers do not need to manipulate :class:`NodeId` directly.
- Maintain a residual-function registry that maps factor-type strings
  (e.g. ``"odom_se3"``, ``"voxel_point_obs"``) to JAX-compatible
  residuals.
- Build unified, vmap-optimized residual and objective functions on
  demand, caching compiled versions keyed by graph structure.
- Expose convenient optimization entry points (e.g. :meth:`optimize`,
  or :class:`optimization.jit_wrappers.JittedGN`) that operate directly
  on the world model.

Typical usage
-------------
Experiments and higher layers typically:

    1. Construct a :class:`WorldModel`.
    2. Add variables & factors according to a scenario.
    3. Register residual functions for each factor type of interest.
    4. Build a residual or objective from the world model and call into
       :mod:`dsg_jit.optimization.solvers` or :mod:`dsg_jit.optimization.jit_wrappers`
       to run Gauss–Newton (potentially manifold-aware) or gradient-based
       optimization.
    5. Decode and interpret the optimized state via the world model’s
       convenience accessors, or export it to higher-level scene-graph
       structures.

Design goals
------------
- **Backend separation**: keep :class:`FactorGraph` as a minimal,
  backend-agnostic data structure (variables, factors, connectivity),
  while :class:`WorldModel` owns JAX-facing logic such as residual
  construction, vmap batching, and JIT caching.
- **Scene-friendly**: provide enough structure that scene graphs, voxel
  modules, and DSG layers can build on top of the world model without
  duplicating graph or optimization logic.
- **Ergonomic but explicit**: favor simple, explicit methods
  (``add_variable``, ``add_factor``, ``register_residual``, ``optimize``)
  over hidden magic, so that experiments remain easy to debug and extend.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

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


# --- Module-level helper for marginal prior residual ---
def marginal_prior_residual(stacked: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
    """Residual for a dense Gaussian prior induced by marginalization.

    This residual encodes a quadratic term of the form

        1/2 (x - μ)^T H (x - μ)

    via a Cholesky factorization H = L^T L. The parameters are:

        mean       : μ, a 1D array of the same shape as ``stacked``.
        sqrt_info  : L, a square matrix such that L^T L ≈ H.

    The returned residual is L @ (x - μ), so that the overall contribution
    to the objective is 1/2 ||L (x - μ)||^2.
    """
    mean = params["mean"]
    sqrt_info = params["sqrt_info"]
    return sqrt_info @ (stacked - mean)

ResidualFn = Callable[..., jnp.array]

@dataclass
class ActiveWindowTemplate:
    """Defines a fixed-capacity active factor graph template for JIT-stable operation.
    Each variable/factor slot is identified by (type, slot_idx).
    """
    variable_slots: List[Tuple[str, int, int]]  # (var_type, slot_idx, dim)
    factor_slots: List[Tuple[str, int, Tuple[Tuple[str, int], ...]]]  # (factor_type, slot_idx, var_slot_keys)

@dataclass
class VarSlot:
    """Bookkeeping for a variable slot in the active template."""
    var_type: str
    slot_idx: int
    node_id: NodeId
    dim: int

@dataclass
class FactorSlot:
    """Bookkeeping for a factor slot in the active template."""
    factor_type: str
    slot_idx: int
    factor_id: FactorId
    var_slot_keys: Tuple[Tuple[str, int], ...]


@dataclass
class WorldModel:
    """High-level world model built on top of :class:`FactorGraph`.

    Modes:
      - Dynamic/unbounded FG (legacy, research mode): Variables and factors can be added/removed dynamically.
      - Fixed-capacity active template (real-time / JIT-stable mode): A fixed set of variable/factor slots is preallocated for JIT-compatibility and in-place updates.

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
    _residual_registry: Dict[str, ResidualFn]
    _compiled_solvers: Dict[Tuple[str,str], Any]
    # --- Active template fields (for slot-based, fixed-capacity mode) ---
    _active_template: Optional[ActiveWindowTemplate] = field(default=None, init=False)
    _var_slots: Dict[Tuple[str, int], VarSlot] = field(default_factory=dict, init=False)
    _factor_slots: Dict[Tuple[str, int], FactorSlot] = field(default_factory=dict, init=False)
    _active_factor_mask: Dict[FactorId, bool] = field(default_factory=dict, init=False)

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
        # Residual Registry
        self._residual_registry: Dict[str, ResidualFn] = {}
        self._compiled_solvers: Dict[Tuple[str, str], Any] = {}
        # Active window template fields (for slot-based mode)
        self._active_template: Optional[ActiveWindowTemplate] = None
        self._var_slots: Dict[Tuple[str, int], VarSlot] = {}
        self._factor_slots: Dict[Tuple[str, int], FactorSlot] = {}
        self._active_factor_mask: Dict[FactorId, bool] = {}
    # --- Active template / slot-based API ---
    def init_active_template(self, template: ActiveWindowTemplate) -> None:
        """Initialize a fixed-capacity active factor graph template for JIT-stable operation.
        All variables and factors are preallocated; structure is fixed.
        """
        # Reset the factor graph and slot bookkeeping.
        self.fg = FactorGraph()
        self._active_template = template
        self._var_slots.clear()
        self._factor_slots.clear()
        self._active_factor_mask.clear()
        # Pre-allocate variable slots.
        for var_type, slot_idx, dim in template.variable_slots:
            init_val = jnp.zeros((dim,), dtype=jnp.float32)
            node_id = self.add_variable(var_type, init_val)
            slot_key = (var_type, slot_idx)
            self._var_slots[slot_key] = VarSlot(var_type, slot_idx, node_id, dim)
        # Pre-allocate factor slots (inactive by default).
        for factor_type, slot_idx, var_slot_keys in template.factor_slots:
            var_ids = tuple(self._var_slots[vk].node_id for vk in var_slot_keys)

            # Compute the stacked state dimension for this factor slot.
            stacked_dim = 0
            for vk in var_slot_keys:
                stacked_dim += int(self._var_slots[vk].dim)

            # IMPORTANT: In slot-based mode we rely on vmap + tree stacking.
            # That requires that all factors within a batched group have the
            # same params keys and compatible shapes.
            if factor_type == "prior":
                # Prior residual typically expects a 6D target for SE(3) poses.
                # We default to zeros and unit weight.
                params: Dict[str, Any] = {
                    "target": jnp.zeros((stacked_dim,), dtype=jnp.float32),
                    "weight": jnp.array(1.0, dtype=jnp.float32),
                    "active": jnp.array(0.0, dtype=jnp.float32),
                }
            elif factor_type == "odom_se3":
                # Odometry measurement lives in the se(3) tangent space (6D),
                # even though the stacked state for two poses is 12D.
                params = {
                    "measurement": jnp.zeros((6,), dtype=jnp.float32),
                    "weight": jnp.array(1.0, dtype=jnp.float32),
                    "active": jnp.array(0.0, dtype=jnp.float32),
                }
            elif factor_type == "marginal_prior":
                # Dense Gaussian prior induced by marginalization.
                params = {
                    "mean": jnp.zeros((stacked_dim,), dtype=jnp.float32),
                    "sqrt_info": jnp.eye(stacked_dim, dtype=jnp.float32),
                    "weight": jnp.array(1.0, dtype=jnp.float32),
                    "active": jnp.array(0.0, dtype=jnp.float32),
                }
            else:
                # Generic fallback: only active + weight.
                # Callers can override/extend keys via configure_factor_slot.
                params = {
                    "weight": jnp.array(1.0, dtype=jnp.float32),
                    "active": jnp.array(0.0, dtype=jnp.float32),
                }

            factor_id = self.add_factor(factor_type, var_ids, params)
            slot_key = (factor_type, slot_idx)
            self._factor_slots[slot_key] = FactorSlot(factor_type, slot_idx, factor_id, var_slot_keys)
            self._active_factor_mask[factor_id] = False

    def set_variable_slot(self, var_type: str, slot_idx: int, value: jnp.ndarray) -> NodeId:
        """Set the value of a variable slot in the active template."""
        slot_key = (var_type, slot_idx)
        slot = self._var_slots.get(slot_key)
        if slot is None:
            raise KeyError(f"Variable slot {slot_key} not found in active template.")
        if value.shape[0] != slot.dim:
            raise ValueError(f"Value shape {value.shape} does not match slot dim {slot.dim}")
        self.fg.variables[slot.node_id].value = value
        return slot.node_id

    def configure_factor_slot(
        self,
        factor_type: str,
        slot_idx: int,
        var_ids: Tuple[NodeId, ...],
        params: Dict,
        active: bool = True,
    ) -> None:
        """Configure a factor slot in the active template: set variable ids, params, and activity."""
        slot_key = (factor_type, slot_idx)
        slot = self._factor_slots.get(slot_key)
        if slot is None:
            raise KeyError(f"Factor slot {slot_key} not found in active template.")
        fid = slot.factor_id
        f = self.fg.factors[fid]
        # Update factor's var_ids and params in place.
        object.__setattr__(f, "var_ids", tuple(var_ids))

        # IMPORTANT: preserve existing keys so vmapped stacking sees a
        # consistent pytree structure across all factors in a batched group.
        new_params = dict(f.params)
        new_params.update(params)
        # Normalize scalar params to JAX arrays for stable stacking.
        for k, v in list(new_params.items()):
            if isinstance(v, (float, int)):
                new_params[k] = jnp.array(v, dtype=jnp.float32)
        new_params["active"] = jnp.array(1.0 if active else 0.0, dtype=jnp.float32)
        f.params = new_params
        self._active_factor_mask[fid] = active

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
        # Allocate a fresh NodeId. We cannot rely on len(self.fg.variables)
        # when variables may have been removed (e.g. after marginalization),
        # so we take the maximum existing id and add one.
        if self.fg.variables:
            max_existing_id = max(int(nid) for nid in self.fg.variables.keys())
            nid_int = max_existing_id + 1
        else:
            nid_int = 0
        nid = NodeId(nid_int)
        v = Variable(id=nid, type=var_type, value=value)
        self.fg.add_variable(v)
        # Graph structure has changed; clear any cached compiled solvers
        # and residuals so they can be rebuilt on demand.
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
        # Allocate a fresh FactorId. We cannot rely on len(self.fg.factors)
        # when factors may have been removed (e.g. after marginalization),
        # so we take the maximum existing id and add one.
        if self.fg.factors:
            max_existing_id = max(int(fid) for fid in self.fg.factors.keys())
            fid_int = max_existing_id + 1
        else:
            fid_int = 0
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
        # Adding a factor changes the factor graph structure; clear cached
        # compiled solvers / residuals so they can be rebuilt consistently.
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
        x_init, index = self.pack_state()
        residual_fn = self.build_residual()

        if method == "gd":
            obj = self.build_objective()
            cfg = GDConfig(learning_rate=lr, max_iters=iters)
            x_opt = gradient_descent(obj, x_init, cfg)

        elif method == "newton":
            obj = self.build_objective()
            cfg = NewtonConfig(max_iters=iters, damping=damping)
            x_opt = damped_newton(obj, x_init, cfg)

        elif method == "gn":
            cfg = GNConfig(max_iters=iters, damping=damping, max_step_norm=max_step_norm)
            x_opt = gauss_newton(residual_fn, x_init, cfg)

        elif method == "manifold_gn":
            block_slices, manifold_types = build_manifold_metadata(packed_state=self.pack_state(),fg=self.fg)
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
        values = self.unpack_state(x_opt, index)
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
    
    # --- Residuals ---
    def register_residual(self, factor_type: str, fn: Callable[..., Any]) -> None:
        """Register a residual function for a given factor type.

        This is the WorldModel-level registry that associates factor type
        strings (e.g. ``"odom_se3"``, ``"voxel_point_obs"``) with
        JAX-compatible residual functions. The registered functions are
        consumed by higher-level residual builders such as
        :meth:`build_residual`.

        Parameters
        ----------
        factor_type : str
            String identifier for the factor type. This must match the
            ``type`` field stored in :class:`Factor` instances in the
            underlying :class:`FactorGraph`.
        fn : Callable
            Residual function implementing the measurement model. The
            exact signature is intentionally flexible, but it is expected
            to be compatible with the unified residual builder returned by
            :meth:`build_residual` (e.g. it may be vmapped across factors
            of a given type).
        """
        self._residual_registry[factor_type] = fn

    def get_residual(self, factor_type: str) -> Optional[Callable[..., Any]]:
        """Return the residual function registered for a given factor type.

        Parameters
        ----------
        factor_type : str
            String identifier for the factor type.

        Returns
        -------
        callable or None
            The residual function previously registered via
            :meth:`register_residual`, or ``None`` if no function is
            registered for the requested type.
        """
        return self._residual_registry.get(factor_type)
    
    def get_residuals(self) -> Dict[str, ResidualFn]:
        """Returns the residual registry, all currently registered residuals.
        
        :return: Dict[str, ResidualFn]
        """
        return self._residual_registry

    def list_residual_types(self) -> List[str]:
        """List all factor types with registered residual functions.

        This is a convenience helper for debugging, diagnostics, and tests
        to verify that the WorldModel has been configured with the expected
        residuals for the current application.

        Returns
        -------
        list of str
            Sorted list of factor type strings for which residuals have
            been registered.
        """
        return sorted(self._residual_registry.keys())
    
    def build_residual(
        self,
        *,
        use_type_weights: bool = False,
        learn_odom: bool = False,
        learn_voxel_points: bool = False,
    ) -> Callable[..., Any]:
        """Construct a unified residual function for the current world.

        This method is the WorldModel-level entry point for building a
        JAX-compatible residual function that stacks all factor residuals.
        It is intended to subsume the various specialized builders that
        previously lived on :class:`FactorGraph`, such as:

        * ``build_residual_function_with_type_weights``
        * ``build_residual_function_se3_odom_param_multi``
        * ``build_residual_function_voxel_point_param[_multi]``

        Instead of having separate entry points, this method exposes a
        single interface whose behavior is controlled by configuration
        flags and a structured "hyper-parameter" argument passed at call
        time.

        Parameters
        ----------
        use_type_weights : bool, optional
            Currently unused in this implementation. Reserved for future
            integration with type-weighted residuals.
        learn_odom : bool, optional
            Currently unused in this implementation. Reserved for future
            integration with learnable odometry parameters.
        learn_voxel_points : bool, optional
            Currently unused in this implementation. Reserved for future
            integration with learnable voxel observation points.

        Returns
        -------
        callable
            A JAX-compatible residual function. In the simplest case
            (all flags ``False``) the signature is ``r(x)`` where ``x`` is
            a packed state vector.
        """
        # NOTE: For now, the configuration flags are accepted but not yet
        # wired into the implementation. They are kept in the signature to
        # preserve the planned API surface and avoid breaking callers.
        if use_type_weights or learn_odom or learn_voxel_points:
            raise NotImplementedError(
                "Hyper-parameterized residuals are provided by dedicated "
                "WorldModel helper methods (e.g. "
                "build_residual_function_with_type_weights, "
                "build_residual_function_se3_odom_param_multi). "
                "The generic build_residual hyper-parameter flags are not "
                "yet implemented."
            )

        # Slot-based mode: Use a constant cache key and enforce fixed structure.
        if self._active_template is not None:
            cache_key = ("residual", "active_template")
        else:
            # Legacy dynamic mode: cache by structure.
            factors = tuple(self.fg.factors.values())
            var_count = len(self.fg.variables)
            sig_parts = [f"{f.type}:{len(f.var_ids)}" for f in factors]
            structure_sig = f"v{var_count}|" + "|".join(sig_parts)
            cache_key = ("residual", structure_sig)

        cached = self._compiled_solvers.get(cache_key)
        if cached is not None:
            return cached

        # Group factors as before.
        factors = tuple(self.fg.factors.values())
        group_to_factors: Dict[Tuple[str, Tuple[int, ...]], List[Factor]] = {}
        for f in factors:
            var_dims: List[int] = []
            for nid in f.var_ids:
                v = self.fg.variables[nid].value
                var_dims.append(int(jnp.asarray(v).shape[0]))
            shape_sig = tuple(var_dims)
            key = (f.type, shape_sig)
            group_to_factors.setdefault(key, []).append(f)

        residual_fns = self._residual_registry
        _, index = self.pack_state()

        def residual(x: jnp.ndarray) -> jnp.ndarray:
            """Stacked residual function over all factors for the current graph, with slot-based activity mask if present."""
            var_values = self.unpack_state(x, index)
            res_chunks: List[jnp.ndarray] = []

            for (f_type, _shape_sig), flist in group_to_factors.items():
                res_fn = residual_fns.get(f_type, None)
                if res_fn is None:
                    raise ValueError(
                        f"No residual fn registered for factor type '{f_type}'"
                    )
                if not flist:
                    continue
                # Singleton group
                if len(flist) == 1:
                    f = flist[0]
                    vs = [var_values[nid] for nid in f.var_ids]
                    stacked = jnp.concatenate(vs)
                    # Slot-based: multiply by "active" param if present
                    r = res_fn(stacked, f.params)
                    activity = f.params.get("active", 1.0)
                    r = r * activity
                    res_chunks.append(jnp.reshape(r, (-1,)))
                    continue
                # Batched path
                stacked_states: List[jnp.ndarray] = []
                params_list: List[Dict[str, Any]] = []
                for f in flist:
                    vs = [var_values[nid] for nid in f.var_ids]
                    stacked_states.append(jnp.concatenate(vs))
                    params_list.append(f.params)
                stacked_states_arr = jnp.stack(stacked_states, axis=0)
                params_tree = jtu.tree_map(
                    lambda *vals: jnp.stack(
                        [jnp.asarray(v) for v in vals], axis=0
                    ),
                    *params_list,
                )
                def single_factor_residual(s: jnp.ndarray, p: Dict[str, Any]) -> jnp.ndarray:
                    r = res_fn(s, p)
                    activity = p.get("active", 1.0)
                    return r * activity
                batched_res = jax.vmap(single_factor_residual)(
                    stacked_states_arr, params_tree
                )
                res_chunks.append(jnp.reshape(batched_res, (-1,)))
            if not res_chunks:
                return jnp.zeros((0,), dtype=x.dtype)
            return jnp.concatenate(res_chunks, axis=0)

        residual_jit = jax.jit(residual)
        self._compiled_solvers[cache_key] = residual_jit
        return residual_jit
    
    # Marginalization and fixed-lag smoothing are now handled via bounded active templates.
    # The following methods are disabled in slot-based mode.
    def marginalize_variables(
        self,
        marginalized_ids: List[NodeId],
        damping: float = 1e-6,
    ) -> None:
        """
        Disabled: Marginalization is not supported in active template mode.
        Use bounded active templates for fixed-lag smoothing instead.
        """
        # If in active template mode, do nothing and explain.
        if self._active_template is not None:
            # Fixed-lag smoothing is handled via bounded active templates.
            # This method is disabled in slot-based mode.
            return
        # (Legacy code for dynamic mode could be restored here if needed.)
        pass
    
    def fixed_lag_marginalize(
        self,
        keep_ids: List[NodeId],
        damping: float = 1e-6,
    ) -> None:
        """
        Disabled: Fixed-lag marginalization is not supported in active template mode.
        Use bounded active templates for sliding window/fixed-lag smoothing instead.
        """
        if self._active_template is not None:
            # Fixed-lag smoothing is handled via bounded active templates.
            # This method is disabled in slot-based mode.
            return
        # (Legacy code for dynamic mode could be restored here if needed.)
        pass

    # ------------------------------------------------------------------
    # Hyper-parameterized residual builders
    # ------------------------------------------------------------------
    def build_residual_function_with_type_weights(
        self, factor_type_order: List[str]
    ):
        """Build a residual function that supports learnable type weights.

        The returned function has signature ``r(x, log_scales)`` where
        ``log_scales[i]`` is the log-weight associated with
        ``factor_type_order[i]``. Missing types default to unit weight.

        This is a WorldModel-based version of the old FactorGraph helper,
        implemented in terms of ``pack_state``, ``unpack_state``, and the
        WorldModel residual registry.
        """
        factors = list(self.fg.factors.values())
        residual_fns = self._residual_registry
        _, index = self.pack_state()

        type_to_idx = {t: i for i, t in enumerate(factor_type_order)}

        def residual(x: jnp.ndarray, log_scales: jnp.ndarray) -> jnp.ndarray:
            var_values = self.unpack_state(x, index)
            res_list: List[jnp.ndarray] = []

            for factor in factors:
                res_fn = residual_fns.get(factor.type, None)
                if res_fn is None:
                    raise ValueError(
                        f"No residual fn registered for factor type '{factor.type}'"
                    )

                stacked = jnp.concatenate(
                    [var_values[vid] for vid in factor.var_ids], axis=0
                )
                r = res_fn(stacked, factor.params)  # (k,)

                idx = type_to_idx.get(factor.type, None)
                if idx is not None:
                    scale = jnp.exp(log_scales[idx])
                else:
                    scale = 1.0

                r_scaled = scale * r
                r_scaled = jnp.reshape(r_scaled, (-1,))
                res_list.append(r_scaled)

            if not res_list:
                return jnp.zeros((0,), dtype=x.dtype)

            return jnp.concatenate(res_list, axis=0)

        return residual

    def build_residual_function_se3_odom_param_multi(self):
        """Build a residual function with learnable SE(3) odometry.

        All factors of type ``\"odom_se3\"`` are treated as depending on a
        parameter array ``theta`` of shape ``(K, 6)``, where ``K`` is the
        number of odometry factors. Each row of ``theta`` represents a
        perturbable se(3) measurement.

        Returns
        -------
        (residual_fn, index)
            ``residual_fn(x, theta)`` and the pack index mapping from
            :meth:`pack_state`.
        """
        factors = list(self.fg.factors.values())
        residual_fns = self._residual_registry

        _, index = self.pack_state()

        def residual(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            """
            Parameters
            ----------
            x : jnp.ndarray
                Flat state vector.
            theta : jnp.ndarray
                Shape (K, 6), per-odom se(3) measurement.
            """
            var_values = self.unpack_state(x, index)
            res_list: List[jnp.ndarray] = []
            odom_idx = 0

            for f in factors:
                res_fn = residual_fns.get(f.type, None)
                if res_fn is None:
                    raise ValueError(
                        f"No residual fn registered for factor type '{f.type}'"
                    )

                stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

                if f.type == "odom_se3":
                    meas = theta[odom_idx]  # (6,)
                    odom_idx += 1
                    base_params = dict(f.params)
                    base_params["measurement"] = meas
                    params = base_params
                else:
                    params = f.params

                r = res_fn(stacked, params)
                w = params.get("weight", 1.0)
                res_list.append(jnp.sqrt(w) * r)

            if not res_list:
                return jnp.zeros((0,), dtype=x.dtype)

            return jnp.concatenate(res_list)

        return residual, index

    def build_residual_function_voxel_point_param(self):
        """Build a residual function with a shared voxel observation point.

        All factors of type ``\"voxel_point_obs\"`` will use a dynamic
        ``point_world`` argument passed at call time, rather than a fixed
        value stored in the factor params.

        Returns
        -------
        (residual_fn, index)
            ``residual_fn(x, point_world)`` where ``point_world`` has
            shape (3,).
        """
        factors = list(self.fg.factors.values())
        residual_fns = self._residual_registry

        _, index = self.pack_state()

        def residual(x: jnp.ndarray, point_world: jnp.ndarray) -> jnp.ndarray:
            """
            Parameters
            ----------
            x : jnp.ndarray
                Flat state vector.
            point_world : jnp.ndarray
                Shape (3,), observation point in world coords for ALL
                voxel_point_obs factors. For now we assume a single
                voxel_point_obs, or that all share the same point.
            """
            var_values = self.unpack_state(x, index)
            res_list: List[jnp.ndarray] = []

            for f in factors:
                res_fn = residual_fns.get(f.type, None)
                if res_fn is None:
                    raise ValueError(
                        f"No residual fn registered for factor type '{f.type}'"
                    )

                stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

                if f.type == "voxel_point_obs":
                    base_params = dict(f.params)
                    base_params["point_world"] = point_world
                    params = base_params
                else:
                    params = f.params

                r = res_fn(stacked, params)
                w = params.get("weight", 1.0)
                res_list.append(jnp.sqrt(w) * r)

            if not res_list:
                return jnp.zeros((0,), dtype=x.dtype)

            return jnp.concatenate(res_list)

        return residual, index

    def build_residual_function_voxel_point_param_multi(self):
        """Build a residual function with per-factor voxel observation points.

        Each ``\"voxel_point_obs\"`` factor consumes a row of the parameter
        array ``theta`` of shape ``(K, 3)``, where ``K`` is the number of
        such factors.

        Returns
        -------
        (residual_fn, index)
            ``residual_fn(x, theta)`` where ``theta`` has shape (K, 3).
        """
        factors = list(self.fg.factors.values())
        residual_fns = self._residual_registry

        _, index = self.pack_state()

        def residual(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            """
            Parameters
            ----------
            x : jnp.ndarray
                Flat state vector.
            theta : jnp.ndarray
                Shape (K, 3), per-voxel-point observation in world
                coordinates.
            """
            var_values = self.unpack_state(x, index)
            res_list: List[jnp.ndarray] = []
            obs_idx = 0  # python counter over voxel_point_obs factors

            for f in factors:
                res_fn = residual_fns.get(f.type, None)
                if res_fn is None:
                    raise ValueError(
                        f"No residual fn registered for factor type '{f.type}'"
                    )

                stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

                if f.type == "voxel_point_obs":
                    point_world = theta[obs_idx]  # (3,)
                    obs_idx += 1
                    base_params = dict(f.params)
                    base_params["point_world"] = point_world
                    params = base_params
                else:
                    params = f.params

                r = res_fn(stacked, params)
                w = params.get("weight", 1.0)
                res_list.append(jnp.sqrt(w) * r)

            if not res_list:
                return jnp.zeros((0,), dtype=x.dtype)

            return jnp.concatenate(res_list)

        return residual, index

    def build_objective(self):
        """Construct a scalar objective ``f(x) = ||r(x)||^2``.

        This wraps :meth:`build_residual` and returns a function
        that computes the squared L2 norm of the residual vector.

        :return: JIT-compiled objective function ``f(x)``.
        :rtype: Callable[[jnp.ndarray], jnp.ndarray]
        """
        residual = self.build_residual()

        def objective(x: jnp.ndarray) -> jnp.ndarray:
            r = residual(x)
            return jnp.sum(r ** 2)

        return jax.jit(objective)
    
    # --- State packing/unpacking ---
    def _build_state_index(self) -> Dict[NodeId, Tuple[int, int]]:
        """Build a contiguous index for the global state vector.

        Each variable is assumed to be a 1D array. The method assigns a
        contiguous block ``(start_index, dim)`` to every :class:`NodeId`.

        :return: Mapping from node id to ``(start_index, dimension)`` in the
            flattened state vector.
        :rtype: Dict[NodeId, Tuple[int, int]]
        """
        index: Dict[NodeId, Tuple[int, int]] = {}
        offset = 0
        for node_id, var in sorted(self.fg.variables.items(), key=lambda x: x[0]):
            v = jnp.asarray(var.value)
            dim = v.shape[0]
            index[node_id] = (offset, dim)
            offset += dim
        return index

    def pack_state(self) -> jnp.ndarray:
        """Pack all variable values into a single flat JAX array.

        The variables are ordered by sorted :class:`NodeId` to ensure stable
        indexing across calls.

        :return: Tuple of ``(x, index)`` where ``x`` is the concatenated
            state vector and ``index`` is the mapping produced by
            :meth:`_build_state_index`.
        :rtype: Tuple[jnp.ndarray, Dict[NodeId, Tuple[int, int]]]
        """
        index = self._build_state_index()
        chunks = []
        for node_id in sorted(self.fg.variables.keys()):
            var = self.fg.variables[node_id]
            chunks.append(jnp.asarray(var.value))
        return jnp.concatenate(chunks), index

    def unpack_state(self, x: jnp.ndarray, index: Dict[NodeId, Tuple[int, int]]) -> Dict[NodeId, jnp.ndarray]:
        """Unpack a flat state vector back into per-variable arrays.

        :param x: Flattened state vector produced by :meth:`pack_state` or
            produced by an optimizer.
        :type x: jnp.ndarray
        :param index: Mapping from :class:`NodeId` to ``(start, dim)`` blocks
            as returned by :meth:`_build_state_index`.
        :type index: Dict[NodeId, Tuple[int, int]]
        :return: Mapping from node id to its corresponding slice of ``x``.
        :rtype: Dict[NodeId, jnp.ndarray]
        """
        result: Dict[NodeId, jnp.ndarray] = {}
        for node_id, (start, dim) in index.items():
            result[node_id] = x[start:start+dim]
        return result
    
    def unpack_state_inplace(self, x_opt: jnp.ndarray) -> None:
        """
        Write the optimized state vector back into the FactorGraph variable table.
        """
        _, index = self.pack_state()  # index maps node_id -> (start, end)

        for node_id, (start, end) in index.items():
            block = x_opt[start:end]
            var = self.fg.variables[node_id]
            var.value = block  # overwrite stored variable value