# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Differentiable factor graph engine for DSG-JIT.

This module implements the central structure of the system: a dynamically
constructed factor graph capable of producing fully JIT-compiled residual and
objective functions. These serve as inputs to Gauss–Newton or gradient-based
solvers inside `optimization/solvers.py`.

The FactorGraph stores:
    - Variables (nodes in the optimization graph)
    - Factors (constraints between variables)
    - Registered residual functions (by factor type)

Key Features
------------
• JIT-compiled residual graph
    The graph is converted into a single fused residual function
    `r(x) : ℝ^N → ℝ^M`, where N = total variable DOFs.

• Automatic Jacobians
    Since `r(x)` is written in JAX, Jacobians are derived via autodiff.

• Type-weighted residuals
    The graph supports learning log-scales for different factor types
    (e.g., odometry, voxel observations), enabling meta-learning of cost
    structure.

• Parameter-differentiable factors
    Several builder functions allow factors to depend on dynamic parameters
    rather than static ones (e.g., SE3 odometry measurement learning,
    voxel point observation learning).

Primary Methods
---------------
pack_state()
    Concatenates all variable values into a single flat JAX array.

unpack_state(x)
    Splits a flat state vector back into per-variable blocks.

build_residual_function()
    Returns a fully JIT-compiled residual function suitable for SLAM or
    voxel optimization.

build_objective()
    Returns a scalar objective function `f(x) = ||r(x)||²`.

build_residual_function_with_type_weights(...)
    Extends the graph to accept learned log-weights for each factor type.

build_residual_function_se3_odom_param_multi()
    Generates a residual function where SE3 odometry measurements themselves
    are learnable parameters (used in Phase 4 of DSG-JIT).

build_residual_function_voxel_point_param_multi()
    Generates a residual function where voxel world-points are learnable
    parameters.

Notes
-----
The FactorGraph is intentionally implemented as a Python object for usability.
All heavy computation is done through JAX-generated functions, making the
system both flexible and extremely fast when JIT-compiled.

This module is the mathematical heart of DSG-JIT: all SLAM, voxel grid,
and hybrid scene-graph optimization flows through the functions defined here.

API Overview
------------
- ``add_variable`` / ``add_factor``: Build up the graph structure.
- ``pack_state`` / ``unpack_state``: Convert between dict-of-nodes and flat JAX arrays.
- ``build_residual_function`` / ``build_objective``: Produce JIT-compiled residuals and objectives.
- ``build_residual_function_with_type_weights``: Add learnable log-scales per factor type.
- ``build_residual_function_se3_odom_param_multi``: Treat SE(3) odometry as learnable parameters.
- ``build_residual_function_voxel_point_param_multi``: Treat voxel observation points as learnable parameters.
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Callable, Tuple, List

import jax
import jax.numpy as jnp

from dsg_jit.core.types import NodeId, FactorId, Variable, Factor


# Type aliases for clarity
ResidualFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]


@dataclass
class FactorGraph:
    """Abstract factor graph for DSG-JIT.

    This class stores variables, factors, and residual functions and exposes
    helpers to pack/unpack the global state vector and to build JIT-compiled
    residual/objective functions.

    :param variables: Mapping from :class:`NodeId` to :class:`Variable` instances
        representing nodes in the factor graph.
    :type variables: Dict[NodeId, Variable]
    :param factors: Mapping from :class:`FactorId` to :class:`Factor` instances
        encoding constraints between variables.
    :type factors: Dict[FactorId, Factor]
    :param residual_fns: Registry mapping factor ``type`` strings to residual
        functions of the form ``fn(stacked_state, params) -> residual``.
    :type residual_fns: Dict[str, ResidualFn]
    """
    variables: Dict[NodeId, Variable] = field(default_factory=dict)
    factors: Dict[FactorId, Factor] = field(default_factory=dict)
    residual_fns: Dict[str, ResidualFn] = field(default_factory=dict)
    #TODO add NeRF's(radiance) to factors as a field

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

    def register_residual(self, factor_type: str, fn: ResidualFn) -> None:
        """Register a residual function for a given factor type.

        :param factor_type: String identifier for the factor type
            (e.g. ``"odom_se3"``, ``"voxel_point_obs"``).
        :type factor_type: str
        :param fn: Residual function implementing the measurement model.
            It must accept ``(stacked_state, params)`` and return a residual
            vector.
        :type fn: ResidualFn
        """
        self.residual_fns[factor_type] = fn

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
        for node_id, var in sorted(self.variables.items(), key=lambda x: x[0]):
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
        for node_id in sorted(self.variables.keys()):
            var = self.variables[node_id]
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

    # --- Objective ---

    def build_residual_function(self):
        """Construct a fused residual function for the entire graph.

        The returned function has signature ``r(x) -> residual`` where ``x``
        is the packed state vector. It concatenates the residuals of all
        registered factors in a fixed order.

        :return: JIT-compiled residual function ``r(x)`` mapping a flat state
            vector to the stacked residuals.
        :rtype: Callable[[jnp.ndarray], jnp.ndarray]
        """
        # Freeze index and factor list inside the closure
        _, index = self.pack_state()
        factors = tuple(self.factors.values())
        residual_fns = dict(self.residual_fns)

        def residual(x: jnp.ndarray) -> jnp.ndarray:
            var_values = self.unpack_state(x, index)
            res_list = []

            for factor in factors:
                residual_fn = residual_fns.get(factor.type, None)
                if residual_fn is None:
                    raise ValueError(f"No residual fn registered for factor type '{factor.type}'")

                vs = [var_values[nid] for nid in factor.var_ids]
                stacked = jnp.concatenate(vs)

                res = residual_fn(stacked, factor.params)
                res_list.append(res)

            if not res_list:
                return jnp.zeros((0,), dtype=x.dtype)

            return jnp.concatenate(res_list)

        return jax.jit(residual)

    def build_objective(self):
        """Construct a scalar objective ``f(x) = ||r(x)||^2``.

        This wraps :meth:`build_residual_function` and returns a function
        that computes the squared L2 norm of the residual vector.

        :return: JIT-compiled objective function ``f(x)``.
        :rtype: Callable[[jnp.ndarray], jnp.ndarray]
        """
        residual = self.build_residual_function()

        def objective(x: jnp.ndarray) -> jnp.ndarray:
            r = residual(x)
            return jnp.sum(r ** 2)

        return jax.jit(objective)
    
    # src/core/factor_graph.py

    # src/core/factor_graph.py

    def build_residual_function_with_type_weights(
        self, factor_type_order: List[str]
    ):
        """Build a residual function that supports learnable type weights.

        The returned function has signature ``r(x, log_scales)`` where
        ``log_scales[i]`` is the log-weight associated with
        ``factor_type_order[i]``. Missing types default to unit weight.

        :param factor_type_order: Ordered list of factor type strings for
            which log-scales will be provided.
        :type factor_type_order: List[str]
        :return: Residual function ``r(x, log_scales)`` that scales each
            factor's residual by ``exp(log_scale)`` according to its type.
        :rtype: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        """
        factors = list(self.factors.values())
        residual_fns = self.residual_fns
        _, index = self.pack_state()

        type_to_idx = {t: i for i, t in enumerate(factor_type_order)}

        def residual(x: jnp.ndarray, log_scales: jnp.ndarray) -> jnp.ndarray:
            var_values = self.unpack_state(x, index)
            res_list = []

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

            return jnp.concatenate(res_list, axis=0)

        return residual
    
    def build_residual_function_se3_odom_param_multi(self):
        """Build a residual function with learnable SE(3) odometry.

        All factors of type ``"odom_se3"`` are treated as depending on a
        parameter array ``theta`` of shape ``(K, 6)``, where ``K`` is the
        number of odometry factors. Each row of ``theta`` represents a
        perturbable se(3) measurement.

        :return: Tuple ``(residual_fn, index)`` where ``residual_fn`` has
            signature ``residual_fn(x, theta)`` and ``index`` is the pack
            index mapping from :meth:`pack_state`.
        :rtype: Tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], Dict[NodeId, Tuple[int, int]]]
        """
        factors = list(self.factors.values())
        residual_fns = self.residual_fns

        _, index = self.pack_state()

        def residual(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            """
            x: flat state vector
            theta: shape (K, 6), per-odom se(3) measurement
            """
            var_values = self.unpack_state(x, index)
            res_list = []
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

            return jnp.concatenate(res_list)

        return residual, index
    
    def build_residual_function_voxel_point_param(self):
        """Build a residual function with a shared voxel observation point.

        All factors of type ``"voxel_point_obs"`` will use a dynamic
        ``point_world`` argument passed at call time, rather than a fixed
        value stored in the factor params.

        :return: Tuple ``(residual_fn, index)`` where ``residual_fn`` has
            signature ``residual_fn(x, point_world)``.
        :rtype: Tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], Dict[NodeId, Tuple[int, int]]]
        """
        # Capture factors, residual fns, and index at build time
        factors = list(self.factors.values())
        residual_fns = self.residual_fns

        _, index = self.pack_state()

        def residual(x: jnp.ndarray, point_world: jnp.ndarray) -> jnp.ndarray:
            """
            x: flat state vector
            point_world: shape (3,), observation point in world coords for ALL voxel_point_obs factors.
                         (For now we assume a single voxel_point_obs, or that all share the same point.)
            """
            var_values = self.unpack_state(x, index)
            res_list = []

            for f in factors:
                res_fn = residual_fns.get(f.type, None)
                if res_fn is None:
                    raise ValueError(f"No residual fn registered for factor type '{f.type}'")

                # Stack variable values in the same order as var_ids
                stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

                # Build params, overriding 'point_world' for voxel_point_obs
                if f.type == "voxel_point_obs":
                    # Copy params but replace point_world with dynamic argument
                    base_params = dict(f.params)
                    base_params["point_world"] = point_world
                    params = base_params
                else:
                    params = f.params

                r = res_fn(stacked, params)
                w = params.get("weight", 1.0)
                res_list.append(jnp.sqrt(w) * r)

            return jnp.concatenate(res_list)

        return residual, index
    
    def build_residual_function_voxel_point_param_multi(self):
        """Build a residual function with per-factor voxel observation points.

        Each ``"voxel_point_obs"`` factor consumes a row of the parameter
        array ``theta`` of shape ``(K, 3)``, where ``K`` is the number of
        such factors.

        :return: Tuple ``(residual_fn, index)`` where ``residual_fn`` has
            signature ``residual_fn(x, theta)``.
        :rtype: Tuple[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], Dict[NodeId, Tuple[int, int]]]
        """
        factors = list(self.factors.values())
        residual_fns = self.residual_fns

        _, index = self.pack_state()

        def residual(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            """
            x: flat state vector
            theta: shape (K, 3), per-voxel-point observation in world coordinates
            """
            var_values = self.unpack_state(x, index)
            res_list = []
            obs_idx = 0  # python counter over voxel_point_obs factors

            for f in factors:
                res_fn = residual_fns.get(f.type, None)
                if res_fn is None:
                    raise ValueError(
                        f"No residual fn registered for factor type '{f.type}'"
                    )

                stacked = jnp.concatenate([var_values[vid] for vid in f.var_ids])

                if f.type == "voxel_point_obs":
                    # Take corresponding row of theta as the point_world
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

            return jnp.concatenate(res_list)

        return residual, index