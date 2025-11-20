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
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Callable, Tuple

import jax
import jax.numpy as jnp

from .types import NodeId, FactorId, Variable, Factor


# Type aliases for clarity
ResidualFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]


@dataclass
class FactorGraph:
    """
    Abstract factor graph.

    - variables: mapping from NodeId -> Variable
    - factors: mapping from FactorId -> Factor
    - residual_fns: mapping factor.type -> callable that computes residuals

    We maintain a single flattened state vector for optimization.
    """
    variables: Dict[NodeId, Variable] = field(default_factory=dict)
    factors: Dict[FactorId, Factor] = field(default_factory=dict)
    residual_fns: Dict[str, ResidualFn] = field(default_factory=dict)

    def add_variable(self, var: Variable) -> None:
        assert var.id not in self.variables
        self.variables[var.id] = var

    def add_factor(self, factor: Factor) -> None:
        assert factor.id not in self.factors
        self.factors[factor.id] = factor

    def register_residual(self, factor_type: str, fn: ResidualFn) -> None:
        self.residual_fns[factor_type] = fn

    # --- State packing/unpacking ---

    def _build_state_index(self) -> Dict[NodeId, Tuple[int, int]]:
        """
        Returns a mapping: NodeId -> (start_index, dim)
        For now we assume all variable.value are 1D arrays.
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
        index = self._build_state_index()
        chunks = []
        for node_id in sorted(self.variables.keys()):
            var = self.variables[node_id]
            chunks.append(jnp.asarray(var.value))
        return jnp.concatenate(chunks), index

    def unpack_state(self, x: jnp.ndarray, index: Dict[NodeId, Tuple[int, int]]) -> Dict[NodeId, jnp.ndarray]:
        result: Dict[NodeId, jnp.ndarray] = {}
        for node_id, (start, dim) in index.items():
            result[node_id] = x[start:start+dim]
        return result

    # --- Objective ---

    def build_residual_function(self):
        """
        Returns a JIT-able function r(x) -> residual vector,
        where x is the packed state.

        This is the core for Gauss-Newton: we can compute J = dr/dx.
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
        """
        Returns a JIT-able function f(x) -> scalar loss = ||r(x)||^2,
        where r(x) is the stacked residual vector.
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
        """
        Build a residual function that treats ALL odom_se3_geodesic 'measurement'
        as explicit parameters.

        Assumptions:
          - There are K odom_se3_geodesic factors in this graph.
          - The parameter 'theta' passed to residual(x, theta) has shape (K, 6),
            where theta[k] is the se(3) measurement for the k-th odom factor
            encountered in self.factors.values() order.

        Returns:
            residual_fn(x, theta): (n_vars,),(K,6) -> (n_residuals,)
            index: NodeId -> index info used by pack_state
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
        """
        Build a residual function that treats voxel_point_obs 'point_world'
        as an explicit parameter.

        Returns:
            residual_fn(x, point_world): jnp.ndarray x, jnp.ndarray point_world -> residual vector
            index: NodeId -> slice/tuple used by pack_state (for consistency)
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
        """
        Build a residual function that treats ALL voxel_point_obs 'point_world'
        as explicit parameters.

        We assume:
          - There are K voxel_point_obs factors in this graph.
          - The parameter 'theta' passed to residual(x, theta) has shape (K, 3),
            where theta[k] is the point_world for the k-th voxel_point_obs
            encountered in self.factors.values() order.

        Returns:
            residual_fn(x, theta): (n_vars,),(K,3) -> (n_residuals,)
            index: NodeId -> index info used by pack_state
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