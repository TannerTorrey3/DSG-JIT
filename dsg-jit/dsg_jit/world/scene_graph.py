# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Dynamic 3D scene graph utilities built on top of the world model.

This module provides a `SceneGraphWorld` abstraction that organizes
poses, places, rooms, objects, and agents into a *dynamic scene graph*
backed by the differentiable factor graph.

Conceptually, this layer is responsible for:

    • Creating typed nodes:
        - Robot / agent poses (SE3)
        - Places / topological nodes (1D)
        - Rooms / regions
        - Objects (points / positions in space)
    • Adding semantic and metric relationships between them via factors:
        - Pose priors
        - SE3 odometry / loop closures
        - Pose–place attachments
        - Pose–object / object–place relations
    • Maintaining lightweight indexing:
        - Maps from (agent, time) → pose NodeId
        - Collections of place / room / object node ids
        - Optional trajectory dictionaries

What it does **not** do:
    • It does not implement the optimizer itself.
    • It does not hard-code SE3 math or Jacobians.
    • It does not perform rendering or perception.

All numerical optimization is delegated to:

    - `world.model.WorldModel` (and its `FactorGraph`)
    - `optimization.solvers` (Gauss–Newton / manifold variants)
    - `slam.manifold` and `slam.measurements` for geometry and residuals

Typical usage
-------------
Experiments in `experiments/exp0X_*.py` follow a common pattern:

    1. Construct a `SceneGraphWorld`.
    2. Add a small chain of poses, places, and objects.
    3. Attach priors and odometry factors.
    4. Optionally attach voxel or observation factors.
    5. Optimize via Gauss–Newton (JIT or non-JIT).
    6. Inspect the resulting scene graph state.

Design goals
------------
- **Ergonomics**: hide raw `NodeId` and factor wiring behind friendly
  helpers like “add pose”, “add agent pose”, “attach place”, etc.
- **Differentiable backbone**: everything created here remains compatible
  with JAX JIT and automatic differentiation downstream.
- **Extensibility**: easy to add new relation types and node types
  without changing the optimizer or lower-level infrastructure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import jax.numpy as jnp

from dsg_jit.core.types import NodeId, FactorId, Factor
from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    odom_se3_residual,
    odom_se3_geodesic_residual,
    pose_place_attachment_residual,
    object_at_pose_residual,
    pose_temporal_smoothness_residual,
    sigma_to_weight,
    pose_landmark_relative_residual,
    pose_landmark_bearing_residual,
    pose_voxel_point_residual,
    voxel_smoothness_residual,
    voxel_point_observation_residual,
    range_residual
)

@dataclass
class SceneGraphNoiseConfig:
    """
    Default noise (standard deviation) per factor type.

    These are in the same units as the residuals:
      - prior / odom / smoothness: R^6 pose (m, m, m, rad, rad, rad)
      - pose_place / object_at_pose: R^1 or R^3 (m)
    """
    prior_pose_sigma: float = 1e-3      # very strong prior on initial pose
    odom_se3_sigma: float = 0.05       # odom: ~5cm std dev
    smooth_pose_sigma: float = 0.5     # temporal smoothness: weak (50cm)
    pose_place_sigma: float = 0.05     # place attachments: 5cm
    object_at_pose_sigma: float = 0.05 # object <-> pose: 5cm
    pose_landmark_sigma: float = 0.05 # relative XYZ
    pose_landmark_bearing_sigma: float = 0.05  #for bearing-only
    pose_voxel_point_sigma: float = 0.05 # pose voxel point
    voxel_smoothness_sigma: float = 0.1 #  voxel smoothness 10cm
    voxel_point_obs_sigma: float = 0.05 #voxel point observation


class SceneGraphWorld:
    """
    World-level dynamic scene graph wrapper that manages typed nodes and semantic relationships,
    built atop the WorldModel. Provides ergonomic helpers for creating and connecting SE(3) poses,
    places, rooms, objects, and agents, and maintains convenient indexing for scene-graph
    experiments. All optimization and factor-graph math is delegated to the underlying WorldModel.
    """
    wm: WorldModel
    pose_trajectory: Dict[Tuple[str, int], int] = field(default_factory=dict)
    noise: SceneGraphNoiseConfig
    # --- Named semantic node indexes ---
    room_nodes: Dict[str, int]
    place_nodes: Dict[str, int]
    object_nodes: Dict[str, int]
    room_place_edges: List[Tuple[int, int]]
    def __init__(self) -> None:
        self.wm = WorldModel()
        self.pose_trajectory = {}
        self.noise = SceneGraphNoiseConfig()

        # --- Named semantic node indexes ---
        self.room_nodes = {}
        self.place_nodes = {}
        self.object_nodes = {}

        # --- Semantic adjacency (for visualization / topology) ---
        self.room_place_edges = []

        # --- Global residuals registry ---
        self.wm.fg.register_residual("prior", prior_residual)
        self.wm.fg.register_residual("odom_se3", odom_se3_residual)
        self.wm.fg.register_residual("odom_se3_geodesic", odom_se3_geodesic_residual)
        self.wm.fg.register_residual("pose_place_attachment", pose_place_attachment_residual)
        self.wm.fg.register_residual("object_at_pose", object_at_pose_residual)
        self.wm.fg.register_residual("pose_temporal_smoothness", pose_temporal_smoothness_residual)
        self.wm.fg.register_residual("pose_landmark_relative", pose_landmark_relative_residual)
        self.wm.fg.register_residual("pose_landmark_bearing", pose_landmark_bearing_residual)
        self.wm.fg.register_residual("pose_voxel_point", pose_voxel_point_residual)
        self.wm.fg.register_residual("voxel_smoothness", voxel_smoothness_residual)
        self.wm.fg.register_residual("voxel_point_obs", voxel_point_observation_residual)
        self.wm.fg.register_residual("range", range_residual)

    # --- Variable helpers ---

    def add_pose_se3(self, value: jnp.ndarray) -> int:
        """
        Add a generic SE(3) pose variable.

        :param value: Length-6 array-like se(3) vector [tx, ty, tz, rx, ry, rz].
        :return: Integer node id of the created pose variable.
        """
        return self.wm.add_variable("pose_se3", value)

    def add_place1d(self, x: float) -> int:
        """
        Add a 1D place variable.

        :param x: Scalar position along a 1D axis (e.g. corridor coordinate).
        :return: Integer node id of the created place variable.
        """
        return self.wm.add_variable("place1d", jnp.array([x]))

    def add_room1d(self, x: jnp.ndarray) -> int:
        """
        Add a 1D 'room' variable (just a scalar, wrapped as a length-1 vector).

        The room is stored in :attr:`room_nodes` using an auto-generated
        string key of the form ``"room1d_{k}"`` where ``k`` is the current
        number of rooms.

        :param x: 1D coordinate, shape ``(1,)`` or a scalar float.
        :return: Integer node id of the created room variable.
        """
        # Normalize to a length-1 float32 vector.
        if isinstance(x, float) or (hasattr(x, "ndim") and x.ndim == 0):
            x = jnp.array([float(x)], dtype=jnp.float32)

        x = jnp.array(x, dtype=jnp.float32).reshape((1,))

        nid = self.wm.add_variable("room1d", x)  # scalar room variable
        nid_int = int(nid)

        # Auto-generate a name for bookkeeping, e.g. "room1d_0", "room1d_1", ...
        name = f"room1d_{len(self.room_nodes)}"
        self.room_nodes[name] = nid_int

        return nid_int

    def add_place3d(self, name: str, xyz) -> int:
        """
        Add a 3D place node (R^3) with a human-readable name.

        This is a semantic helper for dynamic scene-graph style usage.

        :param name: Identifier for the place (for example, ``"place_A"``).
        :param xyz: Iterable of length 3 giving the world-frame position.
        :return: Integer node id of the created place variable.
        """
        value = jnp.array(xyz, dtype=jnp.float32).reshape(3,)
        nid = self.wm.add_variable("place3d", value)
        nid_int = int(nid)
        self.place_nodes[name] = nid_int
        return nid_int

    def add_room(self, name: str, center) -> int:
        """
        Add a 3D room node (R^3 center) with a semantic name.

        This is a thin wrapper around a Euclidean variable, but exposes a
        room-level abstraction for dynamic scene-graph experiments.

        :param name: Identifier for the room (for example, ``"room_A"``).
        :param center: Iterable of length 3 giving the approximate room
            centroid in world coordinates.
        :return: Integer node id of the created room variable.
        """
        value = jnp.array(center, dtype=jnp.float32).reshape(3,)
        nid = self.wm.add_variable("room3d", value)
        nid_int = int(nid)
        self.room_nodes[name] = nid_int
        return nid_int
    
    def add_object3d(self, xyz) -> int:
        """
        Add an object with 3D position (R^3).

        :param xyz: Iterable of length 3 giving the object position in
            world coordinates.
        :return: Integer node id of the created object variable.
        """
        xyz = jnp.array(xyz, dtype=jnp.float32).reshape(3,)
        nid = self.wm.add_variable("object3d", xyz)
        return int(nid)

    def add_named_object3d(self, name: str, xyz) -> int:
        """
        Add a 3D object and register it under a semantic name.

        :param name: Identifier for the object (for example, ``"chair_1"``).
        :param xyz: Iterable of length 3 giving the world-frame position.
        :return: Integer node id of the created object variable.
        """
        obj_id = self.add_object3d(xyz)
        self.object_nodes[name] = obj_id
        return obj_id
    
    def add_agent_pose_se3(self, agent: str, t: int, value: jnp.ndarray) -> int:
        """
        Add an SE(3) pose for a given agent at a specific timestep.

        :param agent: Agent identifier (for example, a robot name).
        :param t: Integer timestep index.
        :param value: Length-6 array-like se(3) vector for the pose.
        :return: Integer node id of the created pose variable.
        """
        nid = self.wm.add_variable("pose_se3", value)
        nid_int = int(nid)
        self.pose_trajectory[(agent, t)] = nid_int
        return nid_int

    # --- Factor helpers ---

    def add_prior_pose_identity(self, pose_id: int) -> int:
        sigma = self.noise.prior_pose_sigma
        weight = sigma_to_weight(sigma)  # scalar

        return int(
            self.wm.add_factor(
                "prior",
                (pose_id,),
                {
                    "target": jnp.zeros(6),
                    "weight": weight,
                },
            )
        )

    def add_odom_se3_additive(
        self,
        pose_i: int,
        pose_j: int,
        dx: float,
        sigma: float | None = None,
    ) -> int:
        """
        Add an additive SE(3) odometry factor in R^6.

        The measurement is a translation along the x-axis plus zero rotation.

        :param pose_i: Node id of the source pose.
        :param pose_j: Node id of the destination pose.
        :param dx: Translation along the x-axis in meters.
        :param sigma: Optional standard deviation for the odometry noise. If
            ``None``, :attr:`SceneGraphNoiseConfig.odom_se3_sigma` is used.
        :return: Integer factor id of the created odometry constraint.
        """
        meas = jnp.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0])

        if sigma is None:
            sigma = self.noise.odom_se3_sigma

        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "odom_se3",
                (pose_i, pose_j),
                {
                    "measurement": meas,
                    "weight": weight,
                },
            )
        )

    def add_odom_se3_geodesic(
        self,
        pose_i: int,
        pose_j: int,
        dx: float,
        yaw: float = 0.0,
        sigma: float | None = None,
    ) -> int:
        """
        Add a geodesic SE(3) odometry factor.

        The measurement is parameterized as translation + yaw in se(3).

        :param pose_i: Node id of the source pose.
        :param pose_j: Node id of the destination pose.
        :param dx: Translation along the x-axis in meters.
        :param yaw: Heading change around the z-axis in radians.
        :param sigma: Optional standard deviation for the odometry noise. If
            ``None``, :attr:`SceneGraphNoiseConfig.odom_se3_sigma` is used.
        :return: Integer factor id of the created odometry constraint.
        """
        meas = jnp.array([dx, 0.0, 0.0, 0.0, 0.0, yaw])

        if sigma is None:
            sigma = self.noise.odom_se3_sigma

        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "odom_se3_geodesic",
                (pose_i, pose_j),
                {
                    "measurement": meas,
                    "weight": weight,
                },
            )
        )
    
    def add_range_measurement(
        self,
        pose_nid: int,
        target_nid: int,
        measured_range: float,
        sigma: float | None = None,
        weight: float | None = None,
    ) -> int:
        """
        Add a range-only sensor factor between a pose and a 3D target.

        This creates a factor of type ``"range"`` whose residual is:

            r = ||target - pose|| - measured_range

        The underlying residual is implemented in ``slam.measurements.range_residual``.

        :param pose_nid: NodeId of the pose (pose_se3) variable.
        :param target_nid: NodeId of the target variable (e.g. place3d, voxel_cell, object3d).
        :param measured_range: Observed distance (same units as world coordinates).
        :param sigma: Optional standard deviation of the measurement noise. If provided,
                      it will be converted to a weight as 1 / sigma^2.
        :param weight: Optional explicit weight. If both ``sigma`` and ``weight`` are given,
                       ``weight`` takes precedence.
        :return: Integer factor id of the created range factor.
        """
        if weight is not None:
            w = float(weight)
        elif sigma is not None:
            w = sigma_to_weight(sigma)
        else:
            w = 1.0

        meas = jnp.array([float(measured_range)], dtype=jnp.float32)
        fid = self.wm.add_factor(
            "range",
            (pose_nid, target_nid),
            {"range": meas, "weight": w},
        )
        return int(fid)

    def add_agent_range_measurement(
        self,
        agent: str,
        t: int,
        target_nid: int,
        measured_range: float,
        sigma: float | None = None,
        weight: float | None = None,
    ) -> int:
        """
        Add a range-only factor using an agent's pose at a given timestep.

        This is a convenience wrapper around :meth:`add_range_measurement`
        that looks up the pose node id from :attr:`pose_trajectory` using
        ``(agent, t)`` and then creates a ``"range"`` factor to a target node.

        :param agent: Agent identifier (for example, a robot name).
        :param t: Integer timestep index for the agent pose.
        :param target_nid: NodeId of the target variable (for example, ``place3d``,
            ``voxel_cell`` or ``object3d``).
        :param measured_range: Observed distance (same units as the world coordinates).
        :param sigma: Optional standard deviation of the measurement noise. If
            provided (and ``weight`` is ``None``), it is converted to a weight via
            :func:`slam.measurements.sigma_to_weight`.
        :param weight: Optional explicit weight. If both ``sigma`` and ``weight``
            are given, ``weight`` takes precedence.
        :return: Integer factor id of the created range factor.
        :raises KeyError: If no pose has been registered for ``(agent, t)``.
        """
        pose_key = (agent, t)
        if pose_key not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t}")

        pose_nid = self.pose_trajectory[pose_key]
        return self.add_range_measurement(
            pose_nid=pose_nid,
            target_nid=target_nid,
            measured_range=measured_range,
            sigma=sigma,
            weight=weight,
        )

    def add_agent_pose_place_attachment(
        self,
        agent: str,
        t: int,
        place_id: int,
        coord_index: int = 0,
        sigma: float | None = None,
    ) -> int:
        """
        Attach an agent pose at time ``t`` to a place node.

        This is a higher-level wrapper around :meth:`add_place_attachment`
        which resolves the pose id via :attr:`pose_trajectory`.

        :param agent: Agent identifier.
        :param t: Integer timestep index.
        :param place_id: Node id of the place variable (1D or 3D).
        :param coord_index: Index of the pose coordinate to tie to the place
            (typically 0 for x, 1 for y, etc.). Defaults to 0.
        :param sigma: Optional noise standard deviation. If ``None``, falls back
            to :attr:`SceneGraphNoiseConfig.pose_place_sigma`.
        :return: Integer factor id of the created attachment constraint.
        :raises KeyError: If no pose has been registered for ``(agent, t)``.
        """
        pose_key = (agent, t)
        if pose_key not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t}")

        pose_id = self.pose_trajectory[pose_key]
        return self.add_place_attachment(
            pose_id=pose_id,
            place_id=place_id,
            coord_index=coord_index,
            sigma=sigma,
        )

    def add_agent_temporal_smoothness(
        self,
        agent: str,
        t: int,
        sigma: float | None = None,
    ) -> int:
        """
        Enforce temporal smoothness between successive poses of a given agent.

        This enforces a smoothness constraint between the poses at timesteps
        ``t`` and ``t+1`` for the specified agent, using
        :meth:`add_temporal_smoothness` internally.

        :param agent: Agent identifier.
        :param t: Timestep index for the first pose in the pair.
        :param sigma: Optional standard deviation controlling smoothness. If
            ``None``, falls back to :attr:`SceneGraphNoiseConfig.smooth_pose_sigma`.
        :return: Integer factor id of the created smoothness constraint.
        :raises KeyError: If either pose ``(agent, t)`` or ``(agent, t+1)`` has
            not been registered.
        """
        key_t = (agent, t)
        key_t1 = (agent, t + 1)

        if key_t not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t}")
        if key_t1 not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t+1}")

        pose_id_t = self.pose_trajectory[key_t]
        pose_id_t1 = self.pose_trajectory[key_t1]
        return self.add_temporal_smoothness(
            pose_id_t=pose_id_t,
            pose_id_t1=pose_id_t1,
            sigma=sigma,
        )

    def add_agent_pose_landmark_relative(
        self,
        agent: str,
        t: int,
        landmark_id: int,
        measurement,
        sigma: float | None = None,
    ) -> int:
        """
        Add a relative pose–landmark constraint for an agent at time ``t``.

        This is a small ergonomic wrapper around
        :meth:`add_pose_landmark_relative` that resolves the pose id using
        :attr:`pose_trajectory`.

        :param agent: Agent identifier.
        :param t: Timestep index for the pose.
        :param landmark_id: Node id of the 3D landmark variable.
        :param measurement: Iterable of length 3 giving the expected landmark
            position in the pose frame.
        :param sigma: Optional noise standard deviation. If ``None``, falls back
            to :attr:`SceneGraphNoiseConfig.pose_landmark_sigma`.
        :return: Integer factor id of the created relative landmark constraint.
        :raises KeyError: If no pose has been registered for ``(agent, t)``.
        """
        key = (agent, t)
        if key not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t}")

        pose_id = self.pose_trajectory[key]
        return self.add_pose_landmark_relative(
            pose_id=pose_id,
            landmark_id=landmark_id,
            measurement=measurement,
            sigma=sigma,
        )

    def add_agent_pose_landmark_bearing(
        self,
        agent: str,
        t: int,
        landmark_id: int,
        bearing,
        sigma: float | None = None,
    ) -> int:
        """
        Add a bearing-only pose–landmark constraint for an agent at time ``t``.

        This wraps :meth:`add_pose_landmark_bearing` and resolves the pose id
        from :attr:`pose_trajectory`.

        :param agent: Agent identifier.
        :param t: Timestep index for the pose.
        :param landmark_id: Node id of the 3D landmark variable.
        :param bearing: Iterable of length 3 giving the bearing vector in the
            pose frame (it will be normalized internally).
        :param sigma: Optional noise standard deviation. If ``None``, falls back
            to :attr:`SceneGraphNoiseConfig.pose_landmark_bearing_sigma`.
        :return: Integer factor id of the created bearing constraint.
        :raises KeyError: If no pose has been registered for ``(agent, t)``.
        """
        key = (agent, t)
        if key not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t}")

        pose_id = self.pose_trajectory[key]
        return self.add_pose_landmark_bearing(
            pose_id=pose_id,
            landmark_id=landmark_id,
            bearing=bearing,
            sigma=sigma,
        )

    def add_agent_pose_voxel_point(
        self,
        agent: str,
        t: int,
        voxel_id: int,
        point_meas,
        sigma: float | None = None,
    ) -> int:
        """
        Constrain a voxel cell using a point measurement from an agent pose.

        This wraps :meth:`add_pose_voxel_point` and resolves the pose id from
        :attr:`pose_trajectory`.

        :param agent: Agent identifier.
        :param t: Timestep index for the pose.
        :param voxel_id: Node id of the voxel cell variable.
        :param point_meas: Iterable of length 3 giving a point in the pose
            frame (for example, a back-projected LiDAR or depth sample).
        :param sigma: Optional noise standard deviation. If ``None``, falls
            back to :attr:`SceneGraphNoiseConfig.pose_voxel_point_sigma`.
        :return: Integer factor id of the created voxel-point constraint.
        :raises KeyError: If no pose has been registered for ``(agent, t)``.
        """
        key = (agent, t)
        if key not in self.pose_trajectory:
            raise KeyError(f"No pose registered for agent={agent!r}, t={t}")

        pose_id = self.pose_trajectory[key]
        return self.add_pose_voxel_point(
            pose_id=pose_id,
            voxel_id=voxel_id,
            point_meas=point_meas,
            sigma=sigma,
        )

    def attach_pose_to_place_x(self, pose_id: int, place_id: int) -> int:
        """
        Attach a pose to a 1D place along the x-coordinate.

        This is a low-level helper that assumes a 6D pose and 1D place.

        :param pose_id: Node id of the SE(3) pose variable.
        :param place_id: Node id of the 1D place variable.
        :return: Integer factor id of the created attachment constraint.
        """
        pose_dim = jnp.array(6)
        place_dim = jnp.array(1)
        pose_coord_index = jnp.array(0)

        sigma = self.noise.pose_place_sigma
        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "pose_place_attachment",
                (pose_id, place_id),
                {
                    "pose_dim": pose_dim,
                    "place_dim": place_dim,
                    "pose_coord_index": pose_coord_index,
                    "weight": weight,
                },
            )
        )

    def attach_pose_to_room_x(self, pose_id: int, room_id: int) -> int:
        """
        Attach a pose to a 1D room along the x-coordinate.

        This is analogous to :meth:`attach_pose_to_place_x` but uses a room
        node instead of a place node.

        :param pose_id: Node id of the SE(3) pose variable.
        :param room_id: Node id of the 1D room variable.
        :return: Integer factor id of the created attachment constraint.
        """
        pose_dim = jnp.array(6)
        place_dim = jnp.array(1)
        pose_coord_index = jnp.array(0)

        sigma = self.noise.pose_place_sigma
        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "pose_place_attachment",
                (pose_id, room_id),
                {
                    "pose_dim": pose_dim,
                    "place_dim": place_dim,
                    "pose_coord_index": pose_coord_index,
                    "weight": weight,
                },
            )
        )

    def add_place_attachment(
        self,
        pose_id: int,
        place_id: int,
        coord_index: int = 0,
        sigma: float | None = None,
    ) -> int:
        """
        Attach a SE(3) pose to a place node (1D or 3D).

        This is a higher-level, dimension-aware wrapper around the
        ``pose_place_attachment`` residual, and is intended for scene-graph
        style experiments where places may be either 1D (topological) or
        3D (metric positions).

        :param pose_id: Node id of the SE(3) pose variable.
        :param place_id: Node id of the place variable. The underlying state
            dimension is inferred at runtime from the factor graph (for
            example, 1 for ``place1d`` or 3 for ``place3d``).
        :param coord_index: Index of the pose coordinate to tie to the place
            (typically 0 for x, 1 for y, etc.). Defaults to 0.
        :param sigma: Optional noise standard deviation. If ``None``, falls
            back to :attr:`SceneGraphNoiseConfig.pose_place_sigma`.
        :return: Integer factor id of the created attachment constraint.
        """
        # Infer place dimensionality from the underlying variable.
        place_nid = NodeId(place_id)
        place_var = self.wm.fg.variables[place_nid]
        place_dim_val = place_var.value.shape[0]

        pose_dim = jnp.array(6)
        place_dim = jnp.array(place_dim_val)
        pose_coord_index = jnp.array(coord_index)

        if sigma is None:
            sigma = self.noise.pose_place_sigma
        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "pose_place_attachment",
                (pose_id, place_id),
                {
                    "pose_dim": pose_dim,
                    "place_dim": place_dim,
                    "pose_coord_index": pose_coord_index,
                    "weight": weight,
                },
            )
        )
    
    def add_room_place_edge(self, room_id: int, place_id: int) -> None:
        """
        Register a semantic edge between a room node and a place node.

        This helper is intentionally lightweight: it does *not* add a numeric
        factor to the underlying factor graph. Instead it records topological
        connectivity for visualization and higher-level reasoning, similar to
        classic dynamic scene-graph frameworks.

        :param room_id: Integer node id of the room variable.
        :param place_id: Integer node id of the place variable.
        :return: None.
        """
        self.room_place_edges.append((int(room_id), int(place_id)))
    
    def attach_object_to_pose(
        self,
        pose_id: int,
        obj_id: int,
        offset=(0.0, 0.0, 0.0),
        sigma: float | None = None,
    ) -> int:
        """
        Attach an object to a pose with an optional 3D offset.

        :param pose_id: Node id of the SE(3) pose variable.
        :param obj_id: Node id of the 3D object variable.
        :param offset: Iterable of length 3 giving the offset from the pose
            frame to the object in pose coordinates.
        :param sigma: Optional noise standard deviation. If ``None``, falls
            back to :attr:`SceneGraphNoiseConfig.object_at_pose_sigma`.
        :return: Integer factor id of the created object-at-pose constraint.
        """
        pose_dim = jnp.array(6)
        obj_dim = jnp.array(3)
        offset_arr = jnp.array(offset, dtype=jnp.float32).reshape(3,)

        if sigma is None:
            sigma = self.noise.object_at_pose_sigma
        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "object_at_pose",
                (pose_id, obj_id),
                {
                    "pose_dim": pose_dim,
                    "obj_dim": obj_dim,
                    "offset": offset_arr,
                    "weight": weight,
                },
            )
        )

    def get_object3d(self, obj_id: int) -> jnp.ndarray:
        """
        Return the current 3D position of an object.

        :param obj_id: Integer node id of the object variable.
        :return: JAX array of shape ``(3,)`` giving the object position.
        """
        from core.types import NodeId
        nid = NodeId(obj_id)
        return self.wm.fg.variables[nid].value
    
    def add_temporal_smoothness(
        self,
        pose_id_t: int,
        pose_id_t1: int,
        sigma: float | None = None,
    ) -> int:
        """
        Enforce smoothness between successive poses.

        :param pose_id_t: Node id of the pose at time ``t``.
        :param pose_id_t1: Node id of the pose at time ``t+1``.
        :param sigma: Optional standard deviation of the pose difference; a
            larger value gives weaker smoothness. If ``None``,
            :attr:`SceneGraphNoiseConfig.smooth_pose_sigma` is used.
        :return: Integer factor id of the created smoothness constraint.
        """
        if sigma is None:
            sigma = self.noise.smooth_pose_sigma
        weight = sigma_to_weight(sigma)

        return int(
            self.wm.add_factor(
                "pose_temporal_smoothness",
                (pose_id_t, pose_id_t1),
                {"weight": weight},
            )
        )
    
    def add_pose_landmark_relative(
        self,
        pose_id: int,
        landmark_id: int,
        measurement,
        sigma: float | None = None,
    ) -> int:
        """
        Add a relative measurement between a pose and a 3D landmark.

        The measurement is expressed in the pose frame.

        :param pose_id: Node id of the SE(3) pose variable.
        :param landmark_id: Node id of the 3D landmark variable.
        :param measurement: Iterable of length 3 giving the expected landmark
            position in the pose frame.
        :param sigma: Optional noise standard deviation. If ``None``,
            :attr:`SceneGraphNoiseConfig.pose_landmark_sigma` is used.
        :return: Integer factor id of the created relative landmark constraint.
        """
        meas = jnp.array(measurement, dtype=jnp.float32).reshape(3,)

        if sigma is None:
            sigma = self.noise.pose_landmark_sigma
        weight = sigma_to_weight(sigma)

        fid = self.wm.add_factor(
            "pose_landmark_relative",
            (pose_id, landmark_id),
            {
                "measurement": meas,
                "weight": weight,
            },
        )
        return int(fid)
    
    # ---- Landmark helpers ----

    def add_landmark3d(self, xyz) -> int:
        """
        Add a 3D landmark node (R^3).

        :param xyz: Iterable of length 3 giving world coordinates.
        :return: Integer node id of the created landmark variable.
        """
        value = jnp.array(xyz, dtype=jnp.float32).reshape(3,)
        nid = self.wm.add_variable("landmark3d", value)
        return int(nid)

    def add_pose_landmark_relative(
        self,
        pose_id: int,
        landmark_id: int,
        measurement,
        sigma: float | None = None,
    ) -> int:
        """
        Add a relative measurement between a pose and a 3D landmark.

        The measurement is expressed in the pose frame.

        :param pose_id: Node id of the SE(3) pose variable.
        :param landmark_id: Node id of the 3D landmark variable.
        :param measurement: Iterable of length 3 giving the expected landmark
            position in the pose frame.
        :param sigma: Optional noise standard deviation. If ``None``,
            :attr:`SceneGraphNoiseConfig.pose_landmark_sigma` is used.
        :return: Integer factor id of the created relative landmark constraint.
        """
        meas = jnp.array(measurement, dtype=jnp.float32).reshape(3,)

        if sigma is None:
            sigma = self.noise.pose_landmark_sigma
        weight = sigma_to_weight(sigma)

        fid = self.wm.add_factor(
            "pose_landmark_relative",
            (pose_id, landmark_id),
            {
                "measurement": meas,
                "weight": weight,
            },
        )
        return int(fid)

    def add_pose_landmark_bearing(
        self,
        pose_id: int,
        landmark_id: int,
        bearing,
        sigma: float | None = None,
    ) -> int:
        """
        Add a bearing-only constraint from pose to landmark.

        :param pose_id: Node id of the SE(3) pose variable.
        :param landmark_id: Node id of the 3D landmark variable.
        :param bearing: Iterable of length 3 giving the bearing vector in the
            pose frame (will be normalized internally).
        :param sigma: Optional noise standard deviation. If ``None``,
            :attr:`SceneGraphNoiseConfig.pose_landmark_bearing_sigma` is used.
        :return: Integer factor id of the created bearing constraint.
        """
        b = jnp.array(bearing, dtype=jnp.float32).reshape(3,)
        b = b / (jnp.linalg.norm(b) + 1e-8)

        if sigma is None:
            sigma = self.noise.pose_landmark_bearing_sigma
        weight = sigma_to_weight(sigma)

        fid = self.wm.add_factor(
            "pose_landmark_bearing",
            (pose_id, landmark_id),
            {
                "bearing_meas": b,
                "weight": weight,
            },
        )
        return int(fid)
    
        # ---- Voxel helpers ----

    def add_voxel_cell(self, xyz) -> int:
        """
        Add a voxel cell center in world coordinates (R^3).

        :param xyz: Iterable of length 3 giving the voxel center position.
        :return: Integer node id of the created voxel variable.
        """
        value = jnp.array(xyz, dtype=jnp.float32).reshape(3,)
        nid = self.wm.add_variable("voxel_cell", value)
        return int(nid)

    def add_pose_voxel_point(
        self,
        pose_id: int,
        voxel_id: int,
        point_meas,
        sigma: float | None = None,
    ) -> int:
        """
        Constrain a voxel cell to align with a point measurement seen from a pose.

        :param pose_id: Node id of the SE(3) pose variable.
        :param voxel_id: Node id of the voxel cell variable.
        :param point_meas: Iterable of length 3 giving a point in the pose
            frame (for example, a back-projected depth sample).
        :param sigma: Optional noise standard deviation. If ``None``,
            :attr:`SceneGraphNoiseConfig.pose_voxel_point_sigma` is used.
        :return: Integer factor id of the created voxel-point constraint.
        """
        point_meas = jnp.array(point_meas, dtype=jnp.float32).reshape(3,)

        if sigma is None:
            sigma = self.noise.pose_voxel_point_sigma
        weight = sigma_to_weight(sigma)

        fid = self.wm.add_factor(
            "pose_voxel_point",
            (pose_id, voxel_id),
            {
                "point_meas": point_meas,
                "weight": weight,
            },
        )
        return int(fid)

    def add_voxel_smoothness(
        self,
        voxel_i_id: int,
        voxel_j_id: int,
        offset,
        sigma: float | None = None,
    ) -> int:
        """
        Enforce grid-like spacing between two voxel centers.

        :param voxel_i_id: Node id of the first voxel cell.
        :param voxel_j_id: Node id of the second voxel cell.
        :param offset: Iterable of length 3 giving the expected vector from
            voxel ``i`` to voxel ``j`` (for example, ``[dx, 0, 0]``).
        :param sigma: Optional noise standard deviation. If ``None``,
            :attr:`SceneGraphNoiseConfig.voxel_smoothness_sigma` is used.
        :return: Integer factor id of the created smoothness constraint.
        """
        offset = jnp.array(offset, dtype=jnp.float32).reshape(3,)

        if sigma is None:
            sigma = self.noise.voxel_smoothness_sigma
        weight = sigma_to_weight(sigma)

        fid = self.wm.add_factor(
            "voxel_smoothness",
            (voxel_i_id, voxel_j_id),
            {
                "offset": offset,
                "weight": weight,
            },
        )
        return int(fid)
    
    # ---- Voxel observation helpers ----

    def add_voxel_point_observation(
        self,
        voxel_id: int,
        point_world,
        sigma: float | None = None,
    ) -> int:
        """
        Add an observation tying a voxel center to a 3D point in world coordinates.

        :param voxel_id: Node id of the voxel cell variable.
        :param point_world: Iterable of length 3 giving a world-frame point
            (for example, from fused depth or a point cloud).
        :param sigma: Optional noise standard deviation. If ``None``,
            :attr:`SceneGraphNoiseConfig.voxel_point_obs_sigma` is used.
        :return: Integer factor id of the created observation constraint.
        """
        point_world = jnp.array(point_world, dtype=jnp.float32).reshape(3,)

        if sigma is None:
            sigma = self.noise.voxel_point_obs_sigma
        weight = sigma_to_weight(sigma)

        fid = self.wm.add_factor(
            "voxel_point_obs",
            (voxel_id,),
            {
                "point_world": point_world,
                "weight": weight,
            },
        )
        return int(fid)

    # --- Optimization / access ---

    def optimize(self, method: str = "gn", iters: int = 40) -> None:
        """
        Run nonlinear optimization over the current factor graph.

        :param method: Optimization method name (currently ``"gn"`` for
            Gauss–Newton).
        :param iters: Maximum number of iterations to run.
        :return: ``None``. The internal world model state is updated in-place.
        """
        self.wm.optimize(method=method, iters=iters, damping=1e-3, max_step_norm=0.5)

    def get_pose(self, pose_id: int) -> jnp.ndarray:
        """
        Return the current SE(3) pose value.

        :param pose_id: Integer node id of the pose variable.
        :return: JAX array of shape ``(6,)`` containing the se(3) vector.
        """
        nid = NodeId(pose_id)
        return self.wm.fg.variables[nid].value

    def get_place(self, place_id: int) -> float:
        """
        Return the current scalar value of a 1D place.

        :param place_id: Integer node id of the place variable.
        :return: Floating-point scalar position.
        """
        nid = NodeId(place_id)
        return float(self.wm.fg.variables[nid].value[0])

    def dump_state(self) -> Dict[int, jnp.ndarray]:
        """
        Return a snapshot of all variable values in the world.

        :return: Dictionary mapping integer node ids to JAX arrays of values.
        """
        return {int(nid): v.value for nid, v in self.wm.fg.variables.items()}