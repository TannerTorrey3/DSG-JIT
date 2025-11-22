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
from typing import Dict, Tuple

import jax.numpy as jnp

from core.types import NodeId
from world.model import WorldModel
from slam.measurements import (
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
    voxel_point_observation_residual
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
    def __init__(self) -> None:
        self.wm = WorldModel()
        self.pose_trajectory = {}
        self.noise = SceneGraphNoiseConfig()

        # --- Named semantic node indexes ---
        self.room_nodes = {}
        self.place_nodes = {}
        self.object_nodes = {}

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

    def add_room1d(self, x: float) -> int:
        """
        Add a 1D room variable.

        This currently uses the same underlying type as a 1D place but is
        kept semantically distinct for higher-level reasoning.

        :param x: Scalar position along a 1D axis.
        :return: Integer node id of the created room variable.
        """
        # identical type to place1d for now, but semantically different
        return self.wm.add_variable("place1d", jnp.array([x]))

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