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
    wm: WorldModel
    pose_trajectory: Dict[Tuple[str, int], int] = field(default_factory=dict)
    noise: SceneGraphNoiseConfig
    def __init__(self) -> None:
        self.wm = WorldModel()
        self.pose_trajectory = {}
        self.noise = SceneGraphNoiseConfig()

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
        return self.wm.add_variable("pose_se3", value)

    def add_place1d(self, x: float) -> int:
        return self.wm.add_variable("place1d", jnp.array([x]))

    def add_room1d(self, x: float) -> int:
        # identical type to place1d for now, but semantically different
        return self.wm.add_variable("place1d", jnp.array([x]))
    
    def add_object3d(self, xyz) -> int:
        """
        Add an object with 3D position (R^3).
        xyz can be list/tuple/array length 3.
        """
        xyz = jnp.array(xyz, dtype=jnp.float32).reshape(3,)
        nid = self.wm.add_variable("object3d", xyz)
        return int(nid)
    
    def add_agent_pose_se3(self, agent: str, t: int, value: jnp.ndarray) -> int:
        """
        Add a SE(3) pose for a given agent at timestep t.

        Returns the underlying node id (int).
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
        Add an additive SE(3) odom factor in R^6.

        dx: translation along x (m)
        sigma: optional override for odom noise; if None, use config.
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
    
    def attach_object_to_pose(
        self,
        pose_id: int,
        obj_id: int,
        offset=(0.0, 0.0, 0.0),
        sigma: float | None = None,
    ) -> int:
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

        sigma: std dev of pose difference; larger sigma => weaker smoothness.
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

        measurement: 3D vector in the *pose frame* that we expect for this landmark.
                     (e.g., ground truth landmark position expressed in pose frame.)

        residual enforces:
            se3_exp(pose)^(-1) * landmark  ≈ measurement
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

        xyz: iterable of length 3, world coordinates.
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
        Relative pose-landmark constraint in pose frame (XYZ).

        measurement: R^3, landmark position in the pose frame.
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
        Bearing-only constraint from pose to landmark.

        bearing: R^3 vector in the pose frame (will be normalized).
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

        point_meas: R^3 point in the pose frame (e.g. back-projected depth).
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

        offset: expected vector from voxel_i to voxel_j (e.g. [dx, 0, 0]).
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
        Add an observation tying a voxel center to a 3D point in *world* coordinates.

        voxel_id: id of a voxel_cell variable
        point_world: length-3 array-like, world-frame point (e.g. from fused depth)
        sigma: optional noise std; if None, use config.voxel_point_obs_sigma
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
        self.wm.optimize(method=method, iters=iters, damping=1e-3, max_step_norm=0.5)

    def get_pose(self, pose_id: int) -> jnp.ndarray:
        nid = NodeId(pose_id)
        return self.wm.fg.variables[nid].value

    def get_place(self, place_id: int) -> float:
        nid = NodeId(place_id)
        return float(self.wm.fg.variables[nid].value[0])

    def dump_state(self) -> Dict[int, jnp.ndarray]:
        return {int(nid): v.value for nid, v in self.wm.fg.variables.items()}