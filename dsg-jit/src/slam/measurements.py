# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Residual models (measurement factors) for DSG-JIT.

This module defines the *measurement-level* building blocks used by the
factor graph:

    • Each function here implements a residual:
          r(x; params) ∈ ℝᵏ
      compatible with JAX differentiation and JIT compilation.

    • Factor types in the graph (e.g. "prior", "odom_se3_geodesic",
      "voxel_point_obs") are mapped to these residual functions via
      `FactorGraph.register_residual`.

Broadly, the residuals fall into several families:

1. Priors and Simple Euclidean Factors
--------------------------------------
    • `prior_residual`:
        Generic prior on any variable:
            r = x − target

    Useful for:
        - anchoring poses (pose0 ≈ identity)
        - clamping scalar variables (places, rooms, weights, etc.)

2. SE(3) / SLAM-Style Motion Factors
------------------------------------
    • `odom_se3_geodesic_residual`:
        SE(3) relative pose constraint using the group logarithm:
            r = log( meas⁻¹ ∘ (T_i⁻¹ ∘ T_j) )

        Works on "pose_se3" variables and lives in se(3) (6D tangent).

    • (Optionally) additive variants:
        - `odom_se3_additive_residual`
          for simpler experiments where translation/rotation are treated
          additively in ℝ⁶.

These encode frame-to-frame odometry, loop closures, and generic
relative pose constraints between SE(3) nodes.

3. Landmark and Attachment Factors
----------------------------------
    • `pose_landmark_relative_residual`:
        Relative pose between a SE(3) pose and a landmark position,
        typically enforcing:
            T_pose ∘ landmark ≈ measurement

    • `pose_landmark_bearing_residual`:
        Bearing-only constraint between a pose and a landmark (e.g.,
        enforcing angular consistency between measurement and predicted
        direction).

    • `pose_place_attachment_residual`:
        Softly attaches a pose coordinate (e.g. x) to a 1D "place"
        variable, used for 1D topological / metric alignment.

These connect metric states (poses, landmarks, places) into a coherent
SLAM + scene-graph representation.

4. Voxel Grid / Volumetric Factors
----------------------------------
    • `voxel_smoothness_residual`:
        Encourages neighboring voxel centers to form a smooth chain or
        grid. Used to regularize voxel grids representing surfaces or
        1D/2D/3D structures.

    • `voxel_point_observation_residual`:
        Ties a voxel cell to an observed point in world coordinates,
        often used for learning voxel positions from point-like
        observations.

These factors are key to the differentiable voxel experiments and
hybrid SE3 + voxel benchmarks.

5. Weighting and Noise Models
-----------------------------
Most residuals support per-factor weightings via a shared helper:

    • `_apply_weight(r, params)`:
        Applies scalar or diagonal weights to a residual, enabling:

            - Hand-tuned noise models (e.g. σ⁻¹)
            - Learnable factor-type weights (via log_scales)
            - Consistent scaling for multi-term objectives

This is what allows the engine to support *learnable* factor weights in
Phase 4 experiments (e.g. learning odom vs. observation trade-offs).

Design Goals
------------
• **Clear factor semantics**:
    Each residual corresponds to a named factor type used throughout
    tests and experiments, so it’s obvious what each factor is doing.

• **Differentiable and JIT-friendly**:
    All residuals are written to be compatible with `jax.jit` and
    `jax.grad`, enabling higher-level meta-learning and end-to-end
    differentiable training loops.

• **Composable**:
    Residuals do not own the factor graph logic; they simply implement
    r(x; params). All graph structure, manifold handling, and joint
    optimization is handled in `core.factor_graph`, `slam.manifold`,
    and `optimization.solvers`.

Notes
-----
When adding a new factor type:

    1. Implement a residual here:
           def my_factor_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray

    2. Register it with the factor graph:
           fg.register_residual("my_factor", my_factor_residual)

    3. (Optionally) add tests under `tests/` and, if relevant, a
       differentiable experiment under `experiments/`.

This pattern keeps the measurement models centralized and makes the
engine easy to extend for new research ideas.
"""

from __future__ import annotations
from typing import Dict

import jax.numpy as jnp

from core.math3d import relative_pose_se3, se3_exp

def _apply_weight(residual: jnp.ndarray, params: dict, key: str = "weight") -> jnp.ndarray:
    """
    Optional weighting of residuals.

    If params[key] is:
      - missing: no change
      - scalar:  r' = sqrt(w) * r          (scalar weight)
      - vector:  r' = w * r                (per-component sqrt-info)
    """
    w = params.get(key, None)
    if w is None:
        return residual

    w = jnp.asarray(w)

    if w.ndim == 0:
        # scalar weight; use sqrt to interpret as information
        return jnp.sqrt(w) * residual
    else:
        # assume w is already per-component sqrt-info vector
        return w * residual
    
def sigma_to_weight(sigma):
    """
    Convert standard deviation sigma (or vector of sigmas) to a weight usable
    by _apply_weight.

    For scalar sigma:
        w = 1 / sigma^2

    For vector sigma (per-component std devs):
        w[i] = 1 / sigma[i]^2
    """
    s = jnp.asarray(sigma)
    return 1.0 / (s * s)

def prior_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Simple prior on a single variable:
        residual = x - target
    Works for any vector dimension.
    """
    target = params["target"]
    r = x - target
    return _apply_weight(r, params)


def odom_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Original 1D/nd odometry-style residual:

        x = [pose0, pose1]
        residual = (pose1 - pose0) - measurement

    This is still used by your existing 1D tests.
    """
    dim = x.shape[0] // 2
    pose0 = x[:dim]
    pose1 = x[dim:]
    meas = params["measurement"]
    return (pose1 - pose0) - meas


def odom_se3_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    True SE(3) odometry residual:

        pose0, pose1 in R^6: [tx, ty, tz, wx, wy, wz]
        measurement in R^6: desired relative pose from 0 -> 1

    We compute:
      xi_est = relative_pose_se3(pose0, pose1)
      residual = xi_est - measurement
    """
    #residual true Newton Solver for pure SE(3)
    #TODO replace with true Newton Solver
    dim = x.shape[0] // 2  # should be 6
    pose_i = x[:dim]
    pose_j = x[dim:]
    meas = params["measurement"]
    r = (pose_j - pose_i) - meas
    return r

def odom_se3_geodesic_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Experimental: true SE(3) geodesic residual using relative_pose_se3.

    pose0, pose1 in R^6: [tx, ty, tz, wx, wy, wz]
    measurement in R^6: desired relative pose from 0 -> 1 (in se(3))

    residual = relative_pose_se3(pose0, pose1) - measurement

    NOTE:
      - This is NOT yet used in the main pipeline because don't
        have a robust manifold-aware optimizer wired in.
      - here for future Gauss–Newton-on-Lie-group work.
    """
    assert x.shape[0] == 12, "odom_se3_geodesic_residual expects two 6D poses stacked."

    pose0 = x[:6]
    pose1 = x[6:]
    meas = params["measurement"]

    xi_est = relative_pose_se3(pose0, pose1)
    r = xi_est - meas
    return _apply_weight(r, params)
    
def pose_place_attachment_residual(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Residual tying a scalar place variable to one coordinate of a pose.

    x: stacked vector [pose, place]
       - pose_dim:  length of pose block (e.g. 6 for SE(3))
       - place_dim: length of place block (e.g. 1)

    params:
        "pose_dim"          : jnp.array(int)
        "place_dim"         : jnp.array(int)
        "pose_coord_index"  : jnp.array(int)   # which coordinate of pose to use

    Returns:
        r: shape (1,) so it concatenates cleanly with other residuals.
           r[0] = place[0] - pose[pose_coord_index]
    """
    pose_dim = int(params["pose_dim"])
    place_dim = int(params["place_dim"])
    coord_idx = int(params["pose_coord_index"])

    assert x.shape[0] == pose_dim + place_dim

    pose = x[:pose_dim]
    place = x[pose_dim : pose_dim + place_dim]

    # Make it 1D of length 1, not scalar
    r = jnp.array([place[0] - pose[coord_idx]])
    return _apply_weight(r, params)

def object_at_pose_residual(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Tie a 3D object position to a pose translation (optionally with a fixed offset).

    x: stacked [pose, object]
       pose_dim: 6 => [tx, ty, tz, wx, wy, wz]
       obj_dim:  3 => [ox, oy, oz]

    params:
        "pose_dim" : jnp.array(int)
        "obj_dim"  : jnp.array(int)
        "offset"   : jnp.array(3,)  (optional, default zeros)

    residual:
        r = obj - (pose_t + offset)   in R^3
    """
    pose_dim = int(params["pose_dim"])
    obj_dim = int(params["obj_dim"])

    assert x.shape[0] == pose_dim + obj_dim

    pose = x[:pose_dim]
    obj = x[pose_dim : pose_dim + obj_dim]

    offset = params.get("offset", jnp.zeros(3))
    offset = jnp.asarray(offset).reshape(3,)

    t = pose[:3]  # tx, ty, tz
    r = obj - (t + offset)
    return _apply_weight(r, params)

def pose_temporal_smoothness_residual(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Simple temporal smoothness between two SE(3) poses in R^6.

    x: stacked [pose_t, pose_t1] with each in R^6.
    params: ignored for now, placeholder for future weights.

    residual:
        r = pose_t1 - pose_t    (in R^6)
    """
    dim = x.shape[0] // 2
    pose_t = x[:dim]
    pose_t1 = x[dim:]
    r = pose_t1 - pose_t
    return _apply_weight(r, params)

def pose_landmark_relative_residual(
    x: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Residual for a relative pose–landmark measurement in SE(3).

    x is a flat vector formed by concatenating:
      - pose in se(3): [tx, ty, tz, wx, wy, wz]
      - landmark in R^3: [lx, ly, lz]

    params:
      - "measurement": expected landmark position in the *pose frame* (R^3)
      - "weight" (optional): scalar or vector, applied by _apply_weight upstream.

    We compute:
        T = se3_exp(pose)          # world_T_pose
        R, t = T[:3,:3], T[:3,3]
        landmark_world = landmark

        landmark_pose = R^T (landmark_world - t)

        residual_raw = landmark_pose - measurement
    """
    pose = x[:6]
    landmark = x[6:9]

    meas = params["measurement"]  # (3,)
    T = se3_exp(pose)
    R = T[:3, :3]
    t = T[:3, 3]

    # landmark expressed in pose frame
    landmark_pose = R.T @ (landmark - t)

    residual = landmark_pose - meas
    return residual  # weight is applied via _apply_weight in FactorGraph

def pose_landmark_bearing_residual(
    x: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Bearing-only residual between a pose and a 3D landmark.

    x: concatenated [pose(6), landmark(3)]

    params:
      - "bearing_meas": measured bearing in pose frame, R^3
                        (will be treated as unit vector)
      - "weight": handled upstream in FactorGraph

    We compute:
      T = se3_exp(pose)
      R, t = T[:3,:3], T[:3,3]
      landmark_world = landmark

      landmark_pose = R^T (landmark_world - t)
      bearing_pred = normalize(landmark_pose)
      bearing_meas = normalize(bearing_meas)

      residual = bearing_pred - bearing_meas
    """
    pose = x[:6]
    landmark = x[6:9]

    bearing_meas = params["bearing_meas"]  # (3,)

    T = se3_exp(pose)
    R = T[:3, :3]
    t = T[:3, 3]

    landmark_pose = R.T @ (landmark - t)

    def safe_normalize(v):
        n = jnp.linalg.norm(v)
        return v / (n + 1e-8)

    bearing_pred = safe_normalize(landmark_pose)
    bearing_meas = safe_normalize(bearing_meas)

    residual = bearing_pred - bearing_meas
    return residual

def pose_voxel_point_residual(
    x: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Residual between a pose and a voxel center, given a 3D point measurement
    in the pose frame.

    x: [pose(6), voxel_center(3)]

    params:
      - "point_meas": R^3 point in the pose frame
      - "weight": handled upstream in FactorGraph
    """
    pose = x[:6]
    voxel = x[6:9]  # voxel center in world frame

    point_meas = params["point_meas"]  # (3,)

    T = se3_exp(pose)          # 4x4
    R = T[:3, :3]
    t = T[:3, 3]

    world_point = R @ point_meas + t  # predicted world point from this measurement

    residual = voxel - world_point
    return residual

def voxel_smoothness_residual(
    x: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Smoothness / grid regularity constraint between two voxel centers.

    x: [voxel_i(3), voxel_j(3)]

    params:
      - "offset": R^3, expected offset from voxel_i to voxel_j (e.g. [dx, 0, 0])
      - "weight": handled upstream
    """
    voxel_i = x[:3]
    voxel_j = x[3:6]

    offset = params["offset"]  # (3,)

    residual = (voxel_j - voxel_i) - offset
    return residual
    
def voxel_point_observation_residual(
    x: jnp.ndarray,
    params: dict,
) -> jnp.ndarray:
    """
    Observation factor tying a voxel center to a 3D point in *world* coordinates.

    x: [voxel_center(3)]
    params:
      - "point_world": R^3, observed point in world frame
      - "weight": handled upstream in FactorGraph (if present)

    residual = voxel_center - point_world
    """
    voxel = x[:3]
    point_world = params["point_world"]  # (3,)
    return voxel - point_world