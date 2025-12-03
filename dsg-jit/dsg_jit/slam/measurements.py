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

from dsg_jit.core.math3d import relative_pose_se3, se3_exp

def _apply_weight(residual: jnp.ndarray, params: dict, key: str = "weight") -> jnp.ndarray:
    """
    Optional weighting of residuals.

    Interprets an optional weight entry in ``params`` and rescales the
    residual accordingly:

    * If ``params[key]`` is missing, the residual is returned unchanged.
    * If ``params[key]`` is a scalar ``w``, the residual is scaled as
      ``sqrt(w) * residual`` (scalar square-root information).
    * If ``params[key]`` is a vector, it is treated as a per-component
      square-root information vector and applied elementwise.

    :param residual: Raw residual vector.
    :type residual: jnp.ndarray
    :param params: Factor parameter dictionary, optionally containing a
        weight entry under ``key``.
    :type params: dict
    :param key: Dictionary key under which the weight is stored.
    :type key: str
    :return: Reweighted residual vector.
    :rtype: jnp.ndarray
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
    Convert standard deviation(s) to an information-style weight.

    For a scalar standard deviation ``sigma``, this returns ``1 / sigma**2``.
    For a vector of standard deviations, it returns the elementwise
    inverse-variance ``1 / sigma[i]**2``.

    :param sigma: Scalar or vector of standard deviations.
    :type sigma: Union[float, jnp.ndarray]
    :return: Scalar or vector of weights ``1 / sigma**2``.
    :rtype: jnp.ndarray
    """
    s = jnp.asarray(sigma)
    return 1.0 / (s * s)

def prior_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Simple prior on a single variable.

    Computes ``residual = x - target`` for any vector dimension.

    :param x: Current variable value (flattened state block).
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"target"`` and
        optionally a weight understood by :func:`_apply_weight`.
    :type params: Dict[str, jnp.ndarray]
    :return: Prior residual ``x - target`` (possibly reweighted).
    :rtype: jnp.ndarray
    """
    target = params["target"]
    r = x - target
    return _apply_weight(r, params)


def odom_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Simple odometry-style residual in Euclidean space.

    Interprets ``x`` as a concatenation of two poses ``pose0`` and
    ``pose1`` in R^d and enforces an additive odometry relation

    ``(pose1 - pose0) - measurement = 0``.

    :param x: Stacked pose vector ``[pose0, pose1]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"measurement"`` with
        the desired relative displacement.
    :type params: Dict[str, jnp.ndarray]
    :return: Euclidean odometry residual.
    :rtype: jnp.ndarray
    """
    dim = x.shape[0] // 2
    pose0 = x[:dim]
    pose1 = x[dim:]
    meas = params["measurement"]
    return (pose1 - pose0) - meas


def odom_se3_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    SE(3)-style odometry residual in a 6D vector parameterization.

    Treats each pose as a 6-vector ``[tx, ty, tz, wx, wy, wz]`` and a
    6D measurement in the same parameterization. The residual is

    ``(pose_j - pose_i) - measurement``.

    This is a simple additive model in R^6 and is used as the workhorse
    SE(3) chain factor in many experiments.

    :param x: Stacked pose vector ``[pose_i(6), pose_j(6)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"measurement"`` with
        the desired relative pose in R^6.
    :type params: Dict[str, jnp.ndarray]
    :return: SE(3) odometry residual in R^6.
    :rtype: jnp.ndarray
    """
    #residual true Newton Solver for pure SE(3)
    #TODO replace with true Newton Solver
    dim = x.shape[0] // 2  # should be 6
    pose_i = x[:dim]
    pose_j = x[dim:]
    meas = params["measurement"]
    r = (pose_j - pose_i) - meas
    return r


# Alias for SE(3) chain / odometry residual used in visualization experiments.
def se3_chain_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Alias for SE(3) chain / odometry residual used in visualization.

    This is a thin wrapper around :func:`odom_se3_residual`, so that
    experiments and visualization code can refer to a semantically
    descriptive name ("se3_chain") without duplicating logic.

    :param x: Stacked pose vector ``[pose_i(6), pose_j(6)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"measurement"`` with
        the desired relative pose in R^6.
    :type params: Dict[str, jnp.ndarray]
    :return: SE(3) chain residual produced by :func:`odom_se3_residual`.
    :rtype: jnp.ndarray
    """
    return odom_se3_residual(x, params)

def odom_se3_geodesic_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Experimental SE(3) geodesic residual using ``relative_pose_se3``.

    Interprets ``x`` as two 6D poses in se(3) and uses
    :func:`core.math3d.relative_pose_se3` to compute the estimated
    relative pose before subtracting the provided measurement.

    :param x: Stacked pose vector ``[pose0(6), pose1(6)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"measurement"`` with
        the desired relative pose in se(3), and optionally a weight
        understood by :func:`_apply_weight`.
    :type params: Dict[str, jnp.ndarray]
    :return: Geodesic SE(3) odometry residual in se(3).
    :rtype: jnp.ndarray
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

    Interprets ``x`` as ``[pose, place]`` and enforces that the place
    value tracks a particular coordinate of the pose (e.g., x-position).

    :param x: Stacked state block ``[pose, place]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary with integer entries
        ``"pose_dim"``, ``"place_dim"``, and ``"pose_coord_index"``
        indicating the layout of ``x`` and which pose coordinate to
        attach to. May also contain a weight handled by
        :func:`_apply_weight`.
    :type params: dict
    :return: 1D residual enforcing ``place[0] ≈ pose[pose_coord_index]``.
    :rtype: jnp.ndarray
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
    Residual tying a 3D object position to a pose translation.

    Interprets ``x`` as ``[pose(6), object(3)]`` and encourages the
    object position to coincide with the pose translation plus an
    optional fixed offset.

    :param x: Stacked state block ``[pose(6), object(3)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing integer fields
        ``"pose_dim"`` and ``"obj_dim"``, and optionally ``"offset"``
        (a 3D vector) and a weight handled by :func:`_apply_weight`.
    :type params: dict
    :return: 3D residual ``object - (pose_translation + offset)``.
    :rtype: jnp.ndarray
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
    Temporal smoothness residual between two SE(3) poses.

    Interprets ``x`` as ``[pose_t, pose_t1]`` in R^6 and penalizes the
    difference ``pose_t1 - pose_t``.

    :param x: Stacked state block ``[pose_t(6), pose_t1(6)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary, optionally containing a weight
        handled by :func:`_apply_weight`.
    :type params: dict
    :return: 6D temporal smoothness residual.
    :rtype: jnp.ndarray
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
    Relative pose–landmark residual in SE(3).

    Interprets ``x`` as ``[pose(6), landmark(3)]`` and enforces that the
    landmark, expressed in the pose frame, matches a measured 3D point.

    :param x: Stacked state block ``[pose(6), landmark(3)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"measurement"``
        (a 3D point in the pose frame). Any weighting is applied
        upstream by :func:`_apply_weight`.
    :type params: dict
    :return: 3D residual between predicted and measured landmark
        positions in the pose frame.
    :rtype: jnp.ndarray
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

    Interprets ``x`` as ``[pose(6), landmark(3)]`` and compares the
    predicted bearing from the pose to the landmark against a measured
    bearing vector.

    :param x: Stacked state block ``[pose(6), landmark(3)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"bearing_meas"``
        (a 3D bearing vector in the pose frame). Any weighting is
        applied upstream.
    :type params: dict
    :return: 3D residual ``bearing_pred - bearing_meas``.
    :rtype: jnp.ndarray
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
    Residual between a pose and a voxel center given a point measurement.

    Interprets ``x`` as ``[pose(6), voxel_center(3)]``. The measurement
    is a point expressed in the pose frame; it is projected into the
    world frame and compared against the voxel center.

    :param x: Stacked state block ``[pose(6), voxel_center(3)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"point_meas"``
        (a 3D point in the pose frame). Any weighting is applied
        upstream by :func:`_apply_weight`.
    :type params: dict
    :return: 3D residual ``voxel_center - predicted_world_point``.
    :rtype: jnp.ndarray
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

    Interprets ``x`` as ``[voxel_i(3), voxel_j(3)]`` and penalizes the
    deviation from an expected offset between neighboring voxels.

    :param x: Stacked state block ``[voxel_i(3), voxel_j(3)]``.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"offset"`` (a 3D
        expected difference ``voxel_j - voxel_i``) and optionally a
        weight handled by :func:`_apply_weight`.
    :type params: dict
    :return: 3D residual ``(voxel_j - voxel_i) - offset``.
    :rtype: jnp.ndarray
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
    Observation factor tying a voxel center to a world-frame point.

    Interprets ``x`` as ``[voxel_center(3)]`` and encourages it to match
    an observed point in world coordinates.

    :param x: State block containing a single voxel center.
    :type x: jnp.ndarray
    :param params: Parameter dictionary containing ``"point_world"``
        (a 3D point in the world frame). Any weighting is applied
        upstream by :func:`_apply_weight`.
    :type params: dict
    :return: 3D residual ``voxel_center - point_world``.
    :rtype: jnp.ndarray
    """
    voxel = x[:3]
    point_world = params["point_world"]  # (3,)
    return voxel - point_world

def range_residual(x: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Range-only residual between a pose and a 3D target.

    This residual assumes that ``x`` is the concatenation of a 6D SE(3)
    pose (in se(3) vector form) and a 3D target position::

        x = [pose_se3(6), target(3)]

    Only the translational part of the pose is used. The residual is::

        r = ||target - t|| - r_meas

    where ``t`` is the pose translation and ``r_meas`` is the measured
    range. A scalar weight is applied in the same way as other residuals
    via :func:`_apply_weight`.

    :param x: Concatenated pose and target state, shape ``(9,)``.
    :param params: Parameter dictionary with keys:
        - ``"range"``: scalar or length-1 array containing the
          measured range.
        - ``"weight"`` (optional): scalar weight to apply. If omitted,
          a weight of ``1.0`` is used by :func:`_apply_weight`.
    :return: Residual vector of shape ``(1,)`` (after weighting).
    """
    # Split state: first 6 are se(3) (pose), last 3 are 3D target position.
    pose = x[:6]
    target = x[6:9]

    # Translation component of the pose.
    t = pose[:3]

    # Euclidean distance between pose translation and target.
    diff = target - t
    dist = jnp.linalg.norm(diff)

    # Measured range can be a scalar or length-1 array.
    r_meas = params["range"]
    r_meas = jnp.array(r_meas, dtype=jnp.float32).reshape(())

    # Residual: predicted - measured.
    r = dist - r_meas

    # Wrap as 1D vector and apply weight.
    r_vec = jnp.array([r], dtype=jnp.float32)
    return _apply_weight(r_vec, params)