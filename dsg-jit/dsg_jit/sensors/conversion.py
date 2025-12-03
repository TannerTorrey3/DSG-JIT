# Copyright (c) 2025 Tanner Kocher
# SPDX-License-Identifier: MIT

"""
Conversion utilities from sensor measurements to factor-graph factors.

This module implements the "measurement conversion layer" for DSG-JIT: given
typed sensor measurements (IMU, LiDAR, cameras, simple range sensors, etc.),
it produces factor descriptions that can be attached to the core
:class:`core.factor_graph.FactorGraph` or to higher-level world/scene-graph
abstractions.

The core idea is to keep sensor-facing code and factor-graph-facing code
decoupled:

* Sensor modules (``sensors.camera``, ``sensors.imu``, ``sensors.lidar``,
  ``sensors.streams``, …) produce strongly-typed measurement objects.
* This module converts those measurements into small
  :class:`MeasurementFactor` records describing:
  
    - ``factor_type`` (string key for the residual)
    - ``var_ids`` (tuple of variable node ids to connect)
    - ``params`` (dictionary passed into the residual function)

Downstream code can then:

* Construct :class:`core.types.Factor` objects from these records.
* Call :meth:`core.factor_graph.FactorGraph.add_factor`.
* Or wrap them in higher-level helpers in :mod:`world.scene_graph`.

This keeps sensor integration "plug-and-play" while preserving a clean,
minimal interface for the optimization engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Mapping, Optional

import jax.numpy as jnp
import time
import numpy as np

from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.core.types import Factor


from dsg_jit.sensors.camera import CameraMeasurement, CameraFrame  
from dsg_jit.sensors.imu import IMUMeasurement        
from dsg_jit.sensors.lidar import LidarMeasurement    


@dataclass
class MeasurementFactor:
    """
    Lightweight description of a factor generated from a sensor measurement.

    This is intentionally decoupled from :class:`core.types.Factor` so that
    the conversion layer does not depend on internal factor-graph details.
    Call :func:`measurement_factors_to_graph_factors` to turn these into
    concrete :class:`Factor` instances, or use
    :func:`apply_measurement_factors` to add them directly to a
    :class:`FactorGraph`.

    :param factor_type: String key for the residual function
        (e.g. ``"range_1d"``, ``"pose_landmark_bearing"``,
        ``"voxel_point_obs"``, etc.). This must correspond to a type
        previously registered via
        :meth:`core.factor_graph.FactorGraph.register_residual`.
    :type factor_type: str
    :param var_ids: Tuple of variable node ids that this factor connects,
        in the order expected by the residual function.
    :type var_ids: tuple[int, ...]
    :param params: Parameter dictionary passed into the residual function.
        Entries are typically NumPy/JAX arrays or scalar floats
        (e.g. ``{"measurement": ..., "sigma": ...}``).
    :type params: dict[str, Any]
    """

    factor_type: str
    var_ids: Tuple[int, ...]
    params: Dict[str, Any]


# ---------------------------------------------------------------------------
# Range / 1D distance measurements
# ---------------------------------------------------------------------------

def range_1d_to_factor(
    pose_id: int,
    place_id: int,
    distance: float,
    sigma: float = 0.05,
    factor_type: str = "range_1d",
) -> MeasurementFactor:
    """
    Convert a scalar range measurement (1D) into a factor description.

    This is the generic bridge used by simple range sensors (e.g. the
    1D range DSG experiment), where you have:

        pose_x - place_x ≈ distance

    and a residual function of type ``factor_type`` that expects a
    ``"measurement"`` and optionally a ``"sigma"`` or ``"weight"`` in
    its parameter dictionary.

    :param pose_id: Node id of the pose variable in the factor graph.
    :type pose_id: int
    :param place_id: Node id of the place or landmark variable in the
        factor graph.
    :type place_id: int
    :param distance: Measured distance between the pose and the place.
    :type distance: float
    :param sigma: Measurement noise standard deviation. If your residual
        uses ``weight`` instead, you can convert this using
        ``1.0 / (sigma ** 2)`` when creating the factor.
    :type sigma: float
    :param factor_type: String key for the residual function. This must
        match a factor type registered into the :class:`FactorGraph`,
        for example ``"range_1d"``.
    :type factor_type: str
    :return: A :class:`MeasurementFactor` describing the range constraint.
    :rtype: MeasurementFactor
    """
    params: Dict[str, Any] = {
        "measurement": jnp.array([distance], dtype=jnp.float32),
        "sigma": float(sigma),
    }
    return MeasurementFactor(
        factor_type=factor_type,
        var_ids=(pose_id, place_id),
        params=params,
    )


# ---------------------------------------------------------------------------
# Bearing measurements (camera-like, 2D/3D directions)
# ---------------------------------------------------------------------------

def bearing_to_factor(
    pose_id: int,
    landmark_id: int,
    bearing_vec: jnp.ndarray,
    sigma: float = 0.01,
    factor_type: str = "pose_landmark_bearing",
) -> MeasurementFactor:
    """
    Convert a bearing vector to a factor description.

    This is suitable for camera-like observations where you have a unit
    bearing vector from the camera pose to a landmark in either camera
    or world coordinates, and a residual function that enforces angular
    consistency between the predicted bearing and the measured one.

    :param pose_id: Node id of the camera/pose variable.
    :type pose_id: int
    :param landmark_id: Node id of the landmark/point variable.
    :type landmark_id: int
    :param bearing_vec: Measured bearing direction as a vector. This
        should typically be normalized (unit length) and have shape
        ``(2,)`` or ``(3,)``, depending on the residual implementation.
    :type bearing_vec: jax.numpy.ndarray
    :param sigma: Measurement noise standard deviation in angular units
        (radians or an equivalent bearing metric).
    :type sigma: float
    :param factor_type: String key for the residual function (e.g.
        ``"pose_landmark_bearing"``). Must match a registered factor
        type in the :class:`FactorGraph`.
    :type factor_type: str
    :return: A :class:`MeasurementFactor` describing the bearing
        constraint.
    :rtype: MeasurementFactor
    """
    bearing_vec = jnp.asarray(bearing_vec, dtype=jnp.float32)
    params: Dict[str, Any] = {
        "bearing_meas": bearing_vec,
        "sigma": float(sigma),
    }
    return MeasurementFactor(
        factor_type=factor_type,
        var_ids=(pose_id, landmark_id),
        params=params,
    )


# ---------------------------------------------------------------------------
# Camera measurements (bearing-style observations)
# ---------------------------------------------------------------------------

def camera_bearings_to_factors(
    pose_id: int,
    landmark_ids: Sequence[int],
    measurement: CameraMeasurement,
    sigma: float = 0.01,
    factor_type: str = "pose_landmark_bearing",
) -> List[MeasurementFactor]:
    """
    Convert a :class:`sensors.camera.CameraMeasurement` containing bearing
    directions into a list of measurement factors.

    This helper assumes that the camera front-end has already extracted
    unit bearing vectors from image data (e.g. via feature detection and
    calibration), and that these bearings have been associated with a set
    of landmark ids. For each ``landmark_id`` and corresponding row in
    ``measurement.bearings``, we construct a bearing factor using
    :func:`bearing_to_factor`.

    :param pose_id: Node id of the camera/pose variable in the factor graph.
    :type pose_id: int
    :param landmark_ids: Sequence of landmark node ids, one per bearing
        vector in ``measurement.bearings``.
    :type landmark_ids: Sequence[int]
    :param measurement: Camera measurement containing an array of bearing
        vectors in ``measurement.bearings`` with shape ``(N, D)`` where
        ``D`` is typically 2 or 3.
    :type measurement: CameraMeasurement
    :param sigma: Bearing noise standard deviation passed through to
        :func:`bearing_to_factor`.
    :type sigma: float
    :param factor_type: Factor type string for the bearing residual,
        usually ``"pose_landmark_bearing"``.
    :type factor_type: str
    :return: List of :class:`MeasurementFactor` objects, one per
        (pose, landmark) bearing observation.
    :rtype: list[MeasurementFactor]
    :raises ValueError: If the number of landmark ids does not match the
        number of bearing vectors stored in the measurement.
    """
    bearings = jnp.asarray(measurement.bearings, dtype=jnp.float32)
    if bearings.ndim != 2:
        raise ValueError(
            f"measurement.bearings must have shape (N, D), got {bearings.shape}"
        )
    if len(landmark_ids) != bearings.shape[0]:
        raise ValueError(
            f"len(landmark_ids)={len(landmark_ids)} does not match "
            f"measurement.bearings.shape[0]={bearings.shape[0]}"
        )

    factors: List[MeasurementFactor] = []
    for lid, b in zip(landmark_ids, bearings):
        factors.append(
            bearing_to_factor(
                pose_id=pose_id,
                landmark_id=int(lid),
                bearing_vec=b,
                sigma=sigma,
                factor_type=factor_type,
            )
        )
    return factors


# ---------------------------------------------------------------------------
# Voxel / point-based observations (e.g. LiDAR, depth, point clouds)
# ---------------------------------------------------------------------------

def voxel_point_obs_factor(
    voxel_id: int,
    point_world: jnp.ndarray,
    sigma: float = 0.05,
    factor_type: str = "voxel_point_obs",
) -> MeasurementFactor:
    """
    Convert a single 3D point observation into a voxel-point factor.

    This is intended for mapping-style sensors (LiDAR, depth cameras,
    RGB-D, stereo) where you receive one or more 3D points in world
    coordinates and want to attach them to a voxel cell center.

    :param voxel_id: Node id of the voxel variable (``voxel_cell``)
        in the factor graph.
    :type voxel_id: int
    :param point_world: Observed 3D point in world coordinates with shape
        ``(3,)``.
    :type point_world: jax.numpy.ndarray
    :param sigma: Noise level for this observation in world units
        (meters, etc.).
    :type sigma: float
    :param factor_type: Factor type string (e.g. ``"voxel_point_obs"``)
        corresponding to the voxel-point residual used in your
        measurement model.
    :type factor_type: str
    :return: A :class:`MeasurementFactor` describing the voxel-point
        observation.
    :rtype: MeasurementFactor
    """
    point_world = jnp.asarray(point_world, dtype=jnp.float32).reshape(3,)
    params: Dict[str, Any] = {
        "point_world": point_world,
        "sigma": float(sigma),
    }
    return MeasurementFactor(
        factor_type=factor_type,
        var_ids=(voxel_id,),
        params=params,
    )


def lidar_scan_to_voxel_factors(
    voxel_ids: Sequence[int],
    points_world: jnp.ndarray,
    sigma: float = 0.05,
    factor_type: str = "voxel_point_obs",
) -> List[MeasurementFactor]:
    """
    Convert a LiDAR point cloud into per-voxel observation factors.

    This helper assumes that a pre-processing step has already:

    * Projected the LiDAR ranges into 3D points in world coordinates.
    * Associated each point with a voxel id (e.g. via a voxel grid index).

    Given a list of voxel node ids and a matching array of 3D points,
    this function returns one :class:`MeasurementFactor` per point.

    :param voxel_ids: Iterable of voxel node ids, one per point.
    :type voxel_ids: Sequence[int]
    :param points_world: Array of 3D points in world coordinates with
        shape ``(N, 3)``, where ``N == len(voxel_ids)``.
    :type points_world: jax.numpy.ndarray
    :param sigma: Noise level for each point in world units.
    :type sigma: float
    :param factor_type: Factor type string used for all voxel-point
        factors, typically ``"voxel_point_obs"``.
    :type factor_type: str
    :return: A list of :class:`MeasurementFactor` objects, one for each
        point/voxel pair.
    :rtype: list[MeasurementFactor]
    """
    points_world = jnp.asarray(points_world, dtype=jnp.float32)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"points_world must have shape (N, 3), got {points_world.shape}"
        )
    if len(voxel_ids) != points_world.shape[0]:
        raise ValueError(
            f"voxel_ids length {len(voxel_ids)} does not match "
            f"points_world.shape[0] {points_world.shape[0]}"
        )

    factors: List[MeasurementFactor] = []
    for vid, pt in zip(voxel_ids, points_world):
        factors.append(
            voxel_point_obs_factor(
                voxel_id=int(vid),
                point_world=pt,
                sigma=sigma,
                factor_type=factor_type,
            )
        )
    return factors


# ---------------------------------------------------------------------------
# LiDAR scan measurements
# ---------------------------------------------------------------------------

def lidar_measurement_to_voxel_factors(
    measurement: LidarMeasurement,
    sigma: float = 0.05,
    factor_type: str = "voxel_point_obs",
) -> List[MeasurementFactor]:
    """
    Convert a :class:`sensors.lidar.LidarMeasurement` into voxel-point
    observation factors.

    This helper assumes that the LiDAR front-end has already projected raw
    ranges into 3D world coordinates and, optionally, associated each
    point with a voxel id. If ``measurement.voxel_ids`` is provided, it is
    used directly; otherwise, the caller is expected to supply voxel
    associations separately.

    Concretely, this is a thin wrapper around
    :func:`lidar_scan_to_voxel_factors`, using
    ``measurement.points_world`` and ``measurement.voxel_ids``.

    :param measurement: LiDAR measurement containing a point cloud in
        world coordinates and optional voxel assignments.
    :type measurement: LidarMeasurement
    :param sigma: Noise level for each point in world units, forwarded to
        :func:`voxel_point_obs_factor`.
    :type sigma: float
    :param factor_type: Factor type string used for all voxel-point
        factors, typically ``"voxel_point_obs"``.
    :type factor_type: str
    :return: List of :class:`MeasurementFactor` objects describing the
        LiDAR point cloud constraints.
    :rtype: list[MeasurementFactor]
    :raises ValueError: If ``measurement.voxel_ids`` is ``None`` or its
        length does not match the number of points.
    """
    points_world = jnp.asarray(measurement.points_world, dtype=jnp.float32)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(
            f"measurement.points_world must have shape (N, 3), got {points_world.shape}"
        )

    if measurement.voxel_ids is None:
        raise ValueError(
            "measurement.voxel_ids is None; voxel associations are required "
            "to build voxel-point factors."
        )

    voxel_ids_seq: Sequence[int] = list(measurement.voxel_ids)
    if len(voxel_ids_seq) != points_world.shape[0]:
        raise ValueError(
            f"len(measurement.voxel_ids)={len(voxel_ids_seq)} does not match "
            f"measurement.points_world.shape[0]={points_world.shape[0]}"
        )

    return lidar_scan_to_voxel_factors(
        voxel_ids=voxel_ids_seq,
        points_world=points_world,
        sigma=sigma,
        factor_type=factor_type,
    )


# ---------------------------------------------------------------------------
# IMU measurements (placeholder / future preintegration)
# ---------------------------------------------------------------------------

def imu_to_factors_placeholder(
    pose_ids: Sequence[int],
    imu_meas: IMUMeasurement,
) -> List[MeasurementFactor]:
    """
    Placeholder conversion from IMU measurement to factor descriptions.

    In a full SLAM system, IMU data is typically handled via *preintegration*
    over multiple high-rate samples to produce a single inertial factor
    between two poses. That logic is non-trivial and highly application
    specific, so this function serves as a placeholder and example.

    For now, it returns an empty list and is intended to be replaced with
    a proper preintegration pipeline in future work.

    :param pose_ids: Sequence of pose node ids between which an IMU factor
        would be created (e.g. previous pose id and current pose id).
    :type pose_ids: Sequence[int]
    :param imu_meas: A single IMU measurement containing accelerometer and
        gyroscope readings along with a timestamp.
    :type imu_meas: IMUMeasurement
    :return: An empty list. Replace with application-specific IMU factor
        generation as needed.
    :rtype: list[MeasurementFactor]
    """
    _ = pose_ids, imu_meas  # avoid unused variable warnings
    return []


# ---------------------------------------------------------------------------
# Helpers to apply MeasurementFactor objects to a FactorGraph
# ---------------------------------------------------------------------------

def measurement_factors_to_graph_factors(
    meas_factors: Iterable[MeasurementFactor],
) -> List[Factor]:
    """
    Convert a sequence of :class:`MeasurementFactor` objects to concrete
    :class:`core.types.Factor` instances.

    Unique factor ids are generated using a simple running index; if you
    need stable ids, you can post-process the factors or construct
    them manually instead.

    :param meas_factors: Iterable of :class:`MeasurementFactor` objects
        returned by the conversion helpers in this module.
    :type meas_factors: Iterable[MeasurementFactor]
    :return: A list of :class:`Factor` instances suitable for adding to
        a :class:`FactorGraph`.
    :rtype: list[Factor]
    """
    graph_factors: List[Factor] = []
    for idx, mf in enumerate(meas_factors):
        fid = f"meas_{idx}"
        graph_factors.append(
            Factor(
                id=fid,
                type=mf.factor_type,
                var_ids=tuple(mf.var_ids),
                params=dict(mf.params),
            )
        )
    return graph_factors


def apply_measurement_factors(
    fg: FactorGraph,
    meas_factors: Iterable[MeasurementFactor],
) -> None:
    """
    Add a sequence of measurement-derived factors to a factor graph.

    This is a thin convenience wrapper around
    :func:`measurement_factors_to_graph_factors` and
    :meth:`core.factor_graph.FactorGraph.add_factor`.

    :param fg: The factor graph to which the new factors should be added.
    :type fg: FactorGraph
    :param meas_factors: Iterable of :class:`MeasurementFactor` objects.
    :type meas_factors: Iterable[MeasurementFactor]
    :return: This function has no return value; it mutates ``fg`` in-place
        by adding new factors.
    :rtype: None
    """
    for factor in measurement_factors_to_graph_factors(meas_factors):
        fg.add_factor(factor)

# ---------------------------------------------------------------------------
# IMU → SE3 delta / odometry-style increment
# ---------------------------------------------------------------------------

def integrate_imu_delta(
    imu: IMUMeasurement,
    dt: float,
    gravity: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Integrate a single IMU sample into a small SE(3) increment (se(3) vector).

    This is a **very** simple, single-step integrator meant as a starting
    point / placeholder. For real applications, a proper preintegration
    scheme or filter should be used instead.

    The returned 6D vector is in the form::

        [dtx, dty, dtz, drx, dry, drz]

    where ``dt*`` is the approximate translational displacement in the IMU
    frame and ``dr*`` is a small-angle approximation for the rotation.

    :param imu: IMU measurement containing linear acceleration and angular
        velocity in the sensor frame.
    :param dt: Time step in seconds between this sample and the previous one.
    :param gravity: Optional gravity vector (in sensor frame). If provided,
        it is subtracted from the measured acceleration before integration.
        If ``None``, no gravity compensation is performed.
    :returns: A length-6 JAX array representing a small SE(3) increment in
        the IMU frame.
    """
    acc = jnp.asarray(imu.accel, dtype=jnp.float32)
    gyro = jnp.asarray(imu.gyro, dtype=jnp.float32)

    if gravity is not None:
        gravity = jnp.asarray(gravity, dtype=jnp.float32)
        acc = acc - gravity

    # Very crude single-step integration:
    #   v ≈ a * dt
    #   p ≈ 0.5 * a * dt^2
    #   θ ≈ ω * dt
    dp = 0.5 * acc * (dt ** 2)
    dtheta = gyro * dt

    return jnp.concatenate([dp, dtheta], axis=0)


# ---------------------------------------------------------------------------
# RANGE → 3D point
# ---------------------------------------------------------------------------

@dataclass
class RangeMeasurement:
    """
    Simple 1D range measurement along a known unit ray.

    :param distance: Measured distance along the ray, in meters.
    :param ray_dir: Unit direction vector in the sensor frame, shape (3,).
    """
    distance: float
    ray_dir: jnp.ndarray


def range_to_point_sensor(m: RangeMeasurement) -> jnp.ndarray:
    """
    Convert a range measurement into a 3D point in the sensor frame.

    :param m: Range measurement with a scalar distance and unit ray direction
        in the sensor frame.
    :returns: A 3D point (shape ``(3,)``) in the sensor frame.
    """
    d = float(m.distance)
    dir_s = jnp.asarray(m.ray_dir, dtype=jnp.float32)
    return d * dir_s


def range_to_point_world(
    m: RangeMeasurement,
    T_world_sensor: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convert a range measurement into a 3D point in world coordinates.

    This assumes you already have the sensor pose ``T_world_sensor`` as a
    homogeneous transform matrix of shape ``(4, 4)``. The point is first
    constructed in the sensor frame and then transformed into the world.

    :param m: Range measurement in the sensor frame.
    :param T_world_sensor: Homogeneous transform from sensor to world,
        shape ``(4, 4)``.
    :returns: A 3D point (shape ``(3,)``) in world coordinates.
    """
    p_s = range_to_point_sensor(m)
    p_s_h = jnp.concatenate([p_s, jnp.array([1.0], dtype=jnp.float32)], axis=0)
    p_w_h = T_world_sensor @ p_s_h
    return p_w_h[:3]


# ---------------------------------------------------------------------------
# LIDAR → point cloud
# ---------------------------------------------------------------------------

def lidar_scan_to_points_sensor(
    scan: LidarMeasurement,
) -> jnp.ndarray:
    """
    Convert a planar LiDAR scan into 3D points in the sensor frame.

    This function assumes a 2D planar LiDAR mounted in the ``x-y`` plane,
    with all points lying at ``z=0`` in the sensor frame::

        x = r * cos(theta)
        y = r * sin(theta)
        z = 0

    :param scan: LiDAR measurement containing per-beam ranges and angles.
    :returns: Array of points of shape ``(N, 3)`` in the sensor frame.
    """
    ranges = jnp.asarray(scan.ranges, dtype=jnp.float32)
    angles = jnp.asarray(scan.angles, dtype=jnp.float32)

    x = ranges * jnp.cos(angles)
    y = ranges * jnp.sin(angles)
    z = jnp.zeros_like(x)

    return jnp.stack([x, y, z], axis=-1)


def lidar_scan_to_points_world(
    scan: LidarMeasurement,
    T_world_sensor: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convert a planar LiDAR scan into 3D points in world coordinates.

    :param scan: LiDAR measurement in the sensor frame.
    :param T_world_sensor: Homogeneous transform from sensor to world,
        shape ``(4, 4)``.
    :returns: Array of points of shape ``(N, 3)`` in world coordinates.
    """
    pts_s = lidar_scan_to_points_sensor(scan)  # (N, 3)
    ones = jnp.ones((pts_s.shape[0], 1), dtype=jnp.float32)
    pts_s_h = jnp.concatenate([pts_s, ones], axis=1)  # (N, 4)
    pts_w_h = (T_world_sensor @ pts_s_h.T).T  # (N, 4)
    return pts_w_h[:, :3]


# ---------------------------------------------------------------------------
# CAMERA → rays / bearings
# ---------------------------------------------------------------------------

def pixel_to_ray_camera(
    cam: CameraMeasurement,
    u: float,
    v: float,
) -> jnp.ndarray:
    """
    Convert a pixel coordinate into a normalized ray in the camera frame.

    The intrinsics are assumed to follow the usual pinhole model::

        x = (u - cx) / fx
        y = (v - cy) / fy
        ray_cam = [x, y, 1]
        ray_cam /= ||ray_cam||

    :param cam: Camera measurement object providing intrinsics.
    :param u: Pixel x-coordinate (column index).
    :param v: Pixel y-coordinate (row index).
    :returns: A unit 3D vector (shape ``(3,)``) in the camera frame.
    """
    fx = float(cam.intrinsics.fx)
    fy = float(cam.intrinsics.fy)
    cx = float(cam.intrinsics.cx)
    cy = float(cam.intrinsics.cy)

    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = jnp.array([x, y, 1.0], dtype=jnp.float32)
    ray = ray / jnp.linalg.norm(ray)
    return ray


def pixels_to_rays_camera(
    cam: CameraMeasurement,
    pixels: Iterable[Tuple[float, float]],
) -> jnp.ndarray:
    """
    Convert a collection of pixel coordinates into rays in the camera frame.

    :param cam: Camera measurement object providing intrinsics.
    :param pixels: Iterable of ``(u, v)`` pixel coordinates.
    :returns: Array of unit 3D vectors of shape ``(N, 3)`` in the camera frame.
    """
    rays = [pixel_to_ray_camera(cam, u, v) for (u, v) in pixels]
    return jnp.stack(rays, axis=0)


def camera_depth_to_points_sensor(
    cam: CameraMeasurement,
    depth: jnp.ndarray,
) -> jnp.ndarray:
    """
    Convert a depth image into a 3D point cloud in the camera frame.

    The depth image is assumed to have the same width/height as specified
    by the camera intrinsics. For each pixel ``(u, v)`` with depth ``d``,
    we compute a ray via :func:`pixel_to_ray_camera` and place the point
    at ``d * ray``.

    :param cam: Camera measurement with intrinsics.
    :param depth: Depth map of shape ``(H, W)`` in meters.
    :returns: Array of points of shape ``(H*W, 3)`` in the camera frame.
        Invalid or zero depths are skipped.
    """
    H, W = depth.shape
    points = []

    for v in range(H):
        for u in range(W):
            d = float(depth[v, u])
            if d <= 0.0:
                continue
            ray = pixel_to_ray_camera(cam, float(u), float(v))
            p = d * ray
            points.append(p)

    if not points:
        return jnp.zeros((0, 3), dtype=jnp.float32)

    return jnp.stack(points, axis=0)

# Conversion helpers

def raw_sample_to_camera_measurement(
    sample: Mapping[str, Any],
    sensor_id: str = "cam0",
    T_cam_body: Optional[np.ndarray] = None,
    seq: Optional[int] = None,
) -> CameraMeasurement:
    """
    Convert a raw camera sample dictionary into a CameraMeasurement.

    Expected keys in ``sample``:

    - ``"t"`` (optional): float timestamp. If missing, the current time is used.
    - ``"frame_id"`` (optional): string frame identifier.
    - ONE of the following image-like entries (optional):
        * ``"image"``: (H, W) or (H, W, 3) array-like.
    - Optional directional data (for feature-based SLAM):
        * ``"bearings"``: (N, 3) array-like of unit directions.
        * ``"dirs"``: (N, 3) array-like.
        * ``"rays"``: (N, 3) array-like.

    Any directional data is stored in the returned measurement's ``metadata``
    under the corresponding key (e.g. ``metadata["bearings"]``).

    :param sample:
        Raw sample dictionary produced by a sensor stream (e.g. ReadingStream
        or FunctionStream) in the experiments.
    :param sensor_id:
        Identifier for this camera (e.g. ``"cam0"``).
    :param T_cam_body:
        Optional 4x4 homogeneous transform from body frame to camera frame.
    :param seq:
        Optional sequence index (frame counter).

    :returns:
        A :class:`CameraMeasurement` containing a :class:`CameraFrame` plus
        any directional data in the ``metadata`` field.
    """
    # Timestamp / frame id
    t = float(sample.get("t", time.time()))
    frame_id = sample.get("frame_id", None)

    # Image (optional)
    if "image" in sample:
        img = np.asarray(sample["image"])
    else:
        # If no image is present, create a dummy 1x1 grayscale image.
        # This lets us still pass a valid CameraFrame downstream.
        img = np.zeros((1, 1), dtype=np.float32)

    # Decide color space from shape
    if img.ndim == 3 and img.shape[-1] == 3:
        color_space = "rgb"
    else:
        color_space = "gray"

    frame = CameraFrame(
        image=img,
        timestamp=t,
        frame_id=frame_id,
        color_space=color_space,
    )

    # Collect directional data (bearings, dirs, rays) into metadata
    metadata: Dict[str, Any] = {}

    for key in ("bearings", "dirs", "rays"):
        if key in sample:
            metadata[key] = np.asarray(sample[key])

    # Optionally keep any other extra keys under metadata["extra"]
    for k, v in sample.items():
        if k not in ("t", "frame_id", "image", "bearings", "dirs", "rays"):
            metadata.setdefault("extra", {})[k] = v

    if not metadata:
        metadata = None  # keep clean if there is nothing to store

    return CameraMeasurement(
        frame=frame,
        sensor_id=sensor_id,
        T_cam_body=T_cam_body,
        seq=seq,
        metadata=metadata,
    )

def raw_sample_to_lidar_measurement(sample: Mapping[str, Any]) -> LidarMeasurement:
    """Convert a raw LiDAR sample dictionary into a :class:`LidarMeasurement`.

    Expected keys in ``sample``:

    - ``"ranges"``: 1D array-like of LiDAR ranges. (required)
    - ``"angles"``: 1D array-like of bearing angles (radians), same length as ``ranges``. (optional)
    - ``"directions"``: (N, 3) array-like of direction vectors. (optional)
    - ``"t"`` (optional): float timestamp.
    - ``"frame_id"`` (optional): string frame identifier.

    This is consistent with the rest of the LiDAR utilities in this module,
    which assume a planar LiDAR described by ``ranges`` and ``angles``.

    :param sample: Raw sample dictionary produced by a sensor stream.
    :returns: A :class:`LidarMeasurement` instance.
    :raises KeyError: If required keys are missing from the sample.
    """
    if "ranges" not in sample:
        raise KeyError("raw_sample_to_lidar_measurement expected 'ranges' in sample")

    ranges = jnp.array(sample["ranges"], dtype=jnp.float32)

    # Determine directions (dirs) if present, else compute from angles if present, else None
    dirs = None
    if "directions" in sample:
        dirs = jnp.array(sample["directions"], dtype=jnp.float32)
    elif "angles" in sample:
        angles = jnp.array(sample["angles"], dtype=jnp.float32)
        # Planar lidar: directions in the x-y plane
        dirs = jnp.stack(
            [jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], axis=-1
        )
    else:
        dirs = None

    return LidarMeasurement(
        ranges=ranges,
        directions=dirs,
        t=sample.get("t", None),
        frame_id=sample.get("frame_id", "lidar"),
        metadata=None,
    )

def raw_sample_to_imu_measurement(sample: Mapping[str, Any]) -> IMUMeasurement:
    """
    Convert a raw dict from a stream into an IMUMeasurement.

    Expected keys in ``sample``:

    - "t": float timestamp (seconds).
    - "accel": array-like linear acceleration in the IMU frame.
    - "gyro": array-like angular velocity in the IMU frame.
    - "dt" (optional): time step in seconds since the previous sample.
      If omitted, a default value is used.

    :param sample: Raw sample dictionary produced by a sensor stream.
    :return: An :class:`IMUMeasurement` instance.
    """
    t = float(sample.get("t", 0.0))
    accel = jnp.array(sample["accel"], dtype=jnp.float32)
    gyro = jnp.array(sample["gyro"], dtype=jnp.float32)

    # Use provided dt if available, otherwise fall back to a reasonable default
    dt = float(sample.get("dt", 0.1))

    return IMUMeasurement(
        timestamp=t,
        accel=accel,
        gyro=gyro,
        dt=dt,
    )