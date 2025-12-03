# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
LiDAR sensor abstractions and utilities for DSG-JIT.

This module defines a lightweight representation of LiDAR data—either
1D range scans, 2D planar scans, or sparse 3D point samples—and provides
a simple, JAX-friendly interface for integrating such scans into the
factor graph or dynamic scene graph layers.

The module includes:

  * **LiDARScan**
      A minimal container representing a single LiDAR measurement:
        - Ranges (1D, 2D, or 3D depending on sensor type)
        - Optional beam angles or directions
        - Optional timestamp
      The structure is intentionally simple so that downstream algorithms
      (e.g., geometry-based localization, place association, or occupancy
      grid inference) can build on top of it without being locked into a
      particular LiDAR model.

  * **LiDARSensor**
      A wrapper around any user-provided LiDAR capture function. This allows
      plugging in real hardware, ROS topics, simulation engines, or synthetic
      data generators. The wrapper returns ``LiDARScan`` objects and is fully
      compatible with both synchronous and asynchronous streaming utilities.

  * **Helper transforms**
      Utilities for:
        - Converting range/angle scans into 2D or 3D point clouds.
        - Projecting LiDAR points into world coordinates given a robot pose.
        - Preparing LiDAR-derived factors for integration into the factor
          graph (e.g., bearing-range residuals, scan-matching constraints).

Design philosophy:

  * Keep the LiDAR model minimal and general.
  * Allow users to choose how scans translate into DSG elements (places,
    objects, layout nodes, occupancy voxels).
  * Cleanly interoperate with the broader DSG-JIT world model, enabling:
        - scan→point cloud→place association
        - scan→object detection integration
        - scan→unknown-space discovery
        - scan→range constraints between robot poses and map nodes

This module intentionally avoids tying LiDAR data to a specific SLAM method
(e.g., ICP, NDT, LOAM); instead, it provides a consistent base layer for
future plugins and extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp

from dsg_jit.core.types import Factor
from dsg_jit.slam.measurements import sigma_to_weight
from dsg_jit.world.model import WorldModel
from dsg_jit.world.dynamic_scene_graph import DynamicSceneGraph
from dsg_jit.world.scene_graph import SceneGraphWorld
from dsg_jit.sensors.base import Sensor, SensorReading

@dataclass
class LidarMeasurement:
    """
    Lightweight container for a single LiDAR scan.

    This structure is deliberately minimal and JAX-friendly. It can be used
    for 1D range scans, 2D planar scans, or sparse 3D point samples, and
    is meant to serve as an intermediate representation between raw sensor
    data and DSG-JIT factors (e.g., range/bearing residuals, voxel updates).

    :param ranges:
        LiDAR ranges in sensor coordinates.

        Typical shapes:

        * 1D scan: ``(N,)`` where each entry is a range sample.
        * 2D grid: ``(H, W)`` for image-like range sensors.
        * Sparse 3D: ``(N,)`` used together with ``directions`` below.

        Values are typically in meters.
    :type ranges: jax.numpy.ndarray

    :param directions:
        Optional unit vectors giving the direction of each range sample in
        the sensor frame.

        Typical shapes:

        * 1D scan: ``(N, 3)`` where each row is a 3D direction vector.
        * 2D grid: ``(H, W, 3)`` matching the shape of ``ranges``.

        If ``None``, downstream code is expected to infer directions based
        on sensor intrinsics (e.g., azimuth/elevation tables or a pinhole
        camera model).
    :type directions: Optional[jax.numpy.ndarray]

    :param t:
        Optional timestamp (e.g., in seconds). This can be used to align the
        measurement with the dynamic scene graph time index or other sensors.
    :type t: Optional[float]

    :param frame_id:
        Identifier for the sensor frame from which this measurement was
        taken (e.g. ``"lidar_front"``). This is useful when a robot carries
        multiple LiDAR units or when extrinsic calibration is maintained
        per frame.
    :type frame_id: str

    :param metadata:
        Optional dictionary for any additional information (e.g., intensity
        values, per-beam noise estimates, or scan ID). This field is not
        used by the core library but can be useful for higher-level
        perception algorithms.
    :type metadata: Optional[dict]
    """

    ranges: jnp.ndarray
    directions: Optional[jnp.ndarray] = None
    t: Optional[float] = None
    frame_id: str = "lidar"
    metadata: Optional[dict] = None
    
@dataclass
class RangeSensorConfig:
    """
    Configuration for a simple range-only sensor.

    :param noise_sigma: Standard deviation of the range measurement, in
        the same units as the world coordinates (typically meters).
    :type noise_sigma: float
    :param target_node: Node id of the 3D target in the
        :class:`SceneGraphWorld` (e.g. a room or object center).
    :type target_node: int
    """
    noise_sigma: float
    target_node: int


class RangeSensor(Sensor):
    """
    Simple range-only sensor attached to an agent.

    Given a :class:`SensorReading` whose ``data`` field is a scalar
    range, this sensor produces a single ``"range"`` factor connecting
    the agent's pose at time ``t`` to a 3D target node.

    It expects:

    * the agent trajectory to be registered in the
      :class:`DynamicSceneGraph` under the same ``agent_id`` used to
      construct this sensor, and
    * the target node id to be present in the underlying
      :class:`SceneGraphWorld`.

    The factor uses :func:`slam.measurements.range_residual` and is
    registered under the factor type ``"range"``.
    """

    def __init__(
        self,
        name: str,
        agent_id: str,
        config: RangeSensorConfig,
        sg: SceneGraphWorld,
    ) -> None:
        super().__init__(name=name, agent_id=agent_id)
        self.config = config
        self.sg = sg  # used to look up the target's variable index

    def build_factors(
        self,
        wm: WorldModel,
        dsg: DynamicSceneGraph,
        reading: SensorReading,
    ) -> List[Factor]:
        """
        Build a single range factor from a reading.

        :param wm: World model whose factor graph already contains the
            agent poses and the target node declared in the config.
        :type wm: world.model.WorldModel
        :param dsg: Dynamic scene graph providing the pose node id for
            ``(agent_id, reading.t)``.
        :type dsg: world.dynamic_scene_graph.DynamicSceneGraph
        :param reading: Range measurement; ``reading.data`` is assumed
            to be a scalar float.
        :type reading: sensors.base.SensorReading
        :return: List containing a single ``"range"`` factor.
        :rtype: list[core.types.Factor]
        """
        t = reading.t
        range_meas = float(reading.data)

        # Resolve pose node id from the DSG trajectory.
        pose_nid = dsg.world.pose_trajectory[(self.agent_id, t)]

        # Variable ids: [pose, target]
        var_ids = (pose_nid, self.config.target_node)

        # Params for range_residual.
        params: Dict[str, jnp.ndarray] = {
            "range": jnp.array(range_meas, dtype=jnp.float32),
            "weight": sigma_to_weight(self.config.noise_sigma),
        }

        factor = Factor(
            id=len(wm.fg.factors),
            type="range",
            var_ids=var_ids,
            params=params,
        )
        return [factor]