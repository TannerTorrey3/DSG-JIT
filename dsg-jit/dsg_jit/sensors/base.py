# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Abstract base classes and shared interfaces for DSG-JIT sensor modules.

This module defines the foundational API that all sensor types in DSG-JIT
(IMU, LiDAR, RGB cameras, range sensors, etc.) are expected to implement.
By unifying the contract between sensor objects and the rest of the system,
DSG-JIT enables plug-and-play multi-sensor integration without requiring
special-case logic for each modality.

The goal of this module is to answer a simple question:

    “What does it mean to be a sensor in DSG-JIT?”

-------------------------------------------------------------------------------
Core Components
-------------------------------------------------------------------------------

**BaseSensor**
    The abstract parent class for all sensors. It typically defines:

      - ``initialize()``  — optional setup before streaming begins  
      - ``read()``        — return a *single typed measurement*  
      - ``close()``       — release hardware or simulation resources  

    Subclasses (e.g., ``IMUSensor``, ``LiDARSensor``, ``CameraSensor``)
    implement their own measurement-specific logic, but all present
    a consistent surface to the rest of DSG-JIT.

**BaseMeasurement**
    A typed container for the output of a sensor.  
    Every measurement has:

      - ``timestamp``  
      - device-specific payload (e.g., accelerations, images, point clouds)  

    The purpose of this common type is to make downstream modules—
    factor graphs, dynamic scene graphs, training loops, logging tools—
    treat all measurements uniformly.

-------------------------------------------------------------------------------
Design Goals
-------------------------------------------------------------------------------

1. **Unified sensor API**  
   All sensors behave the same from the perspective of DSG-JIT’s
   world model, optimization routines, and streaming helpers.

2. **Compatibility with synchronous & asynchronous streams**  
   The interfaces defined here are intentionally minimal so that
   ``streams.py`` can wrap *any* sensor using either Python loops or
   asyncio-based background tasks.

3. **Future-proof extensibility**  
   The goal is for users to implement custom sensors (GPS, UWB, RADAR,
   event cameras, tactile sensors, etc.) by subclassing
   ``BaseSensor`` and returning custom ``BaseMeasurement`` types.

4. **Separation of concerns**  
   This module defines *contracts*, not implementation logic.
   Sensor-specific math, calibration, conversion, or projection
   live in their respective modules (e.g., ``lidar.py``, ``camera.py``).

-------------------------------------------------------------------------------
Summary
-------------------------------------------------------------------------------

The ``base`` module provides the essential abstraction layer required for
building robust, modular, multi-sensor dynamic scene graphs. It ensures that
all data entering DSG-JIT—regardless of modality—flows through a consistent,
well-structured interface suitable for high-frequency SLAM, perception,
and future real-time scene generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from dsg_jit.core.types import Factor
from dsg_jit.world.model import WorldModel
from dsg_jit.world.dynamic_scene_graph import DynamicSceneGraph

@dataclass
class BaseMeasurement:
    """
    Generic typed measurement used by all DSG-JIT sensor backends.

    This class represents a single sensor sample emitted by a device or
    by a stream wrapper. All concrete sensor measurement types
    (e.g., :class:`CameraMeasurement`, :class:`LidarMeasurement`,
    :class:`IMUMeasurement`) should inherit from this class.

    :param t: Discrete timestamp or frame index associated with the sample.
    :type t: int
    :param source: Name of the sensor that produced this measurement
        (e.g., ``"front_cam"``, ``"lidar_0"``, ``"imu_main"``).
    :type source: str
    :param data: Raw modality-specific payload. Subclasses may refine this type.
        For example, a camera may store an ndarray, LiDAR may store a point cloud,
        and IMU may store (accel, gyro) tuples.
    :type data: Any
    :param meta: Optional metadata, such as exposure time, pose hints, or flags.
    :type meta: dict or None
    """

    t: int
    source: str
    data: Any
    meta: Optional[dict] = None

@dataclass
class SensorReading:
    """
    Lightweight container for a single sensor measurement.

    :param t: Discrete time index or timestamp at which the reading was
        taken. For now this is typically an integer matching the DSG's
        time index.
    :type t: int
    :param data: Raw or minimally processed sensor payload. The exact
        structure depends on the sensor type (e.g. a scalar range,
        a 3D point, an image array, etc.).
    :type data: Any
    """
    t: int
    data: Any


class Sensor:
    """
    Abstract base class for all DSG-JIT sensors.

    Concrete subclasses implement :meth:`build_factors` to turn a
    :class:`SensorReading` into one or more :class:`core.types.Factor`
    instances, which can then be added to a :class:`world.model.WorldModel`.

    Sensors are intended to be *stateless* with respect to optimization:
    they describe how to map readings into factors, but do not own any
    variables themselves.

    Typical usage pattern::

        sensor = SomeSensor(agent_id="robot0", ...)
        reading = SensorReading(t=3, data=raw_measurement)
        factors = sensor.build_factors(world_model, dsg, reading)
        for f in factors:
            world_model.fg.add_factor(f)

    :param name: Human-readable name for this sensor instance.
    :type name: str
    :param agent_id: Identifier of the agent this sensor is mounted on,
        e.g. ``"robot0"``. Used to resolve pose nodes in the DSG.
    :type agent_id: str
    """

    def __init__(self, name: str, agent_id: str) -> None:
        self.name = name
        self.agent_id = agent_id

    # You can keep this as a simple duck-typed interface instead of
    # using abc.ABC to avoid import overheads.
    def build_factors(
        self,
        wm: WorldModel,
        dsg: DynamicSceneGraph,
        reading: SensorReading,
    ) -> List[Factor]:
        """
        Convert a sensor reading into factor(s) to be added to the world.

        Subclasses must implement this to return a list of
        :class:`core.types.Factor` objects whose ``var_ids`` and
        ``params`` are consistent with the residuals registered in the
        world's factor graph.

        :param wm: World model into which new factors will be added.
        :type wm: world.model.WorldModel
        :param dsg: Dynamic scene graph providing access to pose node ids
            for this sensor's agent at the requested time index.
        :type dsg: world.dynamic_scene_graph.DynamicSceneGraph
        :param reading: Sensor reading to convert.
        :type reading: SensorReading
        :return: List of factor objects ready to be added to
            :attr:`wm.fg`.
        :rtype: list[core.types.Factor]
        """
        raise NotImplementedError