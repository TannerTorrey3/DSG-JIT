# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

"""Sensor fusion utilities for DSG-JIT.

This module provides a small, opinionated *fusion core* that sits between
raw sensor streams (e.g. LiDAR, cameras, IMUs) and the rest of the
DSG-JIT world/scene-graph stack.

The goals of this layer are:

- Provide a common abstraction to register named sensors and pull data
  from synchronous streams.
- Convert raw samples into strongly-typed measurement objects defined in
  :mod:`sensors.camera`, :mod:`sensors.lidar`, and :mod:`sensors.imu`.
- Dispatch the resulting measurements to user-provided callbacks that
  can update a :class:`world.model.WorldModel` or
  :class:`world.dynamic_scene_graph.DynamicSceneGraph`.

This is intentionally lightweight:

- It does **not** assume any particular factor-graph structure.
- It does **not** run its own background threads or event loops.
- It is designed so that experiments can start simple (single-threaded
  polling) while leaving room to grow into a more complex async or
  multi-rate fusion system later on.

Typical usage from an experiment::

    from sensors.streams import ReadingStream
    from sensors.camera import CameraMeasurement
    from sensors.lidar import LidarMeasurement
    from sensors.fusion import SensorFusionManager

    fusion = SensorFusionManager()

    # 1) Register a camera and a lidar stream
    fusion.register_sensor(
        name="cam0",
        modality="camera",
        stream=ReadingStream(camera_read_fn),
    )
    fusion.register_sensor(
        name="lidar0",
        modality="lidar",
        stream=ReadingStream(lidar_read_fn),
    )

    # 2) Connect fusion to the world model via a callback
    def on_measurement(meas):
        if isinstance(meas, CameraMeasurement):
            # add a bearing / reprojection factor, etc.
            ...
        elif isinstance(meas, LidarMeasurement):
            # add range factors or occupancy updates
            ...

    fusion.register_callback(on_measurement)

    # 3) Poll sensors in your main loop
    for step in range(100):
        fusion.poll_once()
        # run optimization, render, etc.

In the future we can add:

- Async helpers that drive :class:`sensors.streams.AReadingStream`.
- Tighter integration with the dynamic scene graph (e.g. per-agent queues).
- Higher-level policies (e.g. downsampling, time sync, etc.).
"""

from __future__ import annotations

import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
from typing import TYPE_CHECKING

from dsg_jit.sensors.camera import CameraMeasurement
from dsg_jit.sensors.lidar import LidarMeasurement
from dsg_jit.sensors.imu import IMUMeasurement

if TYPE_CHECKING:
    from dsg_jit.world.model import WorldModel

from dsg_jit.sensors.streams import BaseSensorStream, ReadingStream
from dsg_jit.sensors.base import BaseMeasurement

from dsg_jit.sensors.conversion import (
    raw_sample_to_camera_measurement,
    raw_sample_to_lidar_measurement,
    imu_to_factors_placeholder,
)

@dataclass
class FusedPoseEstimate:
    """
    Fused SE(3) pose estimate produced by SensorFusionManager.

    :param t: Time index (discrete step or timestamp) associated with this
        estimate.
    :param pose_se3: 6D se(3) pose vector in world coordinates, typically
        ``(tx, ty, tz, rx, ry, rz)``.
    :param covariance: Optional 6x6 covariance matrix for the fused pose.
    :param source_counts: Optional dictionary tracking how many measurements
        contributed from each sensor type (e.g. ``{"imu": 10, "lidar": 2}``).
    """

    t: float | int
    pose_se3: jnp.ndarray
    covariance: Optional[jnp.ndarray] = None
    source_counts: Dict[str, int] = field(default_factory=dict)

@dataclass
class RegisteredSensor:
    """Bookkeeping structure for a registered sensor.

    :param name: Logical name of the sensor (e.g. ``"lidar0"``).
    :param modality: String describing the modality, e.g. ``"camera"``,
        ``"lidar"``, or ``"imu"``. This is used to choose a default
        converter when one is not provided.
    :param stream: Underlying sensor stream used to pull raw samples.
    :param converter: Function that maps a raw sample from ``stream`` to
        a :class:`~sensors.base.BaseMeasurement` instance.
    """

    name: str
    modality: str
    stream: BaseSensorStream
    converter: Callable[[Any], BaseMeasurement]


class SensorFusionManager:
    """Central registry and dispatcher for sensor measurements.

    The fusion manager is intentionally minimal: it owns no threads and does
    not know about factor graphs or optimization. It simply:

    - Keeps track of registered sensors.
    - Polls synchronous streams on demand.
    - Converts raw samples to measurement objects.
    - Broadcasts those measurements to user callbacks.

    Experiments are free to use this in a tight, single-threaded loop, or
    to build more advanced async / multi-rate infrastructure on top.

    :param default_callbacks: Optional iterable of callbacks to register
        at construction time. Each callback will be called as
        ``callback(measurement)`` whenever a new
        :class:`~sensors.base.BaseMeasurement` is produced.
    """

    def __init__(
        self,
        default_callbacks: Optional[Iterable[Callable[[BaseMeasurement], None]]] = None,
        world_model: Optional["WorldModel"] = None,
        auto_register_world_callbacks: bool = True,
    ) -> None:
        """Create a new fusion manager.

        :param default_callbacks: Optional iterable of callbacks to register
            immediately. Each callback will be invoked as ``cb(measurement)``
            for every measurement produced by :meth:`poll_once` or
            :meth:`push_measurement`.
        :param world_model: Optional :class:`world.model.WorldModel` instance
            to be associated with this fusion manager. If provided and
            ``auto_register_world_callbacks`` is ``True``, the manager will
            automatically register per-modality callbacks that forward
            measurements into the world model.
        :param auto_register_world_callbacks: If ``True`` and ``world_model``
            is not ``None``, register default world-model callbacks for
            camera, LiDAR, and IMU measurements (when the corresponding
            handler methods exist on the world model).
        """

        self._sensors: Dict[str, RegisteredSensor] = {}
        self._callbacks: List[Callable[[BaseMeasurement], None]] = []
        self._fused_history: List[
            tuple[float | int, jnp.ndarray, Optional[jnp.ndarray], Dict[str, int]]
        ] = []
        self._world_model: Optional["WorldModel"] = world_model

        if default_callbacks is not None:
            for cb in default_callbacks:
                self.register_callback(cb)

        if self._world_model is not None and auto_register_world_callbacks:
            # Automatically hook camera / LiDAR / IMU measurements into the
            # world model using any available handler methods.
            self.attach_world_model(self._world_model, register_default_callbacks=True)
    def record_fused_pose(
        self,
        t: float | int,
        pose_se3: jnp.ndarray,
        covariance: Optional[jnp.ndarray] = None,
        source_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Append a fused SE(3) pose estimate to the internal history buffer.

        This does not perform any fusion by itself; instead it allows an
        external estimator (e.g. an EKF or factor-graph solver) to publish
        its current best pose into the fusion manager so that callers can
        retrieve it via :meth:`get_latest_pose`.

        :param t: Time index or timestamp associated with the estimate.
        :param pose_se3: 6D se(3) pose vector in world coordinates.
        :param covariance: Optional 6x6 covariance matrix for the estimate.
        :param source_counts: Optional dictionary describing how many
            measurements from each sensor type contributed to this pose.
        """
        if source_counts is None:
            source_counts = {}
        self._fused_history.append((t, pose_se3, covariance, source_counts))

    # ------------------------------------------------------------------
    # World model integration hooks
    # ------------------------------------------------------------------

    def attach_world_model(
        self,
        world_model: "WorldModel",
        register_default_callbacks: bool = True,
    ) -> None:
        """Attach a :class:`world.model.WorldModel` to this fusion manager.

        When a world model is attached and ``register_default_callbacks`` is
        ``True``, the manager will install per-modality callbacks that route
        camera, LiDAR, and IMU measurements into the world model using any
        available handler methods.

        Expected handler names on ``world_model`` are, for example:

        * ``apply_camera_measurement`` or ``add_camera_measurement``
        * ``apply_lidar_measurement`` or ``add_lidar_measurement``
        * ``apply_imu_measurement`` or ``add_imu_measurement``

        Only handlers that actually exist and are callable will be used.

        :param world_model: World model instance to associate.
        :param register_default_callbacks: If ``True``, automatically
            register default callbacks that forward measurements into the
            world model.
        """

        self._world_model = world_model
        if register_default_callbacks:
            self._register_world_callbacks()

    @staticmethod
    def _resolve_world_handler(world_model: Any, names: List[str]) -> Optional[Callable[..., Any]]:
        """Return the first callable attribute on ``world_model`` matching ``names``.

        :param world_model: World model object to inspect.
        :param names: Ordered list of attribute names to try.
        :return: The first callable attribute found, or ``None`` if no
            suitable handler exists.
        """

        for name in names:
            if hasattr(world_model, name):
                fn = getattr(world_model, name)
                if callable(fn):
                    return fn
        return None

    def _register_world_callbacks(self) -> None:
        """Register per-modality callbacks that forward into the world model.

        This inspects the attached world model (if any) for known handler
        methods and wraps them in measurement-type checks. It is safe to
        call repeatedly; new callbacks are simply appended to the internal
        callback list.
        """

        wm = self._world_model
        if wm is None:
            return

        # Camera measurements
        cam_handler = self._resolve_world_handler(
            wm,
            ["apply_camera_measurement", "add_camera_measurement"],
        )
        if cam_handler is not None:

            def _cb_camera(meas: BaseMeasurement) -> None:
                if isinstance(meas, CameraMeasurement):
                    cam_handler(meas)

            self.register_callback(_cb_camera)

        # LiDAR measurements
        lidar_handler = self._resolve_world_handler(
            wm,
            ["apply_lidar_measurement", "add_lidar_measurement"],
        )
        if lidar_handler is not None:

            def _cb_lidar(meas: BaseMeasurement) -> None:
                if isinstance(meas, LidarMeasurement):
                    lidar_handler(meas)

            self.register_callback(_cb_lidar)

        # IMU measurements
        imu_handler = self._resolve_world_handler(
            wm,
            ["apply_imu_measurement", "add_imu_measurement"],
        )
        if imu_handler is not None:

            def _cb_imu(meas: BaseMeasurement) -> None:
                if isinstance(meas, IMUMeasurement):
                    imu_handler(meas)

            self.register_callback(_cb_imu)

    # ------------------------------------------------------------------
    # Registration and configuration
    # ------------------------------------------------------------------

    def register_sensor(
        self,
        name: str,
        modality: str,
        stream: BaseSensorStream,
        converter: Optional[Callable[[Any], BaseMeasurement]] = None,
    ) -> None:
        """Register a new sensor with the fusion manager.

        If ``converter`` is not provided, a default converter will be
        chosen based on ``modality``.

        :param name: Logical sensor name, e.g. ``"cam0"`` or ``"lidar_front"``.
        :param modality: Modality string (``"camera"``, ``"lidar"``,
            ``"imu"``, or a custom value). Custom values must provide an
            explicit ``converter``.
        :param stream: Sensor stream object used to read raw samples.
        :param converter: Optional function that converts raw samples from
            ``stream`` into measurement objects.

        :raises ValueError: If a sensor with the same name is already
            registered, or if no default converter exists for the given
            modality and no explicit converter is provided.
        """

        if name in self._sensors:
            raise ValueError(f"Sensor '{name}' is already registered")

        if converter is None:
            converter = self._infer_default_converter(modality)

        self._sensors[name] = RegisteredSensor(
            name=name,
            modality=modality,
            stream=stream,
            converter=converter,
        )

    def _infer_default_converter(self, modality: str) -> Callable[[Any], BaseMeasurement]:
        """Return a default converter for a given modality.

        :param modality: Modality string such as ``"camera"``, ``"lidar"``,
            or ``"imu"``.
        :return: Callable that converts raw samples for the given modality
            into measurement objects.

        :raises ValueError: If a default converter is not known for the
            given modality.
        """

        m = modality.lower()
        if m == "camera":
            return raw_sample_to_camera_measurement
        if m == "lidar":
            return raw_sample_to_lidar_measurement
        if m == "imu":
            return imu_to_factors_placeholder

        raise ValueError(
            "No default converter for modality '" + modality + "'. "
            "Please provide an explicit converter when registering this sensor."
        )

    def register_callback(self, callback: Callable[[BaseMeasurement], None]) -> None:
        """Register a callback to receive all fused measurements.

        :param callback: Function that accepts a
            :class:`~sensors.base.BaseMeasurement` instance. It will be
            called for *every* measurement produced by
            :meth:`poll_once` or :meth:`push_measurement`.
        """

        self._callbacks.append(callback)

    # ------------------------------------------------------------------
    # Core polling / dispatch API
    # ------------------------------------------------------------------

    def poll_once(self) -> int:
        """Poll all registered *synchronous* streams exactly once.

        For each sensor whose stream is an instance of
        :class:`~sensors.streams.ReadingStream`, this method will:

        1. Call ``stream.read()`` to obtain a raw sample.
        2. If a non-``None`` sample is returned, convert it to a
           measurement via the registered converter.
        3. Dispatch the measurement to all registered callbacks.

        Asynchronous streams are not handled here; they can be consumed
        separately using their own ``async for`` loops.

        :return: The total number of measurements produced and
            dispatched during this call.
        """

        count = 0
        for rs in self._sensors.values():
            if not isinstance(rs.stream, ReadingStream):
                # Async streams are handled outside of this helper.
                continue

            sample = rs.stream.read()
            if sample is None:
                continue

            meas = rs.converter(sample)
            self._dispatch(meas)
            count += 1

        return count

    def push_measurement(self, measurement: BaseMeasurement) -> None:
        """Inject a pre-constructed measurement into the fusion pipeline.

        This is useful when the experiment already builds measurement
        objects directly (e.g. from a simulator) and only wants to reuse
        the callback dispatch mechanism.

        :param measurement: Measurement instance to dispatch to all
            registered callbacks.
        """

        self._dispatch(measurement)

    def get_latest_pose(self) -> Optional[FusedPoseEstimate]:
        """
        Return the most recent fused SE(3) pose estimate, if available.

        This is a thin convenience wrapper used by integration layers
        (e.g. world models or dynamic scene graphs) to pull a single,
        canonical pose update out of the fusion buffer.

        :returns: A :class:`FusedPoseEstimate` instance if any fused result
            has been produced, or ``None`` if the manager has not yet
            emitted a pose.
        """
        if not self._fused_history:
            return None

        # Assuming you internally store a list of (t, pose, cov, meta).
        t, pose_se3, cov, source_counts = self._fused_history[-1]

        return FusedPoseEstimate(
            t=t,
            pose_se3=pose_se3,
            covariance=cov,
            source_counts=source_counts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(self, measurement: BaseMeasurement) -> None:
        """Call all registered callbacks with ``measurement``.

        :param measurement: Measurement instance to forward to callbacks.
        """

        for cb in self._callbacks:
            cb(measurement)


