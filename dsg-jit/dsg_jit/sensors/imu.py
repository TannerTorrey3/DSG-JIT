# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Inertial measurement unit (IMU) abstractions and simple integration utilities.

This module provides a small set of IMU-related types and helpers that make
it easy to wire IMU data into DSG-JIT experiments and factor graphs. The
focus is on **structured measurements** and **simple integration**, rather
than on a full production-grade inertial navigation system.

The module typically exposes:

  * ``IMUMeasurement``:
      A lightweight container for a single IMU sample, including body-frame
      linear acceleration, angular velocity, and an associated ``dt`` (time
      delta). A timestamp can also be stored for logging or alignment with
      other sensors.

  * ``IMUSensor``:
      A thin wrapper around a user-defined sampling function. The sampling
      function returns ``IMUMeasurement`` instances, and the wrapper
      provides:
        - A ``read()`` method for synchronous polling.
        - An iterator interface for use in simple loops.
        - Compatibility with the generic sensor streaming helpers.

  * ``integrate_imu_naive``:
      A toy integrator that demonstrates how to accumulate IMU measurements
      into a position and velocity estimate. It assumes that:
        - Acceleration is already expressed in the world frame, and
        - Gravity has been compensated externally.
      This function is intended for didactic experiments and not as a
      replacement for a full IMU preintegration pipeline.

Usage patterns:

  * Wrap any IMU source (hardware driver, simulator, dataset) with an
    ``IMUSensor`` so that you always work with ``IMUMeasurement`` objects.
  * For demonstration or unit tests, use ``integrate_imu_naive`` to produce
    rough pose/velocity estimates and then inject those as odometry factors
    into a factor graph.
  * For more advanced use cases, replace the naive integrator with a
    preintegration module, but keep the same ``IMUMeasurement`` and
    ``IMUSensor`` interfaces so the rest of the system remains unchanged.

The design goal is to make IMU handling:

  * Explicit and transparent (no hidden global state).
  * Composable with other sensors (cameras, range sensors, etc.).
  * Easy to plug into the existing DSG-JIT world model and optimization
    stack without requiring changes to core solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import jax.numpy as jnp
import numpy as np

@dataclass
class IMUMeasurement:
    """Single IMU sample consisting of specific force and angular velocity.

    This is a lightweight container for raw IMU readings that can be produced by
    hardware drivers, simulators, or log readers and then consumed by the
    factor-graph/DSG layers.

    The convention assumed here is:

    * ``accel`` is specific force in the IMU body frame, in ``m/s^2``.
    * ``gyro`` is angular velocity in the IMU body frame, in ``rad/s``.
    * ``dt`` is the time delta since the previous sample, in seconds.
    * ``timestamp`` is an optional absolute time in seconds.

    :param accel: Linear acceleration in the IMU/body frame, shape ``(3,)``.
    :param gyro: Angular velocity in the IMU/body frame, shape ``(3,)``.
    :param dt: Time delta since the previous sample, in seconds.
    :param timestamp: Optional absolute timestamp in seconds.
    """

    accel: jnp.ndarray
    gyro: jnp.ndarray
    dt: float
    timestamp: Optional[float] = None

    def as_numpy(self) -> tuple["np.ndarray", "np.ndarray"]:  # type: ignore[name-defined]
        """Return the acceleration and gyro as NumPy arrays.

        This is a small convenience helper for users who want to interoperate
        with NumPy-based tooling. If NumPy is not available, a :class:`RuntimeError`
        is raised.

        :return: Tuple ``(accel_np, gyro_np)`` as NumPy arrays.
        :raises RuntimeError: If NumPy is not installed or import failed.
        """

        if np is None:  # type: ignore[truthy-function]
            raise RuntimeError("NumPy is not available in this environment.")
        return np.asarray(self.accel), np.asarray(self.gyro)


class IMUSampleFn(Protocol):
    """Protocol for callables that produce :class:`IMUMeasurement` samples.

    This is primarily used to type-annotate IMU sensor wrappers.

    :return: A new :class:`IMUMeasurement` instance.
    """

    def __call__(self) -> IMUMeasurement:
        ...


# ``BaseSensor`` is optional here to keep this module usable in isolation.
try:  # pragma: no cover - import is environment dependent
    from .base import BaseSensor  # type: ignore
except Exception:  # pragma: no cover
    BaseSensor = object  # type: ignore[misc,assignment]


class IMUSensor(BaseSensor):  # type: ignore[misc]
    """Generic IMU sensor wrapper.

    This class adapts any callable that returns :class:`IMUMeasurement` into the
    common sensor interface used by DSG-JIT. It is intentionally minimal: it does
    not attempt to manage threads, buffering, or synchronization—those concerns
    are handled by the sensor stream utilities in :mod:`sensors.streams`.

    :param name: Human-readable sensor name.
    :param sample_fn: Callable producing a single :class:`IMUMeasurement` per call.
    """

    def __init__(self, name: str, sample_fn: IMUSampleFn) -> None:
        # ``BaseSensor`` may or may not define an ``__init__``; call it if present.
        if hasattr(super(), "__init__"):
            try:
                super().__init__(name=name)  # type: ignore[call-arg]
            except TypeError:
                # Fallback if BaseSensor has a different signature.
                super().__init__()  # type: ignore[misc]

        self.name: str = name
        self._sample_fn: IMUSampleFn = sample_fn

    def read(self) -> IMUMeasurement:
        """Read a single IMU measurement.

        This will typically perform a blocking read from a hardware device, a
        simulator, or a log file, depending on how ``sample_fn`` is implemented.

        :return: The next :class:`IMUMeasurement` sample.
        """

        return self._sample_fn()

    def __iter__(self) -> Iterable[IMUMeasurement]:
        """Create an iterator over IMU samples.

        This is primarily useful when coupling the sensor with an asynchronous
        or synchronous sensor stream helper that consumes an iterator.

        :return: Infinite iterator yielding :class:`IMUMeasurement` objects.
        """

        while True:
            yield self.read()


def integrate_imu_naive(
    measurements: Iterable[IMUMeasurement],
    v0: Optional[jnp.ndarray] = None,
    p0: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Naively integrate IMU measurements to update position and velocity.

    This helper performs a very simple, orientation-agnostic integration of a
    sequence of IMU samples. It assumes that the acceleration vectors are
    already expressed in the world frame and that gravity has been compensated
    for. As such, **it is not a replacement for proper IMU preintegration**, but
    it is convenient for toy 1D/3D experiments and testing the data flow from
    sensors into the optimizer.

    The integration scheme is:

    .. math::

        v_{k+1} &= v_k + a_k \\delta t_k\\
        p_{k+1} &= p_k + v_{k+1} \\delta t_k\\

    :param measurements: Iterable of :class:`IMUMeasurement` objects, in time order.
    :param v0: Optional initial velocity, shape ``(3,)``. Defaults to zeros.
    :param p0: Optional initial position, shape ``(3,)``. Defaults to zeros.
    :return: Tuple ``(p, v)`` with the final position and velocity.
    """

    if v0 is None:
        v = jnp.zeros(3, dtype=jnp.float32)
    else:
        v = jnp.asarray(v0, dtype=jnp.float32)

    if p0 is None:
        p = jnp.zeros(3, dtype=jnp.float32)
    else:
        p = jnp.asarray(p0, dtype=jnp.float32)

    for meas in measurements:
        a = jnp.asarray(meas.accel, dtype=jnp.float32)
        dt = float(meas.dt)
        v = v + a * dt
        p = p + v * dt

    return p, v
