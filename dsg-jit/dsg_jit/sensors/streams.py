# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Generic sensor streaming utilities for DSG-JIT.

This module provides a unified abstraction layer for handling live or
simulated sensor data streams—synchronous or asynchronous—so that all
sensor types (IMU, LiDAR, cameras, range sensors, etc.) can plug into
DSG-JIT’s dynamic scene graph pipeline without requiring unique logic for
each device.

The key goal is to decouple **how data is produced** (polling, callbacks,
asynchronous feeds) from **how DSG-JIT consumes that data** (factor graph
updates, node creation, temporal linking).

The module typically defines:

  * **SynchronousStreams**
      - A minimal wrapper around a sensor object that exposes a ``read()``
        method.
      - Designed for simple for-loops or offline dataset playback.
      - Ensures each call returns a typed measurement (e.g., IMUMeasurement,
        CameraFrame, LiDARScan).

  * **AsynchronousStreams**
      - Provides asyncio-based background tasks that fetch sensor data
        without blocking the main SLAM/DSG loop.
      - Each sensor type is polled in a separate coroutine, and the latest
        measurement is stored internally for consumption.
      - Enables future real-time extensions (e.g., multi-sensor fusion,
        asynchronous factor graph updates).

  * **StreamHandle**
      - An ergonomic wrapper exposing:
          ``.latest()`` – retrieve most recent measurement.
          ``.reset()`` – clear buffer.
          ``.close()`` – stop background tasks.
      - Useful for synchronized fusion pipelines where multiple sensors must
        provide data simultaneously.

Design philosophy:

  * Treat *all* sensors uniformly—same streaming API regardless of modality.
  * Allow both synchronous and asynchronous execution with zero change to
    downstream SLAM/DSG logic.
  * Enable future “plug-and-play” sensor integration for real robots,
    simulators, prerecorded logs, or unit test data.

This module does not itself perform SLAM or scene graph updates—it only
defines the mechanism by which sensors deliver data into such pipelines.
"""

from typing import Iterator, AsyncIterator, Optional, Any
from abc import ABC
from dsg_jit.sensors.base import SensorReading

class BaseSensorStream(ABC):
    """
    Abstract base class for all sensor streams used by DSG-JIT.

    This defines the minimal interface that synchronous and asynchronous
    sensor streams must support so they can be registered with
    :class:`sensors.fusion.SensorFusionManager`.

    Subclasses may implement *either* sync or async behavior:

    - Synchronous streams must implement :meth:`read`.
    - Asynchronous streams must implement :meth:`__aiter__`.

    Implementations may choose to support both, but it is not required.
    """

    def read(self) -> Optional[Any]:
        """
        Return the next sample from the stream, or ``None`` if no sample
        is currently available.

        Sync-only streams MUST implement this. Async-only streams may raise
        ``NotImplementedError``.
        """
        raise NotImplementedError("This stream does not support sync read().")

    def __aiter__(self) -> AsyncIterator[Any]:
        """
        Asynchronous iteration interface. Async-only streams MUST implement
        this. Sync-only streams may raise ``NotImplementedError``.
        """
        raise NotImplementedError("This stream does not support async iteration.")

    def close(self) -> None:
        """
        Optional cleanup hook.

        Streams that hold system resources (files, sockets, hardware handles)
        should override this to release them.
        """
        return None

class ReadingStream(BaseSensorStream):
    """Synchronous stream of :class:`SensorReading` objects.

    Concrete subclasses implement :meth:`__iter__` to yield readings one by one.

    Typical usage::

        stream = FileRangeStream("ranges.txt")
        for reading in stream:
            process(reading)
    """

    def __init__(self) -> None:
        """
        Initialize a new reading stream.

        Subclasses typically only need to override :meth:`__iter__`. This base
        constructor sets up internal state for :meth:`read`.
        """
        self._iter: Optional[Iterator[SensorReading]] = None

    def read(self) -> Optional[SensorReading]:
        """
        Return the next sensor reading from the stream, or ``None`` if the
        underlying iterator is exhausted.

        This method adapts the iterator protocol (``__iter__``) to the
        :class:`BaseSensorStream` synchronous ``read`` API, so that
        :class:`~sensors.fusion.SensorFusionManager` and other callers can
        treat all synchronous streams uniformly.

        :return: Next sensor reading, or ``None`` if no further data is
                 available.
        :rtype: Optional[SensorReading]
        """
        if self._iter is None:
            self._iter = iter(self)
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def __iter__(self) -> Iterator[SensorReading]:
        """Iterate over sensor readings.

        :return: Iterator over :class:`SensorReading` objects.
        :rtype: Iterator[SensorReading]
        """
        raise NotImplementedError

class FunctionStream(ReadingStream):
    """
    Synchronous stream that pulls samples from a user-provided callback
    function instead of a file or iterator.

    The callback must return either:
        - a SensorReading
        - None (to indicate end of data)
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __iter__(self):
        while True:
            out = self.fn()
            if out is None:
                return
            yield out

class FileRangeStream(ReadingStream):
    """Synchronous stream backed by a plain-text file.

    Each line in the file is interpreted as a single floating-point range
    measurement. The line index is used as the time step ``t``.

    :param path: Path to a text file containing one numeric value per line.
    :type path: str
    """

    def __init__(self, path: str):
        """Initialize the file-backed range stream.

        :param path: Path to a text file containing one numeric value per line.
        :type path: str
        """
        super().__init__()
        self.path = path

    def __iter__(self) -> Iterator[SensorReading]:
        """Yield sensor readings from the underlying file.

        Each yielded reading has ``t`` equal to the zero-based line index and
        ``data`` equal to the parsed floating-point value.

        :return: Iterator over :class:`SensorReading` objects.
        :rtype: Iterator[SensorReading]
        """
        with open(self.path) as f:
            for t, line in enumerate(f):
                yield SensorReading(t=t, data=float(line.strip()))


class AsyncReadingStream(BaseSensorStream):
    """Asynchronous stream of :class:`SensorReading` objects.

    Subclasses implement :meth:`__aiter__` to provide an async iterator over
    sensor readings. This is useful when integrating with an asyncio-based
    event loop or real-time sensor drivers.
    """

    def __aiter__(self) -> AsyncIterator[SensorReading]:
        """Return an asynchronous iterator over sensor readings.

        :return: Asynchronous iterator over :class:`SensorReading` objects.
        :rtype: AsyncIterator[SensorReading]
        """
        raise NotImplementedError


import asyncio


class AsyncFileRangeStream(AsyncReadingStream):
    """Asynchronous file-backed range stream.

    This behaves like :class:`FileRangeStream`, but exposes an async iterator
    interface so it can be consumed with ``async for``. An optional ``delay``
    can be used to simulate a sensor publishing at a fixed rate.

    .. note::

       File I/O is still performed using the standard blocking ``open`` call.
       For small logs and simple simulations this is usually fine, but for
       large files or strict real-time requirements you may prefer to replace
       this with a true async file reader.

    :param path: Path to a text file containing one numeric value per line.
    :type path: str
    :param delay: Optional delay between consecutive readings, in seconds.
    :type delay: float or None
    """

    def __init__(self, path: str, delay: Optional[float] = None):
        """Initialize the asynchronous file-backed range stream.

        :param path: Path to a text file containing one numeric value per line.
        :type path: str
        :param delay: Optional delay between consecutive readings, in seconds.
        :type delay: float or None
        """
        self.path = path
        self.delay = delay

    async def __aiter__(self) -> AsyncIterator[SensorReading]:
        """Asynchronously iterate over sensor readings.

        If ``delay`` is set, the coroutine sleeps for that many seconds
        between consecutive readings.

        :return: Asynchronous iterator over :class:`SensorReading` objects.
        :rtype: AsyncIterator[SensorReading]
        """
        # Note: we use regular file I/O here; for most offline logs this is
        # sufficient and keeps the dependency surface small.
        with open(self.path) as f:
            for t, line in enumerate(f):
                if self.delay is not None:
                    await asyncio.sleep(self.delay)
                yield SensorReading(t=t, data=float(line.strip()))