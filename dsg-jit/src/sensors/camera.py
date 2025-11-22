# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Camera sensor abstractions and utilities for DSG-JIT.

This module defines lightweight, JAX-friendly camera interfaces that can be
plugged into dynamic scene graph (DSG) pipelines. The goal is to provide a
clean separation between:

  * **Raw image acquisition** (e.g., from a hardware driver, simulator, or
    prerecorded dataset), and
  * **Downstream SLAM / DSG consumers** that only need structured frames
    (RGB or grayscale) with timestamps and metadata.

The module typically exposes:

  * ``CameraFrame``:
      A simple dataclass-like structure representing a single image frame.
      It usually stores image data (RGB or grayscale), optional depth, and
      a timestamp or frame index.

  * ``CameraSensor``:
      A wrapper around an arbitrary user-provided capture function. The
      capture function may return NumPy arrays or JAX arrays; the wrapper
      normalizes these and provides a consistent interface for reading
      frames in synchronous loops or via the generic sensor streams.

  * Optional utilities for:
      - Converting RGB frames to grayscale.
      - Normalizing/typing images for consumption by JAX or downstream
        vision modules.
      - Integrating with the generic sensor streaming API (synchronous or
        asynchronous).

These camera abstractions are intentionally minimal and do *not* perform
full visual odometry or object detection. Instead, they are designed as
building blocks for higher-level modules (e.g., visual SLAM, semantic
DSG layers) that can interpret camera data and inject new nodes and
factors into the scene graph or factor graph.

The design philosophy is:

  * Keep camera handling **stateless** where possible.
  * Make it easy to wrap *any* existing camera source (OpenCV, ROS, custom
    simulator, etc.) behind a small capture function.
  * Ensure that integration with DSG-JIT remains explicit and composable,
    so users can swap cameras or add multiple sensors without changing
    core SLAM logic.
"""

from typing import Optional, Callable, AsyncIterator, Dict, Any
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsics.

    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param fx: Focal length in x direction.
    :param fy: Focal length in y direction.
    :param cx: Principal point x coordinate.
    :param cy: Principal point y coordinate.
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraFrame:
    """
    A single camera frame.

    :param image: The image array.
    :param timestamp: Timestamp of the frame capture.
    :param frame_id: Optional frame identifier.
    :param color_space: Color space of the image, either "rgb" or "gray".
    """
    image: np.ndarray
    timestamp: float
    frame_id: Optional[str]
    color_space: str

@dataclass
class CameraMeasurement:
    """
    High-level camera measurement suitable for feeding into SLAM / DSG layers.

    This wraps a low-level :class:`CameraFrame` with additional metadata
    such as sensor ID, extrinsics, and an optional sequence index. It is
    intentionally minimal and can be extended by applications as needed.

    :param frame:
        The underlying camera frame (image, timestamp, color space, etc.).
    :param sensor_id:
        Identifier for the camera (e.g., ``"cam0"``). Useful when multiple
        cameras are present.
    :param T_cam_body:
        Optional 4x4 homogeneous transform from body frame to camera frame.
        If omitted, downstream consumers may assume an identity transform or
        use a configured default.
    :param seq:
        Optional sequence index (e.g., frame counter) for convenience.
    :param metadata:
        Optional free-form dictionary for extra information
        (exposure, gain, rolling-shutter parameters, etc.).
    """
    frame: "CameraFrame"
    sensor_id: str = "cam0"
    T_cam_body: Optional[np.ndarray] = None
    seq: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def is_rgb(frame: CameraFrame) -> bool:
    """
    Check if the frame is in RGB color space.

    :param frame: The camera frame to check.
    :returns: True if the frame's color space is "rgb" (case-insensitive), False otherwise.
    """
    return frame.color_space.lower() == "rgb"


def is_gray(frame: CameraFrame) -> bool:
    """
    Check if the frame is in grayscale color space.

    :param frame: The camera frame to check.
    :returns: True if the frame's color space is one of "gray", "grey", or "grayscale" (case-insensitive), False otherwise.
    """
    return frame.color_space.lower() in ("gray", "grey", "grayscale")


def to_grayscale(frame: CameraFrame) -> CameraFrame:
    """
    Convert an RGB frame to grayscale. If the frame is already grayscale, returns it unchanged.

    For RGB frames, assumes image shape is (H, W, 3) and applies standard luminance weights [0.299, 0.587, 0.114]
    to convert to grayscale with shape (H, W). The returned image is float32 in [0, 1] if the input was uint8.

    :param frame: The input camera frame.
    :returns: A new CameraFrame in grayscale color space with the same timestamp and frame_id.
    """
    if is_gray(frame):
        return frame
    img = np.asarray(frame.image)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return CameraFrame(
        image=gray_img,
        timestamp=frame.timestamp,
        frame_id=frame.frame_id,
        color_space="gray"
    )


def to_rgb(frame: CameraFrame) -> CameraFrame:
    """
    Convert a grayscale frame to RGB by stacking the gray channel three times along the last axis.
    If the frame is already RGB, returns it unchanged.

    :param frame: The input camera frame.
    :returns: A new CameraFrame in RGB color space with the same timestamp and frame_id.
    """
    if is_rgb(frame):
        return frame
    img = np.asarray(frame.image)
    if img.ndim == 2:
        rgb_img = np.stack([img, img, img], axis=-1)
    else:
        rgb_img = img
    return CameraFrame(
        image=rgb_img,
        timestamp=frame.timestamp,
        frame_id=frame.frame_id,
        color_space="rgb"
    )


@dataclass
class SyncCamera:
    """
    Synchronous camera source.

    :param intrinsics: Camera intrinsics.
    :param read_fn: Callable that returns an image as a numpy array.
    :param color_space: Color space of the images produced by read_fn, default is "rgb".
    """
    intrinsics: CameraIntrinsics
    read_fn: Callable[[], np.ndarray]
    color_space: str = "rgb"

    def read(self) -> CameraFrame:
        """
        Read a single frame from the camera.

        :returns: A CameraFrame containing the image, current timestamp, no frame_id, and the configured color_space.
        """
        img = self.read_fn()
        return CameraFrame(
            image=img,
            timestamp=time.time(),
            frame_id=None,
            color_space=self.color_space
        )


@dataclass
class AsyncCamera:
    """
    Asynchronous camera source.

    :param intrinsics: Camera intrinsics.
    :param aiter_fn: Callable that returns an async iterator yielding images as numpy arrays.
    :param color_space: Color space of the images produced by aiter_fn, default is "rgb".
    """
    intrinsics: CameraIntrinsics
    aiter_fn: Callable[[], AsyncIterator[np.ndarray]]
    color_space: str = "rgb"

    async def frames(self) -> AsyncIterator[CameraFrame]:
        """
        Asynchronously iterate over frames from the camera.

        :returns: An async iterator yielding CameraFrame objects with current timestamp, no frame_id, and the configured color_space.
        """
        async for img in self.aiter_fn():
            yield CameraFrame(
                image=img,
                timestamp=time.time(),
                frame_id=None,
                color_space=self.color_space
            )
