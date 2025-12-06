# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.

"""
exp21_sensor_fusion_demo.py

Minimal sensor fusion demo for DSG-JIT.

This experiment shows how to:

  - Define simple synthetic sensor "streams" for a camera, LiDAR, and IMU.
  - Wrap them in FunctionStream (for synthetic generators) or FileRangeStream (for file playback).
  - Register them with SensorFusionManager.
  - Use a callback to accumulate a toy 1D pose estimate from IMU data.
  - Inspect / print the fused results.

This also demonstrates how to mirror fused IMU poses into a WorldModel-backed
factor graph, so the standard residual / optimization pipeline can be used
later if desired. It remains a lightweight sandbox for the new `sensors.*`
layer.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List

import jax.numpy as jnp
import matplotlib.pyplot as plt

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual

from dsg_jit.sensors.streams import FunctionStream
from dsg_jit.sensors.fusion import SensorFusionManager
from dsg_jit.sensors.base import BaseMeasurement
from dsg_jit.sensors.camera import CameraMeasurement
from dsg_jit.sensors.lidar import LidarMeasurement
from dsg_jit.sensors.imu import IMUMeasurement
from dsg_jit.sensors.conversion import raw_sample_to_camera_measurement, raw_sample_to_imu_measurement, raw_sample_to_lidar_measurement


# ---------------------------------------------------------------------------
# Synthetic sensor "hardware"
# ---------------------------------------------------------------------------

def make_camera_read_fn(landmark_ids):
    """
    Return a generator-style read() function that yields synthetic 
    bearing measurements compatible with raw_sample_to_camera_measurement.
    """
    # Toy example: 2 landmarks, constant bearings
    samples = [
        {
            "t": 0.0,
            "frame_id": 0,
            # This is the key raw_sample_to_camera_measurement expects:
            "bearings": jnp.array(
                [
                    [1.0, 0.0, 0.0],    # bearing to landmark 0
                    [1.0, 0.1, 0.0],    # bearing to landmark 1
                ],
                dtype=jnp.float32,
            ),
            "landmark_ids": landmark_ids,           # e.g. [lid0, lid1]
            "sensor_pose": jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32),
        },
        {
            "t": 1.0,
            "frame_id": 1,
            # Slightly perturbed bearings at t=1
            "bearings": jnp.array(
                [
                    [1.0, 0.05, 0.0],
                    [1.0, 0.15, 0.0],
                ],
                dtype=jnp.float32,
            ),
            "landmark_ids": landmark_ids,
            "sensor_pose": jnp.array([0.5, 0.0, 0, 0, 0, 0], dtype=jnp.float32),
        },
        # add as many time steps as you like
    ]

    it = iter(samples)

    def read():
        try:
            return next(it)
        except StopIteration:
            return None  # signals end of stream to SensorFusionManager

    return read


def make_lidar_read_fn() -> callable:
    """
    Build a synthetic LiDAR read() function.

    We simulate a simple 2D planar scanner with a single "wall" at x=5.0.
    """

    t = 0.0

    def read() -> Dict[str, Any]:
        nonlocal t
        t += 0.1  # 10 Hz lidar
        num_beams = 16
        # Angles from -45 to +45 degrees
        angles = jnp.linspace(-math.pi / 4, math.pi / 4, num_beams)
        # Flat wall at ~5m in front
        ranges = 5.0 * jnp.ones_like(angles, dtype=jnp.float32)

        return {
            "t": t,
            "frame_id": "lidar0",
            "angles": angles,
            "ranges": ranges,
            "rays": jnp.stack([angles, ranges], axis=1),
        }

    return read


def make_imu_read_fn() -> callable:
    """
    Build a synthetic IMU read() function.

    We simulate constant acceleration in +x, which we will integrate into
    a toy 1D position estimate.
    """

    t = 0.0

    def read() -> Dict[str, Any]:
        nonlocal t
        t += 0.05  # 20 Hz IMU
        dt = 0.05

        # Simple motion: a(t) = 0.5 m/s^2 along x, zero elsewhere.
        accel = jnp.array([0.5, 0.0, 0.0], dtype=jnp.float32)
        gyro = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

        return {
            "t": t,
            "dt": dt,
            "frame_id": "imu0",
            "accel": accel,
            "gyro": gyro,
        }

    return read


# ---------------------------------------------------------------------------
# Simple "fusion" on top of SensorFusionManager
# ---------------------------------------------------------------------------

class ToyImuIntegrator:
    """
    Very simple 1D integrator using IMU measurements.

    This is *not* a real inertial filter; it's just a toy to show that
    SensorFusionManager callbacks can accumulate state and optionally
    publish fused poses back into the manager and into a WorldModel.
    """

    def __init__(self, fusion: SensorFusionManager, wm: WorldModel | None = None) -> None:
        self.fusion = fusion
        self.wm = wm

        self.x = 0.0      # position along x
        self.vx = 0.0     # velocity along x
        self.history_t: List[float] = []
        self.history_x: List[float] = []

        # Optional WorldModel-backed pose chain
        self._last_pose_id = None
        self._last_pose_x = 0.0
        self._pose_ids: List[int] = []

        if self.wm is not None:
            # Register residuals for pose priors and odometry at the WorldModel level.
            self.wm.register_residual("prior", prior_residual)
            self.wm.register_residual("odom_se3", odom_se3_residual)

            # Initialize a base pose at the origin with a prior.
            pose0 = jnp.zeros(6, dtype=jnp.float32)
            pose0_id = self.wm.add_variable(
                var_type="pose_se3",
                value=pose0,
            )
            self.wm.add_factor(
                f_type="prior",
                var_ids=(pose0_id,),
                params={"target": pose0},
            )
            self._last_pose_id = pose0_id
            self._last_pose_x = 0.0
            self._pose_ids.append(pose0_id)

    def __call__(self, meas: BaseMeasurement) -> None:
        # We only care about IMU for this toy integrator
        if isinstance(meas, IMUMeasurement):
            # Expect meas.accel and meas.dt
            a_x = float(meas.accel[0])
            dt = float(meas.dt)

            # Basic integration: v += a * dt, x += v * dt
            self.vx += a_x * dt
            self.x += self.vx * dt

            self.history_t.append(float(meas.timestamp))
            self.history_x.append(self.x)

            # Optionally publish a "fused" pose (only x is used here)
            pose_se3 = jnp.array([self.x, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
            self.fusion.record_fused_pose(
                t=meas.timestamp,
                pose_se3=pose_se3,
                covariance=None,
                source_counts={"imu": 1},
            )

            # Optionally mirror this fused pose into the WorldModel as a pose chain.
            if self.wm is not None and self._last_pose_id is not None:
                # Create a new pose variable at the fused position.
                pose_vec = jnp.array(
                    [self.x, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=jnp.float32,
                )
                pose_id = self.wm.add_variable(
                    var_type="pose_se3",
                    value=pose_vec,
                )

                # Build an odometry measurement between the last pose and this one.
                dx = self.x - self._last_pose_x
                odom_meas = jnp.array(
                    [dx, 0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=jnp.float32,
                )
                self.wm.add_factor(
                    f_type="odom_se3",
                    var_ids=(self._last_pose_id, pose_id),
                    params={"measurement": odom_meas},
                )

                self._last_pose_id = pose_id
                self._last_pose_x = self.x
                self._pose_ids.append(pose_id)

        elif isinstance(meas, CameraMeasurement):
            # Log something lightweight about the camera sample.
            # We try to be robust to different CameraMeasurement layouts.
            if hasattr(meas, "frame") and hasattr(meas.frame, "timestamp"):
                t_val = float(meas.frame.timestamp)
            elif hasattr(meas, "timestamp"):
                t_val = float(meas.timestamp)
            elif hasattr(meas, "t"):
                t_val = float(meas.t)
            else:
                t_val = 0.0

            if hasattr(meas, "frame") and hasattr(meas.frame, "image"):
                shape = tuple(meas.frame.image.shape)
                desc = f"image shape={shape}"
            elif hasattr(meas, "image"):
                shape = tuple(meas.image.shape)
                desc = f"image shape={shape}"
            elif hasattr(meas, "bearings") and meas.bearings is not None:
                shape = tuple(meas.bearings.shape)
                desc = f"bearings shape={shape}"
            else:
                desc = "(no image / bearings)"

            print(f"[Camera] t={t_val:.2f}, {desc}")

        elif isinstance(meas, LidarMeasurement):
            # Again, just log; you might add range / occupancy factors.
            if hasattr(meas, "t"):
                t_val = float(meas.t)
            elif hasattr(meas, "timestamp"):
                t_val = float(meas.timestamp)
            else:
                t_val = 0.0

            if hasattr(meas, "ranges") and meas.ranges is not None:
                mean_range = float(jnp.mean(meas.ranges))
                print(f"[LiDAR]  t={t_val:.2f}, mean range={mean_range:.2f} m")
            else:
                print(f"[LiDAR]  t={t_val:.2f}, (no ranges)")


def main() -> None:
    # 1) Build sensor streams
    #
    # By default we use in-memory synthetic generators wrapped in FunctionStream.
    # This is convenient for testing without any data files.
    landmark_ids = [0, 1] 
    cam_stream = FunctionStream(make_camera_read_fn(landmark_ids=landmark_ids))
    lidar_stream = FunctionStream(make_lidar_read_fn())
    imu_stream = FunctionStream(make_imu_read_fn())

    # If you want to drive this experiment from recorded data instead, you can
    # create an `examples/data` directory and store one sample per line in a
    # simple text/JSON format, then swap the above for something like:
    #
    #   from pathlib import Path
    #   data_root = Path(__file__).resolve().parent.parent / "examples" / "data"
    #   cam_stream = FileRangeStream(data_root / "exp21_camera.txt")
    #   lidar_stream = FileRangeStream(data_root / "exp21_lidar.txt")
    #   imu_stream = FileRangeStream(data_root / "exp21_imu.txt")
    #
    # where each line in the file encodes a dict compatible with the
    # corresponding `*_sample_to_measurement` converter in `sensors.conversion`.

    # 2) Construct fusion manager and register sensors
    fusion = SensorFusionManager()

    fusion.register_sensor(
        name="cam0",
        modality="camera",
        stream=cam_stream,
        converter=raw_sample_to_camera_measurement,  # use default camera_sample_to_measurement
    )
    fusion.register_sensor(
        name="lidar0",
        modality="lidar",
        stream=lidar_stream,
        converter=raw_sample_to_lidar_measurement,  # use default lidar_scan_to_measurement
    )
    fusion.register_sensor(
        name="imu0",
        modality="imu",
        stream=imu_stream,
        converter=raw_sample_to_imu_measurement,  # use default imu_sample_to_measurement
    )

    # 3) Construct a WorldModel and attach our toy integrator as a callback.
    wm = WorldModel()
    integrator = ToyImuIntegrator(fusion, wm=wm)
    fusion.register_callback(integrator)

    # 4) Run a short simulation loop
    num_steps = 100
    for step in range(num_steps):
        n_meas = fusion.poll_once()
        if n_meas == 0:
            # All streams returned None; you could break here in a real app.
            pass

    # WorldModel summary: how many poses/factors were mirrored from the IMU integrator.
    print("\n=== WorldModel summary (exp21 sensor fusion demo) ===")
    print(f"Num variables: {len(wm.fg.variables)}")
    print(f"Num factors:   {len(wm.fg.factors)}")

    # 5) Inspect latest fused pose
    fused = fusion.get_latest_pose()
    if fused is not None:
        print("\n=== Latest fused pose (toy IMU integrator) ===")
        print(f"t = {fused.t}")
        print(f"pose_se3 = {fused.pose_se3}")
        print(f"source_counts = {fused.source_counts}")
    else:
        print("No fused pose produced.")

    # 6) Plot the integrated 1D trajectory
    if integrator.history_t:

        ts = jnp.array(integrator.history_t)
        xs = jnp.array(integrator.history_x)

        plt.figure()
        plt.plot(ts, xs, marker="o")
        plt.xlabel("time [s]")
        plt.ylabel("x position [m]")
        plt.title("Toy IMU-integrated 1D trajectory")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()