"""
exp22_sensor_dsg_mapping.py

End-to-end demo:
- Simulated camera + LiDAR + IMU streams.
- Conversion to typed measurements (CameraMeasurement, LidarMeasurement, IMUMeasurement).
- Conversion to MeasurementFactor objects via sensors.conversion.
- Injection into a WorldModel / FactorGraph.
- Gauss–Newton solve and 3D visualization of the resulting graph.

This is intentionally small and synthetic. The goal is to show *how* the
sensor stack can drive DSG-JIT, not to be a realistic SLAM pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import jax.numpy as jnp
import numpy as np

#SLAM
from slam.manifold  import build_manifold_metadata
# Core / optimization
from core.factor_graph import FactorGraph
from optimization.solvers import gauss_newton_manifold, GNConfig
from core.types import Factor, FactorId

# World & visualization
from world.model import WorldModel
from world.scene_graph import SceneGraphWorld
from world.visualization import plot_factor_graph_3d

# Sensor stack
from sensors.camera import CameraMeasurement, CameraFrame
from sensors.lidar import LidarMeasurement
from sensors.imu import IMUMeasurement
from sensors.streams import FunctionStream
from sensors.conversion import (
    raw_sample_to_camera_measurement,
    raw_sample_to_lidar_measurement,
    raw_sample_to_imu_measurement,
    integrate_imu_delta,
)

# -----------------------------------------------------------------------------
# Simple synthetic world + sensor models
# -----------------------------------------------------------------------------

@dataclass
class ToyLandmark1D:
    """A single landmark at x = L in 1D, embedded in 3D as (L, 0, 0)."""
    x: float = 5.0


def make_camera_read_fn():
    """
    Return a generator function that yields fake camera samples.

    Each sample is just a dummy 1x1 grayscale image and a timestamp.  The
    important thing is the structure of the dict, not the content.
    """

    def _gen() -> Iterable[Dict[str, Any]]:
        t = 0.0
        while True:
            yield {
                "t": t,
                "frame_id": "cam0",
                "image": np.zeros((1, 1), dtype=np.float32),
            }
            t += 1.0

    def read_once() -> Dict[str, Any]:
        # Stateful generator captured via closure for FunctionStream
        if not hasattr(read_once, "_it"):
            read_once._it = iter(_gen())
        return next(read_once._it)

    return read_once


def make_lidar_read_fn(range_val: float = 5.0, num_beams: int = 1):
    """
    Return a generator function that yields fake LiDAR samples.

    We simulate a planar LiDAR with `num_beams` beams, all seeing the same
    constant range `range_val` at different angles. For simplicity we
    default to a single beam straight ahead.
    """

    def _gen() -> Iterable[Dict[str, Any]]:
        t = 0.1
        while True:
            ranges = np.full((num_beams,), range_val, dtype=np.float32)
            if num_beams == 1:
                angles = np.array([0.0], dtype=np.float32)
            else:
                angles = np.linspace(-math.pi / 4, math.pi / 4, num_beams, dtype=np.float32)

            yield {
                "t": t,
                "frame_id": "lidar0",
                "ranges": ranges,
                "angles": angles,
            }
            t += 0.1

    def read_once() -> Dict[str, Any]:
        if not hasattr(read_once, "_it"):
            read_once._it = iter(_gen())
        return next(read_once._it)

    return read_once


def make_imu_read_fn(ax: float = 1.0):
    """
    Return a generator function that yields toy IMU samples.

    We simulate constant acceleration in +x with a small dt. This is
    enough to feed integrate_imu_delta and generate small SE(3) increments.
    """

    def _gen() -> Iterable[Dict[str, Any]]:
        t = 0.0
        dt = 0.1
        while True:
            yield {
                "t": t,
                "accel": np.array([ax, 0.0, 0.0], dtype=np.float32),
                "gyro": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "dt": dt,
            }
            t += dt

    def read_once() -> Dict[str, Any]:
        if not hasattr(read_once, "_it"):
            read_once._it = iter(_gen())
        return next(read_once._it)

    return read_once


# -----------------------------------------------------------------------------
# Factor-graph / world helpers
# -----------------------------------------------------------------------------

def build_world_model(num_poses: int = 5) -> Tuple[WorldModel, SceneGraphWorld, list[int]]:
    """
    Build a simple 1D SE(3) chain of robot poses in a WorldModel + SceneGraphWorld.

    The robot moves along x: 0, 1, 2, ...; we register poses and a simple odometry
    chain (additive).

    Returns:
        sg.wm: WorldModel instance
        sg: SceneGraphWorld view
        pose_ids: list of variable node ids for each pose
    """
    sg = SceneGraphWorld()

    # 2) Add poses and odometry chain in the WorldModel directly.
    pose_ids: List[int] = []

    # Pose 0 at the origin
    x0 = jnp.zeros((6,), dtype=jnp.float32)
    pose0 = sg.add_pose_se3(x0)
    pose_ids.append(pose0)

    # Subsequent poses: 1m steps along +x, with odometry factors
    for k in range(1, num_poses):
        xk = jnp.array([float(k), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        pk = sg.add_pose_se3(xk)
        pose_ids.append(pk)

        dx = 1.0
        sg.add_odom_se3_additive(pose_ids[k - 1], pose_ids[k], dx=dx, sigma=1.0)

    return sg.wm, sg, pose_ids


def add_range_factors_from_lidar(
    fg: FactorGraph,
    pose_ids: List[int],
    landmark_id: int,
    lidar_meas: LidarMeasurement,
    sigma: float = 0.05,
) -> List[FactorId]:
    """
    Use a simple 1D interpretation of LiDAR measurements to create prior
    factors on the landmark position.

    We assume a single beam pointing along +x that measures the distance
    from the robot to the landmark. In this toy setup, we already know
    the robot poses approximately (0, 1, 2, ...) along x, and we use the
    LiDAR range to define a prior on the landmark's x coordinate.

    Concretely, we add factors of type "prior" on the landmark variable,
    with a target position:

        [mean_range, 0, 0]

    so that the prior residual enforces:

        landmark_xyz - target ≈ 0

    :param fg: Underlying factor graph (fg) from the WorldModel.
    :type fg: FactorGraph
    :param pose_ids: List of pose node IDs. We only use the length to
        decide how many priors to add (one per pose, for demonstration).
    :type pose_ids: list[int]
    :param landmark_id: Node ID of the landmark variable in the graph.
    :type landmark_id: int
    :param lidar_meas: LiDAR measurement used to build the priors.
    :type lidar_meas: LidarMeasurement
    :param sigma: Standard deviation for the prior on the landmark
        position (in meters). Smaller values mean a stronger prior.
    :type sigma: float
    :return: List of newly created FactorId values.
    :rtype: list[FactorId]
    """
    # Use the mean range from the LiDAR scan (all 5.0 in this toy example).
    mean_range = float(jnp.mean(lidar_meas.ranges))

    # Target landmark position in world coordinates (x = mean_range, y = z = 0).
    target = jnp.array([mean_range, 0.0, 0.0], dtype=jnp.float32)

    # Simple isotropic weight: 1/sigma^2 (can be scalar; _apply_weight will broadcast).
    w = 1.0 / (sigma ** 2)

    # Allocate fresh factor ids.
    if fg.factors:
        next_id_val = max(fg.factors.keys()) + 1
    else:
        next_id_val = 0

    created_ids: List[FactorId] = []

    # For demonstration, add one prior factor per pose (they all constrain the
    # same landmark variable to the same target).
    for _ in pose_ids:
        factor_id = FactorId(next_id_val)
        next_id_val += 1

        fg.add_factor(
            Factor(
                id=factor_id,
                type="prior",  # uses prior_residual in slam.measurements
                var_ids=[int(landmark_id)],
                params={
                    "target": target,        # required by prior_residual
                    "weight": jnp.array(w),  # optional, scalar or length-3
                },
            )
        )
        created_ids.append(factor_id)

    return created_ids

# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def main():
    # -------------------------------------------------------------------------
    # 1) Build world + scene graph
    # -------------------------------------------------------------------------
    num_poses = 5
    wm, sg, pose_ids = build_world_model(num_poses=num_poses)
    fg = wm.fg  # underlying FactorGraph

    # Add a simple 1D landmark at x = 5.0
    landmark = ToyLandmark1D(x=5.0)
    landmark_var = sg.add_landmark3d(
        jnp.array([landmark.x, 0.0, 0.0], dtype=jnp.float32)
    )

    # -------------------------------------------------------------------------
    # 2) Build sensor streams (like exp21) and take a few samples
    # -------------------------------------------------------------------------
    cam_stream = FunctionStream(make_camera_read_fn())
    lidar_stream = FunctionStream(make_lidar_read_fn(range_val=5.0, num_beams=1))
    imu_stream = FunctionStream(make_imu_read_fn(ax=1.0))

    # One camera frame, one LiDAR scan, and a few IMU samples just to show usage.
    raw_cam = cam_stream.read()
    raw_lidar = lidar_stream.read()

    # Convert to typed measurement objects
    cam_meas: CameraMeasurement = raw_sample_to_camera_measurement(raw_cam)
    lidar_meas: LidarMeasurement = raw_sample_to_lidar_measurement(raw_lidar)

    # Take a couple of IMU steps and integrate them into a single se(3) delta
    imu_deltas = []
    for _ in range(3):
        raw_imu = imu_stream.read()
        imu_meas: IMUMeasurement = raw_sample_to_imu_measurement(raw_imu)
        dxi = integrate_imu_delta(imu_meas, dt=imu_meas.dt)
        imu_deltas.append(dxi)
    if imu_deltas:
        fused_delta = jnp.sum(jnp.stack(imu_deltas, axis=0), axis=0)
    else:
        fused_delta = jnp.zeros((6,), dtype=jnp.float32)

    print("[Camera] one frame from stream "
          f"(shape={cam_meas.frame.image.shape}, t={cam_meas.frame.timestamp:.2f})")
    print("[LiDAR]  one scan from stream "
          f"(mean range={float(jnp.mean(lidar_meas.ranges)):.2f} m)")
    print(f"[IMU]    fused_delta (toy) = {np.array(fused_delta)}")

    # -------------------------------------------------------------------------
    # 3) Use LiDAR data to create range factors to the landmark
    # -------------------------------------------------------------------------
    print("\n=== Creating range priors from LiDAR ===")
    factor_ids = add_range_factors_from_lidar(
        fg=fg,
        pose_ids=pose_ids,
        landmark_id=landmark_var,
        lidar_meas=lidar_meas,
        sigma=0.05,
    )
    print(f"Added {len(factor_ids)} prior factors on landmark {landmark_var}.")

    # -------------------------------------------------------------------------
    # 4) Optimize the factor graph (Gauss–Newton on SE(3)+R^3)
    # -------------------------------------------------------------------------
    x0, index = fg.pack_state()

    # Build manifold_types: we assume:
    #   - pose variables are SE3
    #   - landmark is Euclidean
    manifold_types = []
    for var in fg.variables.values():
        if var.type == "pose_se3":
            manifold_types.append("se3")
        else:
            manifold_types.append("euclidean")
    manifold_types = tuple(manifold_types)
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(
        max_iters=20,
        damping=1e-3,
        max_step_norm=1.0
    )

    block_slices, manifold_types = build_manifold_metadata(fg)

    x_opt = gauss_newton_manifold(
        residual_fn=residual_fn,
        x0=x0,
        manifold_types=manifold_types,
        cfg=cfg,
        block_slices=block_slices
    )

    # Unpack optimized poses/landmarks back into the graph
    values = fg.unpack_state(x_opt, index)

    print("\n=== Optimized poses ===")
    for pid in pose_ids:
        v = values[pid]
        print(f"pose[{pid}]: {np.array(v)}")

    print("\n=== Optimized landmark ===")
    lv = values[landmark_var]
    print(f"landmark[{landmark_var}]: {np.array(lv)}")
    # -------------------------------------------------------------------------
    # 5) Visualize as a 3D factor graph
    # -------------------------------------------------------------------------
    plot_factor_graph_3d(
        fg,
        show_labels=True
    )


if __name__ == "__main__":
    main()