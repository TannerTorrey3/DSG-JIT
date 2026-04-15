# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
KITTI dataset utilities for pose conversion and measurement extraction.

Converts between KITTI's 4x4 homogeneous transforms and DSG-JIT's 6D
pose vectors ``[tx, ty, tz, wx, wy, wz]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from dsg_jit.core.math3d import so3_log, relative_pose_se3


def kitti_mat_to_pose6d(flat16: Tuple[float, ...]) -> jnp.ndarray:
    """Convert a flattened 4x4 KITTI pose to a 6D pose vector.

    :param flat16: 16-element row-major 4x4 homogeneous transform.
    :return: 6D pose vector ``[tx, ty, tz, wx, wy, wz]``.
    """
    T = np.array(flat16, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    # Ensure R is a proper rotation (project to SO(3) via SVD).
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt

    w = np.array(so3_log(jnp.array(R_proj, dtype=jnp.float32)))
    return jnp.array([t[0], t[1], t[2], w[0], w[1], w[2]], dtype=jnp.float32)


def kitti_frames_to_poses6d(frames) -> jnp.ndarray:
    """Convert a list of KittiOdomFrame to a (N, 6) array of 6D poses.

    Frames without ground truth (``T_w_cam0 is None``) are skipped.

    :param frames: List of :class:`KittiOdomFrame`.
    :return: (N, 6) array of poses in DSG-JIT format.
    """
    poses = []
    for f in frames:
        if f.T_w_cam0 is None:
            continue
        poses.append(kitti_mat_to_pose6d(f.T_w_cam0))
    return jnp.stack(poses)


def poses6d_to_relative_measurements(poses: jnp.ndarray) -> jnp.ndarray:
    """Compute relative pose measurements between consecutive poses.

    :param poses: (N, 6) absolute poses.
    :return: (N-1, 6) relative measurements.
    """
    n = poses.shape[0]
    meas = []
    for i in range(n - 1):
        meas.append(relative_pose_se3(poses[i], poses[i + 1]))
    return jnp.stack(meas)


def load_kitti_raw_oxts(oxts_dir: str | Path) -> list[dict]:
    """Load raw OXTS GPS/IMU data from a KITTI raw drive.

    Expects the standard layout::

        oxts_dir/
          data/
            0000000000.txt
            ...
          timestamps.txt

    Each OXTS file contains 30 values per line. The first 6 are:
    ``lat, lon, alt, roll, pitch, yaw`` (degrees/radians as per KITTI spec).

    :param oxts_dir: Path to the ``oxts/`` directory of a raw drive.
    :return: List of dicts with keys ``lat, lon, alt, roll, pitch, yaw, t``.
    """
    oxts_dir = Path(oxts_dir)
    data_dir = oxts_dir / "data"
    ts_file = oxts_dir / "timestamps.txt"

    # Load timestamps.
    timestamps = []
    if ts_file.exists():
        with ts_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # KITTI raw timestamps: "2011-09-30 12:34:56.789012345"
                # We just need relative ordering; parse as float seconds.
                try:
                    from datetime import datetime
                    dt = datetime.strptime(line[:26], "%Y-%m-%d %H:%M:%S.%f")
                    timestamps.append(dt.timestamp())
                except (ValueError, IndexError):
                    timestamps.append(len(timestamps) / 10.0)

    # Load OXTS data files.
    data_files = sorted(data_dir.glob("*.txt"))
    records = []
    for i, fpath in enumerate(data_files):
        with fpath.open() as f:
            vals = [float(x) for x in f.read().strip().split()]
        if len(vals) < 6:
            continue
        t = timestamps[i] if i < len(timestamps) else i / 10.0
        records.append({
            "lat": vals[0],
            "lon": vals[1],
            "alt": vals[2],
            "roll": vals[3],
            "pitch": vals[4],
            "yaw": vals[5],
            "t": t,
        })
    return records


def oxts_to_local_poses(records: list[dict]) -> np.ndarray:
    """Convert raw OXTS GPS records to local ENU poses.

    Uses a Mercator projection centred on the first record to convert
    lat/lon to local x/y in metres.  Returns (N, 6) array of 6D poses.

    :param records: List from :func:`load_kitti_raw_oxts`.
    :return: (N, 6) numpy array of poses ``[x, y, z, roll, pitch, yaw]``.
    """
    if not records:
        return np.zeros((0, 6))

    # Reference point.
    lat0 = np.radians(records[0]["lat"])
    lon0 = np.radians(records[0]["lon"])
    alt0 = records[0]["alt"]

    R_EARTH = 6378137.0  # WGS84 semi-major axis

    poses = np.zeros((len(records), 6), dtype=np.float64)
    for i, rec in enumerate(records):
        lat = np.radians(rec["lat"])
        lon = np.radians(rec["lon"])

        # Mercator projection to local ENU.
        x = R_EARTH * (lon - lon0) * np.cos(lat0)
        y = R_EARTH * (lat - lat0)
        z = rec["alt"] - alt0

        poses[i] = [x, y, z, rec["roll"], rec["pitch"], rec["yaw"]]

    return poses
