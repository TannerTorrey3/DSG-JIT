"""
Lightweight dataset loaders for DSG-JIT.

This module provides convenience helpers to load popular SLAM / VO datasets
into simple Python structures that can be wired into the `sensors.*` and
`world.*` subsystems (e.g. camera, LiDAR, pose sequences).

The goal is to:

* Keep I/O and parsing logic separate from the core optimization engine.
* Provide small, explicit dataclasses for each dataset family.
* Avoid heavy dependencies (no hard requirement on OpenCV, etc.).
"""

from .tum_rgbd import TumRgbdFrame, load_tum_rgbd_sequence
from .kitti_odometry import KittiOdomFrame, load_kitti_odometry_sequence

__all__ = [
    "TumRgbdFrame",
    "load_tum_rgbd_sequence",
    "KittiOdomFrame",
    "load_kitti_odometry_sequence",
]