from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class KittiOdomFrame:
    """
    Single frame from the KITTI Odometry dataset.

    This dataclass provides file paths and optional ground-truth pose for a
    particular frame index in a given sequence.

    :param seq:
        KITTI odometry sequence ID (e.g. ``"00"``, ``"05"``).
    :type seq: str

    :param idx:
        Integer frame index within the sequence.
    :type idx: int

    :param t:
        Approximate timestamp in seconds. Many downstream pipelines assume
        KITTI odometry runs at 10 Hz, so a common convention is
        ``t = idx / 10.0``.
    :type t: float

    :param left_path:
        Path to the left camera image (``image_0``) for this frame.
    :type left_path: str

    :param right_path:
        Optional path to the right camera image (``image_1``). May be ``None``
        if not available or desired.
    :type right_path: Optional[str]

    :param velo_path:
        Optional path to the LiDAR point cloud (``velodyne``). May be ``None``
        if not available or desired.
    :type velo_path: Optional[str]

    :param T_w_cam0:
        Optional 4x4 homogeneous transform from camera-0 to world frame as a
        flattened 16-element tuple in row-major order. This is derived from
        the official ``poses/<seq>.txt`` file if available.
    :type T_w_cam0: Optional[Tuple[float, ...]]
    """

    seq: str
    idx: int
    t: float
    left_path: str
    right_path: Optional[str] = None
    velo_path: Optional[str] = None
    T_w_cam0: Optional[Tuple[float, ...]] = None


def _load_kitti_poses(poses_path: Path) -> List[Tuple[float, ...]]:
    """
    Load KITTI odometry ground-truth poses from a ``poses/<seq>.txt`` file.

    Each line is expected to contain 12 numbers corresponding to the first
    three rows of a 4x4 homogeneous transform:

        ``r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz``

    This helper converts each pose to a flattened 4x4 matrix with a final
    row ``[0, 0, 0, 1]``.

    :param poses_path:
        Path to the KITTI poses file for a sequence.
    :type poses_path: pathlib.Path

    :return:
        List of flattened 4x4 transforms in row-major order, one per frame.
    :rtype: List[Tuple[float, ...]]
    """
    poses: List[Tuple[float, ...]] = []

    if not poses_path.exists():
        return poses

    with poses_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                continue

            r11, r12, r13, tx = vals[0:4]
            r21, r22, r23, ty = vals[4:8]
            r31, r32, r33, tz = vals[8:12]
            mat4 = (
                r11, r12, r13, tx,
                r21, r22, r23, ty,
                r31, r32, r33, tz,
                0.0, 0.0, 0.0, 1.0,
            )
            poses.append(mat4)

    return poses


def load_kitti_odometry_sequence(
    root: str | os.PathLike,
    seq: str,
    load_right: bool = True,
    load_velodyne: bool = False,
    with_poses: bool = True,
    max_frames: Optional[int] = None,
) -> List[KittiOdomFrame]:
    """
    Load a KITTI Odometry sequence into a list of frames.

    This helper assumes the standard KITTI directory structure:

    .. code-block::

        root/
          sequences/
            00/
              image_0/
              image_1/
              velodyne/
              calib.txt
          poses/
            00.txt

    Only metadata (paths and ground-truth transforms) is loaded; images and
    point clouds are not read into memory.

    :param root:
        Path to the KITTI odometry dataset root directory.
    :type root: Union[str, os.PathLike]

    :param seq:
        Sequence ID string (e.g. ``"00"``, ``"01"``).
    :type seq: str

    :param load_right:
        Whether to populate ``right_path`` pointing to ``image_1``. If ``False``,
        the field will always be ``None``.
    :type load_right: bool

    :param load_velodyne:
        Whether to populate ``velo_path`` pointing to ``velodyne`` scans. If
        ``False``, the field will always be ``None``.
    :type load_velodyne: bool

    :param with_poses:
        Whether to attempt to load ground-truth poses from ``poses/<seq>.txt``.
    :type with_poses: bool

    :param max_frames:
        Optional maximum number of frames to return. If ``None``, all available
        frames in the left camera directory are used.
    :type max_frames: Optional[int]

    :return:
        List of :class:`KittiOdomFrame` entries with timestamps, paths, and
        optionally ground-truth transforms.
    :rtype: List[KittiOdomFrame]
    """
    root_path = Path(root)
    seq_str = f"{int(seq):02d}"  # normalize "0" -> "00"

    seq_dir = root_path / "sequences" / seq_str
    left_dir = seq_dir / "image_0"
    right_dir = seq_dir / "image_1"
    velo_dir = seq_dir / "velodyne"

    if not left_dir.exists():
        raise FileNotFoundError(f"Left image directory not found: {left_dir}")

    # Determine frame indices from left camera images
    left_files = sorted(left_dir.glob("*.png"))
    if not left_files:
        raise FileNotFoundError(f"No left images found in {left_dir}")

    # Optional poses
    poses: List[Tuple[float, ...]] = []
    if with_poses:
        poses_path = root_path / "poses" / f"{seq_str}.txt"
        poses = _load_kitti_poses(poses_path)

    frames: List[KittiOdomFrame] = []

    for idx, left_path in enumerate(left_files):
        t = idx / 10.0  # KITTI odometry is ~10 Hz; good enough for indexing.

        right_path: Optional[str] = None
        velo_path: Optional[str] = None
        T_w_cam0: Optional[Tuple[float, ...]] = None

        if load_right:
            candidate = right_dir / left_path.name
            if candidate.exists():
                right_path = str(candidate)

        if load_velodyne:
            # KITTI velodyne uses "*.bin" with same numeric frame index
            stem = left_path.stem  # e.g. "000000"
            candidate = velo_dir / f"{stem}.bin"
            if candidate.exists():
                velo_path = str(candidate)

        if with_poses and idx < len(poses):
            T_w_cam0 = poses[idx]

        frames.append(
            KittiOdomFrame(
                seq=seq_str,
                idx=idx,
                t=t,
                left_path=str(left_path),
                right_path=right_path,
                velo_path=velo_path,
                T_w_cam0=T_w_cam0,
            )
        )

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames