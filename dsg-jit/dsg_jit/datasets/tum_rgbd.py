from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class TumRgbdFrame:
    """
    Single RGB-D frame from a TUM RGB-D sequence.

    This dataclass stores only light-weight metadata: timestamps and relative
    file paths. Consumers are responsible for actually loading images/depth
    (e.g., via OpenCV or Pillow) if desired.

    :param t:
        Timestamp in seconds (as parsed from the TUM text files).
    :type t: float

    :param rgb_path:
        Relative or absolute path to the RGB image file corresponding to this
        frame. Typically something like ``"rgb/1341847980.722988.png"``.
    :type rgb_path: str

    :param depth_path:
        Optional path to the depth image associated with this frame. May be
        ``None`` if depth is not available or ``use_depth=False`` was passed
        to the loader.
    :type depth_path: Optional[str]

    :param pose_quat:
        Optional ground-truth pose as a 7-tuple
        ``(tx, ty, tz, qx, qy, qz, qw)`` in TUM convention. May be ``None``
        if ground truth is unavailable or alignment was disabled.
    :type pose_quat: Optional[Tuple[float, float, float, float, float, float, float]]
    """

    t: float
    rgb_path: str
    depth_path: Optional[str] = None
    pose_quat: Optional[Tuple[float, float, float, float, float, float, float]] = None


def _parse_tum_list_file(path: Path) -> List[Tuple[float, str]]:
    """
    Parse a TUM-style list file (e.g. ``rgb.txt`` or ``depth.txt``).

    Each non-comment line is expected to have:

        ``<timestamp> <relative_path>``

    :param path:
        Path to the TUM list file.
    :type path: pathlib.Path

    :return:
        A list of ``(timestamp, relative_path)`` pairs sorted by time.
    :rtype: List[Tuple[float, str]]
    """
    items: List[Tuple[float, str]] = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            rel = parts[1]
            items.append((t, rel))

    items.sort(key=lambda x: x[0])
    return items


def _parse_tum_groundtruth(path: Path) -> List[Tuple[float, Tuple[float, float, float, float, float, float, float]]]:
    """
    Parse a TUM groundtruth file (``groundtruth.txt``).

    Lines follow the standard format:

        ``timestamp tx ty tz qx qy qz qw``

    :param path:
        Path to ``groundtruth.txt``.
    :type path: pathlib.Path

    :return:
        A list of ``(timestamp, (tx, ty, tz, qx, qy, qz, qw))`` tuples sorted by
        time.
    :rtype: List[Tuple[float, Tuple[float, float, float, float, float, float, float]]]
    """
    poses: List[Tuple[float, Tuple[float, float, float, float, float, float, float]]] = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            t = float(parts[0])
            tx, ty, tz = (float(parts[1]), float(parts[2]), float(parts[3]))
            qx, qy, qz, qw = (float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]))
            poses.append((t, (tx, ty, tz, qx, qy, qz, qw)))

    poses.sort(key=lambda x: x[0])
    return poses


def _associate_by_timestamp(
    primary: Sequence[Tuple[float, str]],
    secondary: Sequence[Tuple[float, str]],
    max_diff: float = 0.02,
) -> Dict[int, Optional[int]]:
    """
    Greedy association of two timestamped streams by nearest neighbor.

    Used to match RGB and depth streams, or frames to ground truth. For each
    index in ``primary``, we find the closest timestamp in ``secondary`` within
    a given maximum difference.

    :param primary:
        Primary sequence of ``(timestamp, path)`` pairs.
    :type primary: Sequence[Tuple[float, str]]

    :param secondary:
        Secondary sequence of ``(timestamp, path)`` pairs.
    :type secondary: Sequence[Tuple[float, str]]

    :param max_diff:
        Maximum allowed time difference in seconds. If the closest match
        exceeds this, the association will be ``None``.
    :type max_diff: float

    :return:
        A dictionary mapping indices in ``primary`` to indices in
        ``secondary`` or ``None`` if no suitable match was found.
    :rtype: Dict[int, Optional[int]]
    """
    assoc: Dict[int, Optional[int]] = {}

    j = 0
    for i, (t0, _) in enumerate(primary):
        # Advance j while secondary[j] is before t0
        while j + 1 < len(secondary) and secondary[j + 1][0] <= t0:
            j += 1
        # Check nearest of j or j+1
        best_j = None
        best_dt = float("inf")
        for jj in (j, j + 1):
            if 0 <= jj < len(secondary):
                dt = abs(secondary[jj][0] - t0)
                if dt < best_dt:
                    best_dt = dt
                    best_j = jj
        assoc[i] = best_j if (best_j is not None and best_dt <= max_diff) else None

    return assoc


def load_tum_rgbd_sequence(
    root: str | os.PathLike,
    use_depth: bool = True,
    use_groundtruth: bool = False,
    max_frames: Optional[int] = None,
    max_time_diff: float = 0.02,
) -> List[TumRgbdFrame]:
    """
    Load a TUM RGB-D sequence directory into a list of frames.

    The directory is expected to contain standard TUM files such as
    ``rgb.txt``, ``depth.txt``, and optionally ``groundtruth.txt``. This
    loader parses metadata and returns a list of :class:`TumRgbdFrame`
    instances, but does not actually load images or depth maps.

    :param root:
        Path to the TUM sequence root directory.
    :type root: Union[str, os.PathLike]

    :param use_depth:
        Whether to attempt to associate depth frames from ``depth.txt`` with
        each RGB frame.
    :type use_depth: bool

    :param use_groundtruth:
        Whether to attempt to associate ground-truth poses from
        ``groundtruth.txt`` with each RGB frame.
    :type use_groundtruth: bool

    :param max_frames:
        Optional maximum number of frames to return. If ``None``, the full
        sequence is loaded.
    :type max_frames: Optional[int]

    :param max_time_diff:
        Maximum allowed absolute difference in timestamps (seconds) when
        associating RGB with depth and ground truth.
    :type max_time_diff: float

    :return:
        A list of TUM RGB-D frames with timestamps, file paths, and optionally
        ground-truth poses.
    :rtype: List[TumRgbdFrame]
    """
    root_path = Path(root)

    rgb_entries = _parse_tum_list_file(root_path / "rgb.txt")

    depth_entries: List[Tuple[float, str]] = []
    if use_depth and (root_path / "depth.txt").exists():
        depth_entries = _parse_tum_list_file(root_path / "depth.txt")

    gt_entries: List[Tuple[float, Tuple[float, float, float, float, float, float, float]]] = []
    if use_groundtruth and (root_path / "groundtruth.txt").exists():
        gt_entries = _parse_tum_groundtruth(root_path / "groundtruth.txt")

    depth_assoc: Dict[int, Optional[int]] = {}
    gt_assoc: Dict[int, Optional[int]] = {}

    if depth_entries:
        depth_assoc = _associate_by_timestamp(rgb_entries, depth_entries, max_diff=max_time_diff)

    if gt_entries:
        # Convert to (t, dummy_path) to reuse association logic
        gt_ts_paths = [(t, "") for (t, _pose) in gt_entries]
        gt_assoc = _associate_by_timestamp(rgb_entries, gt_ts_paths, max_diff=max_time_diff)

    frames: List[TumRgbdFrame] = []

    for i, (t_rgb, rgb_rel) in enumerate(rgb_entries):
        depth_rel: Optional[str] = None
        pose_quat: Optional[Tuple[float, float, float, float, float, float, float]] = None

        if depth_entries and i in depth_assoc and depth_assoc[i] is not None:
            depth_rel = depth_entries[depth_assoc[i]][1]

        if gt_entries and i in gt_assoc and gt_assoc[i] is not None:
            pose_quat = gt_entries[gt_assoc[i]][1]

        frames.append(
            TumRgbdFrame(
                t=t_rgb,
                rgb_path=str(root_path / rgb_rel),
                depth_path=str(root_path / depth_rel) if depth_rel is not None else None,
                pose_quat=pose_quat,
            )
        )

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames