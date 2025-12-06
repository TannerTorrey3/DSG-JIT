"""
High-level SLAM pipelines built on top of DSG-JIT.

This module provides small, composable helpers that glue together:

- WorldModel / SceneGraphWorld
- Sensors + SensorFusionManager
- FactorGraph + Gauss-Newton optimizer

The intent is that experiments (or ROS2 nodes) call into these
functions rather than reimplementing the same boilerplate in every file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp

from dsg_jit.core.types import GNConfig
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.optimization.solvers import gauss_newton_manifold
from dsg_jit.world.model import WorldModel
from dsg_jit.world.visualization import plot_factor_graph_3d
from dsg_jit.slam.manifold import build_manifold_metadata


# ---------------------------------------------------------------------------
# Basic pose-graph SLAM pipeline
# ---------------------------------------------------------------------------


@dataclass
class PoseGraphResult:
    """
    Result of a pose-graph SLAM solve.

    :param x_opt: Optimized stacked state vector.
    :type x_opt: jax.numpy.ndarray
    :param pose_ids: List of node ids corresponding to poses in the graph.
    :type pose_ids: list[int]
    :param landmark_ids: Optional list of landmark node ids, if present.
    :type landmark_ids: list[int] | None
    """
    x_opt: jnp.ndarray
    pose_ids: List[int]
    landmark_ids: List[int] | None = None


def run_pose_graph_slam(
    wm: WorldModel,
    cfg: GNConfig | None = None,
) -> PoseGraphResult:
    """
    Run Gauss-Newton on the pose/landmark graph in ``wm.fg``.

    This treats whatever is currently in the WorldModel's factor graph as
    the SLAM problem. It does not modify ``wm`` in-place; it returns the
    optimized stacked state and helper lists for extracting poses/landmarks.

    Typical usage:

    .. code-block:: python

        result = run_pose_graph_slam(wm)
        poses = [result.x_opt[index[nid]] for nid in result.pose_ids]

    :param wm:
        The world model containing a FactorGraph with SE(3) pose variables
        (and optionally landmark variables) plus factors (odom, priors,
        range/bearing, etc.).
    :type wm: world.model.WorldModel
    :param cfg:
        Configuration for the Gauss-Newton solver. If ``None``, a default
        ``GNConfig`` is used.
    :type cfg: core.types.GNConfig | None

    :return:
        A :class:`PoseGraphResult` containing the optimized stacked state
        vector and node-id lists for poses and landmarks.
    :rtype: PoseGraphResult
    """
    if cfg is None:
        cfg = GNConfig()

    fg: FactorGraph = wm.fg

    # Pack initial state
    x0, _ = wm.pack_state()
    residual_fn = wm.build_residual()

    # Manifold types: we already stored these per-variable in the graph.
    manifold_types = build_manifold_metadata(packed_state=wm.pack_state(),fg=fg)

    

    # Solve
    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        manifold_types,
        cfg,
    )

    # Build convenience lists for poses / landmarks.
    pose_ids: List[int] = []
    landmark_ids: List[int] = []

    for nid, v in wm.fg.variables.items():
        if v.manifold == "se3":
            pose_ids.append(nid)
        elif v.manifold == "R3":
            landmark_ids.append(nid)

    return PoseGraphResult(
        x_opt=x_opt,
        pose_ids=sorted(pose_ids),
        landmark_ids=sorted(landmark_ids) if landmark_ids else None,
    )


def update_worldmodel_from_solution(wm: WorldModel, result: PoseGraphResult) -> None:
    """
    Write optimized variables from a :class:`PoseGraphResult` back into ``wm``.

    This is a small helper so that downstream code (DSG construction,
    visualization, dataset export) can reflect the optimized state.

    :param wm:
        The world model whose factor-graph variables will be updated in-place.
    :type wm: world.model.WorldModel
    :param result:
        Output from :func:`run_pose_graph_slam`, containing the optimized
        stacked state vector.
    :type result: PoseGraphResult
    """
    fg = wm.fg
    x_opt = result.x_opt
    _, index = wm.pack_state()  # re-pack to get consistent slices

    for nid, sl in index.items():
        v = fg.variables[nid]
        v.value = x_opt[sl]


def pose_vectors_from_result(
    wm: WorldModel,
    result: PoseGraphResult,
) -> Dict[int, jnp.ndarray]:
    """
    Extract SE(3) pose vectors from an optimized solution.

    :param wm:
        The world model that owns the variables.
    :type wm: world.model.WorldModel
    :param result:
        Optimization result describing pose node ids and the stacked state.
    :type result: PoseGraphResult

    :return:
        Mapping from pose node id -> pose vector (6,).
    :rtype: dict[int, jax.numpy.ndarray]
    """
    x_opt = result.x_opt
    _, index = wm.pack_state()

    out: Dict[int, jnp.ndarray] = {}
    for nid in result.pose_ids:
        sl = index[nid]
        out[nid] = x_opt[sl]
    return out


def visualize_pose_graph_3d(
    wm: WorldModel,
    title: str | None = None,
) -> None:
    """
    Convenience helper to plot the current factor graph in 3D.

    This simply calls :func:`world.visualization.plot_factor_graph_3d`
    with the world's underlying :class:`FactorGraph`.

    :param wm:
        World model whose factor graph will be visualized.
    :type wm: world.model.WorldModel
    :param title:
        Optional plot title.
    :type title: str | None
    """
    plot_factor_graph_3d(wm.fg)