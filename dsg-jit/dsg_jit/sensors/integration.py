# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the MIT License.
"""
Helpers for wiring sensor fusion results into DSG-JIT world models.

This module provides small, stateless utilities that take fused pose
estimates from :mod:`sensors.fusion` and apply them to a
:class:`world.model.WorldModel` or dynamic scene graph.

By keeping this logic in a separate integration layer, we avoid coupling
the sensor stack directly to the optimization core, while still making it
very easy for users to build real-time or batch pipelines.
"""

from typing import Optional

import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.sensors.fusion import SensorFusionManager, FusedPoseEstimate


def apply_fused_pose_to_world(
    world: WorldModel,
    fusion: SensorFusionManager,
    agent_id: str,
    t: int,
    fused: Optional[FusedPoseEstimate] = None,
) -> int:
    """
    Apply a fused SE(3) pose estimate to the world model for a given agent.

    If a pose for ``(agent_id, t)`` already exists in the world, this function
    **updates** its value in-place. Otherwise, it **creates** a new agent pose
    variable via :meth:`world.model.WorldModel.add_agent_pose`.

    :param world: World model whose factor graph should be updated.
    :param fusion: Sensor fusion manager providing fused pose estimates.
    :param agent_id: String identifier for the agent (e.g. ``"robot0"``).
    :param t: Discrete timestep index for this update.
    :param fused: Optional fused pose estimate. If ``None``, this function
        calls :meth:`SensorFusionManager.get_latest_pose` internally.
    :returns: The integer node id (``int(NodeId)``) corresponding to the
        agent's pose variable at time ``t``.
    :raises ValueError: If no fused estimate is available.
    """
    if fused is None:
        fused = fusion.get_latest_pose()

    if fused is None:
        raise ValueError("No fused pose estimate available to apply to world.")

    pose_vec = jnp.asarray(fused.pose_se3)

    # Check if we already have a pose for this (agent, t).
    if agent_id in world.agent_pose_ids and t in world.agent_pose_ids[agent_id]:
        nid = world.agent_pose_ids[agent_id][t]
        world.fg.variables[nid].value = pose_vec
    else:
        nid = world.add_agent_pose(agent_id=agent_id, t=t, value=pose_vec)

    return int(nid)


def apply_trajectory_to_world(
    world: WorldModel,
    agent_id: str,
    trajectory: dict[int, jnp.ndarray],
) -> None:
    """
    Bulk-apply a discrete trajectory to the world model as agent poses.

    This is useful for offline pipelines, where you already have a fused
    trajectory (e.g. from a batch fusion run) and simply want to seed or
    refresh the world model with those states.

    :param world: World model to update.
    :param agent_id: String identifier for the agent.
    :param trajectory: Mapping from timestep ``t`` to 6D se(3) pose vectors
        in world coordinates.
    :returns: ``None``. All updates are applied in-place.
    """
    for t, pose_vec in sorted(trajectory.items()):
        pose_vec = jnp.asarray(pose_vec)
        if agent_id in world.agent_pose_ids and t in world.agent_pose_ids[agent_id]:
            nid = world.agent_pose_ids[agent_id][t]
            world.fg.variables[nid].value = pose_vec
        else:
            world.add_agent_pose(agent_id=agent_id, t=t, value=pose_vec)