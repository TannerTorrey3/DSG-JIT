# Copyright (c) 2025 Tanner Kocher
# This file is part of DSG-JIT, released under the MIT License.

"""Dynamic scene-graph utilities built on top of :mod:`world.scene_graph`.

This module provides a lightweight wrapper around :class:`world.scene_graph.SceneGraphWorld`
that makes *dynamic* (time-indexed) scene graphs easier to build and reason about.

The goal is to keep all of the optimization and factor-graph logic in the existing
engine, while giving users a small, ergonomic API for working with trajectories and
other time-varying entities.

Design goals
------------
- **Don't duplicate state**: the underlying :class:`SceneGraphWorld` and
  :class:`WorldModel` remain the single source of truth.
- **Time-aware helpers**: convenience functions for adding agent trajectories,
  querying poses across time, and wiring odometry factors between consecutive
  poses.
- **Engine-friendly**: everything ultimately calls into existing
  ``SceneGraphWorld`` methods, so this module is safe to ignore if you want to
  use the lower-level API directly.

Typical usage
-------------

.. code-block:: python

    from world.scene_graph import SceneGraphWorld
    from world.dynamic_scene_graph import DynamicSceneGraph
    import jax.numpy as jnp

    sg = SceneGraphWorld()
    dsg = DynamicSceneGraph(sg)

    agent = "robot0"

    # Add a short trajectory
    dsg.add_agent_pose(agent, t=0, pose_se3=jnp.zeros(6))
    dsg.add_agent_pose(agent, t=1, pose_se3=jnp.array([1.0, 0, 0, 0, 0, 0]))

    # Connect poses with odometry in the x-direction
    dsg.add_odom_tx(agent, t0=0, t1=1, dx=1.0, weight=10.0)

    # Later, after optimization, you can recover the optimized trajectory with
    # dsg.get_agent_trajectory(...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Mapping, Tuple

import jax.numpy as jnp

from dsg_jit.core.types import NodeId
from dsg_jit.world.scene_graph import SceneGraphWorld


#: Key used for time-indexed entities (e.g., agent poses).
TimeKey = Tuple[Hashable, int]


@dataclass
class DynamicSceneGraph:
    """Helper for building dynamic (time-indexed) scene graphs.

    This class is a thin façade over :class:`world.scene_graph.SceneGraphWorld`.
    It does **not** introduce new optimization logic or state; instead it
    organizes common patterns for working with agent trajectories and other
    dynamic structures.

    Parameters
    ----------
    world:
        The underlying :class:`SceneGraphWorld` instance. All variables and
        factors are ultimately added to ``world.wm``.
    agents:
        Optional set of agent identifiers. You usually don't need to pass
        this explicitly; agents are registered lazily when you call
        :meth:`add_agent` or :meth:`add_agent_pose`.
    """

    world: SceneGraphWorld
    agents: set[Hashable] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Agent / trajectory management
    # ------------------------------------------------------------------

    def add_agent(self, agent_id: Hashable) -> Hashable:
        """Register an agent identifier.

        This does not create any variables by itself; it simply tracks the
        identifier so you can discover which agents exist in the graph.

        :param agent_id: Hashable identifier for the agent (for example, ``"robot0"``).
        :type agent_id: Hashable
        :return: The same ``agent_id`` that was passed in, for convenience.
        :rtype: Hashable
        """

        self.agents.add(agent_id)
        return agent_id

    # Note: ``SceneGraphWorld`` already maintains ``pose_trajectory`` as a
    # mapping from ``(agent, t)`` to ``NodeId``. We simply wrap that here.

    def add_agent_pose(self, agent_id: Hashable, t: int, pose_se3: jnp.ndarray) -> NodeId:
        """Add an SE(3) pose variable for a given agent and time.

        This delegates directly to :meth:`SceneGraphWorld.add_agent_pose_se3`
        and records the agent identifier in :attr:`agents`.

        :param agent_id: Identifier for the agent.
        :type agent_id: Hashable
        :param t: Discrete time index (for example, frame or step index).
        :type t: int
        :param pose_se3: 6D se(3) vector ``[tx, ty, tz, rx, ry, rz]``.
        :type pose_se3: jax.numpy.ndarray
        :return: The node identifier of the newly created pose variable.
        :rtype: NodeId
        """

        self.agents.add(agent_id)
        return self.world.add_agent_pose_se3(agent_id, t, pose_se3)


    def add_agent_trajectory(
        self,
        agent_id: Hashable,
        poses_se3: Iterable[jnp.ndarray],
        start_t: int = 0,
        add_odom: bool = True,
        default_dx: float | None = None,
        weight: float = 1.0,
    ) -> List[NodeId]:
        """Add a contiguous trajectory for one agent and optionally wire odometry.

        This is a convenience helper that repeatedly calls :meth:`add_agent_pose`
        and, if ``add_odom`` is ``True``, :meth:`add_odom_tx` between consecutive
        time steps.

        :param agent_id: Identifier for the agent.
        :type agent_id: Hashable
        :param poses_se3: Iterable of se(3) pose vectors. The first element is
            placed at ``t = start_t``, the next at ``t = start_t + 1``, and so on.
        :type poses_se3: Iterable[jax.numpy.ndarray]
        :param start_t: Time index to use for the first pose.
        :type start_t: int
        :param add_odom: If ``True``, automatically connect consecutive poses with
            a 1D odometry factor along ``x`` via :meth:`add_odom_tx`.
        :type add_odom: bool
        :param default_dx: If not ``None``, use this value as the expected
            displacement in ``x`` between each consecutive pair of poses. If
            ``None`` and ``add_odom`` is ``True``, the displacement is inferred as
            ``poses_se3[k+1][0] - poses_se3[k][0]``.
        :type default_dx: float | None
        :param weight: Scalar weight used for each odometry factor when
            ``add_odom`` is enabled.
        :type weight: float
        :return: Node identifiers of all created pose variables, in temporal order.
        :rtype: list[NodeId]
        """

        node_ids: List[NodeId] = []
        t = start_t
        prev_t: int | None = None
        prev_pose: jnp.ndarray | None = None

        for pose in poses_se3:
            nid = self.add_agent_pose(agent_id, t, pose)
            node_ids.append(nid)

            if add_odom and prev_t is not None:
                if default_dx is not None:
                    dx = float(default_dx)
                else:
                    # Infer displacement along x from the raw pose guesses
                    dx = float(pose[0] - prev_pose[0])
                self.add_odom_tx(agent_id, prev_t, t, dx=dx, weight=weight)

            prev_t = t
            prev_pose = pose
            t += 1

        return node_ids
    
    def add_range_obs(
        self,
        agent: str,
        t: int,
        target_nid: int,
        measured_range: float,
        sigma: float | None = 0.1,
    ) -> None:
        """
        Add a range measurement from an agent's pose at time t to a target node.

        This wraps :meth:`SceneGraphWorld.add_range_measurement`, using the
        pose node from ``pose_trajectory[(agent, t)]``.

        :param agent: Agent key, e.g. ``"robot0"``.
        :param t: Integer time step.
        :param target_nid: NodeId of the target (place3d, voxel_cell, object3d, etc.).
        :param measured_range: Observed distance.
        :param sigma: Optional measurement noise standard deviation.
        """
        pose_nid = self.world.pose_trajectory[(agent, t)]
        self.world.add_range_measurement(
            pose_nid=pose_nid,
            target_nid=target_nid,
            measured_range=measured_range,
            sigma=sigma,
        )

    # ------------------------------------------------------------------
    # Odometry helpers
    # ------------------------------------------------------------------

    def add_odom_tx(
        self,
        agent_id: Hashable,
        t0: int,
        t1: int,
        dx: float,
        weight: float = 1.0,
    ) -> None:
        """Connect two consecutive poses with a 1D odometry factor in ``x``.

        This is a convenience wrapper around
        :meth:`SceneGraphWorld.add_odom_se3_additive`, which interprets ``dx`` as a
        translation along the ``x`` axis and assumes identity rotation.

        :param agent_id: Agent identifier.
        :type agent_id: Hashable
        :param t0: Time index of the *from* pose.
        :type t0: int
        :param t1: Time index of the *to* pose.
        :type t1: int
        :param dx: Expected displacement in ``x`` from pose ``(agent_id, t0)`` to
            pose ``(agent_id, t1)``.
        :type dx: float
        :param weight: Scalar weight applied to the odometry residual.
        :type weight: float
        :return: ``None``.
        :rtype: None
        """

        pose_i = self.world.pose_trajectory[(agent_id, t0)]
        pose_j = self.world.pose_trajectory[(agent_id, t1)]
        self.world.add_odom_se3_additive(pose_i, pose_j, dx=dx, sigma=weight)

    # ------------------------------------------------------------------
    # Trajectory queries
    # ------------------------------------------------------------------

    def get_agent_times(self, agent_id: Hashable) -> List[int]:
        """Return the sorted list of time indices for which this agent has poses.

        :param agent_id: Agent identifier.
        :type agent_id: Hashable
        :return: Sorted time indices where ``(agent_id, t)`` exists in
            :attr:`SceneGraphWorld.pose_trajectory`.
        :rtype: list[int]
        """

        times = [t for (a, t) in self.world.pose_trajectory.keys() if a == agent_id]
        return sorted(times)

    def get_agent_pose_nodes(self, agent_id: Hashable) -> List[NodeId]:
        """Return the sequence of pose node IDs for an agent, ordered by time.

        :param agent_id: Agent identifier.
        :type agent_id: Hashable
        :return: Pose node IDs for the given agent, sorted by their time index.
        :rtype: list[NodeId]
        """

        times = self.get_agent_times(agent_id)
        return [self.world.pose_trajectory[(agent_id, t)] for t in times]

    def get_agent_trajectory(
        self,
        agent_id: Hashable,
        x_opt: jnp.ndarray,
        index: Mapping[NodeId, Tuple[int, int]],
    ) -> jnp.ndarray:
        """Extract an optimized trajectory for one agent from a flat state vector.

        :param agent_id: Agent identifier.
        :type agent_id: Hashable
        :param x_opt: Optimized flat state vector produced by one of the
            Gauss–Newton solvers, such as
            :func:`optimization.solvers.gauss_newton_manifold`.
        :type x_opt: jax.numpy.ndarray
        :param index: Mapping from :class:`NodeId` to ``(start, dim)`` tuples as
            returned by :meth:`world.model.WorldModel.pack_state`.
        :type index: Mapping[NodeId, Tuple[int, int]]
        :return: Array of shape ``(T, 6)`` containing the se(3) vectors for each
            time step in chronological order.
        :rtype: jax.numpy.ndarray
        """

        nodes = self.get_agent_pose_nodes(agent_id)
        traj = []
        for nid in nodes:
            start, dim = index[nid]
            traj.append(x_opt[start : start + dim])
        return jnp.stack(traj, axis=0)

    def get_all_trajectories(
        self,
        x_opt: jnp.ndarray,
        index: Mapping[NodeId, Tuple[int, int]],
    ) -> Dict[Hashable, jnp.ndarray]:
        """Extract trajectories for all known agents from an optimized state.

        This is a convenience wrapper around :meth:`get_agent_trajectory` that
        iterates over :attr:`agents` and returns a mapping from agent identifier
        to a ``(T_i, 6)`` array of se(3) poses.

        :param x_opt: Optimized flat state vector produced by one of the
            Gauss–Newton solvers.
        :type x_opt: jax.numpy.ndarray
        :param index: Mapping from :class:`NodeId` to ``(start, dim)`` tuples as
            returned by :meth:`world.model.WorldModel.pack_state`.
        :type index: Mapping[NodeId, Tuple[int, int]]
        :return: Dictionary mapping each agent identifier to its optimized
            trajectory as an array of shape ``(T_i, 6)``.
        :rtype: dict[Hashable, jax.numpy.ndarray]
        """

        trajectories: Dict[Hashable, jnp.ndarray] = {}
        for agent_id in self.agents:
            trajectories[agent_id] = self.get_agent_trajectory(agent_id, x_opt, index)
        return trajectories

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def all_pose_time_keys(self) -> List[TimeKey]:
        """Return all ``(agent, t)`` keys present in the underlying world.

        This is mainly useful for debugging or for building custom visualizations
        and exporters.

        :return: All time-index keys found in
            :attr:`SceneGraphWorld.pose_trajectory`.
        :rtype: list[TimeKey]
        """

        return list(self.world.pose_trajectory.keys())

    def __repr__(self) -> str:  # pragma: no cover - trivial representation
        agents = sorted(map(str, self.agents))
        return f"DynamicSceneGraph(agents={agents}, num_poses={len(self.world.pose_trajectory)})"
