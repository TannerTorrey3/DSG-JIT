"""
Experiment 19 — Dynamic Scene Graph Demo (Multi–Agent, Multi–Room, 3D Render)

This experiment demonstrates:
- Building a multi-agent dynamic scene graph
- Rooms, places, objects with hierarchical relations
- Agent trajectories with odometry factors
- Place attachments (agents -> places) including secondary places
- Optimizing the full factor graph on SE3 + Euclidean manifolds
- Rendering a full 3D scene-graph with visual cues and agent trajectories

Requires:
    - world.scene_graph.SceneGraphWorld
    - world.dynamic_scene_graph.DynamicSceneGraph
    - slam.manifold.build_manifold_metadata
    - optimization.solvers.gauss_newton_manifold
    - world.visualization.plot_scenegraph_3d
"""

import jax.numpy as jnp

from dsg_jit.world.scene_graph import SceneGraphWorld
from dsg_jit.world.dynamic_scene_graph import DynamicSceneGraph
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.world.visualization import plot_scenegraph_3d, plot_dynamic_trajectories_3d


def build_scene_graph():
    """
    Build a small dynamic scene graph with:
      - 3 rooms in a line (A, B, C)
      - 2 places per room (A1,A2,B1,B2,C1,C2)
      - 3 objects anchored to places
      - 2 agents moving through the rooms with odometry
      - Place attachments grounding agent poses to places

    Returns:
        sg:  SceneGraphWorld (static world + factor graph)
        dsg: DynamicSceneGraph (agent trajectories, temporal layer)
        objects: dict[name -> NodeId] for the 3 objects
    """
    sg = SceneGraphWorld()
    dsg = DynamicSceneGraph(sg)

    # ------------------------------
    # 1) ROOMS
    # ------------------------------
    roomA = sg.add_room("room_A", center=jnp.array([0.0, 0.0, 0.0]))
    roomB = sg.add_room("room_B", center=jnp.array([5.0, 0.0, 0.0]))
    roomC = sg.add_room("room_C", center=jnp.array([10.0, 0.0, 0.0]))

    # ------------------------------
    # 2) PLACES (3D, relative to room centers)
    # NOTE: updated API: add_place3d(room_node_id, rel_xyz)
    # ------------------------------
    placeA1 = sg.add_place3d(roomA, jnp.array([0.5, 0.5, 0.0]))
    placeA2 = sg.add_place3d(roomA, jnp.array([-0.5, -0.3, 0.0]))

    placeB1 = sg.add_place3d(roomB, jnp.array([0.5, 0.2, 0.0]))
    placeB2 = sg.add_place3d(roomB, jnp.array([-0.6, -0.3, 0.0]))

    placeC1 = sg.add_place3d(roomC, jnp.array([0.4, 0.1, 0.0]))
    placeC2 = sg.add_place3d(roomC, jnp.array([-0.4, -0.2, 0.0]))

    # ------------------------------
    # 3) OBJECTS (anchored to places)
    # ------------------------------
    # If you want named objects:
    obj1 = sg.add_named_object3d("chair_1", jnp.array([0.8, 0.2, 0.0]))
    obj2 = sg.add_named_object3d("chair_2", jnp.array([1.2, 0.2, 0.0]))
    obj3 = sg.add_named_object3d("plant_1", jnp.array([2.0, -0.3, 0.0]))

    # ------------------------------
    # 4) MULTI-AGENT TRAJECTORIES (Dynamic layer)
    # ------------------------------
    # Agent 0: travels from room A toward room B
    dsg.add_agent_pose("robot0", t=0, pose_se3=jnp.array([0.0, 0, 0, 0, 0, 0]))
    dsg.add_agent_pose("robot0", t=1, pose_se3=jnp.array([1.0, 0, 0, 0, 0, 0]))
    dsg.add_agent_pose("robot0", t=2, pose_se3=jnp.array([2.0, 0, 0, 0, 0, 0]))
    dsg.add_agent_pose("robot0", t=3, pose_se3=jnp.array([3.0, 0, 0, 0, 0, 0]))

    dsg.add_odom_tx("robot0", 0, 1, dx=1.0)
    dsg.add_odom_tx("robot0", 1, 2, dx=1.0)
    dsg.add_odom_tx("robot0", 2, 3, dx=1.0)

    # Agent 1: starts in room B, moves toward room C
    dsg.add_agent_pose("robot1", t=0, pose_se3=jnp.array([5.0, 0, 0, 0, 0, 0]))
    dsg.add_agent_pose("robot1", t=1, pose_se3=jnp.array([6.0, 0, 0, 0, 0, 0]))
    dsg.add_agent_pose("robot1", t=2, pose_se3=jnp.array([7.0, 0, 0, 0, 0, 0]))

    dsg.add_odom_tx("robot1", 0, 1, dx=1.0)
    dsg.add_odom_tx("robot1", 1, 2, dx=1.0)

    # ------------------------------
    # 5) Place attachments for grounding
    #    (agent poses -> places)
    # ------------------------------
    # robot0 is firmly in room A at t=0 and room B vicinity at t=3
    sg.add_place_attachment(dsg.world.pose_trajectory[("robot0", 0)], placeA1)
    sg.add_place_attachment(dsg.world.pose_trajectory[("robot0", 3)], placeB2)

    # robot1 starts in room B and ends near room C
    sg.add_place_attachment(dsg.world.pose_trajectory[("robot1", 0)], placeB1)
    sg.add_place_attachment(dsg.world.pose_trajectory[("robot1", 2)], placeC2)

    # Extra grounding: robot0 passes near placeA2 at t=1
    sg.add_place_attachment(dsg.world.pose_trajectory[("robot0", 1)], placeA2)

    objects = {"chair_1": obj1, "table_1": obj2, "lamp_1": obj3}
    return sg, dsg, objects


def main():
    sg, dsg, objects = build_scene_graph()

    # Access the WorldModel and pack the full state
    wm = sg.wm
    x0, index = wm.pack_state()

    # Build manifold metadata using the WorldModel's packed state and underlying factor graph
    packed_state = (x0, index)
    block_slices, manifold_types = build_manifold_metadata(
        packed_state=packed_state,
        fg=wm.fg,
    )

    # Build residual function from the WorldModel-level residual registry
    residual_fn = wm.build_residual()

    # Solve optimization on the SE3+Euclidean manifold
    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        block_slices,
        manifold_types,
        cfg,
    )

    # Render a full 3D scene graph with temporal agent trajectories
    plot_scenegraph_3d(
        sg,
        x_opt,
        index,
        title="Experiment 19 — Dynamic 3D Scene Graph",
        dsg=dsg,
    )

    plot_dynamic_trajectories_3d(
        dsg,
        x_opt,
        index,
        title="Experiment 19 — Dynamic 3D Scene Graph (time-colored)",
        color_by_time=True,
    )

    # Print trajectories per agent
    for agent in dsg.agents:
        traj = dsg.get_agent_trajectory(agent, x_opt, index)
        print(f"\n=== Agent {agent} trajectory (optimized) ===")
        print(traj)

    # Print final object positions
    print("\n=== Object positions (optimized) ===")
    for name, nid in objects.items():
        start, dim = index[nid]
        val = x_opt[start : start + dim]
        print(f"{name}: {val}")


if __name__ == "__main__":
    main()