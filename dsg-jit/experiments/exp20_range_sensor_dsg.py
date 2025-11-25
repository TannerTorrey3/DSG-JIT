
import jax.numpy as jnp

from world.scene_graph import SceneGraphWorld
from world.dynamic_scene_graph import DynamicSceneGraph
from world.visualization import plot_factor_graph_3d
from optimization.solvers import gauss_newton_manifold, GNConfig
from slam.manifold import build_manifold_metadata


def build_range_dsg(num_steps: int = 5):
    """
    Build a simple dynamic scene graph with:
      - One robot 'robot0'
      - A short pose chain along +x
      - A single place at a fixed location
      - Range measurements from each pose to that place.
    """
    sg = SceneGraphWorld()
    dsg = DynamicSceneGraph(world=sg)

    # --- Create static structure: one room + one place ---
    # room1d just needs an x coordinate, no name
    roomA = sg.add_room1d(x=jnp.array([0.0], dtype=jnp.float32))

    place_center = jnp.array([2.0, 1.0, 0.0], dtype=jnp.float32)
    placeA = sg.add_place3d("place_A", xyz=place_center)
    sg.add_room_place_edge(roomA, placeA)

    # --- Create a robot agent and pose chain ---
    agent = "robot0"
    dsg.add_agent(agent)

    # Pose 0 at origin, then move +1m each step along x
    for t in range(num_steps):
        x = float(t)  # ground-truth x
        pose_vec = jnp.array([x, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        dsg.add_agent_pose(agent, t, pose_vec)

    # Odometry edges between consecutive poses
    for t in range(num_steps - 1):
        dsg.add_odom_tx(agent, t, t + 1, dx=1.0)

    # --- Add range measurements to placeA from each pose ---
    # IMPORTANT: we don't reach into sg.wm / fg here; we know the ground-truth
    # pose positions analytically (x = t, y = 0, z = 0).
    for t in range(num_steps):
        x = float(t)
        pose_pos = jnp.array([x, 0.0, 0.0], dtype=jnp.float32)
        true_range = float(jnp.linalg.norm(place_center - pose_pos))

        # Add small synthetic noise that varies with t
        noisy_range = true_range + 0.05 * (2.0 * (t / max(1, num_steps - 1)) - 1.0)

        dsg.add_range_obs(
            agent=agent,
            t=t,
            target_nid=placeA,
            measured_range=noisy_range,
            sigma=0.1,
        )

    return sg, dsg, placeA


def optimize_world(sg: SceneGraphWorld):
    """
    Run Gauss-Newton manifold optimization on the underlying factor graph.
    """
    wm = sg.wm           # WorldModel
    fg = wm.fg           # Underlying FactorGraph

    x0, index = fg.pack_state()
    block_slices, manifold_types = build_manifold_metadata(fg)
    residual_fn = fg.build_residual_function()

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        block_slices,
        manifold_types,
        cfg,
    )

    values = fg.unpack_state(x_opt, index)

    # Update the world variables in-place for visualization and printing
    for nid, v in values.items():
        fg.variables[nid].value = v

    return values


def main():
    sg, dsg, placeA = build_range_dsg(num_steps=6)
    values = optimize_world(sg)

    print("=== Optimized poses and place (range sensor DSG) ===")
    # Print poses for robot0
    for (agent, t), nid in sorted(
        dsg.world.pose_trajectory.items(), key=lambda kv: kv[0][1]
    ):
        pose = values[nid]
        print(f"pose[{agent}, t={t}]: {pose}")

    place_val = values[placeA]
    print(f"\nOptimized place_A: {place_val}")

    # Visualize factor graph in 3D (poses and the place).
    # plot_factor_graph_3d expects a FactorGraph, so we pass sg.wm.fg.
    plot_factor_graph_3d(sg.wm.fg)


if __name__ == "__main__":
    main()