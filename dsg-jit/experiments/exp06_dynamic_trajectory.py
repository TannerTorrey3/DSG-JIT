from __future__ import annotations
import jax.numpy as jnp

from world.scene_graph import SceneGraphWorld


def run_experiment():
    sg = SceneGraphWorld()
    sg.noise.odom_se3_sigma = 0.05    # strong odom
    sg.noise.smooth_pose_sigma = 2.0  # very weak smoothness
    agent = "robot"

    # --- Poses over time (t=0,1,2) with noise ---
    p0 = sg.add_agent_pose_se3(agent, 0, jnp.array([0.1, -0.1, 0.0, 0.01, -0.02, 0.0]))
    p1 = sg.add_agent_pose_se3(agent, 1, jnp.array([1.1,  0.1, 0.0, 0.05, 0.03, 0.0]))
    p2 = sg.add_agent_pose_se3(agent, 2, jnp.array([2.0, -0.2, 0.0, -0.02, 0.01, 0.0]))

    # --- Static places (landmarks) ---
    place0 = sg.add_place1d(-0.1)
    place1 = sg.add_place1d(1.2)
    place2 = sg.add_place1d(2.2)

    # --- Prior on initial pose ---
    sg.add_prior_pose_identity(p0)

    # --- Odom along x (roughly 1m each step) ---
    sg.add_odom_se3_additive(p0, p1, dx=1.0)
    sg.add_odom_se3_additive(p1, p2, dx=1.0)

    # --- Temporal smoothness between poses ---
    sg.add_temporal_smoothness(p0, p1)
    sg.add_temporal_smoothness(p1, p2)

    # --- Attach places to poses (like before) ---
    sg.attach_pose_to_place_x(p0, place0)
    sg.attach_pose_to_place_x(p1, place1)
    sg.attach_pose_to_place_x(p2, place2)

    print("\n=== INITIAL DYNAMIC STATE ===")
    for nid, var in sg.wm.fg.variables.items():
        print(f"{int(nid)}: {var.value}")

    # Use JIT GN for this dynamic graph
    sg.optimize(method="gn_jit", iters=40)

    print("\n=== OPTIMIZED DYNAMIC STATE ===")
    for nid, var in sg.wm.fg.variables.items():
        print(f"{int(nid)}: {var.value}")

    # Quick summary
    print("\n--- Trajectory summary ---")
    print("p0:", sg.get_pose(p0))
    print("p1:", sg.get_pose(p1))
    print("p2:", sg.get_pose(p2))
    print("place0:", sg.get_place(place0))
    print("place1:", sg.get_place(place1))
    print("place2:", sg.get_place(place2))


if __name__ == "__main__":
    run_experiment()