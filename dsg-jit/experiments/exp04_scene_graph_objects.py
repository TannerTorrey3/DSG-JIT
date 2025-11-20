from __future__ import annotations
import jax.numpy as jnp

from world.scene_graph import SceneGraphWorld


def run_experiment():
    sg = SceneGraphWorld()

    # --- Poses (same pattern as before) ---
    pose0 = sg.add_pose_se3(jnp.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.005]))
    pose1 = sg.add_pose_se3(jnp.array([0.8,  0.1, -0.05, 0.05, 0.03, -0.01]))
    pose2 = sg.add_pose_se3(jnp.array([1.9, -0.2,  0.10, -0.02, 0.01, 0.02]))

    # --- Places / room (1D along x) ---
    place0 = sg.add_place1d(-0.2)
    place1 = sg.add_place1d(1.4)
    place2 = sg.add_place1d(2.1)
    room   = sg.add_room1d(5.0)

    # --- Objects (3D) anchored to the three poses ---
    # e.g. each object is at (tx, 0, 0) in world, via constraint
    obj0 = sg.add_object3d([0.0, 0.5, 0.0])  # initial guesses off
    obj1 = sg.add_object3d([1.3, -0.3, 0.2])
    obj2 = sg.add_object3d([2.4,  0.4, -0.1])

    # --- Factors: prior + odom ---
    sg.add_prior_pose_identity(pose0)

    sg.add_odom_se3_additive(pose0, pose1, dx=1.0)
    sg.add_odom_se3_additive(pose1, pose2, dx=1.0)

    # --- Places attached to x of poses ---
    sg.attach_pose_to_place_x(pose0, place0)
    sg.attach_pose_to_place_x(pose1, place1)
    sg.attach_pose_to_place_x(pose2, place2)

    # --- Room attached to pose1 x ---
    sg.attach_pose_to_room_x(pose1, room)

    # --- Objects attached to pose translations (zero offset for now) ---
    sg.attach_object_to_pose(pose0, obj0, offset=(0.0, 0.0, 0.0))
    sg.attach_object_to_pose(pose1, obj1, offset=(0.0, 0.0, 0.0))
    sg.attach_object_to_pose(pose2, obj2, offset=(0.0, 0.0, 0.0))

    # --- Print initial ---
    print("\n=== INITIAL STATE (SceneGraph+Objects) ===")
    for nid, var in sg.wm.fg.variables.items():
        print(f"{int(nid)}: {var.value}")

    # --- Optimize ---
    sg.optimize(method="gn", iters=40)

    # --- Print optimized ---
    print("\n=== OPTIMIZED STATE (SceneGraph+Objects) ===")
    for nid, var in sg.wm.fg.variables.items():
        print(f"{int(nid)}: {var.value}")

    # Quick summary for sanity
    print("\n--- Summaries ---")
    print("pose0:", sg.get_pose(pose0))
    print("pose1:", sg.get_pose(pose1))
    print("pose2:", sg.get_pose(pose2))

    print("place0:", sg.get_place(place0))
    print("place1:", sg.get_place(place1))
    print("place2:", sg.get_place(place2))
    print("room:",   sg.get_place(room))

    print("obj0:", sg.get_object3d(obj0))
    print("obj1:", sg.get_object3d(obj1))
    print("obj2:", sg.get_object3d(obj2))


if __name__ == "__main__":
    run_experiment()