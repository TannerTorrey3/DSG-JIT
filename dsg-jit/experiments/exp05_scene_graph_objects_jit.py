from __future__ import annotations
import jax.numpy as jnp

from world.scene_graph import SceneGraphWorld


def build_world() -> SceneGraphWorld:
    sg = SceneGraphWorld()

    # Poses
    pose0 = sg.add_pose_se3(jnp.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.005]))
    pose1 = sg.add_pose_se3(jnp.array([0.8,  0.1, -0.05, 0.05, 0.03, -0.01]))
    pose2 = sg.add_pose_se3(jnp.array([1.9, -0.2,  0.10, -0.02, 0.01, 0.02]))

    # Places / room
    place0 = sg.add_place1d(-0.2)
    place1 = sg.add_place1d(1.4)
    place2 = sg.add_place1d(2.1)
    room   = sg.add_room1d(5.0)

    # Objects
    obj0 = sg.add_object3d([0.0, 0.5, 0.0])
    obj1 = sg.add_object3d([1.3, -0.3, 0.2])
    obj2 = sg.add_object3d([2.4,  0.4, -0.1])

    # Factors
    sg.add_prior_pose_identity(pose0)
    sg.add_odom_se3_additive(pose0, pose1, dx=1.0)
    sg.add_odom_se3_additive(pose1, pose2, dx=1.0)

    sg.attach_pose_to_place_x(pose0, place0)
    sg.attach_pose_to_place_x(pose1, place1)
    sg.attach_pose_to_place_x(pose2, place2)
    sg.attach_pose_to_room_x(pose1, room)

    sg.attach_object_to_pose(pose0, obj0)
    sg.attach_object_to_pose(pose1, obj1)
    sg.attach_object_to_pose(pose2, obj2)

    return sg


def run_experiment():
    sg = build_world()

    print("\n=== INITIAL STATE (JIT) ===")
    for nid, var in sg.wm.fg.variables.items():
        print(f"{int(nid)}: {var.value}")

    # JIT-compiled GN solve
    sg.optimize(method="gn_jit", iters=40)

    print("\n=== OPTIMIZED STATE (JIT GN) ===")
    for nid, var in sg.wm.fg.variables.items():
        print(f"{int(nid)}: {var.value}")

    return sg


if __name__ == "__main__":
    run_experiment()