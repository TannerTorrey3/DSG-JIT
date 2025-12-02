from __future__ import annotations

import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual
from dsg_jit.scene_graph.relations import room_centroid_residual


def setup_mini_world() -> WorldModel:
    """
    Build a tiny synthetic world:

      - 3 SE(3) poses along +x:
          pose0 ~ [0, 0, 0, 0, 0, 0]
          pose1 ~ [1, 0, 0, 0, 0, 0]
          pose2 ~ [2, 0, 0, 0, 0, 0]

      - 3 places, each near its pose's translation (1D x in this example).
        We keep them 1D for now to reuse the existing centroid factor cleanly.

      - 1 room, tied to centroid of the three places (so ~ x = 1.0).

    Factors:
      - prior on pose0 (identity)
      - SE(3) odom from pose0->pose1 (1m in x)
      - SE(3) odom from pose1->pose2 (1m in x)
      - priors on each place (0, 1, 2)
      - room_centroid tying room to mean(place0, place1, place2)
    """

    wm = WorldModel()

    # Register residuals
    wm.fg.register_residual("prior", prior_residual)
    wm.fg.register_residual("odom_se3", odom_se3_residual)
    wm.fg.register_residual("room_centroid", room_centroid_residual)

    # -------------------------
    # Poses (SE(3) in R^6)
    # -------------------------
    # Initial guesses are intentionally noisy
    pose0 = wm.add_variable(
        "pose_se3",
        jnp.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.005]),
    )
    pose1 = wm.add_variable(
        "pose_se3",
        jnp.array([0.8, 0.1, -0.05, 0.05, 0.03, -0.01]),
    )
    pose2 = wm.add_variable(
        "pose_se3",
        jnp.array([1.9, -0.2, 0.1, -0.02, 0.01, 0.02]),
    )

    # Ground-truth relative SE(3) motion: 1m along x, zero rotation
    odom01 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    odom12 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Prior on pose0: identity
    wm.add_factor(
        "prior",
        (pose0,),
        {"target": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
    )

    # Odometry constraints
    wm.add_factor("odom_se3", (pose0, pose1), {"measurement": odom01})
    wm.add_factor("odom_se3", (pose1, pose2), {"measurement": odom12})

    # -------------------------
    # Places (1D positions along x)
    # -------------------------
    # We just model them as scalars; conceptually they are attached to each pose.
    place0 = wm.add_variable("place1d", jnp.array([-0.2]))  # noisy around 0
    place1 = wm.add_variable("place1d", jnp.array([1.4]))   # noisy around 1
    place2 = wm.add_variable("place1d", jnp.array([2.1]))   # noisy around 2

    # Priors on places to keep them near 0, 1, 2
    wm.add_factor("prior", (place0,), {"target": jnp.array([0.0])})
    wm.add_factor("prior", (place1,), {"target": jnp.array([1.0])})
    wm.add_factor("prior", (place2,), {"target": jnp.array([2.0])})

    # -------------------------
    # Room centroid (1D)
    # -------------------------
    room = wm.add_variable("room1d", jnp.array([5.0]))  # bad initial guess

    # room_centroid ties room position to mean of the three place positions
    wm.add_factor(
        "room_centroid",
        (room, place0, place1, place2),
        {"dim": jnp.array(1)},
    )

    return wm, (pose0, pose1, pose2), (place0, place1, place2), room


def print_world_state(wm: WorldModel, pose_ids, place_ids, room_id, label: str):
    print(f"\n=== {label} ===")
    for i, pid in enumerate(pose_ids):
        v = wm.fg.variables[pid].value
        tx, ty, tz, wx, wy, wz = [float(x) for x in v]
        print(
            f"pose{i}: t=({tx:.3f}, {ty:.3f}, {tz:.3f}), "
            f"w=({wx:.3f}, {wy:.3f}, {wz:.3f})"
        )

    for i, sid in enumerate(place_ids):
        v = wm.fg.variables[sid].value
        print(f"place{i}: x={float(v[0]):.3f}")

    rv = wm.fg.variables[room_id].value
    print(f"room: x={float(rv[0]):.3f}")


def main():
    wm, pose_ids, place_ids, room_id = setup_mini_world()

    # Print initial state
    print_world_state(wm, pose_ids, place_ids, room_id, label="INITIAL STATE")

    # Optimize joint SLAM + scene graph
    wm.optimize(iters=20, method="gn")

    # Print optimized state
    print_world_state(wm, pose_ids, place_ids, room_id, label="OPTIMIZED STATE")


if __name__ == "__main__":
    main()