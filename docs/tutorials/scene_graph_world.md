# Tutorial: SceneGraphWorld: Building a Simple Semantic World
**Categories:** Static Scene Graphs, Core Concepts, SE(3) & SLAM

This tutorial walks through **Experiment 3** (`exp03_scene_graph_world.py`) and shows how to:

- Build a small **WorldModel** with SE(3) poses and landmark-like variables.
- Wrap it in a **SceneGraphWorld** to get semantic layers (poses, places, rooms).
- Add high‑level factors using scene‑graph helpers instead of wiring residuals by hand.
- Run Gauss–Newton and interpret the optimized scene.

We will stay in **1D** along the x‑axis for clarity, but everything generalizes to full 3D.

---

## 1. Imports and setup

The experiment starts by importing the core world + scene‑graph wrappers:

```python
import jax.numpy as jnp

from world.model import WorldModel
from world.scene_graph import SceneGraphWorld
```

- **`WorldModel`** wraps a low‑level `FactorGraph` and knows how to pack/unpack the optimization state.
- **`SceneGraphWorld`** adds semantic structure (rooms, places, objects, agents) and high‑level helpers
  that internally add the right variables and factors to the `WorldModel`.

We also define a small helper to print vectors as plain Python arrays for readability.

---

## 2. Constructing a minimal SceneGraphWorld

The experiment uses a convenience constructor that builds a **1D SE(3) pose chain**:

```python
def build_scenegraph_world() -> SceneGraphWorld:
    # Create a WorldModel with a small SE(3) chain
    wm = WorldModel.make_se3_chain(num_poses=3, dx=1.0)

    # Wrap it in a SceneGraphWorld for semantic layers
    sg = SceneGraphWorld(wm)
    return sg
```

Conceptually:

- `WorldModel.make_se3_chain(num_poses=3, dx=1.0)` creates 3 pose variables
  `pose0, pose1, pose2` with initial guesses approximately at x = 0, 1, 2.
- `SceneGraphWorld(wm)` does **not** change the underlying `FactorGraph`,
  but adds bookkeeping so that each variable can also live in a semantic layer
  (e.g., pose layer, place layer, room layer).

---

## 3. Adding places and a room in 1D

Next, we add **place nodes** (landmarks) and a **room node** along the same 1D axis:

```python
def add_places_and_room(sg: SceneGraphWorld):
    # Three 1D places near the poses
    place0 = sg.add_place1d(0.1)
    place1 = sg.add_place1d(1.2)
    place2 = sg.add_place1d(2.1)

    # One 1D room center farther out
    room = sg.add_room1d(5.0)

    return place0, place1, place2, room
```

Here:

- **`add_place1d(x)`** creates a landmark‑like variable constrained to a 1D position along x.
- **`add_room1d(x)`** creates a higher‑level node representing a room center in the same 1D space.

These calls both:

1. Allocate a new variable in the underlying `WorldModel`.
2. Register the node in the appropriate semantic layer inside `SceneGraphWorld`.

---

## 4. Wiring semantic factors via helpers

Instead of directly using low‑level residual functions, the tutorial uses high‑level helpers
provided by `SceneGraphWorld`. They internally add correctly‑typed factors to the `WorldModel`.

```python
def add_factors(sg: SceneGraphWorld, pose0: int, pose1: int, pose2: int,
                place0: int, place1: int, place2: int, room: int) -> None:
    # 1) Fix the first pose at the origin (identity prior)
    sg.add_prior_pose_identity(pose0)

    # 2) SE(3) odometry chain: pose0 -> pose1 -> pose2, each +1m along x
    sg.add_odom_se3_additive(pose0, pose1, dx=1.0)
    sg.add_odom_se3_additive(pose1, pose2, dx=1.0)

    # 3) Attach each pose to a nearby place along x
    sg.attach_pose_to_place_x(pose0, place0)
    sg.attach_pose_to_place_x(pose1, place1)
    sg.attach_pose_to_place_x(pose2, place2)

    # 4) Attach one pose to the room center along x
    sg.attach_pose_to_room_x(pose1, room)
```

What each helper means:

- **`add_prior_pose_identity(pose0)`**
  - Adds a **prior factor** that anchors `pose0` to the identity SE(3) transform.
  - This prevents the system from sliding as a whole and fixes the global frame.

- **`add_odom_se3_additive(i, j, dx)`**
  - Adds an SE(3) **odometry factor** between poses `i` and `j`.
  - In this 1D experiment, `dx=1.0` means we expect pose `j` to sit **1 meter** ahead of pose `i` along x.

- **`attach_pose_to_place_x(pose, place)`**
  - Adds a factor enforcing that the x‑coordinate of a pose and a place are consistent
    (up to noise). This is like saying “this pose is currently near this place.”

- **`attach_pose_to_room_x(pose, room)`**
  - Adds a factor tying a pose’s x‑position to a room center’s x‑position.
  - This gives a soft notion of “the robot is inside this room.”

All these helpers ultimately call into `slam.measurements` residuals and register
`Factor` objects with the underlying `FactorGraph`, but you don’t have to wire
those details manually.

---

## 5. Solving the world

Once variables and factors are in place, we can optimize the world:

```python
def solve_and_print(sg: SceneGraphWorld) -> None:
    # Run Gauss–Newton through WorldModel
    values = sg.wm.solve()

    # Retrieve optimized SE(3) poses as 6‑vectors (x, y, z, rx, ry, rz)
    pose0 = sg.wm.get_se3(0, values)
    pose1 = sg.wm.get_se3(1, values)
    pose2 = sg.wm.get_se3(2, values)

    # Retrieve optimized places and room (as 3D vectors, but only x is meaningful here)
    place0 = sg.wm.get_landmark(3, values)
    place1 = sg.wm.get_landmark(4, values)
    place2 = sg.wm.get_landmark(5, values)
    room   = sg.wm.get_landmark(6, values)

    print("=== Optimized Scene ===")
    print(f"pose0: {pose0}")
    print(f"pose1: {pose1}")
    print(f"pose2: {pose2}")
    print(f"place0: {place0}")
    print(f"place1: {place1}")
    print(f"place2: {place2}")
    print(f"room:   {room}")
```

(Exact variable IDs may differ depending on how your world was constructed; in the
original experiment they are chosen to match the creation order.)

Running this prints an **optimized 1D scene** where:

- The **poses** sit near x ≈ 0, 1, 2.
- The **places** are close to their associated poses.
- The **room** is anchored via its attachment to `pose1`.

---

## 6. Putting it all together

A minimal `main()` in the experiment looks like this:

```python
def main() -> None:
    sg = build_scenegraph_world()

    # In this tiny example we know that the first three world variables
    # are the SE(3) poses created by make_se3_chain.
    pose0, pose1, pose2 = 0, 1, 2

    place0, place1, place2, room = add_places_and_room(sg)
    add_factors(sg, pose0, pose1, pose2, place0, place1, place2, room)

    solve_and_print(sg)


if __name__ == "__main__":
    main()
```

This is already a **mini scene graph**:

- SE(3) poses form a **trajectory layer**.
- Places and rooms form **semantic layers** on top.
- Factors express odometry, attachments, and room membership.

All optimization is still done by the same Gauss–Newton engine used everywhere else
in DSG‑JIT.

---

## Summary

In this tutorial you learned how to:

- Construct a 1D `SceneGraphWorld`
- Populate it with poses, places, and a room
- Add priors and odometry factors
- Attach semantic relationships between poses and world elements
- Run optimizations and interpret results

This provides the foundation for creating more complex 2D/3D static and dynamic scene graphs.


