# Tutorial: Range Sensor Dynamic Scene Graph

**Categories:** Sensors & Fusion, Dynamic Scene Graphs

---

## Overview

This tutorial walks through a small but complete example of using a **range sensor** inside a **Dynamic Scene Graph (DSG)** built with `SceneGraphWorld` and `DynamicSceneGraph`.

We will:

- Build a simple world with **one room** and **one place** in 3D.
- Add a **single robot agent** that moves along the x-axis.
- Attach **noisy range measurements** from each robot pose to the fixed place.
- Let the `WorldModel` automatically convert these measurements into factors.
- Run **manifold Gauss–Newton** to jointly refine the robot trajectory and place position.
- Visualize the resulting factor graph in 3D.

Conceptually, this is a minimal example of range-based SLAM in the DSG-JIT engine.

---

## 1. What this experiment sets up

The experiment in `expXX_range_sensor_dsg.py` (or similar) constructs:

- A **static layer** with:
  - One room (1D representation) at x = 0.
  - One place `place_A` in 3D at some fixed coordinate (e.g., `[2.0, 1.0, 0.0]`).
- A **dynamic layer** with:
  - One agent `"robot0"`.
  - A short pose chain for `robot0` along the x-axis: `x = 0, 1, 2, ...`.
- **Range observations** from each pose to `place_A`, with small synthetic noise.

During optimization, the factor graph tries to reconcile:

- Odometry constraints between consecutive robot poses.
- Range constraints between each pose and the place.

The result is a **joint estimate** of robot trajectory and place position that best explains all the measurements.

---

## 2. Building the range sensor DSG

The main construction happens in `build_range_dsg`:

```python
import jax.numpy as jnp

from dsg_jit.world.scene_graph import SceneGraphWorld
from dsg_jit.world.dynamic_scene_graph import DynamicSceneGraph


def build_range_dsg(num_steps: int = 5):
    """Build a simple dynamic scene graph with:
      - One robot 'robot0'
      - A short pose chain along +x
      - A single place at a fixed location
      - Range measurements from each pose to that place.
    """
    sg = SceneGraphWorld()
    dsg = DynamicSceneGraph(world=sg)

    # --- Create static structure: one room + one place ---
    roomA = sg.add_room1d(x=jnp.array([0.0], dtype=jnp.float32))

    place_center = jnp.array([2.0, 1.0, 0.0], dtype=jnp.float32)
    placeA = sg.add_place3d("place_A", xyz=place_center)
    sg.add_room_place_edge(roomA, placeA)

    # ... more to come (agent and range obs)
    return sg, dsg, placeA
```

### 2.1 Static world

- `SceneGraphWorld()` creates a container for the **WorldModel** (variables, factors) plus semantic nodes.
- `add_room1d` builds a minimal room node parameterized by a 1D coordinate (here just x). It is mostly semantic and gives the place a higher-level parent.
- `add_place3d` creates a **3D place node** at `place_center`. This is the range sensor target.
- `add_room_place_edge` connects the room and place in the scene graph.

The static layer now represents **"room A"** with a single place `place_A` floating somewhere in front of the robot.

---

## 3. Adding a robot agent and odometry

Next, we add a single agent and give it a short trajectory along the x-axis:

```python
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
```

### 3.1 Pose representation

Each pose is a 6D `pose_se3` vector:

- `[tx, ty, tz, wx, wy, wz]`, where `t*` is translation and `w*` is a minimal rotation vector.
- For this toy example, we keep rotations zero and only change `tx`.

### 3.2 Odometry edges

- `add_odom_tx(agent, t_from, t_to, dx)` creates an **odometry factor** between consecutive poses.
- Here we use a simple 1D translation `dx = 1.0` along x.
- These factors encourage `pose[t+1].x ≈ pose[t].x + 1`.

This gives us a tiny odom chain that the optimizer will refine.

---

## 4. Injecting range measurements

The interesting part of this experiment is **range-only sensing**.

For each pose at time `t`, we:

1. Compute the ground-truth position of the robot in world coordinates.
2. Compute the true Euclidean distance to the place center.
3. Add small synthetic noise.
4. Call `add_range_obs` to attach a range observation factor.

```python
    # --- Add range measurements to placeA from each pose ---
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
```

### 4.1 Range factors in the WorldModel

When you call `dsg.add_range_obs`, the `DynamicSceneGraph`:

- Looks up the pose node `(agent, t)` in the underlying `WorldModel`.
- Adds a **range factor** between that pose and the static place node `placeA`.
- Encodes the noisy scalar range and its uncertainty `sigma`.

You do **not** need to manipulate the factor graph directly—the DSG helpers create the correct variables and factors for you.

---

## 5. Optimizing the world

Once the scene graph and dynamic layer are constructed, we can optimize the underlying factor graph using **manifold Gauss–Newton**.

The helper `optimize_world` encapsulates this step:

```python
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig
from dsg_jit.slam.manifold import build_manifold_metadata


def optimize_world(sg: SceneGraphWorld):
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
```

### 5.1 Manifold metadata

- `build_manifold_metadata(fg)` inspects each variable type and builds:
  - `block_slices`: how each variable lives in the flat state vector.
  - `manifold_types`: whether a block is Euclidean or SE(3).
- `gauss_newton_manifold` uses this metadata to:
  - Compute tangent-space updates.
  - Retract updates back to the correct manifold (SE(3) for poses, R^n for places).

### 5.2 Updating the world

By writing the optimized values back into `fg.variables[nid].value`, the **scene graph** and **visualization code** see the refined state automatically.

---

## 6. Running the experiment and visualizing

The `main()` function wires everything together:

```python
from dsg_jit.world.visualization import plot_factor_graph_3d


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
    plot_factor_graph_3d(sg.wm.fg)
```

### 6.1 Inspecting the result

- The printed poses should lie roughly along the x-axis, consistent with odometry and range.
- `place_A` should be close to its true position (e.g., `[2.0, 1.0, 0.0]`), adjusted slightly to best fit all noisy ranges.
- The 3D plot shows:
  - Robot poses as SE(3) nodes.
  - The place node.
  - Factors connecting them.

---

## 7. How this fits into the bigger picture

This range-sensor DSG experiment illustrates several key DSG-JIT concepts:

- **Dynamic Scene Graphs:** `DynamicSceneGraph` manages agents and time-indexed poses while delegating factor creation to the `WorldModel`.
- **Sensor-Level APIs:** High-level calls like `add_range_obs` let you think in terms of *measurements* rather than low-level factor wiring.
- **Manifold Optimization:** The same Gauss–Newton machinery used for SLAM also refines semantic nodes (like places) when they are linked to sensor data.

You can build on this pattern to:

- Add multiple places and multiple range targets.
- Combine range with other modalities (camera, LiDAR, IMU) in a single DSG.
- Integrate real dataset loaders that stream range measurements from logs or ROS2.

---

## Summary

In this tutorial, we:

- Constructed a **range-sensor-aware Dynamic Scene Graph** with one agent, one room, and one place.
- Attached **noisy range observations** from each pose to the place via `add_range_obs`.
- Used **manifold-aware Gauss–Newton** to jointly optimize robot trajectory and place location.
- Visualized the resulting factor graph in 3D.

This experiment is a clean starting point for **range-based SLAM** in DSG-JIT and shows how easily sensor modalities can be integrated via the `SceneGraphWorld` / `DynamicSceneGraph` interface.
