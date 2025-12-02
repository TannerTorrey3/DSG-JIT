# Tutorial: Scene Graph Demo: Rooms, Places, Poses, and Objects

**Categories:** Static Scene Graphs, Core Concepts

## Overview

This tutorial walks through a small **static scene graph** built directly as a `FactorGraph` in DSG‑JIT.  
Instead of running SLAM or optimization, we focus purely on **structure and visualization**:

- We create **rooms**, **a shared place (corridor)**, **robot poses**, and **objects**.
- We connect them with **semantic edges** (room–place, place–object) and **structural edges** (pose chain, pose–place attachment).
- We then render both a **top‑down 2D view** and a **3D scene graph view** using the built‑in visualization utilities.

This experiment is a good way to understand how DSG‑JIT can represent **semantic structure** on top of a metric world, even when no optimization is involved.

---

## Building a Scene Graph Inside a FactorGraph

The experiment constructs a `FactorGraph` and then uses it as a **structural scene graph**:

```python
from dsg_jit.core.types import NodeId, FactorId, Variable, Factor
from dsg_jit.core.factor_graph import FactorGraph
from dsg_jit.world.visualization import plot_factor_graph_3d, plot_factor_graph_2d
```

We start by creating the graph and defining two small helper functions:

- `add_var` – creates a `Variable` with a given id, type, and value.
- `add_edge` – creates a `Factor` that acts as a **pure edge** (no residuals) between nodes.

```python
def build_scenegraph_factor_graph() -> FactorGraph:
    fg = FactorGraph()

    # --- Helpers ------------------------------------------------------------
    def add_var(idx: int, vtype: str, value) -> NodeId:
        nid = NodeId(idx)
        fg.add_variable(
            Variable(
                id=nid,
                type=vtype,
                value=jnp.asarray(value, dtype=jnp.float32),
            )
        )
        return nid

    def add_edge(fid: int, var_indices, ftype: str = "scene_edge") -> None:
        fg.add_factor(
            Factor(
                id=FactorId(fid),
                type=ftype,
                var_ids=tuple(NodeId(i) for i in var_indices),
                params={},  # purely structural, no residuals used
            )
        )
```

Notice that the `params` dictionary is empty. These factors are not used in any optimization; they simply encode **connectivity** between nodes so that visualization tools can draw edges.

---

## Step 1 – Adding Rooms and a Shared Place

We first create **room nodes** and a shared **corridor place**. Conceptually, you can think of them as semantic landmarks in a building.

```python
    # --- Rooms (high-level) -------------------------------------------------
    # Positions are (x, y, z) in "world" coordinates
    room_a = add_var(100, "room1d", [2.0, 2.0, 1.2])   # Room A up/right
    room_b = add_var(101, "room1d", [4.5, 2.0, 1.0])   # Room B further right

    # --- Shared place (hallway / doorway) -----------------------------------
    place_corridor = add_var(200, "place1d", [1.0, 0.0, 0.0])
```

Here:

- `room_a` and `room_b` live at higher `y` coordinates, as if they are **up above** the robot corridor.
- `place_corridor` is a shared **place node** (e.g., doorway or hallway intersection).

We then connect these nodes with **semantic “room_place” edges**:

```python
    # Connect rooms to shared place
    eid = 0
    add_edge(eid, [room_a, place_corridor], ftype="room_place")
    eid += 1
    add_edge(eid, [room_b, place_corridor], ftype="room_place")
    eid += 1
```

These edges express that both rooms are **accessible via the same corridor place**.

---

## Step 2 – Robot Trajectory Poses

Next, we add a small chain of **robot poses** near the corridor and connect them with a “pose_chain” edge type:

```python
    # --- Robot trajectory (poses near the place) ----------------------------
    pose_ids = []
    for i, x in enumerate([-1.0, -0.2, 0.6, 1.4, 2.2]):
        # pose_se3: [tx, ty, tz, roll, pitch, yaw]
        p = add_var(10 + i, "pose_se3", [x, 0.0, 0.5, 0.0, 0.0, 0.0])
        pose_ids.append(p)

    # Link poses in a chain (visual odometry edges)
    for i in range(len(pose_ids) - 1):
        add_edge(eid, [pose_ids[i], pose_ids[i + 1]], ftype="pose_chain")
        eid += 1
```

Key ideas:

- Each `pose_se3` is a **6‑DoF pose**, but here we keep everything flat (only `tx` and `z=0.5` change).
- The `pose_chain` factors represent **odometry-style connectivity**, but again, they are purely structural in this demo.

We also connect every pose back to the corridor place with a `pose_place_attachment` edge type:

```python
    # Attach all poses to the corridor place (like a localization prior)
    for pid in pose_ids:
        add_edge(eid, [pid, place_corridor], ftype="pose_place_attachment")
        eid += 1
```

Conceptually, this says: *the robot’s trajectory is localized around this shared corridor place*.

---

## Step 3 – Objects Attached to the Place

We then add a few **object nodes**—represented as small “voxel_cell” variables—near the corridor, and attach them to the place:

```python
    # --- Objects near the place ---------------------------------------------
    obj_chair = add_var(300, "voxel_cell", [0.7, 0.4, 0.6])
    obj_table = add_var(301, "voxel_cell", [1.3, 0.5, 0.7])
    obj_plant = add_var(302, "voxel_cell", [0.9, 0.9, 0.9])

    # Attach objects to the place (semantic containment)
    add_edge(eid, [place_corridor, obj_chair], ftype="place_object")
    eid += 1
    add_edge(eid, [place_corridor, obj_table], ftype="place_object")
    eid += 1
    add_edge(eid, [place_corridor, obj_plant], ftype="place_object")
    eid += 1
```

These “voxel_cell” entries stand in for **localized 3D objects** (e.g., the centroid of a chair point cloud). The edges with type `"place_object"` encode a simple containment or “located at” relationship:

- The chair, table, and plant all “live” at the corridor place.

At this point, the `FactorGraph` encodes a **complete layered scene graph**:

- **Rooms** (high-level regions)
- **Place (corridor)** connecting the rooms
- **Robot trajectory poses** moving through the corridor
- **Objects** attached to the place

No optimization has been run; it is purely a **hand‑crafted scene**.

---

## Visualizing the Scene Graph (2D and 3D)

The `main()` function of the experiment simply builds the graph and calls the visualization helpers:

```python
def main() -> None:
    fg = build_scenegraph_factor_graph()

    # Just visualize – no optimization in this hero scene graph demo.
    print("=== DSG-JIT Scene Graph Demo (exp18) ===")
    print(f"Num variables: {len(fg.variables)}")
    print(f"Num factors:   {len(fg.factors)}")

    # 2D top-down
    plot_factor_graph_2d(fg, show_labels=True)

    # 3D view
    plot_factor_graph_3d(fg, show_labels=True)
```

- `plot_factor_graph_2d` gives a **top‑down view** (e.g., x–y plane) with nodes and edges.
- `plot_factor_graph_3d` renders the full **3D layout**, including the vertical separation between rooms, place, and objects if encoded in the positions.

You can run this experiment from the repository root (after setting `PYTHONPATH=src`) to see both visualizations:

```bash
export PYTHONPATH=src
python3 experiments/expXX_scenegraph_demo.py  # use the actual filename in your repo
```

(Replace `expXX_scenegraph_demo.py` with the real experiment filename if it differs.)

---

## Summary

In this tutorial, we:

- Built a **static scene graph** inside a `FactorGraph` using `Variable` and `Factor` objects.
- Created **rooms**, a **shared corridor place**, **robot poses**, and **objects** with simple numeric positions.
- Connected them with **semantic edge types** (`room_place`, `pose_place_attachment`, `place_object`) and **structural edges** (`pose_chain`).
- Visualized the resulting structure using `plot_factor_graph_2d` and `plot_factor_graph_3d`.

Even without running any optimization, this experiment demonstrates how DSG‑JIT can serve as a **unified representation** for **metric** (positions) and **semantic** (rooms, places, objects) information. In later tutorials, you can combine these ideas with SLAM, sensor fusion, and learning to build fully dynamic, optimized scene graphs.
