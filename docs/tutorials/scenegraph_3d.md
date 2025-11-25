# Tutorial: 3D Scene Graph Construction & Visualization  
**Category:** Dynamic Scene Graphs, SE(3) & SLAM, Visualization

---

## Overview

This tutorial demonstrates how to build, solve, and visualize a **3D Dynamic Scene Graph (DSG)** using DSG‑JIT.  
We walk through:

- Constructing a **SE(3) odometry factor graph**  
- Solving it with **manifold Gauss‑Newton**  
- Exporting the optimized graph to **VisNode / VisEdge** structures  
- Adding a **semantic layer** (rooms, places, objects)  
- Rendering a **layered 3D scene graph** with both metric and semantic edges  

This is a full pipeline example combining *SLAM*, *semantics*, and *DSG visualization*.

---

## 3D Scene Graph Tutorial (Based on `exp18_scenegraph_3d.py`)

### 1. Build the SE(3) Odometry Factor Graph

We construct a simple 1D pose chain:

```
pose0 → pose1 → pose2 → pose3 → pose4
```

Each pose is a **6‑vector se(3)** state, and each edge is an **odom_se3** geodesic residual.

Key components:
- `pose_se3` variables (`[tx, ty, tz, wx, wy, wz]`)
- A strong **prior** on `pose0`
- Consecutive **odom** constraints set to `[1, 0, 0, 0, 0, 0]`

```python
fg.register_residual("odom_se3", odom_se3_geodesic_residual)
fg.register_residual("prior", prior_residual)
```

After adding variables and factors, we pack the state and prepare it for optimization.

---

### 2. Solve Using Manifold Gauss‑Newton

We solve with the DSG-JIT manifold-aware solver:

```python
cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
x_opt = gauss_newton_manifold(residual_fn, x0, block_slices, manifold_types, cfg)
```

Each pose converges to an SE(3) configuration roughly aligned with:

```
pose_i.x ≈ i
```

---

### 3. Convert SLAM Graph → VisNode / VisEdge

DSG-JIT provides:

```python
nodes_fg, edges_fg = export_factor_graph_for_vis(fg)
```

This generates visualization-friendly structures:

- **VisNode**: id, type ("pose"), position ∈ R³
- **VisEdge**: var_ids, factor_type

These form the **metric layer** of the final scene graph.

---

### 4. Add Semantic Structure (Rooms, Places, Objects)

We introduce additional node types:

- **Rooms** (semantic level 0)
- **Places** (semantic level 1)
- **Objects** (semantic level 2)

We place them relative to optimized poses to simulate a building layout.

Example:

```python
room0 = VisNode(... type="room")
place1 = VisNode(... type="place")
obj2 = VisNode(... type="object")
```

Semantic edges are also inserted:

- `room → place`
- `place → object`
- `pose → place` (robot‑at‑place)

This creates a hierarchical spatial graph with mixed metric + semantic structure.

---

### 5. Render the Full 3D Scene Graph

The renderer supports:

- Layered Z‑offsets per type  
- Metric edges (solid black)  
- Semantic edges (dashed colored)  
- Node labels and color‑coding  

Example invocation:

```python
plot_scene_graph_3d_from_nodes(
    all_nodes,
    metric_edges=metric_edges,
    semantic_edges=semantic_edges,
    z_by_type={"room":0.0, "place":0.5, "pose":1.0, "object":1.5},
    show_labels=True,
)
```

This yields a clear 3D layered diagram showing:

- Pose chain (trajectory)
- Semantic structure (rooms, places, objects)
- All edges across layers

---

## Summary

This tutorial demonstrated:

- Building and solving a **manifold SE(3) SLAM factor graph**
- Exporting it into **DSG visualization structures**
- Adding a hierarchical **semantic graph**
- Rendering a **3D dynamic scene graph** with layered organization

This experiment shows how DSG‑JIT can fuse **geometry + semantics** into unified 3D scene graphs suitable for robotics, SLAM, and spatial AI.

---
