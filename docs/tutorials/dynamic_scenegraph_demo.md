

# Tutorial: Dynamic Scene Graph Demo  
**Categories:** Dynamic Scene Graphs, SE(3) & SLAM, 3D Visualization

---

## Overview

This tutorial presents **Experiment 19 — Dynamic Scene Graph Demo**, a fully featured multi‑agent, multi‑room 3D dynamic scene‑graph pipeline.  
You will learn how DSG‑JIT constructs and optimizes a **hierarchical semantic + metric scene graph**, fusing:

- Multiple rooms, places, and named objects
- Multi‑agent pose trajectories over time
- Odometry constraints between time steps
- Place attachments that ground agent poses within the scene structure
- Joint optimization over SE(3) and Euclidean manifolds
- Full 3D rendering of both the static world and temporal trajectories

This experiment showcases the complete end‑to‑end workflow of **Dynamic Scene Graphs** in the DSL‑JIT framework.

---

## Dynamic Scene Graph Construction

We build two layers:

### **1. Static World Layer — `SceneGraphWorld`**
This contains:
- **Rooms** (A, B, C) arranged spatially along the x‑axis  
- **Places** within rooms, using the updated API `add_place3d(room_id, rel_xyz)`
- **Named objects** (chair, table, plant) anchored to places via semantic factors  
- **Place attachments** linking object/pose nodes to their associated places

### **2. Dynamic Layer — `DynamicSceneGraph`**
This contains:
- **Multiple agents** (robot0 and robot1)  
- SE(3) agent poses over time  
- Odometry edges for temporal consistency: `add_odom_tx(agent, t0, t1, dx)`  
- Place attachments grounding agents to relevant places at selected timestamps  

This hybrid structure creates a **multi‑layer, multi‑agent dynamic spatial model**.

---

## Optimization

Once the scene graph is built, the full factor graph is:

- Packed into a state vector  
- Given manifold metadata (SE(3) + Euclidean)
- Optimized using **manifold Gauss‑Newton**:

```python
cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
x_opt = gauss_newton_manifold(
    residual_fn,
    x0,
    block_slices,
    manifold_types,
    cfg,
)
```

This solves simultaneously for:
- Room centers  
- Place positions  
- Object positions  
- All agent trajectories  

---

## 3D Visualization

The tutorial renders two visualizations:

### **1. Full Static + Dynamic 3D Scene Graph**
```python
plot_scenegraph_3d(
    sg,
    x_opt,
    index,
    title="Experiment 19 — Dynamic 3D Scene Graph",
    dsg=dsg,
)
```
This renders:
- Rooms (semantic layer)
- Places
- Named objects
- Optimized agent poses
- Semantic and metric edges

### **2. Time‑Colored Agent Trajectories**
```python
plot_dynamic_trajectories_3d(
    dsg,
    x_opt,
    index,
    title="Experiment 19 — Dynamic 3D Scene Graph (time-colored)",
    color_by_time=True,
)
```

The result is a clear dynamic‑spatiotemporal map of the entire scene.

---

## Summary

This tutorial walked through the full **Dynamic Scene Graph (DSG)** pipeline:

1. Building a multi-room, multi-place semantic scene  
2. Adding named objects  
3. Creating multi-agent temporal trajectories  
4. Linking poses to places with semantic factors  
5. Optimizing the entire system over SE(3) + Euclidean manifolds  
6. Rendering high‑quality 3D visualizations of the scene and trajectories  

Experiment 19 demonstrates the power of DSG‑JIT for robotics, SLAM, and semantic mapping applications where **agents, objects, and environments evolve together over time**.