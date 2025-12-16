# Tutorial: Sliding‑Window Optimization with Active Templates
**Categories:** Dynamic Scene Graphs, SLAM, JAX/JIT

---

## Overview

This tutorial explains how **DSG‑JIT performs real‑time optimization** using a
**sliding window (active template)** while preserving a **persistent global Scene Graph**.

You will learn:

- What an *active window* is and why it is used
- How **ActiveWindowTemplate** enables constant‑shape JIT compilation
- How SceneGraph memory and WorldModel optimization interact
- How to stream data online without recompiling
- How to visualize the final Scene Graph in the browser

This tutorial is derived from *Experiment 23* but is intentionally simplified to
focus on the **core API concepts** rather than benchmark details.

---

## Concept: Sliding Windows vs Persistent Scene Graphs

In DSG‑JIT, **memory and computation are separated**:

- **SceneGraphWorld** stores the *entire history* of the environment
- **WorldModel** optimizes only a *bounded active subset* of that history

A **sliding window** (also called an *active window*) is a fixed‑size subset of
recent variables and factors used for online optimization.

> Older nodes are *not deleted* — they simply become inactive for optimization.

This allows:

- constant problem size
- one‑time JIT compilation
- millisecond‑level solve times

---

## 1. Create a SceneGraphWorld

```python
from dsg_jit.world.scene_graph import SceneGraphWorld

sg = SceneGraphWorld()
wm = sg.wm   # WorldModel backing the scene graph
```

The SceneGraphWorld owns:

- agents, poses, rooms, objects
- semantic relationships
- a persistent memory layer

The WorldModel owns:

- the active factor graph
- residual functions
- JIT‑compiled solvers

---

## 2. Register Residuals at the WorldModel

All optimization logic lives at the **WorldModel** level.
Residuals must be registered *once* before optimization.

```python
from dsg_jit.slam.measurements import prior_residual, odom_se3_residual

wm.register_residual("prior", prior_residual)
wm.register_residual("odom_se3", odom_se3_residual)
```

Residual registration defines *what kinds of constraints* the optimizer understands.

---

## 3. Define an Active Window Template

An **ActiveWindowTemplate** fixes the *shape* of the optimization problem.
This is what enables DSG‑JIT to compile **once** and reuse indefinitely.

```python
from dsg_jit.world.model import ActiveWindowTemplate

WINDOW = 20
POSE_DIM = 6

variable_slots = [
    ("pose_se3", i, POSE_DIM)
    for i in range(WINDOW)
]

factor_slots = [
    ("prior", 0, (("pose_se3", 0),))
]

for k in range(1, WINDOW):
    factor_slots.append(
        ("odom_se3", k, (("pose_se3", k-1), ("pose_se3", k)))
    )

wm.init_active_template(
    ActiveWindowTemplate(variable_slots, factor_slots)
)
```

Once initialized, **the template never changes shape**.
Only values inside slots are updated.

---

## 4. Stream Poses into the Scene Graph

We now add poses over time.
Only the *last W poses* are mapped into the active template.

```python
pose_id = sg.add_agent_pose_se3(
    agent="robot0",
    t=t,
    value=pose_vector
)
```

The SceneGraph keeps *all poses forever*.
The WorldModel only optimizes the most recent ones.

---

## 5. Populate Active Slots and Optimize

At each timestep:

```python
wm.set_variable_slot("pose_se3", slot_idx, pose_value)

wm.configure_factor_slot(
    factor_type="odom_se3",
    slot_idx=k,
    var_ids=(prev_slot, curr_slot),
    params={"measurement": delta, "weight": 1.0},
    active=True,
)

wm.optimize_active_template(iters=1)
```

Key points:

- Slot indices are reused
- No graph growth occurs
- JIT compilation happens **once**
- Optimization cost stays constant

---

## 6. Persistent Scene Graph Memory

Even though only the active window is optimized:

- all poses
- all rooms
- all objects
- all semantic edges

remain stored in the SceneGraphWorld.

This allows:

- global reasoning
- offline batch optimization later
- multi‑agent fusion

---

## 7. Visualize the Scene Graph (Web)

After streaming is complete, launch the web viewer:

```python
sg.visualize_web(port=8000)
```

This renders:

- the full pose history
- room / object hierarchy
- semantic edges

in an interactive Three.js viewer.

---

## Summary

This tutorial demonstrated how DSG‑JIT achieves **real‑time performance** using:

- a fixed‑shape active optimization window
- one‑time JIT compilation
- persistent Scene Graph memory

**SceneGraphWorld** handles *what exists*.
**WorldModel** handles *what is optimized*.

Together, they form a scalable architecture for:

- online SLAM
- multi‑agent mapping
- large‑scale dynamic environments

The same pattern extends naturally to landmarks, voxels, objects, and learned factors.
