# Tutorial: Building a Mini-World Scene Graph (exp01)

This tutorial walks through **Experiment 1** of DSG‑JIT: constructing a tiny
*static* scene graph using the core `SceneGraphWorld` and `WorldModel` APIs.  
The goal is to explain **what** is being built, **why** each component exists,
and **how** the engine interprets the resulting factor graph.

---

## 🧠 What You Will Learn

- What a **SceneGraphWorld** is and why DSG‑JIT uses it  
- The difference between **semantic nodes** and **geometric nodes**  
- How to add **rooms**, **places**, and **objects**  
- How the underlying **WorldModel** stores factor graph variables  
- How to visualize the resulting scene  

This tutorial assumes no prior SLAM knowledge — explanations are provided
inline wherever new terminology appears.

---

## 1. What is a Scene Graph?

A **Scene Graph** is a hierarchical representation of an environment.  
DSG‑JIT uses an adaptation of MIT’s **Dynamic Scene Graph (DSG)**, but this
tutorial focuses on the simplest static case:

```
Room
 ├── Place A
 │     ├── Object A1
 │     └── Object A2
 └── Place B
       └── Object B1
```

- **Room** = large-scale region of space  
- **Place** = a physically meaningful sub‑location  
- **Object** = something contained inside a place  
- **Pose** = geometric position/orientation stored internally as an SE(3) vector  

In Experiment 1, each of these nodes is represented by an entry in the
`WorldModel`’s **factor graph**, meaning they have geometric variables assigned
to them.

---

## 2. The Code Structure

Below is the annotated version of the experiment’s logic.
You can paste this into a Jupyter notebook or run directly.

```python
from world.scene_graph import SceneGraphWorld
from world.visualization import plot_factor_graph_2d
import jax.numpy as jnp

# Create a new world
sg = SceneGraphWorld()

# Add a single room at the origin
roomA = sg.add_room3d(jnp.array([0.0, 0.0, 0.0]))

# Add two places inside the room
placeA = sg.add_place3d(roomA, jnp.array([1.0, 0.0, 0.0]))
placeB = sg.add_place3d(roomA, jnp.array([-1.0, 0.0, 0.0]))

# Add objects inside each place
objA1 = sg.add_object3d(placeA, jnp.array([1.2, 0.1, 0.0]))
objA2 = sg.add_object3d(placeA, jnp.array([0.8, -0.2, 0.0]))
objB1 = sg.add_object3d(placeB, jnp.array([-1.2, 0.2, 0.0]))

# Visualize
plot_factor_graph_2d(sg.wm.fg, title="Mini-World Scene Graph")
```

---

## 3. Why Is This Useful?

This small example illustrates:

### ✔ The semantic hierarchy  
Rooms → Places → Objects form the backbone of DSG‑JIT.

### ✔ Object and place geometry  
Every node gets an SE(3) variable (pose), allowing downstream:
- SLAM optimization  
- Range/bearing factor creation  
- Sensor alignment  
- Dynamic scene graph motion  

### ✔ Factor graph integration  
Even though this example uses simple priors, all nodes are treated as
optimizable geometric variables inside the factor graph.

---

## 4. How DSG-JIT Represents Geometry

Each call such as:

```python
sg.add_place3d(roomA, xyz)
```

creates:

1. A **semantic node** (place) in the scene graph  
2. A **geometric variable** (SE(3) pose) in the factor graph  
3. A parent‑child **semantic edge** linking the place to its region  

You can inspect all nodes with:

```python
print(sg.wm.fg.variables)
```

---

## 5. Final Visualization

The 2D plot generated at the end shows:

- Blue points = pose nodes  
- Red lines = factor constraints  
- Clusters around the room → places → objects  

This view is minimal but immediately useful: it shows how the world model
stores and interprets the relationships between semantic and geometric
elements.

---

## 🎉 Summary

In this tutorial you learned:

- How to build a minimal scene graph  
- How DSG-JIT connects semantic and geometric layers  
- How factor graph variables back the scene graph nodes  
- How to visualize the resulting structure  

You are now ready for **Tutorial 2**, where we extend this into a **temporal**
scene graph with robot poses and odometry constraints.

---
