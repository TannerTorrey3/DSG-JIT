# Tutorial: Scene Graph Objects (JIT)
**Categories:** Static Scene Graphs, JAX & JIT, Core Concepts  
**Based on:** `exp05_scene_graph_objects_jit.py`

---

## Overview
This tutorial demonstrates how *Dynamic Scene Graphs* (DSGs) interact with JAX and `jit` compilation, using DSG‑JIT’s optimized architecture.  
You will learn:

- How object nodes (e.g., places, rooms, agents) are represented.
- How edges encode structural and semantic relations.
- How DSG‑JIT uses JAX structures internally.
- How to construct and manipulate objects while keeping everything JAX‑friendly.
- The difference between standard Python execution and JIT‑compiled execution.

This is a continuation of the earlier Scene Graph tutorials, now showing how these components behave under lightweight compilation.

---

## Scene Graph Objects with JIT

### 1. Importing required modules
We begin by importing the scene graph and any JAX utilities needed for jit-friendly updates.

```python
import jax
import jax.numpy as jnp

from dsg_jit.world.scene_graph import SceneGraphWorld
```

The `SceneGraphWorld` class wraps the dynamic scene graph and world model, providing unified access to nodes, edges, and SLAM structures.

---

### 2. Creating the world and graph
We initialize a new SceneGraphWorld instance and extract its scene graph:

```python
sgw = SceneGraphWorld()
sg = sgw.sg
```

This `sg` object will host rooms, places, objects, agents, and their relational edges.

---

### 3. Adding Nodes (Rooms, Places, Objects)

DSG‑JIT organizes spatial/semantic data hierarchically:

```python
room = sg.add_room1d(x=jnp.array([0.0]))
place = sg.add_place1d(room, x=jnp.array([1.0]))
obj = sg.add_object(place, x=jnp.array([1.0]))
```

- **Rooms** represent larger areas (topology).
- **Places** represent navigational nodes within rooms.
- **Objects** are local semantic entities attached to places.

All positions are expressed as JAX arrays to ensure compatibility with compiled computation.

---

### 4. Adding Edges

Edges encode structural and navigational relationships:

```python
sg.add_edge(room, place)
sg.add_edge(place, obj)
```

These edges are stored in JAX‑friendly containers internally, allowing the graph to be updated or rolled into optimization stages.

---

### 5. JIT‑Compiling Graph Functions

To demonstrate JIT compatibility, the experiment defines a simple JIT‑compiled function that reads node positions:

```python
@jax.jit
def read_node_position(nid, node_table):
    return node_table[nid].x
```

You can now use:

```python
pos = read_node_position(obj, sg.nodes)
print(pos)
```

This confirms the nodes and their attributes are accessible inside compiled JAX functions.

---

## Summary
In this tutorial, you learned how to:

- Create scenes using DSG‑JIT's node primitives (rooms, places, objects).
- Add edges encoding spatial/semantic relationships.
- Work with JAX arrays for all positions.
- Use JIT‑compiled functions to interact with DSG internal structures.

This experiment illustrates a foundational principle of DSG‑JIT: **scene graphs remain fully JAX‑compatible**, enabling high‑performance optimization and real‑time robotics applications.
