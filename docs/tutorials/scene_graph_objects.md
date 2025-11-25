# Tutorial: Working with Objects in a Scene Graph  
**Categories:** Static Scene Graphs, World Modeling

---

## Overview

In this tutorial, we expand on earlier scene-graph examples by adding **semantic objects** into a `SceneGraphWorld`.  
Objects such as chairs or tables are represented as nodes with 3D geometry (positions), and they can be linked to rooms, places, or agents through relational edges.

This tutorial corresponds to *Experiment 04* from the project and explains:

- How objects are created in DSG‑JIT  
- How they relate to other scene‑graph elements (rooms, places, agents)  
- How optimization adjusts their positions  
- How to visualize a small static scene containing semantic objects  

---

## Tutorial

### 1. Create the World Model

Every scene graph begins with a `WorldModel`, the optimization backend:

```python
cfg = GNConfig(max_iters=20)
wm = WorldModel(config=cfg)
```

### 2. Create the Scene Graph World

```python
sg = SceneGraphWorld(wm)
```

### 3. Add a Room

We create a room centered at the origin:

```python
room_id = sg.add_room3d(center=jnp.array([0.0, 0.0, 0.0]))
```

Rooms serve as high‑level spatial partitions for grouping places and objects.

### 4. Add Places Inside the Room

```python
place_ids = [
    sg.add_place3d(jnp.array([1.0, 0.0, 0.0])),
    sg.add_place3d(jnp.array([0.0, 1.0, 0.0])),
]
```

Places act as anchor points for navigation or semantic structure.

### 5. Add Objects

Objects are full 3D nodes that may be associated with places or rooms.

```python
obj1 = sg.add_object3d(jnp.array([1.2, 0.1, 0.0]))   # Chair
obj2 = sg.add_object3d(jnp.array([-0.5, -0.3, 0.0])) # Table
obj3 = sg.add_object3d(jnp.array([0.3, 1.4, 0.0]))   # Lamp
```

### 6. Connect Objects to Room / Places

This establishes the semantic structure:

```python
sg.add_room_object_edge(room_id, obj1)
sg.add_room_object_edge(room_id, obj2)
sg.add_place_object_edge(place_ids[0], obj1)
sg.add_place_object_edge(place_ids[1], obj3)
```

These edges allow the optimizer to reason about object–room and object–place relations.

### 7. Add Simple Priors

A mild prior prevents variables from drifting:

```python
sg.add_prior_point(obj1, jnp.array([1.2, 0.1, 0.0]), sigma=0.1)
sg.add_prior_point(obj2, jnp.array([-0.5, -0.3, 0.0]), sigma=0.1)
sg.add_prior_point(obj3, jnp.array([0.3, 1.4, 0.0]), sigma=0.1)
```

### 8. Optimize

```python
x_opt = wm.optimize()
vals = wm.unpack_state(x_opt)
```

### 9. Inspect the Result

```python
print("Room:", vals[room_id])
for pid in place_ids:
    print("Place:", vals[pid])
for oid in [obj1, obj2, obj3]:
    print("Object:", vals[oid])
```

### 10. Visualize the Object‑Level Scene Graph

```python
plot_scenegraph_3d(
    sg,
    title="Scene Graph with Objects",
    show=True,
)
```

This produces a 3D plot showing rooms, places, and objects connected by semantic edges.

---

## Summary

In this tutorial, you learned how to:

- Create a semantic scene graph with **objects**
- Attach objects to **rooms** and **places**
- Add geometric priors for stability
- Visualize the resulting structure

Objects are foundational for building rich semantic environments.  
Future tutorials will extend this to dynamic scene graphs and multi‑sensor perception.
