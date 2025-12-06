# Tutorial: Hybrid Differentiable Scene Graphs
**Categories:** SE(3) & SLAM, Voxel Grids & Spatial Fields, Dynamic Scene Graphs, Learning & Hybrid Modules

---

## Overview

This tutorial walks through the **HERO hybrid experiment**—a large‑scale differentiable factor‑graph example that combines:

- **6 SE(3) poses**
- **6 voxel centers**
- **Odometry factors (odom_se3)**
- **Voxel point observations (voxel_point_obs)**
- **Learnable odometry measurements**
- **Learnable voxel observation points**
- A fully differentiable **inner optimization** loop  
- A fully differentiable **outer learning loop**

This experiment demonstrates how to jointly learn:
1. The correct **odometry increments** between poses.
2. The correct **observed 3D points** associated with voxels.
3. The optimal **state configuration** of both poses and voxels.

It represents one of the most complete examples of *hybrid SE(3) + voxel learning* in this repository.
Under the hood, this experiment uses a **WorldModel‑backed factor graph**, where residuals are registered with the WorldModel and all state packing/unpacking happens at the WorldModel layer.

---

## Tutorial: Hybrid Joint Learning With SE(3) Poses + Voxels

### 1. What We Build

We construct a **hybrid WorldModel‑backed factor graph** with:

#### **6 SE(3) Poses**
Ground‑truth conceptual targets:
```
pose0: [0, 0, 0, 0, 0, 0]
pose1: [1, 0, 0, 0, 0, 0]
...
pose5: [5, 0, 0, 0, 0, 0]
```

Each pose has a small initial perturbation.

#### **6 Voxel Centers**
Ground‑truth conceptual targets:
```
voxel0: [0, 0, 0]
voxel1: [1, 0, 0]
...
voxel5: [5, 0, 0]
```

Each voxel is initialized with positional errors in x and y.

---

### 2. Factor Types in the Graph

We incorporate the following factors:

#### **Prior Factors**
- A strong prior on `pose0` → anchors the absolute frame.
- A weak prior on `voxel0`.

#### **Odom Factors (pose_i → pose_{i+1})**
These are **learnable**:
- Each odometry measurement is a 6‑vector SE3 increment.
- Initial measurements are intentionally biased.

#### **Voxel Observation Factors (pose_j, voxel_i)**
These are also **learnable**:
- Each factor uses one 3D point (world coordinate).
- These 3D points become the learnable parameters `theta["obs"]`.

---

### 3. Parameterization

We learn two parameter sets:

```
theta = {
    "odom": (n_odom, 6)   # learned SE(3) increments
    "obs":  (n_obs, 3)   # learned 3D observation points
}
```

Both are jointly optimized during the outer loop.

---

### 4. Inner Optimization Loop (Gradient Descent)

We optimize the **state vector** (all poses + all voxels) using:

- A differentiable objective  
- Based on 0.5 * || residuals ||²  
- Gradient descent (small learning rate, 80 iterations)  

Because the inner loop is differentiable, we can backpropagate through it to update `theta`.

---

### 5. Outer Optimization Loop (Learn Parameters)

We optimize `theta` to minimize:

#### **Pose supervision**
Encourage:
```
pose5.tx → 5.0
```

#### **Voxel supervision**
Encourage:
```
voxel_i.x → i
voxel_i.y → 0
```

We compute gradients using JAX:
```
grad_theta = jax.grad(supervised_loss)(theta)
theta ← theta - lr * grad_theta
```

This is effectively a **hybrid differentiable SLAM + mapping system**.

---

## Full Code (With Explanations)

### Build the Hybrid Factor Graph  
```python
def build_hybrid_graph():
    ...
```

This constructs a WorldModel, adds all SE(3) poses, voxels, and factors described above, and returns the WorldModel (and associated pose/voxel ids) used by the rest of the experiment.

---

### Build the Parametric Residual Function  
```python
from dsg_jit.world.model import WorldModel

def build_param_residual(wm: WorldModel):
    ...
```

- A `residual(x, theta)` function built on top of the WorldModel residual registry  
- That injects learned odom & voxel obs parameters into each corresponding factor  
- While using `wm.pack_state()` / `wm.unpack_state()` to manage the stacked state

This keeps the graph structure, residual definitions, and packed state layout centralized in the WorldModel.

---

### Inner Solve (Differentiate Through GD)
```python
def inner_solve(theta):
    x_opt = gradient_descent(objective, x0, cfg)
```

The inner optimization updates all pose/voxel states.

---

### Outer Supervised Loss
```python
def supervised_loss(theta):
    ...
```

This controls learning behavior:
- Move last pose toward 5.0 along x
- Move voxels to correct x positions
- Penalize y drift

---

### Training Loop
```python
for it in range(steps):
    g = grad_fn(theta)
    theta = {
        "odom": theta["odom"] - lr * g["odom"],
        "obs":  theta["obs"] - lr * g["obs"],
    }
```

---

## Summary

In this tutorial you learned how to:

- Construct a **hybrid SE(3) + voxel WorldModel‑backed factor graph**
- Parameterize both odometry and 3D point observations
- Build a **differentiable residual function** with parameter injection
- Implement a **differentiable inner solver**
- Implement an **outer learning loop** to optimize parameters  
- Achieve a complete **end‑to‑end differentiable SLAM + mapping system**

This HERO experiment is one of the most advanced examples in the project and serves as a blueprint for (implemented on top of the WorldModel residual architecture):

- Joint pose + map learning  
- Robust SLAM systems  
- Hybrid neural‑symbolic optimization  
- Differentiable scene‑graph reasoning  
