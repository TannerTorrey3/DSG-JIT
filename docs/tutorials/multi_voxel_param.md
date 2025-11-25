

# Tutorial: Multi‑Voxel Parameter Optimization  
**Categories:** Voxel Grids & Spatial Fields, Learning & Hybrid Modules

---

## Overview

This tutorial demonstrates **multi‑voxel differentiable optimization** using DSG‑JIT.  
It is based on **Experiment 09** and shows how to:

- Construct a **chain of voxel\_cell variables**.
- Add **priors**, **voxel smoothness**, and **voxel→point observation** factors.
- Build a **parameterized residual function** that uses an external learnable parameter `theta`.
- Run **Gauss–Newton (GN) optimization** on the voxel variables.
- Compute **gradients w.r.t. theta**, enabling hybrid learning+optimization pipelines.

The core idea:  
> We treat certain factor parameters (here, observation points) as *differentiable learnable parameters* and compute gradients through Gauss–Newton inference.

This is foundational for **learning‑based SLAM**, **neural fields**, and **probabilistic mapping**.

---

## Building the Multi‑Voxel Chain

We build a simple 1D chain:

```
v0 — v1 — v2
```

Their ground‑truth positions are:

```
v0 = [0, 0, 0]
v1 = [1, 0, 0]
v2 = [2, 0, 0]
```

To make the problem interesting, the initial values are perturbed.

The factor graph includes:

### 1. Priors on the endpoints
These anchor the chain:

- `v0 ~ [0,0,0]`
- `v2 ~ [2,0,0]`

### 2. Voxel Smoothness factors  
These encourage:

```
v1 - v0 ≈ [1,0,0]
v2 - v1 ≈ [1,0,0]
```

This enforces a **regular grid structure**.

### 3. voxel_point_obs factors  
Each voxel observes a point in the world frame:

```python
point_world = theta[i]
```

But instead of fixing these observations, we treat the vector:

```
theta ∈ R^(3×3)
```

as a **differentiable parameter** that will affect the final voxel estimates.

---

## Parametric Residual Function

The factor graph constructs:

```
residual_param_fn(x, theta)
```

This allows the Gauss–Newton solver to optimize voxel positions **conditional on the learnable parameters**.

---

## Objective Function

We define:

```
loss = Σ_i || v_i_opt(theta) – gt_i ||²
```

Then compute:

```
∂loss/∂theta
```

allowing gradient‑based learning of observation parameters.

This is useful for:

- Calibrating sensors  
- Learning correction offsets  
- Optimizing geometric structures  
- Hybrid neural inference loops

---

## Running Optimization and Computing Gradients

We JIT‑compile:

- `loss(theta)`
- `grad(loss)(theta)`

Key operations:

- Perform GN optimization starting from initial state.
- Compute loss from optimized voxel positions.
- Take a gradient step in theta.
- Re‑run optimization to show improved voxel estimates.

---

## Summary

This tutorial illustrates:

- How to construct voxel grids with geometric priors.
- How to integrate smoothness and observation factors.
- How to construct **parametric residual functions**.
- How to differentiate *through* Gauss–Newton optimization.
- How DSG‑JIT supports hybrid **optimization + learning** pipelines.

This forms the foundation for:

- Neural SLAM  
- Learnable spatial networks  
- Self‑supervised map refinement  
- Joint inference‑learning systems

In the next tutorials, we will extend this to **multi‑voxel fields**, **semantic voxels**, and **deep‑learned observation models**.