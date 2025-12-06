

# Tutorial: Multi-Voxel Point Observation Learning
**Categories:** Voxel Grids & Spatial Fields, Learning & Hybrid Modules, Dynamic Scene Graphs

---

## Overview

In voxel-based mapping systems, observed 3D points often serve as soft constraints that pull voxel centers toward actual sensor measurements. When these observations are uncertain or biased, it becomes valuable to **learn** the observation parameters themselves by differentiating through the optimization process.

This tutorial demonstrates a compact example of *trainer-style learning* for voxel point observations. We combine:

- A 3‑voxel chain in 1D-like geometry  
- Smoothness constraints (voxel‑to‑voxel)
- A prior anchor for the first voxel  
- Three learnable voxel-point observations supplied through a parameter matrix `theta`  
- A Gauss‑Newton inner solver operating on voxel variables  
- An outer gradient descent loop updating `theta` via supervision  

The experiment is based on `exp13_trainer_voxel_point_multi.py`.

---

## Building the Voxel Graph

We construct a tiny world model containing three voxel variables:

```python
v0 = Variable(NodeId(0), "voxel_cell", jnp.array([0.0, 0.0, 0.0]))
v1 = Variable(NodeId(1), "voxel_cell", jnp.array([1.2, 0.2, 0.0]))
v2 = Variable(NodeId(2), "voxel_cell", jnp.array([2.3, -0.1, 0.1]))
```

Next we register residuals used by the factors:

```python
wm.register_residual("prior", prior_residual)
wm.register_residual("voxel_smoothness", voxel_smoothness_residual)
wm.register_residual("voxel_point_obs", voxel_point_observation_residual)
```

We add:

- A strong **prior** on `v0`
- **Smoothness factors** between `(v0, v1)` and `(v1, v2)`
- **Three voxel-point observation factors**, each of which receives its real `point_world` from `theta`

This produces a compact, differentiable voxel estimation problem.

---

## Defining the Parameterized Observation Model

Rather than storing fixed `point_world` values in factors, we use:

```
theta ∈ ℝ^(K × 3)
```

where each row of `theta[k]` is injected into the corresponding `voxel_point_obs` factor. This is implemented through:

```python
residual_fn_param, _ = wm.build_residual_function_voxel_point_param_multi()
```

This allows the residual function to depend on both the state `x` and the learnable parameters `theta`.

---

## Inner Optimization: Solving for Voxels

For each proposed value of `theta`, we solve for voxel positions using Gauss‑Newton:

```python
def solve_inner_voxel(wm, theta):
    residual_fn_param, _ = wm.build_residual_function_voxel_point_param_multi()
    x0, _ = wm.pack_state()
    def residual_x(x):
        return residual_fn_param(x, theta)

    cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
    return gauss_newton(residual_x, x0, cfg)
```

This inner loop is fully differentiable.

---

## The Supervised Learning Objective

We specify target voxel centers:

```python
gt_voxels = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0]
])
```

The supervised loss is:

```
L(θ) = ½ Σ_i || v_i(θ) – gt_i ||²
```

Implemented as:

```python
def supervised_loss(theta):
    x_opt = solve_inner_voxel(wm, theta)
    ...
    v_stack = jnp.stack([v0, v1, v2])
    return 0.5 * jnp.sum((v_stack - gt_voxels)**2)
```

---

## Outer Learning Loop

We apply gradient descent to adjust `theta`:

```python
theta = theta0
lr = 0.1
for it in range(20):
    g = grad_fn(theta)
    theta = theta - lr * g
```

Optionally, gradient clipping avoids numerical instability.

Over iterations, the observation points `theta[k]` become more consistent with the ground-truth voxel positions. This, in turn, drives the optimized voxel states closer to the true layout.

---

## Results

After optimization:

- The learned observation points `theta` converge toward ground truth.
- The voxel estimates `(v0, v1, v2)` align closely with their true positions.
- The supervised loss decreases significantly.

This demonstrates one of DSG‑JIT’s core advantages:  
**You can differentiate through a full Gauss‑Newton optimization and learn parameters that influence the system.**

---

## Summary

In this tutorial, you learned how to:

- Build a voxel-based factor graph with smoothness and observation factors.
- Use parameterized voxel-point observations with differentiable residuals.
- Run a Gauss‑Newton inner solver to estimate voxel state.
- Define a supervised objective on voxel positions.
- Use outer-loop gradient descent to learn per-observation parameters.

This trainer-style workflow generalizes to larger voxel grids, learned observation models, or hybrid neural feature extractors. It is a powerful pattern enabled by DSG‑JIT’s differentiable factor graph engine.