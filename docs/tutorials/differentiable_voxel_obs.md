# Tutorial: Differentiable Voxel Observations  
**Categories:** Voxel Grids & Spatial Fields, JAX & JIT, Core Concepts

---

## Overview

This tutorial demonstrates a core idea behind **DSG-JIT**:

> You can make the entire SLAM and spatial reasoning pipeline *differentiable*, including Gauss–Newton optimization, voxel inference, and geometric factors.

We walk through a minimal example where we optimize the position of a single voxel cell that is constrained by:
- A **weak prior** pulling it toward `[0, 0, 0]`
- A **strong voxel-point observation** pulling it toward `[1, 0, 0]`

Then we show how to compute the **gradient of the optimized voxel position with respect to the initial state**, proving that the optimization is end‑to‑end differentiable.

This type of differentiability is essential for:
- Neural SLAM
- Learned mapping systems
- Differentiable robotics
- Implicit neural fields
- Calibration and meta‑optimization

---

## Building a Single‑Voxel Optimization Problem

We start by constructing a tiny factor graph containing:

### **1. A single voxel variable**
A voxel cell is simply a 3‑vector in \(\mathbb{R}^3\).  
We intentionally place it *incorrectly* at:

```
[-0.5, 0.2, 0.0]
```

This allows the optimization to move it toward the true target.

### **2. A weak prior**
A prior factor encourages the voxel to be close to:

```
[0, 0, 0]
```

This prevents degeneracy and ensures the graph is anchored.

### **3. A strong voxel–point observation**

A more confident factor pushes the voxel toward the point:

```
[1, 0, 0]
```

This simulates the effect of a real sensor producing a measurement that “observes” where the voxel should be.

### **4. Registering residuals**

We register two residuals:

- `prior_residual`
- `voxel_point_observation_residual`

These convert factor parameters and values into residual vectors consumed by Gauss–Newton.

### **5. Preparing the manifold metadata**

Voxel cells live in \(\mathbb{R}^3\), so they are Euclidean.  
We call `build_manifold_metadata` to generate:

- Slices for each variable inside the packed vector
- The manifold type (Euclidean in this case)

---

## Running Differentiable Gauss–Newton

We define:

```python
def solve_and_loss(x0):
    x_opt = gauss_newton_manifold(...)
    v_opt = x_opt[voxel_slice]
    return || v_opt - target ||^2
```

This function:

1. Runs Gauss–Newton
2. Retrieves the optimized voxel
3. Computes its squared error from the target

Because everything inside is written in pure JAX, we can do:

```python
loss_jit = jax.jit(solve_and_loss)
grad_fn = jax.jit(jax.grad(solve_and_loss))
```

This allows:
- JIT‑compiled optimization
- Automatic differentiation through solver steps
- True end‑to‑end differentiability

---

## Taking a Gradient Step

We evaluate:

- The initial loss
- The gradient of the loss with respect to the initial voxel value
- A gradient step on the initial voxel estimate

This demonstrates how learning‑based systems could *adapt* voxel initializations, sensor models, or even entire map representations through backpropagation.

---

## Comparing Optimized States

We solve:

- Once from the original initial state
- Once from the gradient‑updated initial state

Because Gauss–Newton is now embedded inside a gradient flow, we observe:

- **Lower loss** when using the gradient‑refined initialization
- **Optimized voxel positions moving closer to the target**

This is the core idea behind:
- meta‑learning initial conditions,
- differentiable mapping,
- amortized optimization.

---

## Summary

In this tutorial, you learned how to:

- Construct a minimal factor graph with voxel variables
- Add priors and voxel-to-point observation factors
- Use the manifold Gauss–Newton solver
- Make the *entire optimization differentiable*
- Compute gradients of SLAM solutions with respect to initial variables

This unlocks powerful capabilities for future DSG‑JIT modules such as:
- differentiable mapping pipelines,
- neural field refinement,
- learned Jacobians,
- and self‑supervised perception systems.

You now have the foundation for building advanced differentiable SLAM systems in DSG‑JIT.
