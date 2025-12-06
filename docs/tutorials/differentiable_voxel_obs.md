# Tutorial: Differentiable Voxel Observations  
**Categories:** Voxel Grids & Spatial Fields, JAX & JIT, Core Concepts

---

## Overview

This tutorial demonstrates a core idea behind **DSG-JIT**:

> You can make the entire SLAM and spatial reasoning pipeline *differentiable*, including Gauss–Newton optimization, voxel inference, and geometric factors.

We walk through a minimal example where we optimize the position of a single voxel cell that is constrained by:

Under the hood, this is represented as a **WorldModel‑backed factor graph**, where residuals are registered with the WorldModel and state packing/unpacking is handled at the WorldModel layer.

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

We start by constructing a tiny **WorldModel‑backed factor graph** containing:

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

We register two residuals with the WorldModel:

- `prior_residual`
- `voxel_point_observation_residual`

These convert factor parameters and values into residual vectors consumed by Gauss–Newton.

### **5. Preparing the manifold metadata**

Voxel cells live in \(\mathbb{R}^3\), so they are Euclidean.  
We call `build_manifold_metadata` to generate:

- Slices for each variable inside the packed vector
- The manifold type (Euclidean in this case)

Concretely, the setup in DSG‑JIT looks like this:

```python
import jax
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import (
    prior_residual,
    voxel_point_observation_residual,
)
from dsg_jit.slam.manifold import build_manifold_metadata
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig

# 1) Construct the WorldModel
wm = WorldModel()

# Register residuals at the WorldModel level
wm.register_residual("prior", prior_residual)
wm.register_residual("voxel_point_obs", voxel_point_observation_residual)

# 2) Add a single voxel variable (incorrect initial position)
voxel_id = wm.add_variable(
    var_type="voxel_cell3d",
    value=jnp.array([-0.5, 0.2, 0.0], dtype=jnp.float32),
)

# 3) Add a weak prior pulling toward [0, 0, 0]
wm.add_factor(
    f_type="prior",
    var_ids=(voxel_id,),
    params={
        "target": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        "weight": 0.1,
    },
)

# 4) Add a strong voxel–point observation toward [1, 0, 0]
target = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
wm.add_factor(
    f_type="voxel_point_obs",
    var_ids=(voxel_id,),
    params={
        "point_world": target,
        "weight": 10.0,
    },
)

# 5) Pack state and build manifold metadata
x0, index = wm.pack_state()             # x0 is the flat voxel state
packed_state = (x0, index)
block_slices, manifold_types = build_manifold_metadata(
    packed_state=packed_state,
    fg=wm.fg,                           # underlying factor graph structure
)

# 6) Build the residual function from the WorldModel registry
residual_fn = wm.build_residual()

# Convenience: slice for the voxel inside x
voxel_slice = slice(*index[voxel_id])
cfg = GNConfig(max_iters=10, damping=1e-3, max_step_norm=1.0)
```

---

## Running Differentiable Gauss–Newton

```python
def solve_and_loss(x0):
    """
    x0 is the packed initial voxel state (a 3‑vector in this example).
    We treat it as differentiable input to the Gauss–Newton solve.
    """
    x_opt = gauss_newton_manifold(
        residual_fn,
        x0,
        block_slices,
        manifold_types,
        cfg,
    )

    # Extract the optimized voxel from the flat state
    v_opt = x_opt[voxel_slice]
    return jnp.sum((v_opt - target) ** 2)
```

This function:

1. Runs Gauss–Newton on the WorldModel‑backed residual function  
2. Retrieves the optimized voxel from the packed state  
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

- Construct a minimal WorldModel‑backed factor graph with voxel variables
- Add priors and voxel-to-point observation factors
- Use the manifold Gauss–Newton solver
- Make the *entire optimization differentiable*
- Compute gradients of SLAM solutions with respect to initial variables

This unlocks powerful capabilities for future DSG‑JIT modules such as:
- differentiable mapping pipelines,
- neural field refinement,
- learned Jacobians,
- and self‑supervised perception systems.

You now have the foundation for building advanced differentiable SLAM systems in DSG‑JIT using the WorldModel residual architecture.
