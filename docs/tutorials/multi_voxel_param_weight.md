# Tutorial: Joint Learning of Voxel Observation Parameters & Type Weights
**Categories:** Learning & Hybrid Modules, Voxel Grids & Spatial Fields

---

## Overview

This tutorial demonstrates a powerful capability of **DSG-JIT**: learning *both*  
1. **Low-level observation parameters** (the world-space points used by voxel observation factors),  
and  
2. **High-level factor-type weights** (the global scale applied to all `voxel_point_obs` residuals).

This experiment shows how DSG-JIT supports **nested optimization**:

- **Inner loop:** solve voxel positions using gradient descent on the factor graph.  
- **Outer loop:** optimize observation parameters + weight scales to minimize supervised loss.

The key takeaway is that **entire factor graphs (including their factor types) can be made differentiable**, enabling gradient-based meta-learning.

---

## The Experiment

This experiment mirrors `exp14_multi_voxel_param_and_weight.py`.

We construct a small graph with:

- **Three voxel_cell3d variables:** `v0, v1, v2`  
- **Weak voxel priors** (pulling voxels toward `[0,1,2]` on x-axis)  
- **Three voxel_point_obs factors** with biased initial observation points  
- **Learnable parameters:**
  - `θ[k]` → world-space point for voxel k  
  - `log_scale_obs` → global learned weight for all observation factors

The goal is to learn both **correct observation positions** and **appropriate weighting** so that solving the factor graph recovers:

```
v0 → [0,0,0]
v1 → [1,0,0]
v2 → [2,0,0]
```

---

## How It Works

### 1. Build the Graph

Each voxel is initialized slightly incorrectly:

```python
v0 = Variable(NodeId(0), "voxel_cell3d", jnp.array([-0.2, 0.1, 0.0], dtype=jnp.float32))
v1 = Variable(NodeId(1), "voxel_cell3d", jnp.array([0.8, -0.3, 0.0], dtype=jnp.float32))
v2 = Variable(NodeId(2), "voxel_cell3d", jnp.array([2.3, 0.2, 0.0], dtype=jnp.float32))
```

Weak priors pull these toward ground truth, but the observations carry most of the corrective force.

The observation parameters `theta_init` contain incorrect measurements:

```python
theta_init = jnp.array([
    [-0.5, 0.1, 0.0],
    [0.7, -0.2, 0.0],
    [2.4, 0.3, 0.0],
])
```

These are the **values we want to learn**.

---

### 2. Residual Function With Learnable Type Weight

We construct:

```
r(x, θ, log_scale_obs)
```

Where:

- `θ[k]` overrides each observation’s `point_world`
- `log_scale_obs` acts as a learned intensity on all observation residuals

Scaling is applied as:

```python
scale_obs = jnp.exp(log_scale_obs)
r = scale_obs * r
```

This allows the system to learn whether observation factors should be **trusted more or less**.

---

### 3. Inner Optimization (Solving for Voxels)

For fixed `θ` and `log_scale_obs`, we solve:

```python
x_opt = gradient_descent(objective, x0, cfg_inner)
```

Where:

```
objective(x) = 0.5 * || r(x, θ, log_scale_obs) ||²
```

This yields voxel positions that reflect the current parameterization.

---

### 4. Outer Optimization (Learning θ and log_scale)

We pack parameters:

```
p = [theta.flatten(), log_scale_obs]
```

Then compute the supervised loss:

```
Loss = MSE(v_opt, ground_truth) + small_regularizer_on_log_scale
```

We differentiate w.r.t. `p`:

```python
g = grad(loss_fn)(p)
p = p - lr * g
```

With gradient clipping + explicit clamping on `log_scale`.

---

### 5. Results

At the end, we print:

- **Learned θ[k]** for each voxel observation  
- **Learned log_scale_obs**  
- **Final voxel positions**  
- Comparison to ground truth  

Typically, the system:

- Moves θ[k] closer to actual voxel centers  
- Adjusts log_scale_obs to balance priors vs. observations  
- Achieves voxel positions very close to `[0,1,2]`  

---

## Summary

This tutorial demonstrated:

- How to jointly learn **observation parameters** and **global factor weights**
- How DSG-JIT supports differentiable nested optimization
- How voxel-based sensor models can be refined through gradient-based meta-learning

This capability allows DSG-JIT to serve as a foundation for:

- Self-calibrating SLAM systems  
- Learnable sensor models  
- Hybrid analytic/learned mapping pipelines  
- End-to-end differentiable robotics optimization  

Experiment 14 shows how **factors themselves can be learned**, not just state variables — a key feature distinguishing DSG-JIT from traditional SLAM libraries.

