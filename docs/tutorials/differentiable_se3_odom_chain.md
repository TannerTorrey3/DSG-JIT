# Tutorial: Differentiable SE(3) Odometry Chain  
**Category:** SE(3) & SLAM, Learning & Hybrid Modules

---

## Overview

This tutorial demonstrates how **DSG‑JIT** supports *differentiable odometry*, enabling SE(3)-based pose-graph optimization where the **odometry measurements themselves are learnable parameters**.  

Unlike the Gauss–Newton–on‑manifold solvers used in earlier tutorials, this experiment uses a **first‑order gradient descent (GD) optimizer**. This avoids the numerical difficulty of backpropagating through repeated linear solves and ensures full differentiability end‑to‑end. Under the hood, we use a **WorldModel‑backed factor graph**, where residuals are registered with the WorldModel and state packing/unpacking is handled at the WorldModel layer.

This experiment (based on `exp10_differentiable_se3_odom_chain.py`) walks through:

- Building a simple SE(3) chain of 3 poses  
- Adding **parametric odometry factors**, where the 6‑D odom deltas are parameters  
- Computing residuals:  
  - a pose prior  
  - additive SE(3) odometry factors  
- Running a differentiable "solve‑then‑loss" loop  
- Computing gradients **with respect to the odometry measurements themselves**  

This pattern is crucial for **learning‑based odometry**, **self‑supervised calibration**, and **model‑based RL with differentiable state estimators**.

---

## The Problem Setup

We create three SE(3) poses, slightly perturbed from the ground truth:

```
pose0 = [0, 0, 0, 0, 0, 0]
pose1 = [1, 0, 0, 0, 0, 0]
pose2 = [2, 0, 0, 0, 0, 0]
```

The state lives in **se(3)** vector form:  
`[tx, ty, tz, wx, wy, wz]`.

### Factors Used

| Factor | Meaning | Notes |
|-------|---------|-------|
| `prior` | Anchors pose0 at identity | Strong prior |
| `odom_se3` | Additive odometry residual | We parametrize the measurement |
| `odom_se3` | Second odometry (pose1 → pose2) | Also parameterized |

### Learnable Parameter

We stack the two odometry measurements into `theta`:

```
theta.shape = (2, 6)
theta[0] = meas(0→1)
theta[1] = meas(1→2)
```

These are the *learnable* inputs.

---

## Build the WorldModel‑Backed Factor Graph

We add the variables and factors exactly as in the experiment:

```python
wm = WorldModel()

# 1) Add pose variables (se(3) vectors)
p0_id = wm.add_variable(
    var_type="pose_se3",
    value=jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
p1_id = wm.add_variable(
    var_type="pose_se3",
    value=jnp.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0]),
)
p2_id = wm.add_variable(
    var_type="pose_se3",
    value=jnp.array([1.7, 0.1, 0.0, 0.0, 0.0, 0.0]),
)

# 2) Add prior and odometry factors
wm.add_factor(
    f_type="prior",
    var_ids=(p0_id,),
    params={"target": jnp.zeros(6), "weight": 1.0},
)

# Odometry between pose0 → pose1 and pose1 → pose2.
# Their measurements will be parameterized by theta, so we start with placeholders.
wm.add_factor(
    f_type="odom_se3",
    var_ids=(p0_id, p1_id),
    params={"measurement": jnp.zeros(6)},
)
wm.add_factor(
    f_type="odom_se3",
    var_ids=(p1_id, p2_id),
    params={"measurement": jnp.zeros(6)},
)

# 3) Register residuals at the WorldModel level
wm.register_residual("prior", prior_residual)
wm.register_residual("odom_se3", odom_se3_residual)
```

Next, we build the **parametric residual function**:

```python
# Helper from the experiment that builds a parametric residual
# r(x, theta) using the WorldModel's residual registry.
residual_param_fn, x_init = build_param_residual(wm)
```

This yields:

```
residuals = r(x, theta)
```

where **theta enters the odom factor residuals** via the WorldModel‑backed builder, enabling full differentiability while keeping the graph structure and residual registry centralized in the WorldModel.

---

## Defining the Loss Function

Here, `x_init` comes from the WorldModel via `build_param_residual(wm)`, which internally calls `wm.pack_state()` to get the initial stacked state.

```python
def solve_and_loss(theta):
    def objective_for_x(x):
        r = residual_param_fn(x, theta)
        return jnp.sum(r * r)

    x_opt = gradient_descent(objective_for_x, x_init, gd_cfg)

    p_stack = jnp.stack([...])
    return jnp.sum((p_stack - gt_stack)**2)
```

---

## Differentiation Through the Solver

We JIT‑compile the loss & compute gradients w.r.t theta:

```python
loss_jit = jax.jit(solve_and_loss)
grad_fn = jax.jit(jax.grad(solve_and_loss))

loss0 = loss_jit(theta0)
g0 = grad_fn(theta0)
```

The key is:

> The entire SE3 optimization process is differentiable w.r.t. the odometry measurements.

---

## Example Results

The experiment prints:

- Initial θ and loss  
- Gradient wrt θ  
- Updated θ after one gradient step  
- Optimized poses for θ₀ and θ₁  

You should see:

- The gradient pushes θ toward `[1,0,0,0,0,0]` for both odometry factors  
- The optimized poses become closer to `[0,1,2]` along x  

---

## Summary

In this tutorial, you learned:

- How to define **parametric SE(3) odometry factors** in DSG‑JIT  
- How to define a differentiable inner optimization loop  
- How to compute gradients wrt measurement parameters  
- Why first‑order solvers (like GD) are advantageous for differentiable pipelines  

This pattern is foundational for:

- Learning odometry / motion models  
- Self-supervised robotics  
- Differentiable simulators  
- Hybrid neural‑optimization architectures  

Continue to the next tutorial to extend differentiable SLAM to more complex SE(3) graphs and learned motion models.
