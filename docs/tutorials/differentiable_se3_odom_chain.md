# Tutorial: Differentiable SE(3) Odometry Chain  
**Category:** SE(3) & SLAM, Learning & Hybrid Modules

---

## Overview

This tutorial demonstrates how **DSG‑JIT** supports *differentiable odometry*, enabling SE(3)-based pose-graph optimization where the **odometry measurements themselves are learnable parameters**.  

Unlike the Gauss–Newton–on‑manifold solvers used in earlier tutorials, this experiment uses a **first‑order gradient descent (GD) optimizer**. This avoids the numerical difficulty of backpropagating through repeated linear solves and ensures full differentiability end‑to‑end.

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

## Build the Factor Graph

We add the variables and factors exactly as in the experiment:

```python
fg.add_variable(p0)
fg.add_variable(p1)
fg.add_variable(p2)

fg.add_factor(f_prior0)
fg.add_factor(f_odom01)
fg.add_factor(f_odom12)

fg.register_residual("prior", prior_residual)
fg.register_residual("odom_se3", odom_se3_residual)
```

Next, we build the **parametric residual function**:

```python
residual_param_fn, _ = fg.build_residual_function_se3_odom_param_multi()
```

This yields:

```
residuals = r(x, theta)
```

where **theta enters the factor residuals**, enabling full differentiability.

---

## Defining the Loss Function

We optimize over **x (the poses)** inside the solve and measure loss on the resulting optimized graph:

```python
loss(theta) = Σ_i || pose_i(θ) − ground_truth_i ||²
```

This is implemented by:

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

