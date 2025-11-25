# Tutorial: Learnable Factor-Type Weights in a Scene Graph
**Categories:** Dynamic Scene Graphs, Learning & Hybrid Modules, SE(3) & SLAM

---

## Overview

In many robotics and SLAM systems, factor graphs contain multiple types of constraints—priors, odometry, attachments, and others. Typically, each factor type is assigned a fixed weight reflecting its reliability. However, in modern differentiable SLAM, we can **learn** optimal per-type weights by differentiating through the optimization process itself.

This tutorial demonstrates:

- How to construct a minimal **SceneGraphWorld** with SE(3) poses.
- How to integrate factors such as priors and odometry.
- How to use **DSGTrainer**, which performs differentiable inner-loop optimization.
- How to define a supervised loss on the final scene graph state.
- How to backpropagate through the entire factor graph to learn factor-type weights.

This experiment corresponds to `exp12_scenegraph_learnable_type_weights.py`.

---

## 1. Problem Setup

We consider a tiny SceneGraph containing:

- **Two SE(3) robot poses** (`pose0`, `pose1`, `pose2`)
- A **prior** on the first pose (fixing it at the origin)
- **Odometry factors** connecting the poses, but with *biased* measurements

The goal is to learn a scalar weight for the `"odom_se3"` factor type such that the optimized `pose2.tx` matches its ground truth value of **2.0**.

If odometry has too much weight, the graph pulls toward biased odometry.
If we learn to *down-weight* odometry, the solution aligns with ground-truth priors.

---

## 2. Building the SceneGraph

```python
sg = SceneGraphWorld()
wm = sg.wm
fg = wm.fg

fg.register_residual("prior", prior_residual)
fg.register_residual("odom_se3", odom_se3_residual)

p0 = sg.add_pose_se3(jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
p1 = sg.add_pose_se3(jnp.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
p2 = sg.add_pose_se3(jnp.array([1.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
```

We insert:

- Weak prior on `p0`
- Additive SE(3) odometry from `p0 → p1` and `p1 → p2`

```python
sg.add_prior_pose_identity(p0)
sg.add_odom_se3_additive(p0, p1, dx=0.5)
sg.add_odom_se3_additive(p1, p2, dx=0.5)
```

---

## 3. Using DSGTrainer for Differentiable Optimization

`DSGTrainer` wraps an inner-loop optimizer (gradient descent), allowing us to differentiate through:

```
solve_state(log_scales) = argmin_x || residuals(x, log_scales) ||²
```

We configure the trainer as follows:

```python
factor_type_order = ["prior", "odom_se3"]
inner_cfg = InnerGDConfig(learning_rate=0.02, max_iters=40, max_step_norm=0.5)
trainer = DSGTrainer(wm, factor_type_order, inner_cfg)
```

This gives us a function:

```
x_opt = trainer.solve_state(log_scales)
```

where `log_scales` has one entry per factor type.

---

## 4. Supervised Objective

We define the supervised loss:

```
L = (pose2.tx - 2.0)²
```

This lets the system learn a log-weight for `"odom_se3"` that moves the odometry-consistent solution toward the ground-truth position.

```python
def supervised_loss_scalar(log_scale_odom):
    log_scales = jnp.array([0.0, log_scale_odom], dtype=jnp.float32)
    x_opt = trainer.solve_state(log_scales)
    values = trainer.unpack_state(x_opt)
    pose2 = values[p2]
    return (pose2[0] - 2.0)**2
```

---

## 5. Outer Optimization Loop

We compute the gradient with respect to the odometry type’s log-scale and update it manually:

```python
grad_fn = jax.grad(supervised_loss_scalar)
log_scale_odom = jnp.array(0.0)

for it in range(50):
    g = grad_fn(log_scale_odom)
    log_scale_odom -= 5.0 * g
```

A large learning rate exaggerates the update, making the effect obvious.

---

## 6. Results and Interpretation

After several iterations:

- The learned weight for `"odom_se3"` decreases.
- The system trusts odometry less.
- `pose1` and `pose2` shift closer to their ground-truth positions.
- The supervised loss decreases.

This illustrates a powerful capability of DSG-JIT:

> You can differentiate *through the entire SLAM system*, allowing factors to learn reliability from data.

---

## Summary

In this tutorial you learned:

- How factor-type weights influence the resulting SceneGraph.
- How to wrap the factor graph into a differentiable trainer.
- How to define outer-loop supervision.
- How to learn per-factor-type log-weights with gradients.
- How to build differentiable SLAM-style pipelines in DSG-JIT.

This completes Tutorial 12 and demonstrates how DSG-JIT supports gradient-based learning on top of dynamic scene graphs.

