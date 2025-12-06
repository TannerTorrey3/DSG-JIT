# Tutorial: Learning Factor-Type Weights  
**Category:** Learning & Hybrid Modules

---

## Overview

In modern SLAM and scene-graph optimization systems, not all measurement types are equally reliable.  
For example:

- Wheel odometry may drift.
- Visual place detections may produce false positives.
- GPS may be noisy in urban canyons.

DSG‑JIT supports *learnable factor‑type weighting*:  
you can jointly optimize the scene variables **and** learn confidence scalings for entire classes of factors, such as `"odom_se3"` or `"loop_closure"`.

Under the hood, this example uses a **WorldModel-backed factor graph**, where residuals are registered with the WorldModel and state packing/unpacking happen at the WorldModel layer.

This tutorial is based on **Experiment 11** (`exp11_learn_type_weights.py`), which demonstrates:

1. A tiny factor graph combining SE(3) robot poses and a 1‑D “place” variable.
2. Competing constraints:
   - Odometry wants `pose1.tx = 0.7`
   - A semantic “attachment” and a prior want `pose1.tx = 1.0`
3. Learning a *type-level weight* for `odom_se3` to reduce its influence.

This is a fully differentiable bilevel optimization example.

---

## Problem Setup

We construct a minimal **WorldModel-backed factor graph**:

**Variables**

| Name   | Type      | Dimension | Meaning |
|--------|-----------|-----------|---------|
| pose0  | pose_se3  | 6         | Robot start pose |
| pose1  | pose_se3  | 6         | Robot second pose |
| place0 | place1d   | 1         | A 1‑D anchor point in the world |

**Factors**

1. `prior(pose0)`  
   Enforces `pose0 = 0`.

2. `odom_se3(pose0, pose1)`  
   A *biased* odometry measurement wanting  
   `pose1.tx = 0.7`.

3. `pose_place_attachment(pose1, place0)`  
   Enforces that `place0` should be near `pose1.tx`.

4. `prior(place0)`  
   Trusted semantic clue: `place0 = 1.0`.

The ground‑truth configuration is:

- `pose0.tx = 0`
- `pose1.tx = 1`
- `place0 = 1`

But odometry tries to pull the system away from this.

---

## Learning a Type Weight

We introduce a **single log‑scale parameter** for the factor type `"odom_se3"`:

```
log_scale["odom_se3"]  ->  scale = exp(log_scale)
```

The residual function becomes:

```
r_w(x, log_scales) = concat_over_factors( scale[f.type] * r_f(x) )
```

We then solve the bi‑level objective:

1. **Inner optimization (solve for x):**

```
x*(log) = argmin_x || r_w(x, log) ||²
```

2. **Outer optimization (learn log):**

```
L(log) = (pose1_tx(x*(log)) - 1.0)²
```

We differentiate **through the entire inner optimization** using JAX and SGD.

---

## Code Walkthrough

### Building the Problem

```python
import jax.numpy as jnp
from dsg_jit.world.model import WorldModel

wm = WorldModel()

# Variables are stored inside the WorldModel's factor graph
pose0_id = wm.add_variable(
    var_type="pose_se3",
    value=jnp.zeros(6, dtype=jnp.float32),
)
pose1_id = wm.add_variable(
    var_type="pose_se3",
    value=jnp.zeros(6, dtype=jnp.float32),
)
place0_id = wm.add_variable(
    var_type="place1d",
    value=jnp.zeros(1, dtype=jnp.float32),
)
```

### Adding Factors

```python
# Prior on pose0: pose0 = 0
wm.add_factor(
    f_type="prior",
    var_ids=(pose0_id,),
    params={"target": jnp.zeros(6, dtype=jnp.float32)},
)

# Biased odometry: wants pose1.tx ≈ 0.7
wm.add_factor(
    f_type="odom_se3",
    var_ids=(pose0_id, pose1_id),
    params={"measurement": biased_meas},
)

# Attachment between pose1 and place0
wm.add_factor(
    f_type="pose_place_attachment",
    var_ids=(pose1_id, place0_id),
    params={...},  # e.g. a weight or scale parameter
)

# Prior on place0: place0 = 1.0
wm.add_factor(
    f_type="prior",
    var_ids=(place0_id,),
    params={"target": jnp.array([1.0], dtype=jnp.float32)},
)
```

### Register Residuals

```python
wm.register_residual("prior", prior_residual)
wm.register_residual("odom_se3", odom_se3_residual)
wm.register_residual("pose_place_attachment", pose_place_attachment_residual)
```

### Building the Weighted Residual Function

```python
factor_type_order = ["odom_se3"]  # we only learn a weight for odometry

# WorldModel provides a helper that builds a type-weighted residual:
residual_w = wm.build_residual_function_with_type_weights(factor_type_order)
```

This produces a callable:

```python
residual_w(x, log_scales)
```

where `log_scales` is a vector of shape `(1,)`, and internally the WorldModel:

- packs/unpacks the state using its own index map, and
- scales each factor's residual according to its type and the provided `log_scales`.

### Outer Loss Function

```python
from dsg_jit.optimization.gradient_descent import gradient_descent

# Initial stacked state from the WorldModel
x_init, index = wm.pack_state()

def solve_and_loss(log_scales):
    """
    Bi-level objective:
      - inner: minimize weighted residuals over x
      - outer: penalize deviation of pose1.tx from 1.0
    """
    def objective_for_x(x):
        r = residual_w(x, log_scales)
        return jnp.sum(r * r)

    x_opt = gradient_descent(objective_for_x, x_init, gd_cfg)

    # Unpack optimized state via the WorldModel
    values = wm.unpack_state(x_opt, index)
    pose1_vec = values[pose1_id]   # 6-vector se(3)
    pose1_tx = pose1_vec[0]        # x-translation component

    return (pose1_tx - 1.0) ** 2
```

We JIT‑compile and differentiate it:

```python
loss_val = solve_and_loss_jit(log_scale_odom)
grad     = grad_log_jit(log_scale_odom)
```

Here, the WorldModel is responsible for managing the packed state and residual registry, while the learnable type weight enters only through the scaled residuals in `residual_w`.

---

## Interpretation

- If `"odom_se3"` is too influential, the estimate for `pose1.tx` will stick near **0.7**.
- The learning step adjusts `log_scale_odom`, effectively down‑weighting odometry.
- After several iterations, `pose1.tx` moves toward **1.0**, aligning with the semantic prior and attachment constraint.

This mechanism mirrors techniques used in:

- Adaptive SLAM
- Robust back‑end optimization
- Meta‑learning measurement confidences
- Learning M‑estimators or robust kernels

---

## Summary

In this tutorial, you learned:

- How DSG‑JIT composes small multi‑variable SLAM problems on top of a WorldModel‑backed factor graph.
- How to introduce *learnable per‑factor‑type weights*.
- How to differentiate through optimization itself (bilevel learning).
- How semantic constraints can correct biased odometry when the system learns the appropriate weight schedule.

This pattern generalizes to:

- Loop closures
- Landmark observations
- IMU residual weights
- Multi‑sensor fusion reliability learning
- Large‑scale SLAM backends with meta‑learned noise models

**You now have the core foundation for building adaptive, differentiable SLAM pipelines in DSG‑JIT.**
