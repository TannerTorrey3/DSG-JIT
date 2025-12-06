# Tutorial Hybrid SE(3) + Voxel Joint Parameter Learning
**Categories:** Dynamic Scene Graphs, SE(3) & SLAM, Voxel Grids & Spatial Fields, Learning & Hybrid Modules

## Overview
This tutorial explores joint optimization over SE(3) poses and voxel states, where both the SE(3) odometry parameters and voxel observation points are themselves learnable. The experiment demonstrates:
Under the hood, this experiment uses a **WorldModel-backed factor graph**, where residuals are registered with the WorldModel and all state packing/unpacking happens at the WorldModel layer.

- Hybrid **WorldModel-backed factor graphs** mixing SE(3) trajectory variables and voxel-grid geometry.
- Building a parametric residual function that depends on both odometry and voxel-observation parameters.
- Differentiable inner–outer loops: optimizing state variables in the inner loop, and optimizing sensor parameters in the outer loop.
- How DSG-JIT enables dense gradient‑based learning across geometric and spatial structures.

This hybrid setup underpins modern neural‑SLAM models and differentiable mapping pipelines.

---

## Hybrid SE3 + Voxel Joint Parameter Learning (exp15)

### 1. Building the Hybrid Factor Graph
We create a **WorldModel-backed factor graph** containing:

- **3 SE3 poses** (each in R⁶)
- **3 voxel centers** (each in R³)
- **priors** over pose0 and voxels 0–2
- **2 odometry factors** whose `measurement` vectors will be replaced by learnable parameters
- **3 voxel_point_obs factors** whose `point_world` values will also be replaced by learnable parameters

This mirrors a mobile robot moving through an environment while simultaneously observing voxelized structure.

### 2. Hybrid Parametric Residual Function
We build:

```
r(x, theta)
```

Where:
- `theta["odom"]` provides learned odometry increments
- `theta["obs"]` provides learned voxel observation points

Each factor reads its parameters from `theta` during runtime, enabling full differentiability.
Concretely, this is implemented via a helper like `build_param_residual(wm)`, which iterates over the WorldModel's factors, uses the residuals registered in the WorldModel's registry, and relies on `wm.pack_state()` / `wm.unpack_state()` to manage the stacked state.

### 3. Inner Optimization
For fixed parameters `theta`, we solve the state optimization problem:

```
x* = argmin_x 0.5 * || r(x, theta) ||²
```

We use first‑order gradient descent for stability.

### 4. Outer Optimization
We define a supervised loss combining:

- Pose supervision (pose2.tx → 2.0)
- Voxel supervision (v0→0, v1→1, v2→2 along x)

We differentiate through the inner optimization to update `theta`:

```
theta ← theta − η ∇_theta L(theta)
```

This jointly learns odometry increments and voxel observations.

---

## Full Experiment Code
```python
# See `experiments/exp15_hybrid_se3_voxel_joint_learning.py` for the full WorldModel-based
# implementation, which:
#   - Builds the hybrid SE(3) + voxel WorldModel-backed factor graph
#   - Uses `build_param_residual(wm)` to construct r(x, theta)
#   - Runs inner/outer optimization loops using JAX and the WorldModel residual registry.
```

---

## Summary
In this tutorial, you learned how to:

- Construct a hybrid SE(3) + voxel **WorldModel-backed factor graph**.
- Create parametric residuals that depend on both odometry and voxel observations.
- Perform nested optimization: inner state‑solving and outer parameter‑learning.
- Build learnable calibration and mapping systems using DSG‑JIT with JAX.

This pattern forms the core of differentiable SLAM, neural mapping, and hybrid geometric‑learning pipelines.

```
