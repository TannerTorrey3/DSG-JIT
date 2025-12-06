# Tutorial: Visualizing a Factor Graph in 3D

**Categories:** Core Concepts, SE(3) & SLAM, JAX & JIT

---

## Overview

This tutorial walks through a minimal but complete example of:

- Building a small **WorldModel-backed factor graph** over SE(3) poses,
- Registering a custom **odometry residual**,
- Solving the resulting nonlinear least‑squares problem with the **manifold Gauss–Newton** solver, and
- **Visualizing** the optimized poses and factors in 3D.

The code is based on the experiment `exp_visual_factor_graph.py` (the snippet below), and is meant as a first introduction to how a *WorldModel-backed factor graph + residuals + solvers + visualization* fit together in DSG-JIT.

---

## 1. Problem setup: a simple SE(3) odometry chain

We start from a very simple SLAM‑style setup: a chain of poses along the x‑axis, connected by odometry constraints.

Conceptually, we want 5 poses

\[
T_0, T_1, T_2, T_3, T_4 \in SE(3)
\]

with each consecutive pair constrained by a relative motion of **1 meter along +x**. In this experiment, we represent each pose in its **minimal se(3) vector form** (6‑vector: translation + rotation), and we add one odometry factor between each pair.

### Building the demo factor graph

```python
import jax.numpy as jnp

from dsg_jit.world.model import WorldModel
from dsg_jit.slam.measurements import se3_chain_residual


def build_demo_world(num_poses: int = 5) -> WorldModel:
    """Build a simple WorldModel-backed factor graph for an SE(3) pose chain."""
    wm = WorldModel()

    # 1. Register the residual type used by our factors
    wm.register_residual("odom_se3", se3_chain_residual)

    # 2. Add pose variables: pose_se3 in R^6
    pose_ids = []
    for _ in range(num_poses):
        vid = wm.add_variable(
            var_type="pose_se3",
            value=jnp.zeros(6, dtype=jnp.float32),  # initial guess: all zeros
        )
        pose_ids.append(vid)

    # 3. Add odometry factors between consecutive poses
    for i in range(num_poses - 1):
        meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1 m along x
        wm.add_factor(
            f_type="odom_se3",
            var_ids=(pose_ids[i], pose_ids[i + 1]),
            params={"measurement": meas, "weight": 1.0},
        )

    return wm
```

**What’s happening here?**

- `WorldModel` is the high-level container that owns:
  - An underlying **factor graph** (`wm.fg`) storing variables and factors,
  - A registry of **residual functions** keyed by factor type names, and
  - Helpers for **state packing/unpacking** and building residual functions for optimization.

- Each SE(3) pose is stored as a variable in the WorldModel-backed factor graph with:
  - A `type` string (`"pose_se3"`), and
  - A 6‑D initial value.

- Each odometry constraint is a factor of type `"odom_se3"` connecting two pose variables in the WorldModel-backed factor graph.

- `wm.register_residual("odom_se3", se3_chain_residual)` tells the WorldModel which residual function to call when evaluating factors of this type.

---

## 2. Packing the state and building manifold metadata

Once the graph structure is built, we need to:

1. **Pack** all variables into a single flat state vector `x`, and
2. Build **manifold metadata** so the solver knows how to treat each variable (Euclidean vs SE(3), etc.).

```python
from dsg_jit.slam.manifold import build_manifold_metadata

wm = build_demo_world(num_poses=5)
fg = wm.fg  # underlying factor graph structure

# Pack current variable values into a flat state vector x0 via the WorldModel
x0, index = wm.pack_state()
packed_state = (x0, index)

# Build manifold metadata for all variables
block_slices, manifold_types = build_manifold_metadata(
    packed_state=packed_state,
    fg=fg,
)

# Build the global residual function r(x) from the WorldModel registry
residual_fn = wm.build_residual()
```

- `build_manifold_metadata(packed_state=..., fg=fg)` inspects the WorldModel-backed factor graph and returns:
  - `block_slices`: a mapping from node ids to slices in the flat vector `x`,
  - `manifold_types`: a mapping from node ids to a manifold tag (e.g. `"se3"` vs `"euclidean"`).

- `wm.pack_state()` collects all variable values into one flat JAX vector `x0`, along with an `index` structure for unpacking later.

- `wm.build_residual()` returns a JAX‑compatible function

  \[
  r(x): \mathbb{R}^n \to \mathbb{R}^m
  \]

  that stacks all factor residuals in a consistent order.

---

## 3. Running Gauss–Newton on the SE(3) chain

We now solve the nonlinear least‑squares problem

\[
x^\* = \arg\min_x \| r(x) \|^2
\]

using the **manifold Gauss–Newton** solver. This uses the manifold metadata to update SE(3) poses properly.

```python
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig

cfg = GNConfig(
    max_iters=10,
    damping=1e-3,      # Levenberg–Marquardt style diagonal damping
    max_step_norm=1.0  # clamp step size for stability
)

x_opt = gauss_newton_manifold(
    residual_fn,
    x0,
    block_slices,
    manifold_types,
    cfg,
)
values = wm.unpack_state(x_opt, index)
```

- `GNConfig` controls:
  - `max_iters`: how many Gauss–Newton iterations to run,
  - `damping`: how much diagonal regularization to add,
  - `max_step_norm`: a safety clamp on the step size.

- `gauss_newton_manifold`:
  - Linearizes the residual at the current `x`,
  - Solves for a step in the **tangent space** of each manifold block,
  - Retracts back onto the manifold (e.g. updates SE(3) poses correctly).

- `wm.unpack_state(x_opt, index)` maps the optimized flat state back to a dict from node ids to arrays.

After solving, we typically push the optimized values back into the factor graph:

```python
print("Optimized poses:")
for nid, v in values.items():
    print(nid, v)
    fg.variables[nid].value = v
```

This is necessary so downstream tools (like visualization) see the updated poses.

Here, `wm` owns the packed state and residuals, while `fg = wm.fg` stores the underlying variable objects used by the visualizer.

---

## 4. Visualizing the factor graph in 3D

With the optimized poses stored in `fg.variables` (where `fg = wm.fg`), we can call the visualization helpers to render the graph layout.

```python
from dsg_jit.world.visualization import plot_factor_graph_3d

# Visualize optimized poses and factors
plot_factor_graph_3d(fg)
```

- `plot_factor_graph_3d(fg)` inspects:
  - All pose variables (e.g. `"pose_se3"`),
  - Factors connecting them (e.g. `odom_se3` edges),
  - And produces a simple 3D matplotlib plot.

- There is also `plot_factor_graph_2d(fg)` for planar (x–y) visualization when you only care about positions in a plane.

In this experiment, since the odometry measurements are all `[1, 0, 0, 0, 0, 0]`, the optimized chain should line up roughly along the x‑axis with unit spacing between consecutive poses.

---

## Summary

In this tutorial you saw how to:

1. **Define a minimal SE(3) odometry chain** as a WorldModel-backed factor graph by adding pose variables and odometry factors via the `WorldModel` API.
2. **Register a residual function** (`se3_chain_residual`) for a factor type (`"odom_se3"`) with the `WorldModel`.
3. **Pack and unpack the state** via `WorldModel.pack_state` / `unpack_state`, and build **manifold metadata** with `build_manifold_metadata(packed_state, fg)`.
4. **Solve** the resulting nonlinear least squares problem with `gauss_newton_manifold` and a simple `GNConfig`.
5. **Visualize** the final optimized poses and factors in 3D using `plot_factor_graph_3d`.

This pattern—build a graph, register residuals, pack state, run a solver, then visualize—is the core workflow that many other DSG‑JIT experiments and higher‑level components build upon. In more advanced tutorials, we’ll extend this to dynamic scene graphs, voxel grids, and sensor‑driven mapping.
