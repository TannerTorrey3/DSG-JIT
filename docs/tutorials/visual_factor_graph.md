# Tutorial: Visualizing a Factor Graph in 3D

**Categories:** Core Concepts, SE(3) & SLAM, JAX & JIT

---

## Overview

This tutorial walks through a minimal but complete example of:

- Building a small **factor graph** over SE(3) poses,
- Registering a custom **odometry residual**,
- Solving the resulting nonlinear least‑squares problem with the **manifold Gauss–Newton** solver, and
- **Visualizing** the optimized poses and factors in 3D.

The code is based on the experiment `exp_visual_factor_graph.py` (the snippet below), and is meant as a first introduction to how *FactorGraph + residuals + solvers + visualization* fit together in DSG-JIT.

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

from core.factor_graph import FactorGraph
from core.types import NodeId, Variable, Factor
from slam.measurements import se3_chain_residual


def build_demo_graph(num_poses: int = 5) -> FactorGraph:
    fg = FactorGraph()

    # 1. Register the residual type used by our factors
    fg.register_residual("odom_se3", se3_chain_residual)

    # 2. Add pose variables: pose_se3 in R^6
    for i in range(num_poses):
        nid = NodeId(i)
        fg.add_variable(
            Variable(
                id=nid,
                type="pose_se3",
                value=jnp.zeros(6),  # initial guess: all zeros
            )
        )

    # 3. Add odometry factors between consecutive poses
    fid = 0
    for i in range(num_poses - 1):
        meas = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 1 m along x
        fg.add_factor(
            Factor(
                id=fid,
                type="odom_se3",
                var_ids=(NodeId(i), NodeId(i + 1)),
                params={"measurement": meas, "weight": 1.0},
            )
        )
        fid += 1

    return fg
```

**What’s happening here?**

- `FactorGraph` is the core container for:
  - **Variables** (node states),
  - **Factors** (constraints / residuals), and
  - A mapping from **factor type names** (like `"odom_se3"`) to residual functions.

- Each SE(3) pose is stored as a `Variable` with:
  - A `NodeId` for bookkeeping,
  - A `type` string (`"pose_se3"`), and
  - A 6‑D initial value.

- Each odometry constraint is a `Factor` of type `"odom_se3"` connecting two variables. Its `params` dictionary carries:
  - `measurement`: the relative motion (here, 1 m along x),
  - `weight`: a scalar that scales that residual in the objective.

- `fg.register_residual("odom_se3", se3_chain_residual)` tells the factor graph **which residual function to call** when evaluating factors of this type.

---

## 2. Packing the state and building manifold metadata

Once the graph structure is built, we need to:

1. **Pack** all variables into a single flat state vector `x`, and
2. Build **manifold metadata** so the solver knows how to treat each variable (Euclidean vs SE(3), etc.).

```python
from slam.manifold import build_manifold_metadata

fg = build_demo_graph(num_poses=5)

# Build manifold metadata for all variables
block_slices, manifold_types = build_manifold_metadata(fg)

# Pack current variable values into a flat state vector x0
x0, index = fg.pack_state()

# Build the global residual function r(x)
residual_fn = fg.build_residual_function()
```

- `build_manifold_metadata(fg)` inspects the graph and returns:
  - `block_slices`: a mapping from `NodeId` to slices in the flat vector `x`,
  - `manifold_types`: a mapping from `NodeId` to a manifold tag (e.g. `"se3"` vs `"euclidean"`).

- `fg.pack_state()` collects all variable `value` arrays into one flat JAX vector `x0`, along with an `index` structure for unpacking later.

- `fg.build_residual_function()` returns a JAX‑compatible function

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
from optimization.solvers import gauss_newton_manifold, GNConfig

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
values = fg.unpack_state(x_opt, index)
```

- `GNConfig` controls:
  - `max_iters`: how many Gauss–Newton iterations to run,
  - `damping`: how much diagonal regularization to add,
  - `max_step_norm`: a safety clamp on the step size.

- `gauss_newton_manifold`:
  - Linearizes the residual at the current `x`,
  - Solves for a step in the **tangent space** of each manifold block,
  - Retracts back onto the manifold (e.g. updates SE(3) poses correctly).

- `fg.unpack_state(x_opt, index)` maps the optimized flat state back to a dict from `NodeId` to arrays.

After solving, we typically push the optimized values back into the factor graph:

```python
print("Optimized poses:")
for nid, v in values.items():
    print(nid, v)
    fg.variables[nid].value = v
```

This is necessary so downstream tools (like visualization) see the updated poses.

---

## 4. Visualizing the factor graph in 3D

With the optimized poses stored in `fg.variables`, we can call the visualization helpers to render the graph layout.

```python
from world.visualization import plot_factor_graph_3d

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

1. **Define a minimal SE(3) odometry chain** as a factor graph using `Variable` and `Factor` objects.
2. **Register a residual function** (`se3_chain_residual`) for a factor type (`"odom_se3"`).
3. **Pack and unpack the state** via `FactorGraph.pack_state` / `unpack_state`, and build **manifold metadata** with `build_manifold_metadata`.
4. **Solve** the resulting nonlinear least squares problem with `gauss_newton_manifold` and a simple `GNConfig`.
5. **Visualize** the final optimized poses and factors in 3D using `plot_factor_graph_3d`.

This pattern—build a graph, register residuals, pack state, run a solver, then visualize—is the core workflow that many other DSG‑JIT experiments and higher‑level components build upon. In more advanced tutorials, we’ll extend this to dynamic scene graphs, voxel grids, and sensor‑driven mapping.
