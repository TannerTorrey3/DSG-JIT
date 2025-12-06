# Tutorial: Dynamic Trajectories  
**Categories:** Static Scene Graphs, SE(3) & SLAM

---

## Overview

This tutorial is based on **Experiment 6: Dynamic Trajectories**, which demonstrates how DSG‑JIT represents and optimizes robot motion over time.  
You will learn:

- What *dynamic trajectories* are within the context of a Scene Graph World  
- How DSG‑JIT represents time‑indexed SE(3) robot poses  
- How to add **odometry**, **priors**, and other motion constraints  
- How to run Gauss‑Newton optimization over a full trajectory  
- How the dynamic aspect fits into the larger DSG pipeline

This tutorial shows *why* trajectories matter, *how* DSG‑JIT builds them, and *what* tools the experiment uses.  
Under the hood, trajectories are represented in a **WorldModel-backed factor graph**, where residuals are registered with the WorldModel and all state packing/unpacking happens at the WorldModel layer.

---

## Understanding Dynamic Trajectories

Robots do not exist at a single fixed pose—they move through time.  
A **dynamic trajectory** is simply a *sequence* of SE(3) poses:

\[
T_0,\; T_1,\; T_2,\; ..., T_N
\]

In DSG‑JIT:

- Each pose corresponds to a **node** in the SceneGraphWorld  
- Consecutive poses are connected using **odometry SE(3) factors**  
- Optional **priors** help constrain drift  
- All poses participate in **joint optimization**, allowing the solver to refine the entire trajectory at once

Experiment 6 walks through a minimal version of this workflow.

---

## Tutorial Body

### 1. Initialize a SceneGraphWorld

The experiment begins by constructing a fresh scene‑graph world model:

```python
from dsg_jit.world.scene_graph import SceneGraphWorld

sg = SceneGraphWorld()
```

This world model will hold:

- Robot trajectories  
- Landmarks  
- Rooms / Places (later experiments)  
- Factors & residuals for optimization  

---

### 2. Add a Robot Node and Trajectory

We introduce a robot agent:

```python
robot_id = sg.add_agent("robot0")
```

Then create a sequence of pose nodes:

```python
pose_ids = []
for t in range(T):
    pid = sg.add_robot_pose_se3("robot0", t)
    pose_ids.append(pid)
```

Each pose is a node of type `"pose"` with 6‑DOF parameters.

---

### 3. Add Odometry Factors (Core of Dynamic Trajectories)

Motion between pose *i* and pose *i+1* is encoded with:

```python
sg.add_odom_se3_additive(
    pose_ids[i], pose_ids[i+1],
    dx=1.0,  # unit step  
    sigma=0.1
)
```

This creates a **Factor(type='odom_se3_additive')** connecting the two poses.  
The solver uses these constraints to “pull” the poses into alignment.

---

### 4. Add a Prior to Anchor the Trajectory

Without a prior, trajectories float freely.  
We typically fix pose 0:

```python
sg.add_prior_se3(pose_ids[0], value=jnp.zeros(6), sigma=1e-6)
```

This stabilizes optimization.

---

### 5. Solve the Trajectory

Experiment 6 uses the standard DSG‑JIT Gauss‑Newton solver on the **WorldModel‑backed factor graph**:

```python
from dsg_jit.optimization.solvers import gauss_newton_manifold, GNConfig
from dsg_jit.slam.manifold import build_manifold_metadata

# Access the WorldModel from the SceneGraphWorld
wm = sg.wm

# 1) Pack the full state from the WorldModel
x0, index = wm.pack_state()
packed_state = (x0, index)

# 2) Build manifold metadata (SE(3) + Euclidean) from the packed state
block_slices, manifold_types = build_manifold_metadata(
    packed_state=packed_state,
    fg=wm.fg,  # underlying factor graph structure
)

# 3) Build the residual function from the WorldModel residual registry
residual_fn = wm.build_residual()

# 4) Run manifold Gauss–Newton
cfg = GNConfig(max_iters=20, damping=1e-3, max_step_norm=1.0)
x_opt = gauss_newton_manifold(
    residual_fn,
    x0,
    block_slices,
    manifold_types,
    cfg,
)
```

After optimization:

```python
values = wm.unpack_state(x_opt, index)
```

Here, the WorldModel is responsible for packing/unpacking the trajectory state and owning the residual registry, while the underlying factor graph `wm.fg` provides the structure needed for manifold metadata.

---

### 6. Visualize the Trajectory

Experiment 6 ends by plotting the 3‑D path:

```python
from dsg_jit.world.visualization import plot_factor_graph_3d

plot_factor_graph_3d(
    sg.wm.fg, values,
    title="Dynamic Trajectory Optimization"
)
```

You should see a straight or slightly corrected path depending on the synthetic odometry.

---

## Summary

This experiment demonstrates the most essential concept in robot SLAM:

> A trajectory is a sequence of time‑indexed poses connected by motion constraints.

In DSG‑JIT:

- Trajectories live inside a unified **SceneGraphWorld**  
- Each pose is an SE(3) variable  
- Odometry builds the WorldModel‑backed factor graph structure  
- Gauss‑Newton refines the entire trajectory jointly  
- The system scales naturally into full SLAM with landmarks and sensor fusion  

Dynamic trajectories are the backbone of DSG‑JIT’s mapping capabilities, and later tutorials build on this by adding LiDAR, cameras, landmarks, and dynamic scene updates.
