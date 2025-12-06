# DSG-JIT Architecture

*A unified, differentiable 3D Dynamic Scene Graph and SLAM engine — built on JAX, manifolds, and factor graphs.*

---

## Overview

DSG-JIT is built around one central idea:

> **A 3D world model should be fully differentiable, jointly optimized, and JIT-compiled for real-time performance.**

To achieve this, the system is organized into **five coordinated layers**, each built on top of a general-purpose factor graph and SE3 manifold engine.

![Architecture](img/dsg-jit-architecture.png)

Each layer is optional, modular, and composable — enabling applications in classical SLAM, neural fields, robotics, and world-modeling.

---

# 1. Core Layer — Differentiable Factor Graph Engine

The **core** provides the primitive mathematical and structural building blocks:

### Core Responsibilities
- Variable management (poses, voxels, arbitrary vectors), via a backend‑agnostic data structure.
- Factor definitions (priors, odometry, smoothness, observations).
- Differentiable **residual registry** (now owned by the WorldModel).
- Backend‑agnostic factor graph interface (supports pluggable backends).
- JIT‑compiled residual builders (vmap‑optimized for performance).
- Unified flat‑state storage and index maps for optimization.
- Automatic Jacobian computation (via JAX autodiff).

### Key Modules
| Module | Purpose |
|--------|---------|
| `world.model` | Central owner of variables, factors, residuals, and state packing |
| `core.factor_graph` | Lightweight backend factor graph used by WorldModel (pluggable) |
| `core.types` | Data structures (Variable, Factor, NodeId, FactorId) |
| `core.math3d` | Vector/rotation utilities, SE3 helpers |

This layer is **geometry-agnostic** — it does not know about voxels, rooms, or trajectories.  
Everything above is built on these abstractions.

---

# 2. Optimization Layer — JIT Gauss–Newton on Manifolds

The optimization layer transforms the factor graph into a **high-performance nonlinear solver**.

### Features
- Gauss–Newton with line-search, fully JIT‑compiled.
- Manifold retractors for SE3 and Euclidean variables.
- Pure JAX solver loop with **vmap‑accelerated residual evaluation**.
- JIT‑compiled objective + residuals with caching of compiled solvers in WorldModel.
- Ultra‑low Python overhead — solver runs nearly entirely in XLA.

### Modules
| Module | Purpose |
|--------|---------|
| `optimization.solvers` | Gauss–Newton, manifold-aware logic |
| `optimization.jit_wrappers` | One‑line JIT interfaces using the new WorldModel residual API |

This layer is responsible for the **50–1000x performance boost** seen in benchmarks.

---

# 3. SLAM Layer — Residual Functions & Manifolds

This layer provides the differentiable measurements registered inside the WorldModel. These residuals implement the geometric logic of SE3 constraints, landmarks, voxel consistency, and smoothness. The factor graph simply stores variable/factor connectivity; the **WorldModel owns the residual functions and packing logic**.

### Supported Manifolds
- **SE3 poses**
- **3D Euclidean points**
- **Voxel nodes (R³)**

### Provided Residuals
| Residual Type | Description |
|---------------|-------------|
| `se3_geodesic` | Pose‑to‑pose geodesic constraint |
| `odom_se3` | Learnable or fixed SE3 odometry factor |
| `pose_landmark_relative` | Relative landmark measurement |
| `pose_voxel_point` | Voxel observation constraint |
| `voxel_smoothness` | Grid regularization term |
| `prior` | Generic variable prior |

All residuals are now registered with `WorldModel.register_residual`. The FactorGraph stores only structure — not residual logic.

### Modules
| Module | Purpose |
|--------|---------|
| `slam.measurements` | All differentiable SLAM residuals |
| `slam.manifold` | Manifold utilities (exp/log maps, Jacobian-safe ops) |

This layer enables **standard SLAM**, **learnable SLAM**, and **hybrid SLAM + voxel** systems.

---

# 4. Scene Graph Layer — 3D Dynamic Scene Graph

This layer introduces the structural and semantic relationships that turn raw geometry into a **world model**.

### Entities

- Poses  
- Places  
- Rooms  
- Agents  
- Voxels  
- Attachments  
- Trajectories  

### Relations

- Pose → Place membership  
- Agent → Pose trajectory  
- Room → Place grouping  
- Voxel → Place attachment  

### Modules
| Module | Purpose |
|--------|---------|
| `scene_graph.entities` | Node classes (PoseNode, PlaceNode, RoomNode, VoxelNode, etc.) |
| `scene_graph.relations` | Relations & constraints that form the DSG |

### Features

- Geometry + semantics in one structure
- Differentiable constraints between DSG entities
- Realtime graph growth (future)

Future extensions allow the Scene Graph to store **multi‑resolution geometric layers** (voxels, meshes, raw points, NeRF latent fields). DSG‑JIT is designed so that future NeRF modules can attach object‑specific neural fields directly to DSG nodes while still participating in global optimization.

The DSG operates as a **high-level structural layer** over the SLAM + voxel system.

---

# 5. World Layer — Unified World Model

The world layer combines SLAM, voxels, and the scene graph into one coherent system.

### `SceneGraphWorld`

`SceneGraphWorld` is the highest‑level API and the primary entry point for users. It manages:

- Variables and factors (via the underlying factor graph backend)
- The residual registry
- JIT‑compiled residual and solver construction
- Optimization of full DSG + voxel + SLAM systems
- Integration with differentiable learning pipelines
- Multi‑resolution storage formats (planned)
- Planned NeRF attachments for objects and rooms

### WorldModel Responsibilities

- Owns variables, factors, manifold types, and residual functions.
- Provides unified `pack_state` / `unpack_state` logic.
- Builds JIT‑optimized residuals via `build_residual` and specialized builders.
- Groups factors by type and applies `vmap` for high‑throughput evaluation.
- Caches compiled solvers for real‑time re‑optimization.
- Supports backend‑pluggable FactorGraph implementations.

### Training & Learning

The world layer exposes the **learnable parameters** used in the experiments:

- Odom SE3 measurements  
- Voxel observation points  
- Factor-type weights  
- Joint hybrid SE3 + voxel models  

### Modules
| Module | Purpose |
|--------|---------|
| `world.model` | SceneGraphWorld implementation |
| `world.voxel_grid` | Voxel layers + smoothness |
| `world.training` | Trainer-style learning loop |

---

# Information Flow Between Layers

```
Sensors → SLAM Residuals → WorldModel (variables + residuals + factor graph)  
→ JIT Solver (vmap‑optimized) → SceneGraphWorld → Applications
```

1. Sensors provide point clouds, images, IMU.  
2. SLAM residuals convert observations to geometric factors.  
3. WorldModel manages variables, residuals, and factor graph structure.  
4. JIT solver (vmap‑optimized) optimizes the entire state.  
5. SceneGraphWorld structures it into a semantic world.  
6. Applications consume the optimized graph (robotics, NeRFs, planning, etc.).

---

# Why This Design?

## 1. Performance
- JIT residuals + JIT Gauss–Newton  
- SE3 on manifolds  
- Minimal Python overhead  
- 50–1000× faster than naïve Python solvers  
- vmap grouping of factor types for large batching speedups

## 2. Differentiability
- JAX grad through:
  - Odom parameters  
  - Voxel obs  
  - Factor weights  
  - Entire scene graph  

## 3. Modularity
- SLAM or NeRF-only systems  
- DSG-only semantic reasoning  
- Hybrid world models  
- Incremental or batch optimization  
- Supports pluggable factor‑graph backends (Python, Rust, C++) as long as they implement the FactorGraph Standard.

## 4. Extensibility
This architecture supports future additions:

- Photometric residuals  
- Neural field appearance models  
- Real-world robotics datasets  
- Differentiable planning & policies  
- Joint SLAM + segmentation  
- NeRF‑augmented objects and rooms (future)
- Multi‑resolution geometric storage (voxels, meshes, point clouds)

---


# The Modern DSG‑JIT Architecture (2025+)

DSG‑JIT has evolved into a **WorldModel‑centric** system:

- The FactorGraph is a backend implementation detail.
- All differentiability, residuals, and packing logic live in the WorldModel.
- Solvers operate on JIT‑compiled, vmap‑batched functions.
- SceneGraphWorld provides a rich, extensible API that will support NeRFs, hierarchical geometry, and large‑scale world‑modeling.

This architecture enables DSG‑JIT to function as a real‑time differentiable world model suitable for robotics, simulation, mapping, and neural field research.
