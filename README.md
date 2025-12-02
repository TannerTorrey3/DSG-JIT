<p align="center">
  <img src="docs/img/dsg-jit-logo.png" alt="DSG-JIT Logo" width="280">
</p>

# DSG-JIT: A JIT‑Compiled, Differentiable 3D Dynamic Scene Graph Engine

<p align="left">
  <a href="https://github.com/TannerTorrey3/DSG-JIT/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/TannerTorrey3/DSG-JIT/tests.yml?label=tests" alt="Tests Status">
  </a>
  <a href="https://github.com/TannerTorrey3/DSG-JIT/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/TannerTorrey3/DSG-JIT" alt="License">
  </a>
  <a href="https://TannerTorrey3.github.io/DSG-JIT">
    <img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Documentation">
  </a>
  <a href="https://github.com/TannerTorrey3/DSG-JIT/stargazers">
    <img src="https://img.shields.io/github/stars/TannerTorrey3/DSG-JIT?style=social" alt="Stars">
  </a>
  <a href="https://github.com/TannerTorrey3/DSG-JIT/blob/main/CITATION.cff">
    <img src="https://img.shields.io/badge/cite%20this-software-blue" alt="Cite this software">
  </a>
</p>

## Overview

Modern spatial intelligence systems—SLAM pipelines, neural rendering models, and 3D scene graph frameworks—remain fragmented.
Each solves part of the perception problem, but none unify:
-	Metric accuracy (SLAM)
-	High-fidelity geometry & appearance (Neural Fields / Gaussians)
-	Semantic structure & reasoning (Scene Graphs)
-	Real-time global consistency (Incremental optimization)
-	End-to-end differentiability (learning cost models, priors, & structure)

DSG-JIT is a new architecture that merges these into one coherent, JIT-compiled, differentiable system.

The goal is simple:

A unified pipeline that builds, optimizes, and reasons over a complete 3D world model in real time—fusing SLAM, neural fields, and dynamic scene graphs into a single optimized computational graph.

This repository serves as the structural roadmap for developing that system.

---

## Installation (PyPI + Local Development)

DSG‑JIT can be installed in two ways:

---

### Option 1 — Install from PyPI (Recommended)

```bash
pip install dsg-jit
```

After installation you can verify:

```python
import dsg_jit
from dsg_jit.core.factor_graph import FactorGraph
print("DSG‑JIT imported successfully!")
```

---

### Option 2 — Local Development Install (Clone Repository)

```bash
git clone https://github.com/TannerTorrey3/DSG-JIT.git
cd DSG-JIT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

This installs DSG‑JIT in editable mode so that changes to the code are reflected immediately.

---

### PYTHONPATH (Legacy Local Workflow)

If you prefer not to use `pip install -e .`, you may still manually add the package to PYTHONPATH:

```bash
export PYTHONPATH=$(pwd)/dsg-jit
```

However, the pip editable installation is now the recommended workflow.

---

## Quickstart

### Install (from source)

If you cloned the repository, install dependencies and enable editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

### Run Tests
```bash
pytest -q
```

### Run a Simple Example

For a guided, hands-on introduction, visit the **Tutorials** section of the documentation:

https://tannertorrey3.github.io/DSG-JIT/tutorials/

These walk through SLAM, voxel fields, dynamic scene graphs, optimization, and JIT acceleration step‑by‑step.
```bash
python experiments/exp06_dynamic_trajectory.py
```

---

## Why This System Must Exist

1. Robotics Needs a Unified Representation

Robots currently run SLAM separately from semantic understanding and separately from scene-graph reasoning.
This creates inconsistencies, duplicated work, and limits closed-loop decision making.

A unified differentiable 3D world model eliminates this fragmentation.

2. Neural Rendering Models Need Structure

NeRFs and 3D Gaussians model appearance well, but lack:
-	Object boundaries
-	Spatial relationships
-	Room / topology structure
-	Multi-agent consistency

A dynamic scene graph provides this missing structure.

3. Scene Graphs Need Modern Optimization

Systems like Kimera and Hydra have proven scene graphs useful, but:
-	They rely on slow, CPU-bound optimization
-	Graph updates are not differentiable
-	They cannot incorporate neural fields
-	Loop closures require expensive hand-coded solvers

A JIT-compiled backend removes these constraints.

4. Differentiable Programming Enables Learning

With a differentiable world model, a system can learn:
-	Sensor models
-	Data association
-	Semantic priors
-	Graph connectivity
-	Object persistence
-	Planning costs

This is impossible with current non-differentiable pipelines.

---

## Vision

A fully integrated spatial intelligence engine—real-time, adaptive, learnable, and structurally grounded—capable of powering next-generation robotics, AR systems, foundational 3D models, and embodied AI.

---

## Core Features

- Differentiable factor graph engine (SE3 + Euclidean)
- JIT‑compiled nonlinear least squares
- SE3 manifold Gauss‑Newton solver
- Voxel grid support with smoothness + observation factors
- Learnable parameters:
  - Odom measurements
  - Voxel observation points
  - Factor‑type weights
- Differentiable Scene Graph structure
- Supports hybrid SE3 + voxel joint optimization

---

## System Architecture

Below is the high-level structural architecture guiding DSG-JIT development.

The system is composed of five major subsystems, each responsible for a specific layer of perception and reasoning.

---

1. Sensor Frontend

Responsible for converting raw sensor data into a structured state suitable for optimization.

Inputs
-	RGB / RGB-D
-	LiDAR / Depth
-	IMU
-	Multimodal (optional)

Outputs
-	Frame-to-frame motion estimates
-	Initial point clouds / depth maps
-	Per-pixel semantics (optional)

Role
Provide fast, incremental measurements that feed directly into SLAM and neural reconstruction modules.

---

2. JIT-Compiled SLAM Backend

A fully differentiable, GPU-accelerated backend that performs:
-	Pose graph optimization
-	Loop closure correction
-	Map deformation via deformation graphs
-	Sparse nonlinear least squares

This replaces traditional C++/GTSAM with JIT-generated solvers (JAX, Taichi, Dr.Jit, TorchInductor).

Why This Matters
-	Kernels fuse automatically
-	Jacobians are auto-derived
-	Massive parallelism (GPU / TPU)
-	Online learning of factor weights and priors
-	Real-time updates even for large-scale scenes

---

3. Neural Field Module (NeRF / Gaussians)

Encodes dense geometry and appearance information.

Responsibilities
-	Maintain neural radiance or Gaussian scene representation
-	Incrementally update the neural field using new sensor data
-	Provide differentiable rendering for optimization and supervision
-	Act as the geometric backbone for object & room segmentation

This module is fully differentiable and JIT-compiled for fast volumetric rendering.

Why This Matters
-	Dense geometry with high visual fidelity
-	Enables photometric residuals in SLAM
-	Supports dynamic objects and multi-agent consistency

---

4. Dynamic 3D Scene Graph Layer

A hierarchical structure that organizes the world into meaningful elements:
-	Places / topology
-	Rooms / corridor structure
-	Objects
-	Agents
-	Structural elements (walls, floors, ceilings)
-	Semantic relations (on, next to, inside, adjacent, etc.)

Key Responsibilities
-	Maintain relationships as the metric map changes
-	Update structure after loop closures
-	Support querying and reasoning
-	Tie semantics directly into optimization processes

This becomes the primary world model for planning and higher-level intelligence.

---

5. Global Optimization & Reasoning Engine

A unified optimization layer that ties modules together.

What it optimizes:
-	Robot trajectory
-	Neural field parameters
-	Object poses
-	Room centroids and topology
-	Graph connectivity
-	Deformation graph nodes
-	Semantic consistency factors
-	Multi-robot alignment (optional)

All of this is JIT-compiled, enabling high-frequency updates unachievable in traditional pipelines.

What it enables:
-	End-to-end differentiable mapping
-	Joint geometric + semantic optimization
-	Real-time global consistency
-	Learning-based priors and graph structures
-	Closed-loop integration with planning/control systems

---

## Architecture (Summary)

- Sensor Frontend
- JIT‑Compiled SLAM Backend
- Neural Field Module
- Dynamic Scene Graph Layer
- Global Optimization & Reasoning Engine

---

## Roadmap & Development Phases

Phase 1 — Core Framework Setup
-	Establish repo structure
-	Define abstract data types (poses, factors, nodes, fields)
-	Integrate JIT backend of choice (JAX or Taichi recommended)

Phase 2 — Minimal SLAM + Scene Graph Prototype
-	Build simple pose graph
-	Add basic room/object segmentation
-	Implement dynamic scene graph updates

Phase 3 — Neural Field Integration
-	Add Gaussian or NeRF reconstruction
-	Enable differentiable rendering
-	Connect neural fields to graph structure

Phase 4 — Unified Optimization
-	Merge SLAM, neural field, and scene graph optimizers
-	Implement end-to-end differentiable update pipeline
-	Add loop closure + graph deformation support

Phase 5 — Scaling & Real-World Validation
-	Multi-robot support
-	Large-scale scenes
-	Real sensor datasets
-	Integration with planning and embodied AI

---

## Intended Outcomes
-	A new class of real-time, differentiable 3D world models
-	A research platform for robotics, AR/VR, and embodied AI
-	A foundation for next-generation, geometry-aware foundation models
-	A future-proof architecture that merges SLAM, neural rendering, and reasoning

---

## Current Status

The differentiable SLAM + voxel + scene‑graph core is operational.
26/26 tests pass, including:
- SE3 chain optimization
- Voxel point learning
- Learnable factor‑type weights
- Hybrid SE3 + voxel joint learning (hero test)

Phase 5 work has begun:
- API cleanup
- Benchmarks
- Documentation and examples

---

## Benchmarks

To validate performance of the JIT‑compiled nonlinear optimizer, DSG‑JIT includes three core benchmarks:

### 1. SE3 Gauss–Newton Benchmark
200‑pose chain, 20 GN iterations.

| Mode     | Time (ms) | Notes |
|----------|-----------:|-------|
| **JIT**      | ~51.8 ms  | After compile |
| **No‑JIT**   | ~376,099 ms | Pure Python/JAX |

**Speedup:** ~7,260×  
**Trajectory error:** near‑zero, poses optimized to [0 … 199] within floating‑point epsilon.

---

### 2. Voxel Chain Gauss–Newton Benchmark
500‑voxel smoothness chain, 20 GN iterations.

| Mode     | Time (ms) | Notes |
|----------|-----------:|-------|
| **JIT**      | ~96 ms   | Fast, stable |
| **No‑JIT**   | ~3,045 ms | CPU‑only solve |

**Speedup:** ~31×  
**Voxel positions:** converge to linear chain with sub‑millimeter error.

---

### 3. Hybrid SE3 + Voxel Benchmark (Hero)
50 SE3 poses + 500 voxels jointly optimized over mixed manifolds.

| Mode     | Time (ms) | Notes |
|----------|-----------:|-------|
| **JIT**      | ~149.8 ms | Includes compile + manifold updates |
| **No‑JIT**   | ~97,500 ms | Extremely slow without JIT |

**Speedup:** ~650×  
**Results:**  
- Poses converge to exact trajectory [0 … 49]  
- Voxels converge to [0 … 499] with small noise (<1e‑3)

---

---

## Contributing

Contributions to **DSG-JIT** are welcome and encouraged.

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/TannerTorrey3/DSG-JIT.git
   cd DSG-JIT
   ```

2. **Create a new feature or fix branch**
   ```bash
   git checkout -b feature/my-enhancement
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   export PYTHONPATH=DSG-JIT/dsg-jit
   ```

4. **Ensure tests pass**
   ```bash
   pytest -q
   ```

5. **Run style checks (optional)**
   ```bash
   ruff check .
   black .
   ```

6. **Submit a Pull Request**

   PRs should:
   - Be focused (one feature/fix per PR)
   - Include tests when applicable
   - Update documentation where relevant
   - Pass continuous integration checks

### Reporting Bugs

Please open an issue using the Bug Report template.  
Include logs, stack traces, and a minimal reproducible example when possible.

### Feature Requests

Use the Feature Request template and describe:
- Motivation
- Proposed API or behavior
- Alternatives considered

### Documentation Contributions

Documentation lives in `docs/`.  
Improvements, corrections, or new examples are appreciated.

### Code of Conduct

By contributing to this project, you agree to maintain a professional, respectful, and collaborative environment.

