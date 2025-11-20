# DSG-JIT: A JIT‑Compiled, Differentiable 3D Dynamic Scene Graph Engine

## Overview

Modern spatial intelligence systems—SLAM pipelines, neural rendering models, and 3D scene graph frameworks—remain fragmented.
Each solves part of the perception problem, but none unify:
	•	Metric accuracy (SLAM)
	•	High-fidelity geometry & appearance (Neural Fields / Gaussians)
	•	Semantic structure & reasoning (Scene Graphs)
	•	Real-time global consistency (Incremental optimization)
	•	End-to-end differentiability (learning cost models, priors, & structure)

DSG-JIT is a new architecture that merges these into one coherent, JIT-compiled, differentiable system.

The goal is simple:

A unified pipeline that builds, optimizes, and reasons over a complete 3D world model in real time—fusing SLAM, neural fields, and dynamic scene graphs into a single optimized computational graph.

This repository serves as the structural roadmap for developing that system.

---

## Quickstart

### Install
```bash
git clone <repo>
cd jit-hydra
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

### Run Tests
```bash
pytest -q
```

### Run a Simple Example
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
	•	Object boundaries
	•	Spatial relationships
	•	Room / topology structure
	•	Multi-agent consistency

A dynamic scene graph provides this missing structure.

3. Scene Graphs Need Modern Optimization

Systems like Kimera and Hydra have proven scene graphs useful, but:
	•	They rely on slow, CPU-bound optimization
	•	Graph updates are not differentiable
	•	They cannot incorporate neural fields
	•	Loop closures require expensive hand-coded solvers

A JIT-compiled backend removes these constraints.

4. Differentiable Programming Enables Learning

With a differentiable world model, a system can learn:
	•	Sensor models
	•	Data association
	•	Semantic priors
	•	Graph connectivity
	•	Object persistence
	•	Planning costs

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
	•	RGB / RGB-D
	•	LiDAR / Depth
	•	IMU
	•	Multimodal (optional)

Outputs
	•	Frame-to-frame motion estimates
	•	Initial point clouds / depth maps
	•	Per-pixel semantics (optional)

Role
Provide fast, incremental measurements that feed directly into SLAM and neural reconstruction modules.

---

2. JIT-Compiled SLAM Backend

A fully differentiable, GPU-accelerated backend that performs:
	•	Pose graph optimization
	•	Loop closure correction
	•	Map deformation via deformation graphs
	•	Sparse nonlinear least squares

This replaces traditional C++/GTSAM with JIT-generated solvers (JAX, Taichi, Dr.Jit, TorchInductor).

Why This Matters
	•	Kernels fuse automatically
	•	Jacobians are auto-derived
	•	Massive parallelism (GPU / TPU)
	•	Online learning of factor weights and priors
	•	Real-time updates even for large-scale scenes

---

3. Neural Field Module (NeRF / Gaussians)

Encodes dense geometry and appearance information.

Responsibilities
	•	Maintain neural radiance or Gaussian scene representation
	•	Incrementally update the neural field using new sensor data
	•	Provide differentiable rendering for optimization and supervision
	•	Act as the geometric backbone for object & room segmentation

This module is fully differentiable and JIT-compiled for fast volumetric rendering.

Why This Matters
	•	Dense geometry with high visual fidelity
	•	Enables photometric residuals in SLAM
	•	Supports dynamic objects and multi-agent consistency

---

4. Dynamic 3D Scene Graph Layer

A hierarchical structure that organizes the world into meaningful elements:
	•	Places / topology
	•	Rooms / corridor structure
	•	Objects
	•	Agents
	•	Structural elements (walls, floors, ceilings)
	•	Semantic relations (on, next to, inside, adjacent, etc.)

Key Responsibilities
	•	Maintain relationships as the metric map changes
	•	Update structure after loop closures
	•	Support querying and reasoning
	•	Tie semantics directly into optimization processes

This becomes the primary world model for planning and higher-level intelligence.

---

5. Global Optimization & Reasoning Engine

A unified optimization layer that ties modules together.

What it optimizes:
	•	Robot trajectory
	•	Neural field parameters
	•	Object poses
	•	Room centroids and topology
	•	Graph connectivity
	•	Deformation graph nodes
	•	Semantic consistency factors
	•	Multi-robot alignment (optional)

All of this is JIT-compiled, enabling high-frequency updates unachievable in traditional pipelines.

What it enables:
	•	End-to-end differentiable mapping
	•	Joint geometric + semantic optimization
	•	Real-time global consistency
	•	Learning-based priors and graph structures
	•	Closed-loop integration with planning/control systems

---

## Architecture (Text Summary)

- Sensor Frontend
- JIT‑Compiled SLAM Backend
- Neural Field Module
- Dynamic Scene Graph Layer
- Global Optimization & Reasoning Engine

---

## Roadmap & Development Phases

Phase 1 — Core Framework Setup
	•	Establish repo structure
	•	Define abstract data types (poses, factors, nodes, fields)
	•	Integrate JIT backend of choice (JAX or Taichi recommended)

Phase 2 — Minimal SLAM + Scene Graph Prototype
	•	Build simple pose graph
	•	Add basic room/object segmentation
	•	Implement dynamic scene graph updates

Phase 3 — Neural Field Integration
	•	Add Gaussian or NeRF reconstruction
	•	Enable differentiable rendering
	•	Connect neural fields to graph structure

Phase 4 — Unified Optimization
	•	Merge SLAM, neural field, and scene graph optimizers
	•	Implement end-to-end differentiable update pipeline
	•	Add loop closure + graph deformation support

Phase 5 — Scaling & Real-World Validation
	•	Multi-robot support
	•	Large-scale scenes
	•	Real sensor datasets
	•	Integration with planning and embodied AI

---

## Intended Outcomes
	•	A new class of real-time, differentiable 3D world models
	•	A research platform for robotics, AR/VR, and embodied AI
	•	A foundation for next-generation, geometry-aware foundation models
	•	A future-proof architecture that merges SLAM, neural rendering, and reasoning

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
