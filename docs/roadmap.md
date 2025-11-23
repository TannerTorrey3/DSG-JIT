# Project Roadmap

This roadmap outlines the upcoming milestones for **DSG‑JIT**, spanning stability, features, optimization, and long‑term research directions.  
It is divided into phases with clear goals, deliverables, and stretch objectives.

---

## 🚀 Phase 1 — Core Stabilization (Completed)
**Status:** ✔️  
**Summary:** Completed foundational work on optimizer, SE(3) manifold, voxel grid operators, scene graph, residuals, and testing suite.

Deliverables:
- JIT‑friendly factor graph engine  
- Full SE(3) geodesic math and differentiable odometry  
- Voxel smoothness, point observation, and multi‑parameter wrappers  
- Unified Gauss‑Newton solver  
- 26/26 passing tests  

---

## 🔧 Phase 2 — Differentiable Scene Graph (Completed)
**Status:** ✔️  
Introduced the world model, scene graph relations, entity system, and DSG‑based optimization hooks.

Deliverables:
- Relational scene graph (parent/child, rigid attachments, etc.)
- Voxel + SE(3) hybrid factors  
- Differentiable world model component  
- Param‑learnable factors for odom & voxel observations  

---

## 🧪 Phase 3 — Experiments & Validation (Completed)
**Status:** ✔️  
All algorithmic experiments defined, executed, and reproduced:
- Learnable type weights  
- Learnable odom measurements  
- Multi‑voxel observation learning  
- Hybrid SE3 + voxel joint learning (hero experiment)

---

## 📈 Phase 4 — Benchmarks & Performance (Completed)
**Status:** ✔️  
Three official benchmarks implemented:
- Pure SE3 factor graph  
- Voxel grid smoothness chain  
- Hybrid SE3 + voxel chain  

JIT speedups: **31–7000×** depending on graph size.

---

## 🤖 Phase 5 — Real‑World Sensors & SLAM Integration (Completed)
Integration of DSG‑JIT into full robotics pipelines.

Planned:
- Real LIDAR factor  
- RGB‑D depth factor  
- Visual landmarks  
- Camera intrinsics/extrinsics calibration via DSG  
- Data loaders for KITTI / TUM RGB‑D  

Stretch:
- IMU pre‑integration  
- Multi‑robot DSG fusion  

---

## 📚 Phase 6 — Public Documentation (In Progress)
**Status:** 🟡  
DSG-JIT current development stage.

Remaining tasks:
- Polish docs (architecture, API, examples, benchmarks)  
- Generate gallery diagrams  
- Ensure docs build cleanly under GitHub Pages  
- Add narrative tutorial series  

Stretch:
- Animated diagrams showing optimization steps  
- Interactive code sandboxes  

---

## 🧩 Phase 7 — Packaging & Distribution (In Progress)
**Status:** ⏳

Planned deliverables:
- `pip install dsg-jit`  
- Versioned releases + changelog  
- Improved import layout  
- Automated lint + format + test pipeline  
- Pre‑commit hooks  
- GitHub Actions for:
  - type-check  
  - tests  
  - benchmark snapshot  
  - docs deploy  
- ROS2 wrapper package  

Stretch:
- Optional CUDA/XLA GPU acceleration  
- Wheels for Mac, Linux, Windows  

---

## 🧬 Phase 8 — Research Extensions (Long‑Term)
- Neural scene graphs  
- Neuro-symbolic factor graphs  
- DSG‑based reinforcement learning  
- Learned Jacobian priors  
- Generative world‑model layers  

Potential publications:
- *Differentiable Scene Graph Optimization via JIT Factor Graphs*  
- *Hybrid SE3–Voxel Graphs for Dense Reconstruction*  
- *End‑to‑End Learnable SLAM via Multi‑Residual Differentiation*

---

## 🏁 Phase 9 — 1.0 Stable Release (Future)
The first fully stable release of DSG‑JIT.

Requirements:
- Complete documentation  
- Full packaging  
- All critical benchmarks validated  
- Public examples + tutorials  
- Long‑term support policy  
- Optimization safety & performance guarantees  

---

## 🏗️ Phase 7 — Advanced DSL & Autogeneration (Planned)
A domain‑specific "DSG Modeling Language" for declarative factor-graph design.

Features:
- YAML/JSON graph definitions  
- Auto‑generated optimization graphs  
- Auto‑differentiated residual templates  
- Scenegraph compiler → JIT graph  

Stretch:
- Visual graph editor  
- Drag‑and‑drop factor construction UI  
- "Graph debugger" visualization  

---

## Contributing
Contributions are welcome!  
Upcoming needs:
- More tests  
- More factor types  
- Benchmark expansions  
- Doc improvements  
- Tutorials + examples  

---

