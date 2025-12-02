

# DSG‑JIT Tutorials Overview

Welcome to the **DSG‑JIT Tutorials Hub** — a structured, hands‑on guide to understanding and using the Differentiable Scene Graph Just‑In‑Time (DSG‑JIT) framework.

These tutorials are designed to move from foundational concepts toward advanced hybrid learning workflows, providing practical, minimal examples for every major subsystem in DSG‑JIT.

---

## What You’ll Learn

### **Core Concepts**
Get familiar with the building blocks of DSG‑JIT:
- How factor graphs work and how states are optimized.
- The SE(3) manifold and Lie‑group operations.
- How semantic scene graphs are constructed and maintained.
- How objects, rooms, places, and agents form a unified spatial abstraction.

### **SE(3) & SLAM**
Dive into:
- Differentiable odometry chains  
- Dynamic trajectories  
- Learnable factor‑type weights  
- Hybrid SE(3) + voxel optimization pipelines  
- End‑to‑end differentiable SLAM examples  

These tutorials bridge classical geometry with modern differentiable optimization.

### **Voxel Grids & Spatial Fields**
Learn how DSG‑JIT handles:
- Voxel observation modeling  
- Multi‑voxel parameter learning  
- Differentiable spatial field estimation  
- Hybrid optimization using both geometric and volumetric cues  

Ideal for researchers working on Neural Fields, occupancy mapping, or sensor‑fusion‑based perception.

### **Static & Dynamic Scene Graphs**
Understand hierarchical world modeling at scale:
- Static scene graph construction  
- Object anchoring and semantic relations  
- Dynamic scene graphs with multi‑agent temporal layers  
- 3D visualization of complex DSGs  

These tutorials show how DSG‑JIT organizes high‑level spatial semantics.

### **Sensors & Fusion**
Explore the sensor stack:
- Synthetic Camera, LiDAR, and IMU simulators  
- Streaming, fusion callbacks, and measurement conversion  
- Range‑based DSG construction  
- End‑to‑end mapping from raw sensor samples  

This layer demonstrates how sensors feed into the world model and factor graph.

### **Learning & Hybrid Modules**
Learn differentiable components tightly integrated into the DSL:
- Learnable factor-type weights  
- Multi‑modal learning (SE(3) + voxel)  
- Joint optimization of geometry and field representations  
- Trainer‑based workflows using JAX + JIT  

Useful for machine‑learning‑based mapping and hybrid perception models.

### **JAX & JIT Workflows**
See how DSG‑JIT leverages JAX to:
- Construct differentiable residuals  
- JIT‑compile optimization routines  
- Build training loops that interleave geometry and learning  

---

## How to Use These Tutorials

Each tutorial includes:
- **Categories** (e.g., Core Concepts, Dynamic Scene Graphs)  
- **Overview** explaining the goal and context  
- **Full code listing** from the experiment  
- **Step‑by‑step explanation** of the logic  
- **Summary** capturing key takeaways  

You can read them in order or jump directly to the area relevant to your research.

---

## Recommended Starting Points

If you're new to DSG‑JIT, begin here:

1. **Mini World Factor Graph** — foundational concepts  
2. **Manifold Geometry SE(3)** — essential mathematical background  
3. **Scene Graph World** — your first semantic world model  
4. **Dynamic Trajectories** — motion estimation and odometry  
5. **Visualizing a Factor Graph in 3D** — debugging and intuition  

---

## Have Suggestions?

If you'd like additional tutorials or expanded examples, feel free to open an issue on GitHub — contributions and requests are welcome!

---

Continue exploring the tutorials using the navigation menu on the left.