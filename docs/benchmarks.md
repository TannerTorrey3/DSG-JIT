

# Benchmarks

This page presents reproducible performance benchmarks for **DSG‑JIT**, demonstrating the efficiency and scalability of our differentiable Scene Graph optimizer across SE(3) pose chains, voxel grids, and hybrid problems.  
All benchmarks were run on macOS (Apple Silicon) using JAX CPU back‑ends.

---

## 🚀 Overview

DSG‑JIT achieves **31-1000× speedups** over pure Python Gauss‑Newton solvers thanks to JAX vectorization, JIT compilation, and careful manifold‑aware optimization.

Benchmarks included:

1. **SE3 Gauss‑Newton Chain Benchmark**  
2. **Voxel Chain Gauss‑Newton Benchmark**  
3. **Hybrid SE3 + Voxel Gauss‑Newton Benchmark**

---

# 1. SE3 Gauss‑Newton Benchmark

Solves a chain of `num_poses` SE(3) variables connected by odometry constraints.

### **200‑Pose Chain (JIT Enabled)**

```
Elapsed time: 51.802 ms
pose0 (opt):   [7.94e-06 4.59e-06 0 ...]
poseN-1 (opt): [1.990e+02 5.88e-04 0 ...]
```

### **200‑Pose Chain (No JIT)**

```
Elapsed time: 376,098 ms   (~6.27 minutes)
```

> **Result:** JIT is ~ **7250× faster**.

---

# 2. Voxel Chain Gauss‑Newton Benchmark

Solves a smooth‑regularized chain of voxel centers.

### **500‑Voxel Chain (JIT Enabled)**

```
Elapsed time: 96.192 ms
voxel0 (opt):   [0.1457 0.0425 0]
voxelN-1 (opt): [494.0813 0.018 0]
```

### **500‑Voxel Chain (No JIT)**

```
Elapsed time: 3,044.991 ms
```

> **Result:** JIT is ~ **31× faster**.

---

# 3. Hybrid SE3 + Voxel Benchmark

Solves both a 50‑pose SE(3) chain **and** a 500‑voxel chain simultaneously.

### **JIT Enabled**

```
Elapsed time: 149.832 ms
pose0:     [-4.19e-19 4.00e-10 0 ...]
poseN-1:   [4.90e+01 1.28e-08 0 ...]
voxelM-1:  [4.989959e+02 -1.199e-03 0]
```

### **No JIT**

```
Elapsed time: 97,500 ms
```

> **Result:** JIT is ~ **650× faster**.

---

# 📊 Summary Table

| Benchmark | JIT Time | Non‑JIT Time | Speedup |
|----------|----------|--------------|---------|
| SE3 Chain (200) | 51.8 ms | 376,098 ms | **~7250×** |
| Voxel Chain (500) | 96 ms | 3,045 ms | **~31×** |
| Hybrid SE3+Voxel | 150 ms | 97,500 ms | **~650×** |

---

# 📦 How to Run Benchmarks

From the project root:

```bash
python3 benchmarks/bench_gauss_newton_se3.py
python3 benchmarks/bench_voxel_chain.py
python3 benchmarks/bench_hybrid_se3_voxel.py
```

---

# 📝 Notes

- The first JIT call includes compilation time (can take minutes for large graphs).  
- Timings reported above exclude compilation and measure **cached execution**.  
- Performance scales linearly with problem size once compiled.

---

# 🔬 Reproducibility

To ensure consistent benchmarking:

```bash
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS=--xla_cpu_enable_fast_math=true
```

---

# ✅ Conclusion

DSG‑JIT dramatically accelerates differentiable scene‑graph optimization, delivering up to **7000× speedups** on real‑world sized problems.

These benchmarks confirm:

- JIT + vectorized Gauss‑Newton is extremely efficient  
- Hybrid problems remain fast and stable  
- DSG‑JIT is ready for large‑scale SLAM and differentiable graphics tasks  

---