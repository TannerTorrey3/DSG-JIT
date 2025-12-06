# Tutorial: Sensor DSG Mapping (End-to-End)
**Category:** Sensors & Fusion, Dynamic Scene Graphs

---

## Overview

This tutorial demonstrates the *complete* pipeline for taking synthetic
camera, LiDAR, and IMU data streams → converting them into typed measurements →
constructing factors → inserting them into a **WorldModel + FactorGraph** →
optimizing with **Gauss–Newton** → and visualizing the resulting DSG-driven map.

This is intentionally lightweight and synthetic — the goal is to show the
**interfaces**, not build a full SLAM system.

---

## Key Concepts

- **Sensor Streams**  
  `FunctionStream` generates synthetic camera, LiDAR, and IMU data.

- **Measurement Conversion**  
  We convert raw dictionaries into:
  - `CameraMeasurement`
  - `LidarMeasurement`
  - `IMUMeasurement`

- **IMU Integration**  
  `integrate_imu_delta()` produces SE(3) increments from IMU acceleration.

- **Range Mapping**  
  LiDAR ranges are converted to **prior factors** that constrain a landmark.

- **World Modeling**  
  `SceneGraphWorld` + `WorldModel` store robot poses, odometry, and landmarks.

- **Optimization**  
  `gauss_newton_manifold()` optimizes SE(3) + Euclidean variables.

- **Visualization**  
  `plot_factor_graph_3d()` renders the final DSG-driven factor graph.

---

## 1. Building the World Model

We create a simple SE(3) pose chain:

- Pose0 = [0,0,0]
- Pose1 = [1,0,0]
- Pose2 = [2,0,0]
- …  
Each step is connected by **additive odometry**.

A static **landmark** is placed at `x = 5.0`.

---

## 2. Creating Synthetic Sensor Streams

We simulate three independent sensors:

### Camera Stream
Produces a dummy \(1 \times 1\) grayscale image and timestamp.

### LiDAR Stream
Simulates a **single-beam scanner** with a fixed range (default 5.0 m).

### IMU Stream
Constant acceleration in +x; enough to generate SE(3) deltas via:

```python
dxi = integrate_imu_delta(imu_meas, dt=imu_meas.dt)
```

---

## 3. Converting Raw Samples to Measurement Types

We parse raw dictionaries into typed messages:

```python
cam_meas   = raw_sample_to_camera_measurement(raw_cam)
lidar_meas = raw_sample_to_lidar_measurement(raw_lidar)
imu_meas   = raw_sample_to_imu_measurement(raw_imu)
```

IMU deltas accumulate into:

```python
fused_delta = sum(dxi)
```

---

## 4. LiDAR → Range Priors (Landmark Constraints)

LiDAR ranges become priors on the landmark:

```python
target = [mean_range, 0, 0]  # in world frame
wm.add_factor(
    f_type="prior",
    var_ids=[landmark_id],
    params={"target": target, "weight": 1/sigma^2}
)
```

One prior is added **per pose** (demonstration only), all constraining the
same landmark.

---

## 5. Solving with Manifold Gauss–Newton

We pack the state and build manifold metadata:

```python
x0, index = wm.pack_state()
block_slices, manifold_types = build_manifold_metadata(packed_state=wm.pack_state(),fg=wm.fg)
```

Then solve:

```python
x_opt = gauss_newton_manifold(
    residual_fn,
    x0,
    block_slices,
    manifold_types,
    cfg,
)
```

The resulting poses and landmark are printed.

---

## 6. Visualizing the Result

The factor graph is rendered in 3D:

```python
plot_factor_graph_3d(fg, show_labels=True)
```

This shows:
- SE(3) pose chain
- Landmark node
- LiDAR–derived prior factors

---

## Full Code (From `exp22_sensor_dsg_mapping.py`)

```python
<the user-provided full experiment code should be shown here if desired>
```

(You may embed the full code or link to the file in your repo.)

---

## Summary

This experiment demonstrates the *full pipeline* for turning synthetic sensor
data into DSG constraints:

1. Generate raw sensor samples  
2. Convert to structured measurements  
3. Build factors  
4. Inject into a live WorldModel  
5. Solve with manifold GN  
6. Visualize the optimized 3D structure  

This tutorial establishes the template for **true full-stack SLAM** in DSG‑JIT,
connecting sensors → factors → optimization → visualization.