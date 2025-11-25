# 📚 Glossary & Key Concepts (Alphabetical)

This glossary lists every DSG‑JIT term in strict alphabetical order for instant lookup.

---

## A

### **Agent**
A time‑varying entity (robot/person) with a pose trajectory over time.

### **Agent Pose Node**
A pose node associated with a specific agent + timestamp in the DynamicSceneGraph.

---

## B

### **Bearing**
A unit‑direction vector from a sensor to a target (camera or LiDAR).

---

## C

### **Calibration (Extrinsics)**
Rigid transform from a sensor frame to the robot/base frame.

### **Camera Intrinsics**
3×3 calibration matrix mapping pixels → normalized rays.

### **CameraMeasurement**
Typed camera measurement containing rays, image, timestamps, etc.

### **Chain Residual (SE3 Chain)**
Residual enforcing that a sequence of SE(3) poses matches integrated motion.

### **Conversion Layer (Sensor Conversion Layer)**
Converts raw sensor samples into typed measurements and measurement factors.

---

## D

### **DSG — Dynamic Scene Graph**
A time‑varying scene graph combining geometry + semantics + temporal structure.

### **DSG Layer**
A semantic layer (rooms, places, objects, agents, voxels, etc).

### **DSGTrainer**
Bi‑level optimization utility (inner GN, outer hyperparameter learning).

### **DynamicSceneGraph**
Holds time‑indexed agent poses & temporal edges.

### **Dynamic Voxel Field**
A voxel grid that changes over time (dynamic/deformable environments).

### **Dataset Stream**
Any sensor input driven by FunctionStream or FileRangeStream.

---

## E

### **Edge (Scene Graph Edge)**
Relationship between two scene‑graph nodes (room–place, place–object, etc).

### **Euclidean Variable**
A variable in ℝⁿ (e.g., landmark, voxel, calibration scalar).

---

## F

### **Factor**
A residual constraint relating one or more variables.

### **Factor Graph**
Bipartite graph of variables ↔ factors.

### **FileRangeStream**
Sensor stream reading structured samples from a text/log file.

### **FunctionStream**
Sensor stream driven by a Python generator function.

---

## G

### **Gauss–Newton Optimization**
Nonlinear least‑squares solver implemented on manifolds in DSG‑JIT.

### **Graph Export (SceneGraph Export)**
Converts graph nodes into VisNode/VisEdge for visualization.

---

## H

### **Hybrid Factor Graph**
A factor graph mixing SE(3), Euclidean, and voxel variables in one optimization.

---

## I

### **IMU Delta Integration (Integrate IMU Delta)**
Creates a small SE(3) increment from raw IMU samples.

### **Inner Optimization (Inner GN / Inner Gradient Descent)**
Solves the factor graph while outer parameters are fixed.

---

## J

### **Jacobian**
Matrix of partial derivatives of residuals wrt variable components.

### **JIT Compilation**
JAX compilation of residual functions for high‑performance SLAM.

---

## L

### **Landmark**
A 3D point in the world (usually Euclidean).

### **Landmark Prior**
Unary factor constraining a landmark near a specific 3D location.

### **Layered Visualization**
Rendering style placing node types on separate z‑planes.

### **LiDAR Scan**
List of ranges/angles or rays converted to LidarMeasurement.

### **Lie Group**
Differentiable group structure for SE(3), SO(3), etc.

---

## M

### **Manifold**
Non‑Euclidean space (like SE(3)) requiring special update rules.

### **Manifold Gauss–Newton**
Lie‑group‑aware Gauss–Newton with retract/exp/log operations.

### **Manifold Metadata**
Describes each variable’s manifold type + slice in packed state vector.

### **Measurement**
Raw or typed sensor output (camera, lidar, imu, odom).

### **MeasurementFactor**
Factor generated from a typed measurement.

---

## N

### **Named Object**
Scene‑graph object with semantic identifier (e.g. “chair_1”).

---

## O

### **Object Node**
A semantic object (chair/table/etc) connected to a place.

### **Observation Model**
Predicts expected sensor measurement given state x.

### **Odometry**
Relative motion factor between two poses.

### **Odom_tx**
1D translation‑only odom helper for quick SE(3) edges.

### **Outer Optimization**
Top‑level hyperparameter update loop above the inner GN solver.

---

## P

### **Place**
A mid‑level scene‑graph node (semantic waypoint, corridor, anchor).

### **Place3D**
3D spatial place node used in realistic indoor environments.

### **Place Attachment**
Semantic constraint linking an agent pose → place.

### **Plot Functions**
Graph/scene‑graph visualization utilities in world.visualization.

### **Pose**
An SE(3) rigid transform.

---

## R

### **Range**
Scalar distance reading (LiDAR, UWB, depth).

### **Range Observation**
Factor derived from a distance measurement.

### **Residual**
Prediction error: r(x) = f(x) − z.

---

## S

### **Scene Graph**
Hierarchical structure representing geometry + semantics.

### **SceneGraphWorld**
High‑level API wrapping factor graph, semantic nodes, and helpers.

### **SE(3) — Special Euclidean Group**
Rigid‑body transform group for robot poses.

### **SO(3)**
Rotation‑only Lie group.

### **Semantic Edge**
High‑level symbolic edge between semantic nodes.

### **SensorFusionManager**
Polls sensors and delivers typed measurements + callbacks.

### **Sensor Streams (FunctionStream / FileRangeStream)**
Abstract data sources for camera/lidar/imu streams.

---

## T

### **Temporal Factor**
Factor connecting time‑indexed nodes.

### **Temporal Layer**
Dynamic layer storing agent trajectories and time relations.

### **Trajectory (Agent Trajectory)**
Ordered sequence of time‑indexed SE(3) poses.

### **Type Weights (Learnable Weights)**
Learned weights for residual types in bi‑level training.

---

## V

### **Variable**
Optimizable unknown in the factor graph.

### **VisNode / VisEdge**
Lightweight structures for 2D/3D visualization.

### **Voxel**
Volumetric grid cell.

### **Voxel Grid**
3D grid of voxels for dense mapping experiments.

### **Voxel Observation Model**
Residual connecting voxel center to observed 3D point.

### **Voxel Smoothness Residual**
Factor connecting adjacent voxels enforcing smooth geometry.

---

## W

### **World Model**
Top‑level container for factor graph, variables, and optimization.

---

## Z

### **Zero‑Mean Gaussian Noise**
Standard additive noise assumption for measurements.