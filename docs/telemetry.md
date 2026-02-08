# Telemetry

*Anonymous usage analytics to help us improve DSG-JIT.*

---

## Why We Collect Telemetry

DSG-JIT is an open-source research project, and telemetry helps us:

- **Understand how people use the library** — Which features are popular? Which are rarely used?
- **Identify common errors** — If many users hit the same bug, we can prioritize fixing it
- **Improve performance** — Timing data helps us find slow operations to optimize
- **Guide development** — Real usage patterns inform our roadmap decisions

Your telemetry data directly helps make DSG-JIT better for everyone.

---

## What We Collect

### Session Information

When you first use DSG-JIT, we record a session start with:

| Attribute | Example | Description |
|-----------|---------|-------------|
| `ix.install_id` | `a1b2c3d4-...` | Random UUID, persistent per machine |
| `ix.session_id` | `e5f6g7h8-...` | Random UUID, per Python process |
| `dsgjit.version` | `0.1.0` | DSG-JIT package version |
| `runtime.python` | `3.11.5` | Python version |
| `runtime.os` | `darwin` | Operating system |
| `runtime.arch` | `arm64` | CPU architecture |
| `dsgjit.backend` | `gpu` | Active JAX backend (cpu/gpu/tpu) |
| `dsgjit.backend_available` | `both` | Available backends (cpu/gpu/both) |

### Operation Metrics

When you call instrumented functions, we record:

| Attribute | Example | Description |
|-----------|---------|-------------|
| `dsgjit.component` | `world` | Module being used |
| `dsgjit.op` | `optimize` | Function being called |
| `dsgjit.status` | `ok` | Success or error |
| Timing | (automatic) | How long it took |

---

## Instrumented Operations

### Scene Graph (`scene_graph`)

**Poses & Entities:**
- `add_pose_se3` — Add SE(3) pose
- `add_place1d`, `add_place3d` — Add 1D/3D place
- `add_room1d`, `add_room` — Add room
- `add_object3d`, `add_named_object3d` — Add 3D object
- `add_landmark3d` — Add landmark
- `add_voxel_cell` — Add voxel cell
- `add_agent_pose_se3` — Add agent pose

**Factors & Constraints:**
- `add_prior_pose_identity` — Identity prior on pose
- `add_odom_se3_additive`, `add_odom_se3_geodesic` — Odometry factors
- `add_range_measurement`, `add_agent_range_measurement` — Range measurements
- `add_temporal_smoothness`, `add_agent_temporal_smoothness` — Smoothness constraints
- `add_pose_landmark_relative`, `add_agent_pose_landmark_relative` — Landmark factors
- `add_pose_landmark_bearing`, `add_agent_pose_landmark_bearing` — Bearing factors
- `add_pose_voxel_point`, `add_agent_pose_voxel_point` — Voxel point factors
- `add_voxel_smoothness` — Voxel smoothness
- `add_voxel_point_observation` — Voxel observations
- `add_place_attachment`, `add_agent_pose_place_attachment` — Place attachments

**Attachments & Edges:**
- `attach_pose_to_place_x`, `attach_pose_to_room_x` — Pose attachments
- `attach_object_to_pose` — Object-pose attachment
- `add_room_place_edge`, `add_object_room_edge` — Graph edges

**Getters & Utilities:**
- `get_pose`, `get_place`, `get_object3d` — Retrieve values
- `dump_state` — Export state
- `optimize` — Run optimization
- `visualize_web` — Web visualization
- `enable_active_template`, `get_factor_memory`, `deactivate_factors_for_vars` — Template management

**Residuals:**
- `room_centroid_residual` — Room centroid computation
- `pose_place_attachment_residual` — Attachment residual

### World Model (`world`)

**Variables:**
- `add_variable` — Add generic variable
- `add_pose`, `add_room`, `add_place`, `add_object` — Add typed variables
- `add_agent_pose` — Add agent pose
- `set_variable_slot` — Set variable in slot
- `get_variable_value` — Retrieve variable

**Factors:**
- `add_factor` — Add generic factor
- `add_camera_bearings` — Camera bearing factors
- `add_lidar_ranges` — LiDAR range factors
- `add_imu_preintegration_factor` — IMU preintegration
- `configure_factor_slot` — Configure factor slot
- `register_residual`, `get_residual`, `get_residuals`, `list_residual_types` — Residual management

**Optimization:**
- `optimize` — Run Gauss-Newton optimization
- `build_objective` — Build objective function
- `build_residual_function_with_type_weights` — Build weighted residuals
- `build_residual_function_se3_odom_param_multi` — Build SE3 odom residuals
- `build_residual_function_voxel_point_param`, `build_residual_function_voxel_point_param_multi` — Build voxel residuals

**State Management:**
- `pack_state`, `unpack_state`, `unpack_state_inplace` — State packing
- `snapshot_state` — State snapshot
- `marginalize_variables`, `fixed_lag_marginalize` — Marginalization
- `init_active_template` — Initialize template

### Experiments (`experiment`)

Each experiment run is tracked:

| Experiment | Description |
|------------|-------------|
| `exp01_mini_world` | Mini world factor graph |
| `exp02_mini_world_se3_geodesic` | SE3 geodesic world |
| `exp03_scene_graph_world` | Scene graph world |
| `exp04_scene_graph_objects` | Scene graph with objects |
| `exp05_scene_graph_objects_jit` | JIT scene graph objects |
| `exp06_dynamic_trajectory` | Dynamic trajectory |
| `exp07_differentiable_voxel_obs` | Differentiable voxel observations |
| `exp08_differentiable_voxel_obs_theta` | Voxel obs with theta |
| `exp09_multi_voxel_param` | Multi-voxel parameters |
| `exp10_differentiable_se3_odom_chain` | SE3 odom chain |
| `exp11_learn_type_weights_odom_vs_attachment` | Learn type weights |
| `exp12_scenegraph_learnable_type_weights` | Scene graph type weights |
| `exp13_trainer_voxel_point_multi` | Voxel point trainer |
| `exp14_multi_voxel_param_and_weight` | Multi-voxel param + weight |
| `exp15_hybrid_se3_voxel_joint_learning` | Hybrid SE3 + voxel learning |
| `exp16_hero_hybrid_dsg` | Hero hybrid DSG |
| `exp17_visual_factor_graph` | Visual factor graph |
| `exp18_scenegraph_3d` | 3D scene graph |
| `exp18_scenegraph_demo` | Scene graph demo |
| `exp19_dynamic_scene_graph_demo` | Dynamic scene graph demo |
| `exp20_range_sensor_dsg` | Range sensor DSG |
| `exp21_sensor_fusion_demo` | Sensor fusion demo |
| `exp22_sensor_dsg_mapping` | Sensor DSG mapping |
| `exp23_sliding_window_sg` | Sliding window scene graph |

---

## Safe Arguments (Bucketed)

Some function arguments are recorded, but **only from an allowlist** and **always bucketed**:

| Argument | What We Record | Example |
|----------|----------------|---------|
| `method` | Optimization method | `"gn"` |
| `iters` | Bucketed iteration count | `"10-99"` (not `47`) |
| `var_type` | Variable type | `"se3"` |
| `factor_type`, `f_type` | Factor type | `"odom"` |
| `coord_index` | Bucketed coordinate index | `"1-9"` |
| `use_type_weights` | Boolean flag | `true` |
| `learn_odom` | Boolean flag | `false` |
| `learn_voxel_points` | Boolean flag | `true` |
| `active` | Boolean flag | `true` |

**Integer bucketing:** Instead of exact values, we record ranges:
`0`, `1-9`, `10-99`, `100-999`, `1k-9k`, `10k-99k`, `100k-999k`, `1M+`

---

## Error Information

When an error occurs:

| Attribute | Example | Description |
|-----------|---------|-------------|
| `error.type` | `ValueError` | Exception class name |
| `error.code` | `invalid_argument` | Categorized error type |
| `error.component` | `world` | Where it happened |
| `error.op` | `optimize` | What operation failed |

**Error codes are categorized**, not raw messages:

| Code | Exception Types |
|------|-----------------|
| `invalid_argument` | ValueError, TypeError, KeyError, IndexError, AttributeError |
| `shape_mismatch` | ShapeError |
| `numerical_issue` | FloatingPointError, OverflowError, ZeroDivisionError |
| `backend_error` | RuntimeError |
| `not_initialized` | NotImplementedError |
| `convergence_failure` | ConvergenceError |
| `io_error` | IOError, OSError, FileNotFoundError |
| `unknown_error` | Everything else |

---

## What We Do NOT Collect

We respect your privacy. We **never** collect:

- Your code or source files
- File paths or directory names
- Variable values, array contents, or model data
- Exact numeric values (always bucketed)
- Error messages or stack traces
- Personal information (name, email, IP address)
- Float values (could be identifying)
- Strings longer than 32 characters

---

## How It Works

Telemetry runs silently in the background:

1. When you use DSG-JIT, the `@telemetry_span` decorator records metrics
2. Data is batched locally (up to 512 spans, batches of 32)
3. Every 5 seconds, batches are sent to our server via OTLP/HTTP
4. On process exit, remaining spans are flushed

Telemetry is designed to be invisible — it won't slow down your code or clutter your output.

---

## Privacy & Data Handling

### Data Sanitization

All data is sanitized before leaving your machine:

- **Integers are bucketed** — `iters=47` becomes `"10-99"`
- **Only allowlisted args** — We can't record arbitrary parameters
- **Errors are categorized** — Just the type, never the message
- **No raw values** — We never transmit actual data from your computations

### Storage & Use

Your telemetry data:

- Is stored securely on our servers
- Is used only for improving DSG-JIT
- Is never sold or shared with third parties
- Is analyzed in aggregate, not individually

---

## Questions?

If you have questions or concerns about telemetry, please open an issue on [GitHub](https://github.com/TannerTorrey3/DSG-JIT/issues).
