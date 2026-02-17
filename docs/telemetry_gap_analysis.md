# Telemetry Gap Analysis & Plan

**Status:** Plan only — no code changes.  
**Branch:** Telemetry work (e.g. `dsg_jit/telemetry/decorators.py`) lives on a **dev branch**, not `main`. Merge to main only when ready.  
**Finding:** On main there is currently **no telemetry system** (no `dsg_jit/telemetry/` package, no decorators, no event emissions). This document identifies where telemetry would be applied once adopted, and recommends a consistent naming scheme and the location for that code.

---

## 1. Files / functions that should be decorated (prioritized)

### 1.1 CLI entrypoints (highest priority)

| File | Function | Rationale |
|------|----------|-----------|
| `dsg_jit/cli/main.py` | `main()` | Single CLI entry; track which commands are run. |
| `dsg_jit/cli/main.py` | `_feedback_cmd()` | Feedback flow usage. |
| `dsg_jit/cli/feedback.py` | `show_questionnaire_popup()` | Questionnaire completion / cancellation. |
| `dsg_jit/cli/feedback.py` | `run_questionnaire()` | Optional: per-question or aggregate (e.g. rating, discovery). |
| `dsg_jit/__init__.py` | (import path) | Optional: package import (e.g. for “first use” or environment). |

### 1.2 Public API (DSGApi and main implementation)

**`dsg_jit/api/dsg_api.py` — DSGApi (abstract public API):**

- **Construction:** `__init__`
- **Agent/trajectory:** `register_agent`, `add_agent_pose`, `add_agent_trajectory`, `get_agent_trajectory`, `get_all_trajectories`
- **Structure:** `add_room`, `add_place`, `attach_place_to_room`, `add_object`, `set_object_pose`, `attach_object_to_place`, `add_relation`
- **Observations / factors:** `add_range_observation`, `add_odom_tx`, `add_voxel_observation`, `add_bearing_observation`, `add_photometric_observation`, `add_nerf_photometric_observation`
- **Optimization (heavy):** `optimize`, `optimize_subgraph`
- **State / checkpoints:** `checkpoint`, `rollback_to`, `apply_patch`, `merge_sessions`
- **Queries:** `get_subgraph`, `get_room_subgraph`, `get_agent_local_subgraph`, `get_fov_subgraph`, `find_objects_in_radius`, `find_places_in_radius`, `get_room_for_point`, `find_nearest_object_of_class`, `compute_node_descriptor`, `compute_subgraph_descriptor`, `find_candidate_loop_closures`, `list_visible_objects`, `list_visible_nodes`
- **Planning:** `plan_topological_path`, `plan_semantic_path`

**`dsg_jit/world/scene_graph.py` — SceneGraphWorld (concrete implementation):**

- **Optimization (heavy):** `optimize()`, `optimize_active_batch()`, `optimize_global_offline()`
- **Structure:** `add_room`, `add_place`, `add_prior_pose_identity`, `add_range_measurement`, `add_agent_pose_place_attachment`, `add_agent_temporal_smoothness`, `add_agent_pose_landmark_relative`, `add_agent_pose_landmark_bearing`, `add_agent_pose_voxel_point`, `attach_pose_to_place_x`, `attach_pose_to_room_x`, `add_place_attachment`, `add_voxel_cell`, `add_pose_voxel_point`, `add_voxel_smoothness`, `add_voxel_point_observation`
- **State / export:** `dump_state`, `visualize_web`

**`dsg_jit/world/model.py` — WorldModel (core graph + solve):**

- **Heavy compute:** `optimize()`, `build_residual()`, `build_objective()`, `pack_state()`, `unpack_state()`, `marginalize_variables()`, `fixed_lag_marginalize()`
- **Graph mutation:** `add_variable()`, `add_pose()`, `add_room()`, `add_place()`, `add_object()`, `add_agent_pose()`, `add_factor()`, `add_camera_bearings()`, `add_lidar_ranges()`, `add_imu_preintegration_factor()`
- **Residual API:** `register_residual()`, `build_residual()`, `build_residual_function_with_type_weights()`, `build_residual_function_voxel_point_param()`, `build_residual_function_voxel_point_param_multi()`

### 1.3 Heavy compute paths (solver / pipeline / JIT)

| File | Function / class | Rationale |
|------|-------------------|-----------|
| `dsg_jit/optimization/solvers.py` | `gradient_descent()` | Core solver; duration + iterations. |
| `dsg_jit/optimization/solvers.py` | `damped_newton()` | Core solver. |
| `dsg_jit/optimization/solvers.py` | `gauss_newton()` | Core solver; very hot path. |
| `dsg_jit/optimization/solvers.py` | `gauss_newton_manifold()` | Primary GN entry for SE3; very hot path. |
| `dsg_jit/optimization/jit_wrappers.py` | `JittedGN.__call__()` | JIT-compiled solve entry. |
| `dsg_jit/optimization/jit_wrappers.py` | `JittedGN.from_world_model()` | Factory for WM-backed solver. |
| `dsg_jit/optimization/jit_wrappers.py` | `DSGTrainer.__call__()` | Training-step entry. |
| `dsg_jit/optimization/jit_wrappers.py` | `DSGTrainer.from_world_model()` | Factory for trainer. |
| `dsg_jit/slam/pipeline.py` | `run_pose_graph_slam()` | High-level SLAM entry. |
| `dsg_jit/slam/pipeline.py` | `update_worldmodel_from_solution()` | Post-solve state update. |
| `dsg_jit/slam/pipeline.py` | `visualize_pose_graph_3d()` | Optional (visualization usage). |

### 1.4 Datasets and sensors (I/O and integration)

| File | Function | Rationale |
|------|----------|-----------|
| `dsg_jit/datasets/tum_rgbd.py` | `load_tum_rgbd_sequence()` | Dataset load; size/duration. |
| `dsg_jit/datasets/kitti_odometry.py` | `load_kitti_odometry_sequence()` | Dataset load. |
| `dsg_jit/sensors/fusion.py` | `SensorFusionManager.poll_once()` | Fusion tick (if used in loops). |
| `dsg_jit/sensors/conversion.py` | `lidar_scan_to_voxel_factors()`, `camera_bearings_to_factors()`, etc. | Optional: high-level conversion entry points. |
| `dsg_jit/sensors/integration.py` | `apply_fused_pose_to_world()`, `apply_trajectory_to_world()` | Integration into world model. |

### 1.5 Visualization and export

| File | Function | Rationale |
|------|----------|-----------|
| `dsg_jit/world/visualization.py` | `plot_factor_graph_3d()`, `plot_scenegraph_3d()`, `plot_dynamic_trajectories_3d()` | Feature usage. |
| `dsg_jit/world/web_viewer.py` | `export_scenegraph_to_threejs()`, `run_scenegraph_web_viewer()` | Viewer usage. |

---

## 2. Modules / folders with zero telemetry coverage

**Current state:** Every module has zero coverage because no telemetry system exists.

If telemetry is added only to the items above, these will still have **no** instrumented surface:

| Module / folder | Notes |
|------------------|--------|
| `dsg_jit/core/` | Types, factor graph, math3d — low-level; optional to decorate (e.g. only if exposing a “build graph” or “solve” facade). |
| `dsg_jit/scene_graph/` | `entities.py`, `relations.py` — residual helpers; usually called from `world`; can stay undecorated or add one facade. |
| `dsg_jit/slam/measurements.py` | Pure residual functions; called from world/solvers; telemetry at solver or WorldModel level is enough. |
| `dsg_jit/slam/manifold.py` | Helpers for manifold metadata; typically used inside pipeline/solver. |
| `dsg_jit/world/dynamic_scene_graph.py` | Thin/extended scene graph; if used as main entry, mirror same events as SceneGraphWorld. |
| `dsg_jit/world/voxel_grid.py` | Grid building helpers; optional. |
| `dsg_jit/world/training.py` | Training dataclasses/helpers; optional unless used as main training entry. |
| `dsg_jit/sensors/base.py` | Abstract base; no need to decorate. |
| `dsg_jit/sensors/camera.py` | Pure image helpers; optional. |
| `dsg_jit/sensors/lidar.py` | Data types; optional. |
| `dsg_jit/sensors/imu.py` | Data types + `integrate_imu_naive`; optional. |
| `dsg_jit/sensors/streams.py` | Stream abstractions; optional. |

So: **all folders** currently have zero coverage; the list above is where coverage would remain lowest if only the prioritized list in §1 is instrumented.

---

## 3. Recommended event naming scheme

There is **no existing telemetry usage** to mirror. The following scheme is suggested so that when you introduce a provider (e.g. decorator + backend), events are consistent.

### 3.1 Format

Use a small, flat namespace with clear hierarchy:

- **Pattern:** `dsg_jit.<module>.<action>` or `dsg_jit.<area>.<action>`
- **Module/area:** `cli`, `api`, `world`, `optimization`, `slam`, `sensors`, `datasets`, `viz`
- **Action:** verb or verb_noun — e.g. `run`, `optimize`, `load_sequence`, `questionnaire_completed`

### 3.2 Examples (aligned with §1)

| Area | Example events | Notes |
|------|----------------|--------|
| CLI | `dsg_jit.cli.entry`, `dsg_jit.cli.feedback_started`, `dsg_jit.cli.feedback_completed`, `dsg_jit.cli.feedback_cancelled` | One event per command or flow step. |
| API | `dsg_jit.api.optimize`, `dsg_jit.api.optimize_subgraph`, `dsg_jit.api.checkpoint`, `dsg_jit.api.add_range_observation`, … | Mirror DSGApi method names or group (e.g. `dsg_jit.api.structure_add`, `dsg_jit.api.query`). |
| World | `dsg_jit.world.optimize`, `dsg_jit.world.optimize_active_batch`, `dsg_jit.world.optimize_global_offline` | Same action names as in §1. |
| Optimization | `dsg_jit.optimization.gauss_newton`, `dsg_jit.optimization.gauss_newton_manifold`, `dsg_jit.optimization.jitted_solve`, `dsg_jit.optimization.trainer_step` | Solver and JIT wrapper entry points. |
| SLAM | `dsg_jit.slam.pose_graph_run`, `dsg_jit.slam.update_world_from_solution` | Pipeline-level only unless you add more. |
| Datasets | `dsg_jit.datasets.load_tum_rgbd`, `dsg_jit.datasets.load_kitti_odometry` | One per public loader. |
| Sensors | `dsg_jit.sensors.fusion_poll`, `dsg_jit.sensors.apply_fused_pose` | Optional. |
| Viz | `dsg_jit.viz.plot_3d`, `dsg_jit.viz.web_viewer_run` | Optional. |

### 3.3 Payload conventions (once you have a backend)

- **Common fields:** `duration_ms`, `success` (bool), `error` (if failed), optional `iterations` for solvers.
- **CLI:** `command`, `exit_code`; for feedback: `rating`, `discovery`, `completed`.
- **Optimization:** `method`, `max_iters`, `n_vars`, `n_factors` (if cheap to pass).
- **Datasets:** `dataset`, `path` (or hashed), `n_frames` / `n_poses`.

Keep payloads small and avoid PII; use the same field names across events for easier analysis.

---

## 4. Plan and checklist

### 4.1 Prerequisites (before adding decorators)

- [ ] **Add a telemetry package** at `dsg-jit/dsg_jit/telemetry/` with:
  - `telemetry/__init__.py` — export the public decorator/helper (e.g. `track`).
  - `telemetry/decorators.py` — implement the decorator and any event-emission logic (so all instrumentation lives in one place and can be disabled or swapped).
- [ ] **Choose a telemetry backend** (e.g. in-process logger, OpenTelemetry, or a product SDK) and add it as an optional dependency.
- [ ] **Define a single decorator or helper** in `dsg_jit/telemetry/decorators.py` (e.g. `@track("dsg_jit.cli.entry")`) that:
  - Wraps the function and emits one “start”/“success”/“error” event (or a single “completed” event with duration and outcome).
  - Can be disabled via env (e.g. `DSG_JIT_NO_TELEMETRY`) or config.
- [ ] **Document** in README or docs that telemetry is collected, what events are sent, and how to opt out.

### 4.2 Implementation order (no code changes in this repo until the above is done)

1. **Phase 1 — CLI and import**
   - [ ] `cli/main.py`: `main()`, `_feedback_cmd()`
   - [ ] `cli/feedback.py`: `show_questionnaire_popup()` (and optionally `run_questionnaire()`)
   - [ ] Optional: package import in `__init__.py`

2. **Phase 2 — Public API and world**
   - [ ] `api/dsg_api.py`: at least `optimize`, `optimize_subgraph`; then other DSGApi methods as needed
   - [ ] `world/scene_graph.py`: `optimize`, `optimize_active_batch`, `optimize_global_offline`; then structure/observation methods
   - [ ] `world/model.py`: `optimize`, `build_residual`, `add_variable`, `add_factor` (or a subset)

3. **Phase 3 — Heavy compute**
   - [ ] `optimization/solvers.py`: `gauss_newton`, `gauss_newton_manifold` (and optionally `gradient_descent`, `damped_newton`)
   - [ ] `optimization/jit_wrappers.py`: `JittedGN.__call__`, `JittedGN.from_world_model`, `DSGTrainer.__call__`, `DSGTrainer.from_world_model`
   - [ ] `slam/pipeline.py`: `run_pose_graph_slam`, `update_worldmodel_from_solution`

4. **Phase 4 — Datasets, sensors, viz (optional)**
   - [ ] `datasets/tum_rgbd.py`: `load_tum_rgbd_sequence`
   - [ ] `datasets/kitti_odometry.py`: `load_kitti_odometry_sequence`
   - [ ] `sensors/fusion.py`, `sensors/integration.py` as needed
   - [ ] `world/visualization.py`, `world/web_viewer.py` as needed

### 4.3 Consistency and maintenance

- [ ] **Naming:** All new events follow `dsg_jit.<module>.<action>`.
- [ ] **Payload:** Use the same common fields (e.g. `duration_ms`, `success`) everywhere.
- [ ] **Tests:** Telemetry is off in CI (e.g. `DSG_JIT_NO_TELEMETRY=1` or `CI=1`), so no test changes required unless you assert on events.
- [ ] **Review:** When adding new public API or CLI commands, add the corresponding event to this list and implement in the same phase as the feature.

---

## 5. Summary

| Priority | Area | Files / entrypoints | Suggested events (examples) |
|----------|------|---------------------|-----------------------------|
| P0 | CLI | `main`, `_feedback_cmd`, `show_questionnaire_popup` | `dsg_jit.cli.entry`, `dsg_jit.cli.feedback_*` |
| P1 | Public API | `DSGApi.optimize`, `optimize_subgraph`, structure/observation methods | `dsg_jit.api.*` |
| P1 | World | `SceneGraphWorld.optimize*`, `WorldModel.optimize`, `build_residual` | `dsg_jit.world.*` |
| P2 | Solvers / JIT | `gauss_newton`, `gauss_newton_manifold`, `JittedGN`, `DSGTrainer` | `dsg_jit.optimization.*` |
| P2 | SLAM pipeline | `run_pose_graph_slam`, `update_worldmodel_from_solution` | `dsg_jit.slam.*` |
| P3 | Datasets / sensors / viz | Loaders, fusion, visualization | `dsg_jit.datasets.*`, `dsg_jit.sensors.*`, `dsg_jit.viz.*` |

**No code has been changed.** This document is the plan and checklist only.
