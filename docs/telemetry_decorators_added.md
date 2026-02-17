# Telemetry Decorators Added — Summary

Telemetry was added using the **existing implementation** from the dev branch (`origin/telemtry/dev`). The telemetry **core** was copied as-is (no modifications). `@telemetry_span` from `dsg_jit.telemetry` was applied across CLI, optimization, SLAM, datasets, and world model.

---

## Dependency

**OpenTelemetry** is required for `import dsg_jit` and tests to succeed:

```bash
pip install opentelemetry-api>=1.20 opentelemetry-sdk>=1.20 opentelemetry-exporter-otlp-proto-http>=1.20
```

Or install the project with deps: `pip install -e .` (pyproject.toml now lists these).

---

## Summary of Added Decorators by Folder

### `dsg_jit/telemetry/` (core — copied from dev, unchanged)

- No new decorators; this is the implementation.
- Files: `__init__.py`, `config.py`, `decorators.py`, `identity.py`, `otel.py`, `sanitize.py`.

---

### `dsg_jit/cli/`

| File        | Function                   | Decorator (`component`, `op`)              | Notes                    |
|------------|----------------------------|--------------------------------------------|--------------------------|
| `main.py`  | `main`                     | `cli`, `main`                              | CLI entrypoint           |
| `main.py`  | `_feedback_cmd`            | `cli`, `feedback`                          | Feedback subcommand      |
| `feedback.py` | `run_questionnaire`     | `cli`, `run_questionnaire`                 | Questionnaire flow       |
| `feedback.py` | `show_questionnaire_popup` | `cli`, `show_questionnaire_popup`       | `safe_args={"show_save_location"}` |

---

### `dsg_jit/optimization/`

| File          | Function / method              | Decorator (`component`, `op`)                    |
|---------------|--------------------------------|--------------------------------------------------|
| `solvers.py`  | `gradient_descent`             | `optimization`, `gradient_descent`               |
| `solvers.py`  | `damped_newton`                | `optimization`, `damped_newton`                  |
| `solvers.py`  | `gauss_newton`                 | `optimization`, `gauss_newton`                   |
| `solvers.py`  | `gauss_newton_manifold`        | `optimization`, `gauss_newton_manifold`         |
| `jit_wrappers.py` | `JittedGN.__call__`          | `optimization`, `jitted_gn_solve`                |
| `jit_wrappers.py` | `JittedGN.from_residual`    | `optimization`, `jitted_gn_from_residual`       |
| `jit_wrappers.py` | `JittedGN.from_world_model` | `optimization`, `jitted_gn_from_world_model`    |
| `jit_wrappers.py` | `JittedGNManifold.__call__` | `optimization`, `jitted_gn_manifold_solve`      |
| `jit_wrappers.py` | `JittedGNManifold.from_residual` | `optimization`, `jitted_gn_manifold_from_residual` |
| `jit_wrappers.py` | `JittedGNManifold.from_world_model` | `optimization`, `jitted_gn_manifold_from_world_model` |

---

### `dsg_jit/slam/`

| File         | Function                         | Decorator (`component`, `op`)                |
|--------------|-----------------------------------|-----------------------------------------------|
| `pipeline.py` | `run_pose_graph_slam`            | `slam`, `run_pose_graph_slam`                 |
| `pipeline.py` | `update_worldmodel_from_solution` | `slam`, `update_worldmodel_from_solution`     |
| `pipeline.py` | `visualize_pose_graph_3d`        | `slam`, `visualize_pose_graph_3d`            |

---

### `dsg_jit/datasets/`

| File                 | Function                      | Decorator (`component`, `op`)             |
|----------------------|-------------------------------|-------------------------------------------|
| `tum_rgbd.py`        | `load_tum_rgbd_sequence`      | `datasets`, `load_tum_rgbd_sequence`      |
| `kitti_odometry.py`  | `load_kitti_odometry_sequence` | `datasets`, `load_kitti_odometry_sequence` |

---

### `dsg_jit/world/`

| File       | Method / function                    | Decorator (`component`, `op`)                    | Safe args / notes        |
|------------|--------------------------------------|--------------------------------------------------|--------------------------|
| `model.py` | `init_active_template`               | `world`, `init_active_template`                  |                          |
| `model.py` | `set_variable_slot`                  | `world`, `set_variable_slot`                     | `var_type`               |
| `model.py` | `configure_factor_slot`              | `world`, `configure_factor_slot`                 | `factor_type`, `active`  |
| `model.py` | `add_variable`                       | `world`, `add_variable`                          | `var_type`               |
| `model.py` | `add_pose`                           | `world`, `add_pose`                              |                          |
| `model.py` | `add_room`                           | `world`, `add_room`                              |                          |
| `model.py` | `add_place`                          | `world`, `add_place`                             |                          |
| `model.py` | `add_object`                         | `world`, `add_object`                            |                          |
| `model.py` | `add_agent_pose`                     | `world`, `add_agent_pose`                        | `var_type`               |
| `model.py` | `add_factor`                         | `world`, `add_factor`                            | `f_type`                 |
| `model.py` | `add_camera_bearings`                | `world`, `add_camera_bearings`                   | `factor_type`            |
| `model.py` | `add_lidar_ranges`                   | `world`, `add_lidar_ranges`                      | `factor_type`            |
| `model.py` | `add_imu_preintegration_factor`      | `world`, `add_imu_preintegration_factor`        | `factor_type`            |
| `model.py` | `optimize`                           | `world`, `optimize`                              | `method`, `iters`        |
| `model.py` | `get_variable_value`                 | `world`, `get_variable_value`                    |                          |
| `model.py` | `snapshot_state`                     | `world`, `snapshot_state`                        |                          |
| `model.py` | `register_residual`                  | `world`, `register_residual`                     | `factor_type`            |
| `model.py` | `get_residual`                       | `world`, `get_residual`                          | `factor_type`            |
| `model.py` | `get_residuals`                      | `world`, `get_residuals`                         |                          |
| `model.py` | `list_residual_types`                | `world`, `list_residual_types`                   |                          |
| `model.py` | `build_residual`                     | `world`, `build_residual`                        | `use_type_weights`, `learn_odom`, `learn_voxel_points` |
| `model.py` | `marginalize_variables`              | `world`, `marginalize_variables`                 |                          |
| `model.py` | `fixed_lag_marginalize`              | `world`, `fixed_lag_marginalize`                 |                          |
| `model.py` | `build_residual_function_with_type_weights` | `world`, `build_residual_function_with_type_weights` |                  |
| `model.py` | `build_residual_function_voxel_point_param` | `world`, `build_residual_function_voxel_point_param` |                  |
| `model.py` | `build_residual_function_voxel_point_param_multi` | `world`, `build_residual_function_voxel_point_param_multi` |            |
| `model.py` | `build_objective`                    | `world`, `build_objective`                      |                          |
| `model.py` | `pack_state`                         | `world`, `pack_state`                            |                          |
| `model.py` | `unpack_state`                       | `world`, `unpack_state`                          |                          |
| `model.py` | `unpack_state_inplace`               | `world`, `unpack_state_inplace`                  |                          |

---

## Event Naming

All spans use the existing convention:

- **Span name:** `dsgjit.{component}.{op}` (e.g. `dsgjit.cli.main`, `dsgjit.world.optimize`, `dsgjit.optimization.gauss_newton_manifold`).
- **Attributes:** As in the existing telemetry implementation (`dsgjit.component`, `dsgjit.op`, `dsgjit.status`, `ix.install_id`, etc.; `dsgjit.args.*` for `safe_args`).

---

## What Was Not Done

- **No changes** to the telemetry core (decorators.py, config.py, otel.py, identity.py, sanitize.py) except for adding the package to this branch.
- **No decorators** on `api/dsg_api.py` (abstract API with `NotImplementedError`).
- **No decorators** on `world/scene_graph.py` in this pass (can be added in a follow-up to match the dev branch).
- **No refactors** of unrelated logic; only imports and `@telemetry_span` lines were added.

---

## Tests and Import

- **Import:** `import dsg_jit` works after installing the project (or OpenTelemetry) deps.
- **Tests:** Run with `pytest dsg-jit/tests` after `pip install -e .` (or installing the optional telemetry deps). Without OpenTelemetry installed, tests fail at import with `ModuleNotFoundError: No module named 'opentelemetry'`.
