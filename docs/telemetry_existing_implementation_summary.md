# Existing Telemetry Implementation — Summary (Reuse This, Do Not Replace)

**Branch:** `origin/telemtry/dev` (remote; typo in name: "telemtry" not "telemetry").  
**Confirmed:** The implementation lives on this development branch only; it is not on `main`.  
**Decision:** Reuse this implementation exactly. Do not create a new telemetry system.

---

## 1. Where It Lives

| Path | Purpose |
|------|--------|
| `dsg-jit/dsg_jit/telemetry/__init__.py` | Public API: exports decorator, config, otel helpers, `bucket_count`, `reset_telemetry_state` |
| `dsg-jit/dsg_jit/telemetry/decorators.py` | `@telemetry_span` implementation; session start; span naming; safe args; error recording |
| `dsg-jit/dsg_jit/telemetry/config.py` | Config singleton from env; level, endpoint, sample rate, debug, tag |
| `dsg-jit/dsg_jit/telemetry/otel.py` | OpenTelemetry setup, tracer, OTLP exporter, shutdown, reset (testing) |
| `dsg-jit/dsg_jit/telemetry/identity.py` | Install ID (persistent), session ID (per process); telemetry identity file |
| `dsg-jit/dsg_jit/telemetry/sanitize.py` | `bucket_count`, `sanitize_safe_args`, `get_error_code`, allowlists |
| `docs/telemetry.md` | User-facing telemetry doc (what we collect, what we don’t, error codes) |
| `docs/api/telemetry.md` | API reference for telemetry module |
| `dsg-jit/tests/test_telemetry_*.py` | Smoke, integration, comprehensive tests |

---

## 2. Decorator: Name and Import

- **Decorator name:** `telemetry_span`
- **Import path (canonical):**
  ```python
  from dsg_jit.telemetry import telemetry_span
  ```
- **Alternative (implementation detail):**
  ```python
  from dsg_jit.telemetry.decorators import telemetry_span
  ```
- **Signature:**
  ```python
  def telemetry_span(
      component: str,
      op: str,
      safe_args: Optional[Set[str]] = None,
      shape_fn: Optional[ShapeFn] = None,
  ) -> Callable[[F], F]:
  ```
  - `component`: e.g. `"world"`, `"scene_graph"`, `"scene_graph"` (for relations), `"experiment"`.
  - `op`: operation name, e.g. `"add_pose"`, `"optimize"`, `"add_prior_pose_identity"`.
  - `safe_args`: optional set of argument names to record (allowlisted and sanitized/bucketed).
  - `shape_fn`: optional callable(args, kwargs) -> dict of string attributes.

---

## 3. Enable / Disable

- **There is no opt-out in the current implementation.** Config explicitly sets `enabled = True` and comments state: *"Telemetry is always enabled - cannot be disabled"*.
- **Tuning only:** Environment variables control behavior; they do not turn telemetry off:
  - `DSGJIT_TELEMETRY_LEVEL`: `minimal` | `standard` (default) | `debug`
  - `DSGJIT_TELEMETRY_SAMPLE_RATE`: float in [0, 1], default `0.20` (20% of success spans; errors always recorded)
  - `DSGJIT_TELEMETRY_DEBUG`: `1` | `0` — debug logging
  - `DSGJIT_TELEMETRY_TAG`: custom tag (e.g. experiment id), max 64 chars, alphanumeric + `_` + `-`
  - `DSGJIT_TELEMETRY_ENDPOINT`: OTLP endpoint URL (default: `https://telemetry.ix-infra.com/v1/traces`)

If an opt-out is required later (e.g. `DSGJIT_NO_TELEMETRY=1`), that would be a small change inside `config.py` and the decorator (skip starting span when disabled); the rest of the implementation would be reused.

---

## 4. Event / Span Naming

- **Pattern:** Span name is `dsgjit.{component}.{op}`.
- **Examples from the branch:**
  - `dsgjit.world.add_pose`, `dsgjit.world.optimize`, `dsgjit.world.add_factor`
  - `dsgjit.scene_graph.add_pose_se3`, `dsgjit.scene_graph.add_odom_se3_geodesic`, `dsgjit.scene_graph.add_voxel_point_observation`
  - `dsgjit.scene_graph.room_centroid_residual`, `dsgjit.scene_graph.pose_place_attachment_residual`
- **Session:** First instrumented call also emits a span `dsgjit.session.start` with install/session id, version, runtime, backend, level, etc.

**Attribute naming (consistent):**

- `ix.install_id`, `ix.session_id`
- `dsgjit.version`, `dsgjit.component`, `dsgjit.op`, `dsgjit.status`, `dsgjit.backend`, `dsgjit.telemetry_level`, `dsgjit.entry_component`, `dsgjit.telemetry_enabled`, optional `dsgjit.tag`
- `dsgjit.args.<argname>` for safe args (e.g. `dsgjit.args.method`, `dsgjit.args.iters` — iters bucketed)
- `runtime.python`, `runtime.os`, `runtime.arch`
- On error: `error.type`, `error.code`, `error.component`, `error.op`

So **event naming** = span name `dsgjit.{component}.{op}` plus these attribute names. New instrumented code should reuse the same `component`/`op` style and the same attribute names.

---

## 5. Existing Instrumentation (Where It’s Already Used)

- **`world/model.py`:** `world` component — init_active_template, set_variable_slot, configure_factor_slot, add_variable, add_pose, add_room, add_place, add_object, add_agent_pose, add_factor, add_camera_bearings, add_lidar_ranges, add_imu_preintegration_factor, optimize, get_variable_value, snapshot_state, register_residual, get_residual, get_residuals, list_residual_types, build_residual (and overloads), marginalize_variables, fixed_lag_marginalize, build_residual_function_*, build_objective, pack_state, unpack_state, unpack_state_inplace.
- **`world/scene_graph.py`:** `scene_graph` component — enable_active_template, get_factor_memory, deactivate_factors_for_vars, add_pose_se3, add_place1d, add_room1d, add_place3d, add_room, add_object3d, add_named_object3d, add_agent_pose_se3, add_prior_pose_identity, add_odom_se3_additive, add_odom_se3_geodesic, (and more per docs: range, temporal smoothness, landmark, voxel, attachments, edges, get_pose, get_place, optimize, visualize_web, etc.).
- **`scene_graph/relations.py`:** `scene_graph` component — room_centroid_residual, pose_place_attachment_residual.

Docs also list an `experiment` component for experiment runs (e.g. exp01_mini_world, exp16_hero_hybrid_dsg, etc.).

---

## 6. Safe Args and Sanitization (Reuse As-Is)

- **Allowlist:** `sanitize.SAFE_SCALAR_ARGS` includes e.g. `method`, `var_type`, `factor_type`, `f_type`, `iters`, `coord_index`, `use_type_weights`, `learn_odom`, `learn_voxel_points`, `active`.
- **Bucketing:** Integers via `bucket_count(n)` → `"0"`, `"1-9"`, `"10-99"`, `"100-999"`, `"1k-9k"`, etc.
- **Errors:** `get_error_code(exception)` maps exception type to a category (e.g. `invalid_argument`, `numerical_issue`, `convergence_failure`); no raw messages.

When adding new `@telemetry_span` calls, only pass `safe_args` that are in this allowlist (or extend the allowlist in `sanitize.py` if needed); do not add new ad-hoc sanitization.

---

## 7. Backend and Initialization

- **OpenTelemetry** with OTLP HTTP exporter; tracer name `"dsg_jit.telemetry"` with version.
- **Auto-init:** First call to `get_tracer()` (e.g. from inside the decorator) calls `setup_telemetry()`; no need for users to call it.
- **Shutdown:** `atexit.register(shutdown_telemetry)`; can also call `shutdown_telemetry()` explicitly to flush.
- **Testing:** `reset_telemetry_state()` (in decorators) and `reset_telemetry()` (in otel) / `reset_config()` (in config) for tests.

---

## 8. Checklist for Any New Changes

- [ ] **Do not** add a second telemetry system or decorator; reuse `@telemetry_span` and `dsg_jit.telemetry` only.
- [ ] **Branch:** Work that depends on telemetry should be done on a branch that includes the telemetry implementation (e.g. merge/cherry-pick from `telemtry/dev` or work on a branch that already has it).
- [ ] **Naming:** New spans = `dsgjit.{component}.{op}`; use existing `component` values (`world`, `scene_graph`, `experiment`) or add new ones consistently; keep `op` short and snake_case.
- [ ] **Attributes:** Use the same attribute names (`dsgjit.*`, `ix.*`, `runtime.*`, `error.*`); use `safe_args` only for allowlisted arg names; use `shape_fn` only for non-sensitive, string attributes.
- [ ] **Config:** Respect existing env vars; if adding opt-out later, add a single flag (e.g. in config) and keep the rest of the implementation unchanged.

---

**Summary:** The existing implementation on `origin/telemtry/dev` is the single source of truth. Use `telemetry_span` from `dsg_jit.telemetry`, span names `dsgjit.{component}.{op}`, and the same config/sanitization/backend. Do not create a new telemetry system.
