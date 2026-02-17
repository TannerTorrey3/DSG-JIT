# Telemetry Export Inspection — Current Behavior

Inspection of `dsg_jit/telemetry/` to determine where and how telemetry data is exported. **No code was changed.**

---

## 1. Protocol and path

| Item | Value |
|------|--------|
| **Protocol** | **OTLP over HTTP** (JSON or protobuf per SDK; the Python OTLP HTTP exporter uses protobuf by default). |
| **gRPC?** | **No.** Only the HTTP exporter is used. |
| **Exporter class** | `opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter` (from `otel.py`). |
| **Default endpoint** | `https://telemetry.ix-infra.com/v1/traces` (full URL including path). |
| **Path** | The default already includes **`/v1/traces`**. If you set `DSGJIT_TELEMETRY_ENDPOINT` to your collector base (e.g. `http://localhost:4318`), the OTLP HTTP exporter may append `/v1/traces` itself depending on SDK version; otherwise you must set the full URL (e.g. `http://localhost:4318/v1/traces`). |

---

## 2. Env vars and config

All configuration is in **`config.py`**, read from the environment on first `get_telemetry_config()` and cached.

| Env var | Config attribute | Default | Description |
|--------|-------------------|---------|-------------|
| `DSGJIT_TELEMETRY_LEVEL` | `level` | `"standard"` | Detail level: `minimal` \| `standard` \| `debug`. Validated; invalid → `standard`. |
| `DSGJIT_TELEMETRY_SAMPLE_RATE` | `sample_rate` | `0.20` | Success-span sampling rate in [0, 1]. Clamped; invalid → 0.20. (Note: sampler currently always returns RECORD_AND_SAMPLE; sampling is not applied at SDK level.) |
| `DSGJIT_TELEMETRY_DEBUG` | `debug` | `False` | Debug logging: `1`/`true`/`yes`/`on` → True. |
| `DSGJIT_TELEMETRY_TAG` | `tag` | `""` | Custom tag (e.g. experiment id). Sanitized: max 64 chars, alphanumeric + `_` + `-`. |
| `DSGJIT_TELEMETRY_ENDPOINT` | `endpoint` | `https://telemetry.ix-infra.com/v1/traces` | Full OTLP HTTP endpoint URL for trace export. No validation or special handling for empty. |

**Not configurable via env (hardcoded in code):**

- **Enable/disable:** `enabled` is always `True` in config. There is **no** env var to turn telemetry off; the comment says "Telemetry is always enabled - cannot be disabled."
- **Protocol:** HTTP only; no env for gRPC.
- **Service name:** Hardcoded in `otel.py` as `SERVICE_NAME: "dsg-jit"` in the Resource.
- **Headers:** Set in `otel.py` in `setup_telemetry()` (see below). No env to add/override headers.
- **Resource attributes:** Fixed set in `Resource.create(...)` (service name, version, install_id, session_id, runtime.*, dsgjit.telemetry_level). No env to add more.
- **Batch processor:** `BatchSpanProcessor(exporter, max_queue_size=512, max_export_batch_size=32, schedule_delay_millis=5000)`. Not configurable.
- **Exporter timeout:** 5 seconds in `OTLPSpanExporter(..., timeout=5)`. Not configurable.

**Headers sent on export (set in `otel.py`):**

- `X-Ix-Install-Id`
- `X-Ix-Session-Id`
- `X-Ix-Pkg-Version`
- `X-Ix-Telemetry-Level`

---

## 3. When no endpoint is configured

- **“Not configured”** here means: user never sets `DSGJIT_TELEMETRY_ENDPOINT`.
- In that case, **the default is always used**: `https://telemetry.ix-infra.com/v1/traces`. So export still happens; there is no “no endpoint” path in code.
- If the user **sets** `DSGJIT_TELEMETRY_ENDPOINT` to an empty string, the code does **not** treat that specially: `endpoint` is passed as `""` to `OTLPSpanExporter(endpoint=config.endpoint, ...)`. Behavior is then defined by the OpenTelemetry SDK (often a failing or no-op export). The current code does **not**:
  - detect empty endpoint and switch to a no-op exporter,
  - drop spans,
  - or raise an error.

So: **no explicit “no endpoint” handling**. Default endpoint is always present when the var is unset; when set to empty, behavior is whatever the SDK does with an empty URL.

---

## 4. Where exporters/providers are initialized and when

| What | Where | When it runs |
|------|--------|---------------|
| **Function that initializes everything** | **`setup_telemetry()`** in **`dsg_jit/telemetry/otel.py`** (lines 85–166). |
| **When it runs** | **Lazily**, on first use of the tracer: **`get_tracer()`** (same file) is called when the first `@telemetry_span`-decorated function runs. `get_tracer()` checks `if not _initialized:` and then calls `setup_telemetry()`. So the first instrumented call (e.g. first call to an instrumented API or CLI) triggers: `get_tracer()` → `setup_telemetry()`. |
| **Idempotency** | `setup_telemetry()` returns immediately if `_initialized` is already True, so it only runs once per process. |
| **Shutdown** | `atexit.register(shutdown_telemetry)` is called at the end of `setup_telemetry()`, so on process exit the tracer provider is shut down and pending spans are flushed. |

**Order of operations inside `setup_telemetry()`:**

1. `get_telemetry_config()` (loads env, builds `TelemetryConfig`).
2. `Resource.create({...})` with service name and resource attributes.
3. `DsgJitSampler()` (currently always RECORD_AND_SAMPLE).
4. `TracerProvider(resource=resource, sampler=sampler)`.
5. `OTLPSpanExporter(endpoint=config.endpoint, headers=headers, timeout=5)`.
6. `BatchSpanProcessor(exporter, max_queue_size=512, max_export_batch_size=32, schedule_delay_millis=5000)`.
7. `_tracer_provider.add_span_processor(processor)`.
8. `trace.set_tracer_provider(_tracer_provider)`.
9. `_tracer = trace.get_tracer("dsg_jit.telemetry", _get_version())`.
10. `atexit.register(shutdown_telemetry)`.

---

## 5. Current behavior summary

- **Export:** Traces are exported over **OTLP HTTP** to the URL in `DSGJIT_TELEMETRY_ENDPOINT` (default: `https://telemetry.ix-infra.com/v1/traces`). The path is part of that URL; for a custom collector you typically set the full URL (e.g. `http://<collector>:4318/v1/traces`).
- **Initialization:** On first use of the tracer (first instrumented call), `get_tracer()` → `setup_telemetry()` builds the TracerProvider, OTLP HTTP exporter, and BatchSpanProcessor, and sets the global tracer provider. No explicit “init at import” step.
- **Config:** All config comes from env (endpoint, level, sample_rate, debug, tag). No enable/disable; service name, headers, resource attributes, batching, and timeout are fixed in code.
- **No “no endpoint” handling:** Default endpoint is always used when the env var is unset; empty endpoint is not special-cased (no drop/no-op/error in our code).

---

## 6. What to change to integrate with an existing collector/pipeline

- **Endpoint:** Set **`DSGJIT_TELEMETRY_ENDPOINT`** to your collector’s OTLP HTTP URL, including path if needed (e.g. `http://localhost:4318/v1/traces` for OTLP HTTP). No code change required for a standard OTLP HTTP collector.
- **Optional code changes for tighter integration:**
  - **Disable when not needed:** Add an env var (e.g. `DSGJIT_TELEMETRY_DISABLED=1`) and, when set, skip creating the real exporter (e.g. use a no-op span processor or no-op exporter) so no data is sent and no connection is made.
  - **Empty endpoint = no export:** If `DSGJIT_TELEMETRY_ENDPOINT` is missing or empty, call `setup_telemetry()` only to set a no-op exporter (or skip adding any exporter), so integration is “set endpoint = use collector; leave unset = no export.”
  - **Override service name / resource / headers:** Add env vars or a small config layer for service name, extra resource attributes, and optional headers so the same code can target different backends (e.g. your collector with a specific service name or API key header).
  - **gRPC:** To use OTLP gRPC instead of HTTP, replace `OTLPSpanExporter` from `opentelemetry.exporter.otlp.proto.http.trace_exporter` with the gRPC trace exporter and pass the gRPC endpoint (e.g. `http://localhost:4317`); gRPC typically does not use `/v1/traces` in the URL.

**Minimal change for “use my collector” today:** Set `DSGJIT_TELEMETRY_ENDPOINT` to your OTLP HTTP endpoint (e.g. `http://<host>:4318/v1/traces`). No code changes required.
