# DSG-JIT Telemetry

This document describes how telemetry works in DSG-JIT, how to configure
or disable it, and how to point it at a custom endpoint.

---

## What is collected

DSG-JIT collects anonymous, low-cardinality usage spans to help improve
the library.  Nothing identifying or user-created is ever sent:

- **Collected:** operation names, bucketed counts, method enums, error
  type categories, backend type (cpu/gpu/tpu), Python/OS version,
  anonymous install ID and session ID.
- **Never collected:** source code, prompts, document text, file paths,
  repo URLs, hostnames, usernames, raw numeric values, or any user-domain
  IDs.

See `dsg-jit/dsg_jit/telemetry/sanitize.py` for the exact allowlist and
bucketing rules.

---

## Architecture at a glance

```
  User code
      │  calls a @telemetry_span-decorated function
      ▼
  decorators.py      – emits session.start on first call, builds span dict
      │
      ▼
  client.py          – sampling decision (errors always kept, successes
      │                 sampled at DSGJIT_TELEMETRY_SAMPLE_RATE)
      ▼
  exporter.py        – BatchSpanProcessor queues spans; background thread
      │                 flushes every 5 s or at 128 spans
      ▼
  queue.py           – BoundedSpanQueue (max 2048); drops non-error spans
      │                 when full, preserves errors
      ▼
  OTLPSpanExporter   – POSTs JSON batch to the configured endpoint;
                       exponential back-off on failure, self-disables
                       after 5 consecutive failures
```

`otel.py` is reserved for future OpenTelemetry SDK integration (see that
file for details).

---

## Configuration

All settings are controlled via environment variables.  The defaults are
shown in parentheses.

| Variable | Values | Default | Effect |
|---|---|---|---|
| `DSGJIT_TELEMETRY` | `1` / `0` | `1` | Master on/off switch |
| `DSGJIT_TELEMETRY_LEVEL` | `minimal` / `standard` / `debug` | `standard` | Verbosity level |
| `DSGJIT_TELEMETRY_ENDPOINT` | URL | `https://telemetry.ix-infra.com` | Where spans are POSTed; set to empty string to disable export |
| `DSGJIT_TELEMETRY_SAMPLE_RATE` | `0.0`–`1.0` | `0.20` | Fraction of *success* spans kept (errors are always kept) |
| `DSGJIT_TELEMETRY_DEBUG` | `1` / `0` | `0` | Print every span to stderr before export |
| `DSGJIT_TELEMETRY_TAG` | string | (empty) | Custom tag added to all spans (e.g., experiment name, run ID) |

### Tag your experiment runs

Use the tag to identify specific experiments or benchmark runs:

```bash
# Running exp01
DSGJIT_TELEMETRY_TAG=exp01_mini_world python experiments/exp01_mini_world.py

# Running a benchmark
DSGJIT_TELEMETRY_TAG=benchmark_se3_v2 python benchmarks/bench_gauss_newton_se3.py
```

The tag appears as `dsgjit.tag` in every span, making it easy to filter
and group telemetry data by experiment.

### Disable telemetry entirely

```bash
export DSGJIT_TELEMETRY=0
```

No client is created, no spans are recorded, and no network calls are
made.

### Point at a custom endpoint

```bash
export DSGJIT_TELEMETRY_ENDPOINT=http://localhost:4318/v1/traces
```

---

## Transport Details

### Protocol

Telemetry is sent via **OTLP/HTTP** (OpenTelemetry Protocol over HTTP):

- **Endpoint:** `POST https://telemetry.ix-infra.com/v1/traces`
- **Content-Type:** `application/json` (OTLP JSON encoding)
- **Max payload:** 64 KB
- **Max spans per request:** 200

### Required Headers

Every request includes these headers for rate limiting and routing:

| Header | Value | Purpose |
|--------|-------|---------|
| `X-Ix-Install-Id` | UUIDv4 | Persistent anonymous install ID |
| `X-Ix-Session-Id` | UUIDv4 | Per-process session ID |
| `X-Ix-Pkg-Version` | semver | DSG-JIT version (e.g., `0.7.1`) |
| `X-Ix-Telemetry-Level` | `minimal`/`standard`/`debug` | Current telemetry level |

### Payload Format (OTLP JSON)

```json
{
  "resourceSpans": [{
    "resource": {
      "attributes": [
        {"key": "service.name", "value": {"stringValue": "dsg-jit"}}
      ]
    },
    "scopeSpans": [{
      "scope": {"name": "dsg_jit.telemetry"},
      "spans": [
        {
          "name": "dsgjit.world.optimize",
          "startTimeUnixNano": "1706000000123000000",
          "endTimeUnixNano": "1706000000165700000",
          "attributes": [
            {"key": "dsgjit.component", "value": {"stringValue": "world"}},
            {"key": "dsgjit.op", "value": {"stringValue": "optimize"}},
            {"key": "dsgjit.status", "value": {"stringValue": "ok"}}
          ]
        }
      ]
    }]
  }]
}
```

---

## Span Attributes

### Common Attributes (on every span)

| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| `ix.install_id` | string | `a1b2c3d4-...` | Persistent anonymous install UUID |
| `ix.session_id` | string | `e5f6g7h8-...` | Per-process session UUID |
| `dsgjit.version` | string | `0.7.1` | Package version |
| `runtime.python` | string | `3.11.7` | Python version |
| `runtime.os` | string | `linux` | Operating system |
| `runtime.arch` | string | `x86_64` | CPU architecture |
| `dsgjit.component` | string | `world` | Component name |
| `dsgjit.op` | string | `optimize` | Operation name |
| `dsgjit.status` | string | `ok`/`error` | Operation status |
| `dsgjit.backend` | string | `cpu`/`gpu`/`tpu` | Compute backend |
| `dsgjit.telemetry_level` | string | `standard` | Current level |

### Error Attributes (only on error spans)

| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| `error.type` | string | `ValueError` | Exception class name (never message) |
| `error.code` | string | `invalid_argument` | Error category code |
| `error.component` | string | `world` | Component where error occurred |
| `error.op` | string | `optimize` | Operation where error occurred |

### Error Code Categories

| Code | Triggered By |
|------|--------------|
| `invalid_argument` | ValueError, TypeError, KeyError, IndexError |
| `shape_mismatch` | ShapeError |
| `not_initialized` | NotImplementedError |
| `convergence_failure` | ConvergenceError |
| `numerical_issue` | FloatingPointError, OverflowError, ZeroDivisionError |
| `backend_error` | RuntimeError |
| `io_error` | IOError, OSError, FileNotFoundError |
| `unknown_error` | Any other exception |

### Bucketed Shape Attributes (optional)

| Attribute | Buckets |
|-----------|---------|
| `dsgjit.graph.nodes_bucket` | `0`, `1-9`, `10-99`, `100-999`, `1k-9k`, `10k-99k`, `100k-999k`, `1M+` |
| `dsgjit.graph.edges_bucket` | (same buckets) |
| `dsgjit.iterations_bucket` | (same buckets) |
| `dsgjit.window.size_bucket` | (same buckets) |

---

## Running the smoke tests

```bash
PYTHONPATH=dsg-jit python -m pytest dsg-jit/tests/test_telemetry_smoke.py -v
```

These tests verify:
- importing DSG-JIT does not start telemetry
- the first instrumented call emits `session.start`
- disabling export or leaving the endpoint empty causes no network
  activity and no crashes
- user content (poison strings) never reaches a serialised span

---

## Adding instrumentation

Decorate a public function with `@telemetry_span`:

```python
from dsg_jit.telemetry import telemetry_span
from dsg_jit.telemetry.sanitize import bucket_count

@telemetry_span(
    component="world",
    op="optimize",
    safe_args={"method"},                          # only "method" is recorded
    shape_fn=lambda self, **kw: {                  # optional bucketed metadata
        "iterations_bucket": bucket_count(kw.get("iters", 40))
    },
)
def optimize(self, method="gn", iters=40):
    ...
```

- `safe_args` – only scalar values from these argument names are
  recorded; integers are automatically bucketed, strings must be ≤ 32
  chars and a valid Python identifier.
- `shape_fn` – called with the same arguments; must return a
  `Dict[str, str]` of already-bucketed values.
