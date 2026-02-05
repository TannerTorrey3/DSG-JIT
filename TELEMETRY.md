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

Spans will be POSTed as a JSON payload to that URL.  The expected body
shape is:

```json
{
  "spans": [
    {
      "name": "dsgjit.<component>.<op>",
      "timestamp": 1706000000.123,
      "duration_ms": 42.7,
      "attributes": { ... }
    }
  ]
}
```

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
