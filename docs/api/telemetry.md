# Telemetry API

This section documents the telemetry module for DSG-JIT instrumentation.

For information about what data is collected and why, see the [Telemetry Overview](../telemetry.md).

---

## Decorator

The primary interface for adding telemetry to functions.

::: telemetry.decorators
    options:
      members:
        - telemetry_span

---

## Configuration

Environment-based configuration for telemetry behavior.

::: telemetry.config
    options:
      members:
        - TelemetryConfig
        - get_telemetry_config

---

## Data Sanitization

Utilities for privacy-safe data collection.

::: telemetry.sanitize
    options:
      members:
        - bucket_count
        - get_error_code
        - SAFE_SCALAR_ARGS
        - ERROR_CODES

---

## OpenTelemetry Integration

Low-level tracer management (usually not needed directly).

::: telemetry.otel
    options:
      members:
        - setup_telemetry
        - shutdown_telemetry
        - get_tracer
