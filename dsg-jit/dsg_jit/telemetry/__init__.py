"""DSG-JIT Telemetry Module.

This module provides anonymous usage telemetry for DSG-JIT to help improve
the library. Telemetry is collected via OpenTelemetry and includes:

- Feature usage patterns (which operations are called)
- Error types (categorized, not raw messages)
- Performance timing
- Runtime environment info (Python version, OS, JAX backend)

**What we do NOT collect:**

- Source code or file paths
- Variable values or model data
- Personal information
- Error messages or stack traces

Configuration is via environment variables:

- ``DSGJIT_TELEMETRY_LEVEL``: minimal|standard|debug (default: standard)
- ``DSGJIT_TELEMETRY_DEBUG``: 1|0 (default: 0)
- ``DSGJIT_TELEMETRY_TAG``: custom tag for experiment identification

Example:
    The telemetry system initializes automatically. To add telemetry
    to your own functions::

        from dsg_jit.telemetry import telemetry_span

        @telemetry_span(component="my_module", op="my_function")
        def my_function():
            ...
"""

from dsg_jit.telemetry.decorators import telemetry_span, reset_telemetry_state
from dsg_jit.telemetry.sanitize import bucket_count
from dsg_jit.telemetry.config import get_telemetry_config, TelemetryConfig
from dsg_jit.telemetry.otel import (
    setup_telemetry,
    shutdown_telemetry,
    reset_telemetry,
    get_tracer,
)

__all__ = [
    # Decorator
    "telemetry_span",
    # Utilities
    "bucket_count",
    # Configuration
    "get_telemetry_config",
    "TelemetryConfig",
    # OpenTelemetry functions
    "setup_telemetry",
    "shutdown_telemetry",
    "reset_telemetry",
    "get_tracer",
    # Testing
    "reset_telemetry_state",
]
