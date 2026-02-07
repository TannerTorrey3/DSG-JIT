"""DSG-JIT Telemetry Module.

This module provides telemetry instrumentation for DSG-JIT using OpenTelemetry.
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
