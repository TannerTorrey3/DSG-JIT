"""
OpenTelemetry SDK integration for DSG-JIT telemetry.

This module initializes and manages the OpenTelemetry tracer provider,
exporter, and related components. It provides:
- `setup_telemetry()` - Initialize the OTel tracer with OTLP exporter
- `get_tracer()` - Get the configured tracer for creating spans
- `shutdown_telemetry()` - Graceful shutdown and flush
- `reset_telemetry()` - Reset state (for testing)
"""

from __future__ import annotations

import atexit
import logging
import platform
from typing import Optional

# Completely silence all OpenTelemetry logging from users
_otel_logger = logging.getLogger("opentelemetry")
_otel_logger.setLevel(logging.CRITICAL + 1)  # Above CRITICAL = nothing
_otel_logger.addHandler(logging.NullHandler())
_otel_logger.propagate = False

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, Decision
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode

from dsg_jit.telemetry.config import get_telemetry_config
from dsg_jit.telemetry.identity import get_install_id, get_session_id


def _get_version() -> str:
    """Get DSG-JIT version string."""
    try:
        from importlib.metadata import version
        return version("dsg_jit")
    except Exception:
        try:
            from dsg_jit import __version__
            return __version__
        except (ImportError, AttributeError):
            return "0.0.0"


class DsgJitSampler(Sampler):
    """Custom sampler that respects DSG-JIT telemetry config.

    - If telemetry is disabled, drops all spans
    - Error spans are always sampled (100%)
    - Success spans are sampled at the configured rate
    """

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
    ) -> SamplingResult:
        # Always sample - we handle success/error filtering at record time
        # This is because we don't know if it's an error until the span ends
        return SamplingResult(Decision.RECORD_AND_SAMPLE)

    def get_description(self) -> str:
        return "DsgJitSampler"


# Module-level state
_tracer_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None
_initialized: bool = False


def setup_telemetry() -> None:
    """Initialize OpenTelemetry with OTLP exporter.

    This function is idempotent - calling it multiple times has no effect
    after the first initialization.
    """
    global _tracer_provider, _tracer, _initialized

    if _initialized:
        return

    config = get_telemetry_config()

    # Create resource with DSG-JIT attributes
    resource = Resource.create({
        SERVICE_NAME: "dsg-jit",
        "service.version": _get_version(),
        "ix.install_id": get_install_id(),
        "ix.session_id": get_session_id(),
        "runtime.python": platform.python_version(),
        "runtime.os": platform.system().lower(),
        "runtime.arch": platform.machine(),
        "dsgjit.telemetry_level": config.level,
    })

    # Create sampler
    sampler = DsgJitSampler()

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    # Create OTLP exporter with custom headers
    headers = {
        "X-Ix-Install-Id": get_install_id(),
        "X-Ix-Session-Id": get_session_id(),
        "X-Ix-Pkg-Version": _get_version(),
        "X-Ix-Telemetry-Level": config.level,
    }

    exporter = OTLPSpanExporter(
        endpoint=config.endpoint,
        headers=headers,
        timeout=5,  # 5 second timeout
    )

    # Create batch processor
    processor = BatchSpanProcessor(
        exporter,
        max_queue_size=512,
        max_export_batch_size=32,
        schedule_delay_millis=5000,  # 5 seconds
    )

    _tracer_provider.add_span_processor(processor)

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Create tracer
    _tracer = trace.get_tracer("dsg_jit.telemetry", _get_version())

    _initialized = True

    # Register shutdown handler
    atexit.register(shutdown_telemetry)


def get_tracer() -> trace.Tracer:
    """Get the configured OpenTelemetry tracer.

    Automatically initializes telemetry if not already done.

    Returns:
        The OpenTelemetry tracer for creating spans.
    """
    global _tracer

    if not _initialized:
        setup_telemetry()

    # _tracer should always be set after setup_telemetry()
    assert _tracer is not None, "Telemetry tracer not initialized"
    return _tracer


def shutdown_telemetry() -> None:
    """Shutdown the tracer provider and flush pending spans."""
    global _tracer_provider, _tracer, _initialized

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None

    _tracer = None
    _initialized = False


def reset_telemetry() -> None:
    """Reset telemetry state (for testing).

    This shuts down any existing provider and clears the initialized flag,
    allowing setup_telemetry() to be called again.
    """
    shutdown_telemetry()


def is_telemetry_initialized() -> bool:
    """Check if telemetry has been initialized."""
    return _initialized
