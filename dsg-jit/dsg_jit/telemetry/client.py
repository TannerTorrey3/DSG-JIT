# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Telemetry client for collecting and exporting spans.

TelemetryClient is the single entry-point for span recording.  It owns
the sampling decision and delegates all queueing and export to the
BatchSpanProcessor / OTLPSpanExporter defined in exporter.py.
"""

from __future__ import annotations

import atexit
from typing import Any, Dict, Optional

from dsg_jit.telemetry.config import get_telemetry_config
from dsg_jit.telemetry.exporter import BatchSpanProcessor, OTLPSpanExporter


class TelemetryClient:
    """Minimal telemetry client for collecting and exporting spans.

    Sampling is applied here; once a span is accepted it is handed off
    to the BatchSpanProcessor which never blocks the caller.
    """

    def __init__(self) -> None:
        self._processor: Optional[BatchSpanProcessor] = None
        self._enabled: bool = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_processor(self) -> BatchSpanProcessor:
        """Lazy-initialize the processor and exporter."""
        if self._processor is None:
            config = get_telemetry_config()
            exporter = OTLPSpanExporter(endpoint=config.endpoint)
            self._processor = BatchSpanProcessor(exporter)
        return self._processor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_span(self, span: Dict[str, Any]) -> None:
        """Record a span for export.

        Errors are always accepted.  Success spans are subject to the
        configured sample rate.

        :param span: The span data dictionary.
        """
        if not self._enabled:
            return

        config = get_telemetry_config()
        if not config.enabled:
            return

        # Apply sampling for success spans (errors always recorded per spec)
        attrs = span.get("attributes", {})
        is_error = attrs.get("dsgjit.status") == "error"
        if not is_error and not config.should_sample_success():
            return

        self._get_processor().add_span(span)

    def disable(self) -> None:
        """Disable the client after repeated failures."""
        self._enabled = False

    def shutdown(self) -> None:
        """Shut down the processor, flushing any pending spans."""
        if self._processor is not None:
            self._processor.shutdown()
            self._processor = None


_client: Optional[TelemetryClient] = None


def _get_client() -> TelemetryClient:
    """Get or create the singleton telemetry client."""
    global _client
    if _client is None:
        _client = TelemetryClient()
    return _client


def reset_client() -> None:
    """Reset client state (mainly for testing)."""
    global _client
    if _client is not None:
        _client.shutdown()
    _client = None


def _shutdown_on_exit() -> None:
    """Flush telemetry on process exit."""
    global _client
    if _client is not None:
        _client.shutdown()
        _client = None


# Register atexit handler to flush telemetry when the process exits
atexit.register(_shutdown_on_exit)
