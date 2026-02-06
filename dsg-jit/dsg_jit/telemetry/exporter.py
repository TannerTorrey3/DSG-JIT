# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Span export and batch processing for telemetry.

OTLPSpanExporter handles sending spans to the configured endpoint.
BatchSpanProcessor manages batching and drives periodic export on a
background thread, using a BoundedSpanQueue for queueing.

Key behaviors:
- Never blocks the caller: spans are queued, export runs on a background thread
- Exponential backoff: on export failure, backs off before retrying
- Fail closed: after repeated failures, the exporter disables itself
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from dsg_jit.telemetry.config import get_telemetry_config
from dsg_jit.telemetry.identity import get_install_id, get_session_id
from dsg_jit.telemetry.queue import BoundedSpanQueue, DEFAULT_MAX_QUEUE_SIZE


def _get_version() -> str:
    """Get DSG-JIT version string for headers."""
    try:
        from importlib.metadata import version
        return version("dsg_jit")
    except Exception:
        try:
            from dsg_jit import __version__
            return __version__
        except (ImportError, AttributeError):
            return "0.0.0"

# --- Export configuration constants ---
DEFAULT_BATCH_SIZE = 128
DEFAULT_EXPORT_INTERVAL = 5.0  # seconds
MAX_EXPORT_FAILURES = 5
INITIAL_BACKOFF_SECS = 1.0
MAX_BACKOFF_SECS = 30.0
MAX_PAYLOAD_BYTES = 64 * 1024  # 64KB per spec
MAX_SPANS_PER_REQUEST = 200  # Gateway enforcement limit per spec


class OTLPSpanExporter:
    """Exports spans to the OTLP endpoint.

    Tracks consecutive failures and applies exponential backoff between
    retries.  After MAX_EXPORT_FAILURES consecutive failures the exporter
    disables itself for the lifetime of the process.
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._consecutive_failures: int = 0
        self._disabled: bool = False
        self._next_retry_at: float = 0.0

    @property
    def disabled(self) -> bool:
        return self._disabled

    def export(self, spans: List[Dict[str, Any]]) -> bool:
        """Attempt to export a batch of spans.

        :param spans: List of span dicts to export.
        :return: True if export succeeded, False on failure or backoff.
        """
        if self._disabled:
            return False

        # No endpoint configured � nothing to send
        if not self._endpoint:
            return False

        # Still within backoff window � skip this attempt
        if time.time() < self._next_retry_at:
            return False

        try:
            config = get_telemetry_config()
            if config.debug:
                import sys
                for span in spans:
                    print(f"[TELEMETRY DEBUG] {span}", file=sys.stderr)

            import json
            import urllib.request

            # Enforce max spans per request (gateway limit per spec)
            if len(spans) > MAX_SPANS_PER_REQUEST:
                spans = spans[:MAX_SPANS_PER_REQUEST]

            # Convert to OTLP JSON format
            otlp_spans = []
            for span in spans:
                attrs = span.get("attributes", {})
                otlp_span = {
                    "name": span.get("name", "unknown"),
                    "startTimeUnixNano": str(int(span.get("timestamp", 0) * 1_000_000_000)),
                    "endTimeUnixNano": str(int((span.get("timestamp", 0) + span.get("duration_ms", 0) / 1000) * 1_000_000_000)),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in attrs.items()
                    ],
                }
                otlp_spans.append(otlp_span)

            otlp_payload = {
                "resourceSpans": [{
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "dsg-jit"}}
                        ]
                    },
                    "scopeSpans": [{
                        "scope": {"name": "dsg_jit.telemetry"},
                        "spans": otlp_spans
                    }]
                }]
            }

            payload = json.dumps(otlp_payload).encode("utf-8")

            # Enforce payload size cap (64KB per spec)
            if len(payload) > MAX_PAYLOAD_BYTES:
                # Payload too large - drop silently per spec
                return False

            # Required headers per telemetry spec
            headers = {
                "Content-Type": "application/json",
                "User-Agent": f"dsg-jit/{_get_version()}",
                "X-Ix-Install-Id": get_install_id(),
                "X-Ix-Session-Id": get_session_id(),
                "X-Ix-Pkg-Version": _get_version(),
                "X-Ix-Telemetry-Level": config.level,
            }

            req = urllib.request.Request(
                self._endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            # 5 s timeout - telemetry must never hang the process
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"endpoint returned {resp.status}")

            self._consecutive_failures = 0
            return True

        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= MAX_EXPORT_FAILURES:
                self._disabled = True
            else:
                backoff = min(
                    INITIAL_BACKOFF_SECS * (2 ** (self._consecutive_failures - 1)),
                    MAX_BACKOFF_SECS,
                )
                self._next_retry_at = time.time() + backoff
            return False


class BatchSpanProcessor:
    """Collects spans via a BoundedSpanQueue and exports them in batches.

    Runs a background daemon thread that flushes the queue when either
    the batch size is reached or the export interval elapses, whichever
    comes first.
    """

    def __init__(
        self,
        exporter: OTLPSpanExporter,
        batch_size: int = DEFAULT_BATCH_SIZE,
        export_interval: float = DEFAULT_EXPORT_INTERVAL,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ) -> None:
        self._exporter = exporter
        self._batch_size = batch_size
        self._export_interval = export_interval
        self._queue = BoundedSpanQueue(max_size=max_queue_size)

        self._shutdown_event = threading.Event()
        self._batch_ready = threading.Event()

        # Daemon thread so it never prevents process exit
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()

    def add_span(self, span: Dict[str, Any]) -> None:
        """Enqueue a span.  Never blocks on export.

        :param span: The span data dictionary.
        """
        if not self._queue.enqueue(span):
            return  # Span was dropped by the queue

        if self._queue.size >= self._batch_size:
            self._batch_ready.set()

    def flush(self) -> None:
        """Drain the queue and export whatever was in it."""
        batch = self._queue.drain()
        if not batch:
            return
        self._batch_ready.clear()

        self._exporter.export(batch)

    def shutdown(self) -> None:
        """Signal the background thread to stop, then flush remaining spans."""
        self._shutdown_event.set()
        self._batch_ready.set()  # Wake the thread if it is waiting
        self._thread.join(timeout=5.0)
        self.flush()  # Final flush of anything remaining

    def _export_loop(self) -> None:
        """Background loop: flush on batch-full signal or interval timeout."""
        while not self._shutdown_event.is_set():
            # Wait for batch-full signal or export interval, whichever first
            self._batch_ready.wait(timeout=self._export_interval)
            if self._shutdown_event.is_set():
                break
            self.flush()
