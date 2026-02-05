# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
OpenTelemetry provider setup and SDK integration.

This module is the future home for wiring the OpenTelemetry SDK into
DSG-JIT's telemetry pipeline.  Nothing here is needed by the current
lightweight implementation, but the structure below documents where each
piece belongs when the time comes to integrate.

-----------------------------------------------------------------------
How this fits into the telemetry package
-----------------------------------------------------------------------

The telemetry pipeline has two sides:

  Sending side (this module)          Receiving side (teammate's server)
  ─────────────────────────────       ──────────────────────────────────
  TracerProvider  ← otel.py           Endpoint URL
      │                                   ↑
      ▼                               configured via
  BatchSpanProcessor ← exporter.py    DSGJIT_TELEMETRY_ENDPOINT
      │                               (default: https://telemetry.ix-infra.com)
      ▼
  OTLPSpanExporter  ← exporter.py
      │
      ▼
  HTTP transport  ← TODO stub in OTLPSpanExporter.export()
      │
      └──────────────────────────────►  endpoint

  - config.py   : reads env vars, provides the endpoint URL
  - identity.py : install_id / session_id
  - client.py   : sampling decisions, hands spans to the processor
  - queue.py    : bounded queue with error-priority eviction
  - exporter.py : BatchSpanProcessor + OTLPSpanExporter
  - decorators.py: the @telemetry_span decorator that user code calls

-----------------------------------------------------------------------
What to implement here for future OTEL SDK integration
-----------------------------------------------------------------------

When the opentelemetry-sdk package is added as a dependency, this module
should provide:

  1. A TracerProvider configured with our BatchSpanProcessor and
     OTLPSpanExporter.  The provider is the single object that owns the
     export pipeline.

  2. A get_tracer(name) function that returns an OTEL Tracer from that
     provider.  Other modules (e.g. decorators.py) would call this
     instead of reaching into client.py directly.

  3. Lifecycle helpers — e.g. a shutdown() that flushes and tears down
     the provider cleanly at process exit.

  Rough skeleton (not yet implemented):

      from opentelemetry.sdk.trace import TracerProvider
      from opentelemetry.sdk.trace.export import BatchSpanProcessor

      _provider: Optional[TracerProvider] = None

      def get_provider() -> TracerProvider:
          ...  # lazy-init: create provider, attach processor + exporter

      def get_tracer(name: str) -> Tracer:
          return get_provider().get_tracer(name)

      def shutdown() -> None:
          ...  # flush and shut down the provider

-----------------------------------------------------------------------
What is NOT this module's job
-----------------------------------------------------------------------

  - Sending spans over the network: that is OTLPSpanExporter in
    exporter.py (currently a stub — see the TODO there).
  - Running the teammate's receiving server: that is a separate service.
    This module only cares about the client-side SDK setup.
  - Deciding what to collect or how to sanitize it: that is
    sanitize.py and decorators.py.
"""