# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Telemetry client compatibility layer.

This module provides backward-compatible functions for code that previously
used the custom telemetry client. All functionality is now delegated to
the OpenTelemetry-based implementation in otel.py.
"""

from __future__ import annotations

from dsg_jit.telemetry.otel import (
    reset_telemetry,
    shutdown_telemetry,
)


def reset_client() -> None:
    """Reset client state (mainly for testing).

    This is a compatibility wrapper around reset_telemetry().
    """
    reset_telemetry()


def shutdown_client() -> None:
    """Shutdown the telemetry client.

    This is a compatibility wrapper around shutdown_telemetry().
    """
    shutdown_telemetry()
