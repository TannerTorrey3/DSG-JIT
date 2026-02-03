# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
DSG-JIT Telemetry Module.

This module provides privacy-preserving telemetry for DSG-JIT, collecting
anonymous usage metrics to help improve the library while ensuring no
user content or identifying information is ever transmitted.

See PRIVACY_TELEMETRY.md for details on what is and isn't collected.
"""

from dsg_jit.telemetry.decorators import telemetry_span
from dsg_jit.telemetry.sanitize import bucket_count
from dsg_jit.telemetry.config import get_telemetry_config, TelemetryConfig

__all__ = [
    "telemetry_span",
    "bucket_count",
    "get_telemetry_config",
    "TelemetryConfig",
]
