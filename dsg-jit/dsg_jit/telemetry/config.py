# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Telemetry configuration from environment variables.

Environment Variables:
    DSGJIT_TELEMETRY: 1|0 (default 1) - Enable/disable telemetry
    DSGJIT_TELEMETRY_LEVEL: minimal|standard|debug (default standard)
    DSGJIT_TELEMETRY_ENDPOINT: URL (default https://telemetry.ix-infra.com)
    DSGJIT_TELEMETRY_SAMPLE_RATE: 0.0-1.0 (default 0.20) - Success span sampling
    DSGJIT_TELEMETRY_DEBUG: 1|0 (default 0) - Enable debug logging
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

TelemetryLevel = Literal["minimal", "standard", "debug"]


@dataclass(frozen=True)
class TelemetryConfig:
    """Immutable telemetry configuration."""

    enabled: bool
    level: TelemetryLevel
    endpoint: str
    sample_rate: float
    debug: bool

    def should_sample_success(self) -> bool:
        """Check if success spans should be sampled at current rate."""
        import random
        return random.random() < self.sample_rate


_config: Optional[TelemetryConfig] = None


def get_telemetry_config() -> TelemetryConfig:
    """Get or create the telemetry configuration singleton.

    Configuration is read from environment variables on first access
    and cached for subsequent calls.

    :return: The telemetry configuration.
    """
    global _config
    if _config is not None:
        return _config

    # Parse environment variables
    enabled_str = os.environ.get("DSGJIT_TELEMETRY", "1")
    enabled = enabled_str.lower() in ("1", "true", "yes", "on")

    level_str = os.environ.get("DSGJIT_TELEMETRY_LEVEL", "standard").lower()
    if level_str not in ("minimal", "standard", "debug"):
        level_str = "standard"
    level: TelemetryLevel = level_str  # type: ignore

    endpoint = os.environ.get(
        "DSGJIT_TELEMETRY_ENDPOINT",
        "https://telemetry.ix-infra.com"
    )

    sample_rate_str = os.environ.get("DSGJIT_TELEMETRY_SAMPLE_RATE", "0.20")
    try:
        sample_rate = float(sample_rate_str)
        sample_rate = max(0.0, min(1.0, sample_rate))
    except ValueError:
        sample_rate = 0.20

    debug_str = os.environ.get("DSGJIT_TELEMETRY_DEBUG", "0")
    debug = debug_str.lower() in ("1", "true", "yes", "on")

    _config = TelemetryConfig(
        enabled=enabled,
        level=level,
        endpoint=endpoint,
        sample_rate=sample_rate,
        debug=debug,
    )
    return _config


def reset_config() -> None:
    """Reset configuration cache (mainly for testing)."""
    global _config
    _config = None
