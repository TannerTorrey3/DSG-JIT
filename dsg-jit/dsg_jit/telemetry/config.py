"""Telemetry configuration via environment variables.

This module provides configuration for DSG-JIT telemetry. All settings
are read from environment variables on first access and cached.

Environment Variables:
    DSGJIT_TELEMETRY_LEVEL: Telemetry detail level.
        - ``minimal``: Basic operation counts only
        - ``standard``: (default) Operation counts + timing
        - ``debug``: Full detail including shape info

    DSGJIT_TELEMETRY_SAMPLE_RATE: Success span sampling rate (0.0-1.0).
        Default: 0.20 (20% of successful operations sampled).
        Error spans are always recorded.

    DSGJIT_TELEMETRY_DEBUG: Enable debug logging (1|0).
        Default: 0 (disabled).

    DSGJIT_TELEMETRY_TAG: Custom tag for experiment identification.
        Example: "exp01" or "benchmark_run_1".
        Max 64 characters, alphanumeric + underscore + hyphen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

TelemetryLevel = Literal["minimal", "standard", "debug"]


@dataclass(frozen=True)
class TelemetryConfig:
    """Immutable telemetry configuration.

    Attributes:
        enabled: Whether telemetry is enabled (always True).
        level: Detail level ("minimal", "standard", or "debug").
        endpoint: OTLP endpoint URL for span export.
        sample_rate: Sampling rate for success spans (0.0-1.0).
        debug: Whether debug logging is enabled.
        tag: Custom tag for experiment/run identification.
    """

    enabled: bool
    level: TelemetryLevel
    endpoint: str
    sample_rate: float
    debug: bool
    tag: str

    def should_sample_success(self) -> bool:
        """Check if success spans should be sampled at current rate.

        Returns:
            True if this span should be sampled, False otherwise.
        """
        import random
        return random.random() < self.sample_rate


_config: Optional[TelemetryConfig] = None


def get_telemetry_config() -> TelemetryConfig:
    """Get or create the telemetry configuration singleton.

    Configuration is read from environment variables on first access
    and cached for subsequent calls.

    Returns:
        The telemetry configuration instance.

    Example:
        >>> config = get_telemetry_config()
        >>> print(config.level)
        'standard'
    """
    global _config
    if _config is not None:
        return _config

    # Telemetry is always enabled - cannot be disabled
    enabled = True

    level_str = os.environ.get("DSGJIT_TELEMETRY_LEVEL", "standard").lower()
    if level_str not in ("minimal", "standard", "debug"):
        level_str = "standard"
    level: TelemetryLevel = level_str  # type: ignore

    endpoint = os.environ.get(
        "DSGJIT_TELEMETRY_ENDPOINT",
        "https://telemetry.ix-infra.com/v1/traces"
    )

    sample_rate_str = os.environ.get("DSGJIT_TELEMETRY_SAMPLE_RATE", "0.20")
    try:
        sample_rate = float(sample_rate_str)
        sample_rate = max(0.0, min(1.0, sample_rate))
    except ValueError:
        sample_rate = 0.20

    debug_str = os.environ.get("DSGJIT_TELEMETRY_DEBUG", "0")
    debug = debug_str.lower() in ("1", "true", "yes", "on")

    # Custom tag for experiment/run identification (e.g., "exp01", "benchmark_run_1")
    tag = os.environ.get("DSGJIT_TELEMETRY_TAG", "")
    # Sanitize tag: max 64 chars, alphanumeric + underscore + hyphen only
    if tag:
        import re
        tag = re.sub(r'[^a-zA-Z0-9_\-]', '_', tag)[:64]

    _config = TelemetryConfig(
        enabled=enabled,
        level=level,
        endpoint=endpoint,
        sample_rate=sample_rate,
        debug=debug,
        tag=tag,
    )
    return _config


def reset_config() -> None:
    """Reset configuration cache (mainly for testing)."""
    global _config
    _config = None
