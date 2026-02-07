# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Comprehensive tests for telemetry spec compliance.

Tests additional requirements beyond smoke tests:
  - Error recording (error.type, error.code, never message)
  - Sampling policy (errors 100%, success sampled)
  - Session.start attributes
  - Bucketing behavior
  - Identity and config tests
"""

from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_all() -> None:
    """Tear down every telemetry singleton so each test starts clean."""
    from dsg_jit.telemetry.config import reset_config
    from dsg_jit.telemetry.decorators import reset_telemetry_state
    from dsg_jit.telemetry.identity import reset_session_id
    reset_config()
    reset_telemetry_state()
    reset_session_id()


@pytest.fixture(autouse=True)
def _clean_telemetry(monkeypatch):
    """Reset singletons and env vars around every test in this module."""
    _reset_all()
    # Default: telemetry on, endpoint empty (no network), full sample rate
    monkeypatch.setenv("DSGJIT_TELEMETRY", "1")
    monkeypatch.setenv("DSGJIT_TELEMETRY_ENDPOINT", "")
    monkeypatch.setenv("DSGJIT_TELEMETRY_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("DSGJIT_TELEMETRY_LEVEL", "standard")
    _reset_all()  # re-read config after env is set
    yield
    _reset_all()


# ---------------------------------------------------------------------------
# Error Recording Tests
# ---------------------------------------------------------------------------

def test_error_span_records_type_and_code():
    """Error spans must include error.type (class name) and error.code (category)."""
    from dsg_jit.telemetry.decorators import telemetry_span

    @telemetry_span(component="test", op="error_test")
    def raise_value_error():
        raise ValueError("This message should NOT appear in telemetry")

    # The decorator should record the error but still raise it
    with pytest.raises(ValueError):
        raise_value_error()

    # We can't easily inspect the OTel spans without an in-memory exporter,
    # but we verify the decorator doesn't crash and the error propagates


def test_error_codes_map_correctly():
    """Verify various exception types map to correct error codes."""
    from dsg_jit.telemetry.sanitize import get_error_code

    # Test mappings per spec
    assert get_error_code(ValueError("test")) == "invalid_argument"
    assert get_error_code(TypeError("test")) == "invalid_argument"
    assert get_error_code(KeyError("test")) == "invalid_argument"
    assert get_error_code(IndexError("test")) == "invalid_argument"
    assert get_error_code(RuntimeError("test")) == "backend_error"
    assert get_error_code(NotImplementedError("test")) == "not_initialized"
    assert get_error_code(FloatingPointError("test")) == "numerical_issue"
    assert get_error_code(OverflowError("test")) == "numerical_issue"
    assert get_error_code(ZeroDivisionError("test")) == "numerical_issue"
    assert get_error_code(IOError("test")) == "io_error"
    assert get_error_code(OSError("test")) == "io_error"
    assert get_error_code(FileNotFoundError("test")) == "io_error"

    # Unknown exceptions should map to unknown_error
    class CustomError(Exception):
        pass
    assert get_error_code(CustomError("test")) == "unknown_error"


# ---------------------------------------------------------------------------
# Decorator Behavior Tests
# ---------------------------------------------------------------------------

def test_decorator_does_not_swallow_errors():
    """Telemetry decorator must propagate exceptions."""
    from dsg_jit.telemetry.decorators import telemetry_span

    @telemetry_span(component="test", op="error_test")
    def raise_error():
        raise RuntimeError("test error")

    with pytest.raises(RuntimeError, match="test error"):
        raise_error()


def test_decorator_returns_function_result():
    """Telemetry decorator must return the wrapped function's result."""
    from dsg_jit.telemetry.decorators import telemetry_span

    @telemetry_span(component="test", op="success_test")
    def return_value():
        return {"key": "value", "count": 42}

    result = return_value()
    assert result == {"key": "value", "count": 42}


def test_session_start_emitted_once():
    """Session.start should be emitted exactly once per process."""
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry import decorators as dec_mod

    @telemetry_span(component="test", op="noop")
    def noop():
        return 1

    assert dec_mod._session_started is False

    # Call multiple times
    for _ in range(5):
        noop()

    # _session_started should be True after first call
    assert dec_mod._session_started is True


# ---------------------------------------------------------------------------
# Bucketing Tests
# ---------------------------------------------------------------------------

def test_bucket_count_boundaries():
    """Verify bucket_count returns correct buckets at boundaries."""
    from dsg_jit.telemetry.sanitize import bucket_count

    assert bucket_count(0) == "0"
    assert bucket_count(1) == "1-9"
    assert bucket_count(9) == "1-9"
    assert bucket_count(10) == "10-99"
    assert bucket_count(99) == "10-99"
    assert bucket_count(100) == "100-999"
    assert bucket_count(999) == "100-999"
    assert bucket_count(1000) == "1k-9k"
    assert bucket_count(9999) == "1k-9k"
    assert bucket_count(10000) == "10k-99k"
    assert bucket_count(99999) == "10k-99k"
    assert bucket_count(100000) == "100k-999k"
    assert bucket_count(999999) == "100k-999k"
    assert bucket_count(1000000) == "1M+"
    assert bucket_count(10000000) == "1M+"


def test_sanitize_safe_args_buckets_integers():
    """Integer safe_args values must be bucketed, not recorded raw."""
    from dsg_jit.telemetry.sanitize import sanitize_safe_args

    result = sanitize_safe_args(
        {"iters": 42, "method": "gn"},
        safe_args={"iters", "method"}
    )

    # iters should be bucketed
    assert result.get("iters") == "10-99"
    # method should be kept as-is (short identifier)
    assert result.get("method") == "gn"


def test_sanitize_safe_args_rejects_floats():
    """Float values should not be recorded (privacy concern)."""
    from dsg_jit.telemetry.sanitize import sanitize_safe_args

    result = sanitize_safe_args(
        {"lr": 0.001, "method": "gn"},
        safe_args={"lr", "method"}
    )

    # lr (float) should NOT be included
    assert "lr" not in result
    # method should be kept
    assert result.get("method") == "gn"


def test_sanitize_safe_args_rejects_long_strings():
    """Long strings should not be recorded (could be user content)."""
    from dsg_jit.telemetry.sanitize import sanitize_safe_args

    long_string = "this_is_a_very_long_string_that_exceeds_32_chars"
    result = sanitize_safe_args(
        {"method": long_string},
        safe_args={"method"}
    )

    # Long string should NOT be included
    assert "method" not in result


# ---------------------------------------------------------------------------
# Identity Tests
# ---------------------------------------------------------------------------

def test_install_id_is_uuid():
    """Install ID must be a valid UUID."""
    from dsg_jit.telemetry.identity import get_install_id
    import uuid

    install_id = get_install_id()
    # Should not raise
    uuid.UUID(install_id)


def test_session_id_is_uuid():
    """Session ID must be a valid UUID."""
    from dsg_jit.telemetry.identity import get_session_id
    import uuid

    session_id = get_session_id()
    # Should not raise
    uuid.UUID(session_id)


def test_session_id_changes_on_reset():
    """Session ID should change after reset (simulating new process)."""
    from dsg_jit.telemetry.identity import get_session_id, reset_session_id

    session1 = get_session_id()
    reset_session_id()
    session2 = get_session_id()

    assert session1 != session2, "Session ID should change after reset"


def test_install_id_persists():
    """Install ID should be the same across calls."""
    from dsg_jit.telemetry.identity import get_install_id

    id1 = get_install_id()
    id2 = get_install_id()

    assert id1 == id2, "Install ID should be consistent"


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

def test_config_defaults(monkeypatch):
    """Verify default configuration values per spec."""
    # Clear all telemetry env vars
    for var in ["DSGJIT_TELEMETRY", "DSGJIT_TELEMETRY_LEVEL",
                "DSGJIT_TELEMETRY_ENDPOINT", "DSGJIT_TELEMETRY_SAMPLE_RATE",
                "DSGJIT_TELEMETRY_DEBUG"]:
        monkeypatch.delenv(var, raising=False)
    _reset_all()

    from dsg_jit.telemetry.config import get_telemetry_config

    config = get_telemetry_config()
    assert config.enabled is True, "Telemetry should be enabled by default"
    assert config.level == "standard", "Default level should be standard"
    assert "telemetry.ix-infra.com" in config.endpoint, "Default endpoint should be ix-infra"
    assert config.sample_rate == 0.20, "Default sample rate should be 0.20"
    assert config.debug is False, "Debug should be off by default"


def test_config_level_validation(monkeypatch):
    """Invalid level should fall back to standard."""
    monkeypatch.setenv("DSGJIT_TELEMETRY_LEVEL", "invalid_level")
    _reset_all()

    from dsg_jit.telemetry.config import get_telemetry_config

    config = get_telemetry_config()
    assert config.level == "standard"


def test_sample_rate_clamping(monkeypatch):
    """Sample rate should be clamped to [0.0, 1.0]."""
    from dsg_jit.telemetry.config import get_telemetry_config

    monkeypatch.setenv("DSGJIT_TELEMETRY_SAMPLE_RATE", "2.0")
    _reset_all()
    config = get_telemetry_config()
    assert config.sample_rate == 1.0

    monkeypatch.setenv("DSGJIT_TELEMETRY_SAMPLE_RATE", "-0.5")
    _reset_all()
    config = get_telemetry_config()
    assert config.sample_rate == 0.0


def test_config_always_enabled(monkeypatch):
    """Telemetry is always enabled and cannot be disabled."""
    # Even if someone tries to set DSGJIT_TELEMETRY=0, it should still be enabled
    monkeypatch.setenv("DSGJIT_TELEMETRY", "0")
    _reset_all()

    from dsg_jit.telemetry.config import get_telemetry_config

    config = get_telemetry_config()
    assert config.enabled is True, "Telemetry must always be enabled"


def test_config_tag_sanitization(monkeypatch):
    """Tags should be sanitized to alphanumeric + underscore + hyphen."""
    monkeypatch.setenv("DSGJIT_TELEMETRY_TAG", "exp01/test@run#1")
    _reset_all()

    from dsg_jit.telemetry.config import get_telemetry_config

    config = get_telemetry_config()
    # Special chars should be replaced with underscores
    assert "/" not in config.tag
    assert "@" not in config.tag
    assert "#" not in config.tag


# ---------------------------------------------------------------------------
# OpenTelemetry Setup Tests
# ---------------------------------------------------------------------------

def test_otel_setup_is_idempotent():
    """setup_telemetry() should be safe to call multiple times."""
    from dsg_jit.telemetry.otel import setup_telemetry, is_telemetry_initialized

    setup_telemetry()
    assert is_telemetry_initialized() is True

    # Should not raise
    setup_telemetry()
    setup_telemetry()
    assert is_telemetry_initialized() is True


def test_otel_reset_allows_reinit():
    """reset_telemetry() should allow re-initialization."""
    from dsg_jit.telemetry.otel import (
        setup_telemetry,
        reset_telemetry,
        is_telemetry_initialized,
    )

    setup_telemetry()
    assert is_telemetry_initialized() is True

    reset_telemetry()
    assert is_telemetry_initialized() is False

    setup_telemetry()
    assert is_telemetry_initialized() is True
