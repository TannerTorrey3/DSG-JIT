# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Comprehensive tests for telemetry spec compliance.

Tests additional requirements beyond smoke tests:
  - Error recording (error.type, error.code, never message)
  - Sampling policy (errors 100%, success sampled)
  - Payload and span limits
  - Session.start attributes
  - Required HTTP headers
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
    from dsg_jit.telemetry.client import reset_client
    from dsg_jit.telemetry.decorators import reset_telemetry_state
    from dsg_jit.telemetry.identity import reset_session_id
    reset_config()
    reset_client()
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
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="test", op="error_test")
    def raise_value_error():
        raise ValueError("This message should NOT appear in telemetry")

    with pytest.raises(ValueError):
        raise_value_error()

    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    # Find the error span (not session.start)
    error_spans = [s for s in spans if s.get("name") == "dsgjit.test.error_test"]
    assert len(error_spans) == 1

    attrs = error_spans[0]["attributes"]
    assert attrs["dsgjit.status"] == "error"
    assert attrs["error.type"] == "ValueError"
    assert attrs["error.code"] == "invalid_argument"
    assert attrs["error.component"] == "test"
    assert attrs["error.op"] == "error_test"

    # Verify message is NOT recorded
    serialized = json.dumps(error_spans[0])
    assert "This message should NOT appear" not in serialized


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
# Sampling Policy Tests
# ---------------------------------------------------------------------------

def test_error_spans_always_recorded(monkeypatch):
    """Error spans must always be recorded regardless of sample rate."""
    monkeypatch.setenv("DSGJIT_TELEMETRY_SAMPLE_RATE", "0.0")  # 0% sampling
    _reset_all()

    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="test", op="error_sampled")
    def raise_error():
        raise RuntimeError("test")

    # Call multiple times
    for _ in range(5):
        with pytest.raises(RuntimeError):
            raise_error()

    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    # All 5 error spans + 1 session.start should be recorded
    error_spans = [s for s in spans if s.get("attributes", {}).get("dsgjit.status") == "error"]
    assert len(error_spans) == 5, "All error spans should be recorded even with 0% sample rate"


def test_success_spans_respect_sample_rate(monkeypatch):
    """Success spans should be sampled at the configured rate."""
    monkeypatch.setenv("DSGJIT_TELEMETRY_SAMPLE_RATE", "0.0")  # 0% sampling
    _reset_all()

    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="test", op="success_sampled")
    def succeed():
        return 42

    # Call multiple times
    for _ in range(10):
        succeed()

    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    # Only session.start should be recorded (it's always recorded)
    # Success spans should be dropped due to 0% sample rate
    success_spans = [s for s in spans if s.get("name") == "dsgjit.test.success_sampled"]
    assert len(success_spans) == 0, "Success spans should be dropped with 0% sample rate"


# ---------------------------------------------------------------------------
# Session.start Tests
# ---------------------------------------------------------------------------

def test_session_start_emitted_once():
    """Session.start should be emitted exactly once per process."""
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="test", op="noop")
    def noop():
        return 1

    # Call multiple times
    for _ in range(5):
        noop()

    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    session_starts = [s for s in spans if s.get("name") == "dsgjit.session.start"]
    assert len(session_starts) == 1, "session.start should be emitted exactly once"


def test_session_start_has_required_attributes():
    """Session.start span must have all required attributes per spec."""
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="world", op="test_op")
    def trigger_session():
        return 1

    trigger_session()

    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    session_starts = [s for s in spans if s.get("name") == "dsgjit.session.start"]
    assert len(session_starts) == 1

    attrs = session_starts[0]["attributes"]

    # Required common attributes
    assert "ix.install_id" in attrs
    assert "ix.session_id" in attrs
    assert "dsgjit.version" in attrs
    assert "runtime.python" in attrs
    assert "runtime.os" in attrs
    assert "runtime.arch" in attrs
    assert attrs["dsgjit.status"] == "ok"

    # Session-specific attributes
    assert attrs["dsgjit.entry_component"] == "world"
    assert "dsgjit.backend_available" in attrs
    assert "dsgjit.telemetry_enabled" in attrs


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


def test_integer_args_are_bucketed():
    """Integer safe_args values must be bucketed, not recorded raw."""
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="test", op="bucket_test", safe_args={"iters"})
    def with_iters(iters=100):
        return iters

    with_iters(iters=42)

    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    op_spans = [s for s in spans if s.get("name") == "dsgjit.test.bucket_test"]
    assert len(op_spans) == 1

    attrs = op_spans[0]["attributes"]
    # Should be bucketed, not raw
    assert attrs.get("dsgjit.args.iters") == "10-99"
    # Raw value should NOT appear as the iters value
    # (can't check full JSON as "42" may appear in UUIDs)
    assert attrs.get("dsgjit.args.iters") != "42"
    assert attrs.get("dsgjit.args.iters") != 42


# ---------------------------------------------------------------------------
# Payload Limits Tests
# ---------------------------------------------------------------------------

def test_max_spans_limit():
    """Exporter should limit spans per request to MAX_SPANS_PER_REQUEST."""
    from dsg_jit.telemetry.exporter import MAX_SPANS_PER_REQUEST

    assert MAX_SPANS_PER_REQUEST == 200, "Max spans should be 200 per spec"


def test_max_payload_size():
    """Exporter should have 64KB payload limit."""
    from dsg_jit.telemetry.exporter import MAX_PAYLOAD_BYTES

    assert MAX_PAYLOAD_BYTES == 64 * 1024, "Max payload should be 64KB per spec"


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


# ---------------------------------------------------------------------------
# Required Headers Test
# ---------------------------------------------------------------------------

def test_exporter_has_required_headers():
    """Verify exporter includes all required X-Ix-* headers."""
    # This is a code inspection test - we verify the headers are defined
    from dsg_jit.telemetry import exporter
    import inspect

    source = inspect.getsource(exporter.OTLPSpanExporter.export)

    assert "X-Ix-Install-Id" in source
    assert "X-Ix-Session-Id" in source
    assert "X-Ix-Pkg-Version" in source
    assert "X-Ix-Telemetry-Level" in source
