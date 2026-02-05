# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Smoke tests for the telemetry package.

Validates the Team A acceptance criteria:
  - importing DSG-JIT does not start telemetry
  - first instrumented call emits session.start
  - disabling export (endpoint empty or telemetry=0) causes no network attempts
  - no user content is ever serialized (poison-string validation)
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
    reset_config()
    reset_client()
    reset_telemetry_state()


@pytest.fixture(autouse=True)
def _clean_telemetry(monkeypatch):
    """Reset singletons and env vars around every test in this module."""
    _reset_all()
    # Default: telemetry on, endpoint empty (no network), full sample rate
    monkeypatch.setenv("DSGJIT_TELEMETRY", "1")
    monkeypatch.setenv("DSGJIT_TELEMETRY_ENDPOINT", "")
    monkeypatch.setenv("DSGJIT_TELEMETRY_SAMPLE_RATE", "1.0")
    _reset_all()  # re-read config after env is set
    yield
    _reset_all()


# ---------------------------------------------------------------------------
# 1. Import does not start telemetry
# ---------------------------------------------------------------------------

def test_import_does_not_create_client():
    """Importing the telemetry package must not instantiate a client."""
    from dsg_jit.telemetry import client as client_mod
    from dsg_jit.telemetry import decorators as dec_mod

    assert client_mod._client is None
    assert dec_mod._session_started is False


# ---------------------------------------------------------------------------
# 2. First instrumented call emits session.start
# ---------------------------------------------------------------------------

def test_first_instrumented_call_sets_session_started():
    """Calling a @telemetry_span function must flip _session_started."""
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry import decorators as dec_mod

    @telemetry_span(component="test", op="noop")
    def noop():
        return 42

    assert dec_mod._session_started is False
    result = noop()
    assert result == 42
    assert dec_mod._session_started is True


# ---------------------------------------------------------------------------
# 3. Disabled export — no network, no crash
# ---------------------------------------------------------------------------

def test_telemetry_disabled_does_not_create_client(monkeypatch):
    """With DSGJIT_TELEMETRY=0 the decorator fast-paths; no client is created."""
    monkeypatch.setenv("DSGJIT_TELEMETRY", "0")
    _reset_all()

    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry import client as client_mod

    @telemetry_span(component="test", op="noop")
    def noop():
        return 42

    assert noop() == 42
    assert client_mod._client is None


def test_empty_endpoint_does_not_crash():
    """Spans are recorded and exported without error even when endpoint is empty."""
    from dsg_jit.telemetry.decorators import telemetry_span

    @telemetry_span(component="test", op="noop")
    def noop():
        return 42

    # Must not raise
    assert noop() == 42


def test_exception_in_decorated_fn_still_propagates():
    """Telemetry must not swallow exceptions raised by user code."""
    from dsg_jit.telemetry.decorators import telemetry_span

    @telemetry_span(component="test", op="boom")
    def boom():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        boom()


# ---------------------------------------------------------------------------
# 4. Poison-string tests — no user content leaks into spans
# ---------------------------------------------------------------------------

# A string that must never appear in any serialised span.  It is longer than
# 32 chars so sanitize_safe_args will reject it even if it lands on a
# safe-arg key, AND it contains characters that make it a non-identifier.
POISON = "SECRET/user-content:do-not-leak-this-value-abc123"


def test_poison_in_non_safe_arg_excluded():
    """sanitize_safe_args must drop any key not in the safe set entirely."""
    from dsg_jit.telemetry.sanitize import sanitize_safe_args

    args = {
        "method": "gn",           # safe, short identifier — should survive
        "secret_data": POISON,    # not in safe set — must be dropped
    }
    result = sanitize_safe_args(args, safe_args={"method"})

    assert "method" in result
    assert "secret_data" not in result
    assert POISON not in json.dumps(result)


def test_poison_in_safe_arg_value_blocked():
    """A poison value on a safe-arg key must be rejected by sanitize."""
    from dsg_jit.telemetry.sanitize import sanitize_safe_args

    # "method" is safe, but the value is long + non-identifier
    args = {"method": POISON}
    result = sanitize_safe_args(args, safe_args={"method"})

    assert POISON not in json.dumps(result)


def test_poison_does_not_reach_span_attributes():
    """End-to-end: poison passed as a non-safe kwarg must not appear in the span."""
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client

    @telemetry_span(component="test", op="poison_e2e", safe_args={"method"})
    def work(method="gn", secret=None):
        return 1

    work(method="gn", secret=POISON)

    # Pull spans directly out of the processor queue for inspection
    client = _get_client()
    spans = client._processor._queue.drain() if client._processor else []

    # session.start + the work() span should both be clean
    for span in spans:
        serialised = json.dumps(span)
        assert POISON not in serialised, f"Poison leaked into span: {serialised}"
