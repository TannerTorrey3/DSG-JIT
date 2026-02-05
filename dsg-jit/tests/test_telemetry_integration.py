#!/usr/bin/env python
# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Integration test for telemetry export to a live server.

This script is designed to be run manually to test that telemetry
is actually being sent to and received by the server.

Usage:
    # Test with default endpoint (telemetry.ix-infra.com)
    DSGJIT_TELEMETRY_DEBUG=1 python tests/test_telemetry_integration.py

    # Test with custom endpoint
    DSGJIT_TELEMETRY_ENDPOINT=http://localhost:4318/v1/traces \
    DSGJIT_TELEMETRY_DEBUG=1 python tests/test_telemetry_integration.py

    # Run as pytest (will skip if no network)
    pytest tests/test_telemetry_integration.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
import json

# Add dsg-jit to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def reset_telemetry():
    """Reset all telemetry state for clean test."""
    from dsg_jit.telemetry.config import reset_config
    from dsg_jit.telemetry.client import reset_client
    from dsg_jit.telemetry.decorators import reset_telemetry_state
    from dsg_jit.telemetry.identity import reset_session_id
    reset_config()
    reset_client()
    reset_telemetry_state()
    reset_session_id()


def test_live_export():
    """Test actual export to the configured endpoint.

    This test will:
    1. Create spans with various scenarios (success, error)
    2. Export them to the server
    3. Verify no exceptions occurred

    The test passes if the export completes without error.
    Server-side verification must be done manually.
    """
    # Force telemetry on and set sample rate to 100%
    os.environ["DSGJIT_TELEMETRY"] = "1"
    os.environ["DSGJIT_TELEMETRY_SAMPLE_RATE"] = "1.0"

    reset_telemetry()

    from dsg_jit.telemetry.config import get_telemetry_config
    from dsg_jit.telemetry.decorators import telemetry_span
    from dsg_jit.telemetry.client import _get_client
    from dsg_jit.telemetry.identity import get_install_id, get_session_id

    config = get_telemetry_config()

    print("\n" + "=" * 60)
    print("DSG-JIT Telemetry Integration Test")
    print("=" * 60)
    print(f"Telemetry enabled: {config.enabled}")
    print(f"Telemetry level: {config.level}")
    print(f"Endpoint: {config.endpoint or '(empty - no export)'}")
    print(f"Sample rate: {config.sample_rate}")
    print(f"Debug mode: {config.debug}")
    print(f"Install ID: {get_install_id()}")
    print(f"Session ID: {get_session_id()}")
    print("=" * 60)

    if not config.endpoint:
        print("\nWARNING: No endpoint configured. Spans will not be exported.")
        print("Set DSGJIT_TELEMETRY_ENDPOINT to test live export.\n")

    # Define test functions with telemetry
    @telemetry_span(component="integration_test", op="success_operation")
    def successful_operation(iterations=10):
        """A successful operation."""
        total = 0
        for i in range(iterations):
            total += i
        return total

    @telemetry_span(component="integration_test", op="error_operation")
    def error_operation():
        """An operation that raises an error."""
        raise ValueError("Intentional test error - should NOT appear in telemetry")

    @telemetry_span(
        component="integration_test",
        op="complex_operation",
        safe_args={"method", "iters"},
        shape_fn=lambda method="gn", iters=40: {
            "graph.nodes_bucket": "100-999",
            "iterations_bucket": "10-99" if iters < 100 else "100-999",
        },
    )
    def complex_operation(method="gn", iters=40, secret_data=None):
        """A complex operation with safe args and shape function."""
        return {"method": method, "iterations": iters}

    print("\n--- Executing test operations ---\n")

    # Test 1: Successful operation
    print("1. Running successful_operation(iterations=50)...")
    result = successful_operation(iterations=50)
    print(f"   Result: {result}")

    # Test 2: Error operation
    print("2. Running error_operation() (expect ValueError)...")
    try:
        error_operation()
    except ValueError as e:
        print(f"   Caught expected error: {type(e).__name__}")

    # Test 3: Complex operation with safe args
    print("3. Running complex_operation(method='lm', iters=75)...")
    result = complex_operation(method="lm", iters=75, secret_data="THIS_SHOULD_NOT_APPEAR")
    print(f"   Result: {result}")

    # Test 4: Multiple rapid calls
    print("4. Running 5 rapid success operations...")
    for i in range(5):
        successful_operation(iterations=i * 10 + 1)
    print("   Done.")

    # Force flush
    print("\n--- Flushing telemetry ---\n")
    client = _get_client()

    # Check what's in the queue before flush
    if client._processor:
        queue_size = client._processor._queue.size
        print(f"Spans in queue before flush: {queue_size}")

    client.shutdown()
    print("Telemetry shutdown complete.")

    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)

    if config.endpoint:
        print(f"\nCheck your server at: {config.endpoint}")
        print("Expected spans:")
        print("  - dsgjit.session.start (1x)")
        print("  - dsgjit.integration_test.success_operation (6x)")
        print("  - dsgjit.integration_test.error_operation (1x, status=error)")
        print("  - dsgjit.integration_test.complex_operation (1x)")
        print("\nExpected headers on request:")
        print("  - X-Ix-Install-Id: UUID")
        print("  - X-Ix-Session-Id: UUID")
        print("  - X-Ix-Pkg-Version: semver")
        print("  - X-Ix-Telemetry-Level: standard|minimal|debug")
    else:
        print("\nNo endpoint configured - spans were recorded but not exported.")

    return True


def test_export_with_debug():
    """Run test with debug mode to see span contents."""
    os.environ["DSGJIT_TELEMETRY_DEBUG"] = "1"
    try:
        result = test_live_export()
        return result
    finally:
        os.environ.pop("DSGJIT_TELEMETRY_DEBUG", None)


# Pytest entry point
def test_integration_export_runs_without_error():
    """Pytest-compatible test that verifies the integration test runs."""
    # Set empty endpoint to avoid actual network calls in CI
    os.environ["DSGJIT_TELEMETRY_ENDPOINT"] = ""
    reset_telemetry()

    # Should complete without raising
    assert test_live_export() is True


if __name__ == "__main__":
    # When run directly, use debug mode
    print("\nRunning telemetry integration test with debug output...\n")

    # Enable debug unless explicitly disabled
    if "DSGJIT_TELEMETRY_DEBUG" not in os.environ:
        os.environ["DSGJIT_TELEMETRY_DEBUG"] = "1"

    test_live_export()
