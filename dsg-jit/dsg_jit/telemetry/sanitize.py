# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""Sanitization and bucketing utilities for telemetry data."""

from __future__ import annotations

from typing import Any, Dict, Set


def bucket_count(n: int) -> str:
    """Convert a count to a bucket string.

    Buckets: 0, 1-9, 10-99, 100-999, 1k-9k, 10k-99k, 100k-999k, 1M+

    :param n: The count to bucket.
    :return: A string representing the bucket.
    """
    if n == 0:
        return "0"
    if n < 10:
        return "1-9"
    if n < 100:
        return "10-99"
    if n < 1000:
        return "100-999"
    if n < 10_000:
        return "1k-9k"
    if n < 100_000:
        return "10k-99k"
    if n < 1_000_000:
        return "100k-999k"
    return "1M+"


# Allowlist of safe attribute keys that can be recorded
SAFE_SCALAR_ARGS: Set[str] = frozenset({
    # Method selection
    "method",
    "var_type",
    "factor_type",
    "f_type",
    # Numeric flags (will be bucketed)
    "iters",
    "coord_index",
    # Boolean flags
    "use_type_weights",
    "learn_odom",
    "learn_voxel_points",
    "active",
})


def is_safe_arg(arg_name: str) -> bool:
    """Check if an argument name is safe to record.

    :param arg_name: The argument name to check.
    :return: True if the argument is safe to record.
    """
    return arg_name in SAFE_SCALAR_ARGS


def sanitize_safe_args(args: Dict[str, Any], safe_args: Set[str]) -> Dict[str, Any]:
    """Extract only safe arguments from a dict, sanitizing values.

    :param args: The full arguments dict.
    :param safe_args: Set of argument names that are safe to include.
    :return: Dict containing only safe arguments with sanitized values.
    """
    result: Dict[str, Any] = {}
    for key in safe_args:
        if key not in args:
            continue
        val = args[key]
        # Only record scalars (str, int, bool, float)
        if isinstance(val, bool):
            result[key] = val
        elif isinstance(val, str):
            # Only record short strings that look like enums
            if len(val) <= 32 and val.isidentifier():
                result[key] = val
        elif isinstance(val, int):
            # Bucket integer values
            result[key] = bucket_count(val)
        elif isinstance(val, float):
            # Don't record raw floats (could be identifying)
            pass
    return result


# Error code categories per spec
ERROR_CODES: Dict[str, str] = {
    "ValueError": "invalid_argument",
    "TypeError": "invalid_argument",
    "KeyError": "invalid_argument",
    "IndexError": "invalid_argument",
    "AttributeError": "invalid_argument",
    "ShapeError": "shape_mismatch",
    "RuntimeError": "backend_error",
    "NotImplementedError": "not_initialized",
    "ConvergenceError": "convergence_failure",
    "FloatingPointError": "numerical_issue",
    "OverflowError": "numerical_issue",
    "ZeroDivisionError": "numerical_issue",
    "IOError": "io_error",
    "OSError": "io_error",
    "FileNotFoundError": "io_error",
}


def get_error_code(exception: BaseException) -> str:
    """Map an exception to an error code category.

    :param exception: The exception instance.
    :return: The error code category string.
    """
    exc_type = type(exception).__name__
    return ERROR_CODES.get(exc_type, "unknown_error")
