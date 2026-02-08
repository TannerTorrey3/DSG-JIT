"""Sanitization and bucketing utilities for telemetry data.

This module provides utilities to ensure telemetry data is privacy-safe:

- **Bucketing**: Integer values are converted to ranges (e.g., 47 -> "10-99")
- **Allowlisting**: Only explicitly safe argument names are recorded
- **Error categorization**: Exceptions are mapped to category codes, not raw messages

The goal is to collect useful aggregate statistics without capturing
any potentially identifying or sensitive information.
"""

from __future__ import annotations

from typing import Any, Dict, Set


def bucket_count(n: int) -> str:
    """Convert an integer count to a privacy-safe bucket string.

    This prevents exact values from being recorded while still providing
    useful magnitude information for analytics.

    Args:
        n: The count to bucket (non-negative integer).

    Returns:
        A bucket string: "0", "1-9", "10-99", "100-999",
        "1k-9k", "10k-99k", "100k-999k", or "1M+".

    Example:
        >>> bucket_count(0)
        '0'
        >>> bucket_count(47)
        '10-99'
        >>> bucket_count(1500)
        '1k-9k'
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


SAFE_SCALAR_ARGS: Set[str] = frozenset({
    # Method selection (strings)
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
"""Allowlist of argument names safe to record in telemetry.

Only arguments with names in this set will be considered for recording.
String values must also be short identifiers (<=32 chars, alphanumeric).
Integer values are automatically bucketed.
"""


def is_safe_arg(arg_name: str) -> bool:
    """Check if an argument name is in the safe allowlist.

    Args:
        arg_name: The argument name to check.

    Returns:
        True if the argument is in SAFE_SCALAR_ARGS.
    """
    return arg_name in SAFE_SCALAR_ARGS


def sanitize_safe_args(args: Dict[str, Any], safe_args: Set[str]) -> Dict[str, Any]:
    """Extract and sanitize only safe arguments from a dict.

    Filters arguments by the safe_args allowlist and applies sanitization:
    - Booleans: recorded as-is
    - Strings: only if <=32 chars and valid identifier
    - Integers: bucketed via bucket_count()
    - Floats: not recorded (could be identifying)

    Args:
        args: The full arguments dict from a function call.
        safe_args: Set of argument names that are safe to include.

    Returns:
        Dict containing only safe arguments with sanitized values.
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
"""Mapping from exception type names to categorized error codes.

Error codes provide useful aggregate statistics without exposing
raw error messages which could contain sensitive information.

Categories:
    - ``invalid_argument``: Input validation errors
    - ``shape_mismatch``: Array/tensor shape errors
    - ``numerical_issue``: Math errors (overflow, NaN, etc.)
    - ``backend_error``: JAX/runtime errors
    - ``convergence_failure``: Optimization did not converge
    - ``io_error``: File/network errors
    - ``unknown_error``: Unmapped exception types
"""


def get_error_code(exception: BaseException) -> str:
    """Map an exception to a categorized error code.

    Args:
        exception: The exception instance.

    Returns:
        A category string like "invalid_argument" or "numerical_issue".
        Returns "unknown_error" for unmapped exception types.

    Example:
        >>> get_error_code(ValueError("bad input"))
        'invalid_argument'
    """
    exc_type = type(exception).__name__
    return ERROR_CODES.get(exc_type, "unknown_error")
