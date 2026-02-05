# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Telemetry decorator for instrumenting public APIs.

The @telemetry_span decorator wraps functions to emit telemetry spans
with timing, status, and optional shape/count information.

Usage:
    @telemetry_span(component="world", op="add_pose")
    def add_pose(self, value, name=None):
        ...

    @telemetry_span(
        component="world",
        op="optimize",
        safe_args={"method"},
        shape_fn=lambda self, **kw: {"iterations_bucket": bucket_count(kw.get("iters", 40))}
    )
    def optimize(self, method="gn", iters=40):
        ...
"""

from __future__ import annotations

import functools
import inspect
import platform
import time
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union

from dsg_jit.telemetry.config import get_telemetry_config
from dsg_jit.telemetry.identity import get_install_id, get_session_id
from dsg_jit.telemetry.client import _get_client, reset_client
from dsg_jit.telemetry.sanitize import (
    bucket_count,
    get_error_code,
    sanitize_safe_args,
)

F = TypeVar("F", bound=Callable[..., Any])

_session_started: bool = False


def _emit_session_start(entry_component: str) -> None:
    """Emit session.start span on first instrumented call."""
    global _session_started
    if _session_started:
        return
    _session_started = True

    config = get_telemetry_config()

    span = {
        "name": "dsgjit.session.start",
        "timestamp": time.time(),
        "duration_ms": 0,
        "attributes": {
            "ix.install_id": get_install_id(),
            "ix.session_id": get_session_id(),
            "dsgjit.version": _get_version(),
            "runtime.python": platform.python_version(),
            "runtime.os": platform.system().lower(),
            "runtime.arch": platform.machine(),
            "dsgjit.component": entry_component,
            "dsgjit.op": "session.start",
            "dsgjit.status": "ok",
            "dsgjit.backend": _get_backend(),
            "dsgjit.backend_available": _get_backend_available(),
            "dsgjit.telemetry_level": config.level,
            "dsgjit.entry_component": entry_component,
            "dsgjit.telemetry_enabled": config.enabled,
        },
    }
    _get_client().record_span(span)


def _get_version() -> str:
    """Get DSG-JIT version string."""
    try:
        from dsg_jit import __version__
        return __version__
    except (ImportError, AttributeError):
        return "0.0.0"


def _get_backend() -> str:
    """Detect the JAX backend (cpu/gpu/tpu/unknown)."""
    try:
        import jax
        devices = jax.devices()
        if not devices:
            return "unknown"
        # Check the platform of the first device
        platform = devices[0].platform.lower()
        if "gpu" in platform or "cuda" in platform:
            return "gpu"
        elif "tpu" in platform:
            return "tpu"
        elif "cpu" in platform:
            return "cpu"
        return "unknown"
    except Exception:
        return "unknown"


def _get_backend_available() -> str:
    """Detect all available backends (cpu|gpu|both|unknown)."""
    try:
        import jax
        devices = jax.devices()
        if not devices:
            return "unknown"
        platforms = {d.platform.lower() for d in devices}
        has_gpu = any("gpu" in p or "cuda" in p for p in platforms)
        has_cpu = any("cpu" in p for p in platforms)
        if has_gpu and has_cpu:
            return "both"
        elif has_gpu:
            return "gpu"
        elif has_cpu:
            return "cpu"
        return "unknown"
    except Exception:
        return "unknown"


ShapeFn = Callable[..., Dict[str, str]]


def telemetry_span(
    component: str,
    op: str,
    safe_args: Optional[Set[str]] = None,
    shape_fn: Optional[ShapeFn] = None,
) -> Callable[[F], F]:
    """Decorator to instrument a function with telemetry.

    :param component: The component name (e.g., "world", "scene_graph").
    :param op: The operation name (e.g., "add_pose", "optimize").
    :param safe_args: Optional set of argument names safe to record.
        Only scalar values (str, int, bool) from these args are recorded.
        Integers are automatically bucketed.
    :param shape_fn: Optional function to compute bucketed shape values.
        Called with the same arguments as the decorated function.
        Should return a dict of attribute_name -> bucketed_string.
    :return: The decorated function.
    """
    if safe_args is None:
        safe_args = set()

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = get_telemetry_config()

            # Fast path: if telemetry disabled, just call the function
            if not config.enabled:
                return func(*args, **kwargs)

            # Emit session.start on first call
            _emit_session_start(component)

            # Build base attributes
            attributes: Dict[str, Any] = {
                "ix.install_id": get_install_id(),
                "ix.session_id": get_session_id(),
                "dsgjit.version": _get_version(),
                "runtime.python": platform.python_version(),
                "runtime.os": platform.system().lower(),
                "runtime.arch": platform.machine(),
                "dsgjit.component": component,
                "dsgjit.op": op,
                "dsgjit.backend": _get_backend(),
                "dsgjit.telemetry_level": config.level,
            }

            # Extract safe args if specified
            if safe_args:
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    safe_values = sanitize_safe_args(dict(bound.arguments), safe_args)
                    for k, v in safe_values.items():
                        attributes[f"dsgjit.args.{k}"] = v
                except (TypeError, ValueError):
                    pass

            # Compute shape attributes if shape_fn provided
            if shape_fn is not None:
                try:
                    shape_attrs = shape_fn(*args, **kwargs)
                    for k, v in shape_attrs.items():
                        attributes[f"dsgjit.{k}"] = v
                except Exception:
                    # Shape computation failed - don't break user code
                    pass

            # Execute the function and time it
            start = time.perf_counter()
            error: Optional[BaseException] = None
            try:
                result = func(*args, **kwargs)
                attributes["dsgjit.status"] = "ok"
                return result
            except BaseException as e:
                error = e
                attributes["dsgjit.status"] = "error"
                # Record error info (class name only, never message)
                attributes["error.type"] = type(e).__name__
                attributes["error.code"] = get_error_code(e)
                attributes["error.component"] = component
                attributes["error.op"] = op
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000

                span = {
                    "name": f"dsgjit.{component}.{op}",
                    "timestamp": time.time(),
                    "duration_ms": duration_ms,
                    "attributes": attributes,
                }

                try:
                    _get_client().record_span(span)
                except Exception:
                    # Telemetry errors should never propagate
                    pass

        return wrapper  # type: ignore

    return decorator


def reset_telemetry_state() -> None:
    """Reset telemetry state (mainly for testing)."""
    global _session_started
    reset_client()
    _session_started = False
