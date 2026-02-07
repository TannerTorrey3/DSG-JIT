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
from typing import Any, Callable, Dict, Optional, Set, TypeVar

from opentelemetry.trace import Status, StatusCode

from dsg_jit.telemetry.config import get_telemetry_config
from dsg_jit.telemetry.identity import get_install_id, get_session_id
from dsg_jit.telemetry.otel import get_tracer
from dsg_jit.telemetry.sanitize import (
    bucket_count,
    get_error_code,
    sanitize_safe_args,
)

F = TypeVar("F", bound=Callable[..., Any])

_session_started: bool = False


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
        plat = devices[0].platform.lower()
        if "gpu" in plat or "cuda" in plat:
            return "gpu"
        elif "tpu" in plat:
            return "tpu"
        elif "cpu" in plat:
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


def _emit_session_start(entry_component: str) -> None:
    """Emit session.start span on first instrumented call."""
    global _session_started
    if _session_started:
        return
    _session_started = True

    config = get_telemetry_config()
    tracer = get_tracer()
    with tracer.start_as_current_span("dsgjit.session.start") as span:
        span.set_attribute("ix.install_id", get_install_id())
        span.set_attribute("ix.session_id", get_session_id())
        span.set_attribute("dsgjit.version", _get_version())
        span.set_attribute("runtime.python", platform.python_version())
        span.set_attribute("runtime.os", platform.system().lower())
        span.set_attribute("runtime.arch", platform.machine())
        span.set_attribute("dsgjit.component", entry_component)
        span.set_attribute("dsgjit.op", "session.start")
        span.set_attribute("dsgjit.status", "ok")
        span.set_attribute("dsgjit.backend", _get_backend())
        span.set_attribute("dsgjit.backend_available", _get_backend_available())
        span.set_attribute("dsgjit.telemetry_level", config.level)
        span.set_attribute("dsgjit.entry_component", entry_component)
        span.set_attribute("dsgjit.telemetry_enabled", config.enabled)

        if config.tag:
            span.set_attribute("dsgjit.tag", config.tag)


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

            # Emit session.start on first call
            _emit_session_start(component)

            tracer = get_tracer()
            span_name = f"dsgjit.{component}.{op}"

            with tracer.start_as_current_span(span_name) as span:
                # Set base attributes
                span.set_attribute("ix.install_id", get_install_id())
                span.set_attribute("ix.session_id", get_session_id())
                span.set_attribute("dsgjit.version", _get_version())
                span.set_attribute("runtime.python", platform.python_version())
                span.set_attribute("runtime.os", platform.system().lower())
                span.set_attribute("runtime.arch", platform.machine())
                span.set_attribute("dsgjit.component", component)
                span.set_attribute("dsgjit.op", op)
                span.set_attribute("dsgjit.backend", _get_backend())
                span.set_attribute("dsgjit.telemetry_level", config.level)

                # Add custom tag if set
                if config.tag:
                    span.set_attribute("dsgjit.tag", config.tag)

                # Extract safe args if specified
                if safe_args:
                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                        bound.apply_defaults()
                        safe_values = sanitize_safe_args(dict(bound.arguments), safe_args)
                        for k, v in safe_values.items():
                            span.set_attribute(f"dsgjit.args.{k}", str(v))
                    except (TypeError, ValueError):
                        pass

                # Compute shape attributes if shape_fn provided
                if shape_fn is not None:
                    try:
                        shape_attrs = shape_fn(*args, **kwargs)
                        for k, v in shape_attrs.items():
                            span.set_attribute(f"dsgjit.{k}", str(v))
                    except Exception:
                        pass

                # Execute the function
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("dsgjit.status", "ok")
                    span.set_status(Status(StatusCode.OK))
                    return result
                except BaseException as e:
                    span.set_attribute("dsgjit.status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.code", get_error_code(e))
                    span.set_attribute("error.component", component)
                    span.set_attribute("error.op", op)
                    span.set_status(Status(StatusCode.ERROR, type(e).__name__))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def reset_telemetry_state() -> None:
    """Reset telemetry state (mainly for testing)."""
    global _session_started
    from dsg_jit.telemetry.otel import reset_telemetry
    reset_telemetry()
    _session_started = False
