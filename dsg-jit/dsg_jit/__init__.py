# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
DSG-JIT: Differentiable Scene Graphs with JAX-based factor graphs and SLAM tooling.
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("dsg_jit")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except ImportError:
    # Python < 3.8 fallback
    __version__ = "0.0.0"

# Optionally prompt for feedback on first import (interactive use only)
try:
    from dsg_jit.cli.feedback import prompt_feedback_on_import
    prompt_feedback_on_import()
except Exception:
    # Never break package import because of feedback prompt.
    pass
