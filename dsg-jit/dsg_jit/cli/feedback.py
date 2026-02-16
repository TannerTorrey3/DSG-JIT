"""User feedback questionnaire for DSG-JIT."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

# Minimum days between on-import prompts
_FEEDBACK_PROMPT_INTERVAL_DAYS = 7


def _get_version() -> str:
    try:
        return _pkg_version("dsg-jit")  # PyPI name might be dsg-jit
    except Exception:
        try:
            return _pkg_version("dsg_jit")
        except Exception:
            return "unknown"


def _get_feedback_dir() -> Path:
    """
    Prefer ~/.dsg_jit; if HOME isn't writable (some sandboxes),
    fall back to a temp directory.
    """
    home = Path.home()
    p = home / ".dsg_jit"
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write_test"
        test.write_text("ok")
        test.unlink(missing_ok=True)  # py>=3.8 ok
        return p
    except Exception:
        tmp = Path(tempfile.gettempdir()) / "dsg_jit"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


def _is_notebook() -> bool:
    """
    Detect Jupyter/IPython notebooks.
    """
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        # Kernel-based environments (Notebook/JupyterLab) usually have IPKernelApp
        return "IPKernelApp" in getattr(ip, "config", {}) or "ipykernel" in sys.modules
    except Exception:
        return "ipykernel" in sys.modules


def _is_interactive() -> bool:
    """
    True for:
    - normal terminals (TTY)
    - notebooks (Jupyter/IPython kernel)
    """
    if _is_notebook():
        return True
    # terminal-like interactive sessions
    try:
        return bool(hasattr(sys.stdin, "isatty") and sys.stdin.isatty())
    except Exception:
        return False


def _should_prompt_on_import() -> bool:
    """
    Show questionnaire on import only if:
    - interactive (terminal or notebook)
    - not opted out (DSG_JIT_NO_FEEDBACK)
    - not CI
    - not pytest
    - rate limit passed
    """
    if os.environ.get("DSG_JIT_NO_FEEDBACK", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    if not _is_interactive():
        return False

    feedback_dir = _get_feedback_dir()
    last_prompt = feedback_dir / "last_import_prompt"
    if last_prompt.exists():
        try:
            age_days = (datetime.now().timestamp() - last_prompt.stat().st_mtime) / 86400
            if age_days < _FEEDBACK_PROMPT_INTERVAL_DAYS:
                return False
        except Exception:
            pass

    return True


def _mark_prompt_shown() -> None:
    feedback_dir = _get_feedback_dir()
    (feedback_dir / "last_import_prompt").touch()


def run_questionnaire() -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("  DSG-JIT User Feedback")
    print("  Thank you for helping us improve!")
    print("=" * 60 + "\n")

    fb: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": _get_version(),
    }

    # rating
    while True:
        try:
            rating = input("How would you rate your experience with DSG-JIT? (1-5): ").strip()
            r = int(rating)
            if 1 <= r <= 5:
                fb["rating"] = r
                break
        except ValueError:
            pass
        print("  Please enter a number between 1 and 5.")

    print("\nWhat are you using DSG-JIT for?")
    print("  1) SLAM / robotics research")
    print("  2) Scene graph / 3D reasoning")
    print("  3) Neural fields integration")
    print("  4) Learning / education")
    print("  5) Other")
    fb["use_case"] = input("Choice (1-5): ").strip() or "5"

    worked = input("\nWhat worked well? (optional, Enter to skip): ").strip()
    if worked:
        fb["what_worked"] = worked

    improve = input("What could we improve? (optional, Enter to skip): ").strip()
    if improve:
        fb["improvements"] = improve

    comments = input("Any other feedback? (optional, Enter to skip): ").strip()
    if comments:
        fb["comments"] = comments

    print("\nThank you for your feedback!\n")
    return fb


def save_feedback(feedback: dict[str, Any]) -> Path:
    feedback_dir = _get_feedback_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = feedback_dir / f"feedback_{ts}.json"
    path.write_text(json.dumps(feedback, indent=2))
    return path


def show_questionnaire_popup() -> bool:
    try:
        feedback = run_questionnaire()
        path = save_feedback(feedback)
        print(f"Feedback saved to: {path}")
        return True
    except (KeyboardInterrupt, EOFError):
        print("\nFeedback cancelled.")
        return False


def maybe_prompt_feedback_on_import() -> None:
    """
    Called automatically by dsg_jit/__init__.py.
    Never breaks import.
    """
    if not _should_prompt_on_import():
        return
    try:
        show_questionnaire_popup()
        _mark_prompt_shown()
    except Exception:
        # Never let feedback break import
        pass
