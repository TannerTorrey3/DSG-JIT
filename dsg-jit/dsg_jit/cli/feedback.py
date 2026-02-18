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

from dsg_jit.telemetry import telemetry_span

# Minimum days between questionnaire prompts
_FEEDBACK_PROMPT_INTERVAL_DAYS = 7

# Number of imports before first questionnaire prompt
_FEEDBACK_PROMPT_MIN_RUNS = 3


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


def _load_state() -> dict:
    """Load persistent feedback state from disk."""
    state_file = _get_feedback_dir() / "feedback_state.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"run_count": 0, "feedback_given": False, "last_prompt_timestamp": None}


def _save_state(state: dict) -> None:
    """Persist feedback state to disk."""
    state_file = _get_feedback_dir() / "feedback_state.json"
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except (IOError, OSError):
        pass


def _is_prompt_suppressed() -> bool:
    """Check if prompting is suppressed by env vars or non-interactive context."""
    if os.environ.get("DSG_JIT_NO_FEEDBACK", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if "pytest" in sys.modules:
        return True
    if not _is_interactive():
        return True
    return False


def _should_prompt_questionnaire(state: dict) -> bool:
    """
    Decide whether to show the full questionnaire.

    - Never prompt if feedback has already been given
    - First prompt on the 3rd run (user has some experience)
    - After that, prompt every 7 days
    """
    if state.get("feedback_given", False):
        return False

    run_count = state.get("run_count", 0)
    if run_count < _FEEDBACK_PROMPT_MIN_RUNS:
        return False

    # First time hitting the threshold — prompt
    last_prompt = state.get("last_prompt_timestamp")
    if last_prompt is None:
        return True

    # After first prompt, respect the 7-day interval
    try:
        last_dt = datetime.fromisoformat(last_prompt)
        age_days = (datetime.now() - last_dt).total_seconds() / 86400
        return age_days >= _FEEDBACK_PROMPT_INTERVAL_DAYS
    except (ValueError, TypeError):
        return True


def _mark_prompt_shown(state: dict) -> None:
    """Record that the questionnaire was shown."""
    state["last_prompt_timestamp"] = datetime.now().isoformat()
    _save_state(state)


def _mark_feedback_given(state: dict) -> None:
    """Record that the user completed the questionnaire."""
    state["feedback_given"] = True
    state["last_prompt_timestamp"] = datetime.now().isoformat()
    _save_state(state)


@telemetry_span(component="cli", op="run_questionnaire")
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

    print("\nHow did you discover DSG-JIT?")
    print("  1) Paper / publication")
    print("  2) GitHub / search")
    print("  3) Colleague / recommendation")
    print("  4) Conference / workshop")
    print("  5) Blog / social media")
    print("  6) Other")
    fb["discovery"] = input("Choice (1-6): ").strip() or "6"

    email = input("\nEmail (optional, for follow-up; Enter to skip): ").strip()
    if email:
        fb["email"] = email

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


@telemetry_span(component="cli", op="show_questionnaire_popup", safe_args={"show_save_location"})
def show_questionnaire_popup(show_save_location: bool = True, _state: dict | None = None) -> bool:
    try:
        feedback = run_questionnaire()
        path = save_feedback(feedback)
        if show_save_location:
            print(f"Feedback saved to: {path}")
        if _state is not None:
            _mark_feedback_given(_state)
        return True
    except (KeyboardInterrupt, EOFError):
        print("\nFeedback cancelled.")
        return False


def _print_banner() -> None:
    """Print a short, non-blocking info line on import."""
    version = _get_version()
    print(f"[DSG-JIT v{version}] Feedback: dsg-jit feedback", file=sys.stderr)


def prompt_feedback_on_import() -> None:
    """
    Called automatically by dsg_jit/__init__.py.
    Shows a one-liner banner on every interactive run, and prompts
    the full questionnaire when conditions are met. Never breaks import.
    """
    if _is_prompt_suppressed():
        return

    # Always show the banner so users know feedback/telemetry exist
    try:
        _print_banner()
    except Exception:
        pass

    try:
        state = _load_state()
        state["run_count"] = state.get("run_count", 0) + 1
        _save_state(state)

        if _should_prompt_questionnaire(state):
            _mark_prompt_shown(state)
            show_questionnaire_popup(show_save_location=False, _state=state)
    except Exception:
        # Never let feedback break import
        pass
