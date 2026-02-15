"""User feedback questionnaire for DSG-JIT."""

from __future__ import annotations

import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path


def _get_version() -> str:
    try:
        return version("dsg_jit")
    except Exception:
        return "0.7.1"


def _get_feedback_dir() -> Path:
    """Return the directory where feedback is stored."""
    return Path.home() / ".dsg_jit"


def run_questionnaire() -> dict:
    """
    Run an interactive feedback questionnaire in the terminal.
    Returns a dict with user responses.
    """
    print("\n" + "=" * 60)
    print("  DSG-JIT User Feedback")
    print("  Thank you for helping us improve!")
    print("=" * 60 + "\n")

    feedback: dict[str, str | int | float] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": _get_version(),
    }

    # Rating (1-5)
    while True:
        try:
            rating = input(
                "How would you rate your experience with DSG-JIT? (1-5, 5=excellent): "
            ).strip()
            r = int(rating)
            if 1 <= r <= 5:
                feedback["rating"] = r
                break
        except ValueError:
            pass
        print("  Please enter a number between 1 and 5.")

    # Use case
    print("\nWhat are you using DSG-JIT for?")
    print("  1) SLAM / robotics research")
    print("  2) Scene graph / 3D reasoning")
    print("  3) Neural fields integration")
    print("  4) Learning / education")
    print("  5) Other")
    use_case = input("Choice (1-5): ").strip() or "5"
    feedback["use_case"] = use_case

    # Optional: what worked well
    worked = input("\nWhat worked well? (optional, press Enter to skip): ").strip()
    if worked:
        feedback["what_worked"] = worked

    # Optional: what could improve
    improve = input("What could we improve? (optional, press Enter to skip): ").strip()
    if improve:
        feedback["improvements"] = improve

    # Optional: additional comments
    comments = input("Any other feedback? (optional, press Enter to skip): ").strip()
    if comments:
        feedback["comments"] = comments

    print("\nThank you for your feedback!\n")
    return feedback


def save_feedback(feedback: dict) -> Path:
    """Save feedback to a local JSON file. Returns the path written."""
    feedback_dir = _get_feedback_dir()
    feedback_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = feedback_dir / f"feedback_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(feedback, f, indent=2)

    return path


def show_questionnaire_popup() -> bool:
    """
    Run the feedback questionnaire and save results.
    Returns True if the user completed the questionnaire.
    """
    try:
        feedback = run_questionnaire()
        path = save_feedback(feedback)
        print(f"Feedback saved to: {path}")
        return True
    except (KeyboardInterrupt, EOFError):
        print("\nFeedback cancelled.")
        return False


# Rate-limit: minimum days between on-import prompts
_FEEDBACK_PROMPT_INTERVAL_DAYS = 7


def _should_prompt_on_import() -> bool:
    """
    Decide whether to show the feedback questionnaire on import.
    Returns True only when:
      - stdout is an interactive TTY
      - DSG_JIT_NO_FEEDBACK env var is not set
      - Not running under pytest/CI
      - We haven't prompted recently (or ever)
    """
    import os
    import sys

    if os.environ.get("DSG_JIT_NO_FEEDBACK", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    feedback_dir = _get_feedback_dir()
    last_prompt_file = feedback_dir / "last_import_prompt"
    if last_prompt_file.exists():
        try:
            mtime = last_prompt_file.stat().st_mtime
            age_days = (datetime.now().timestamp() - mtime) / 86400
            if age_days < _FEEDBACK_PROMPT_INTERVAL_DAYS:
                return False
        except OSError:
            pass

    return True


def _mark_prompt_shown() -> None:
    """Record that we showed the prompt (for rate limiting)."""
    feedback_dir = _get_feedback_dir()
    feedback_dir.mkdir(parents=True, exist_ok=True)
    (feedback_dir / "last_import_prompt").touch()


def maybe_prompt_feedback_on_import() -> None:
    """
    If conditions are met, show the feedback questionnaire on import.
    Called automatically when ``import dsg_jit`` runs. Skips if:
    - DSG_JIT_NO_FEEDBACK is set
    - Not an interactive TTY (e.g. CI, pipes)
    - Already prompted within the last N days
    """
    if not _should_prompt_on_import():
        return
    try:
        show_questionnaire_popup()
        _mark_prompt_shown()
    except Exception:
        # Never let feedback break the import
        pass
