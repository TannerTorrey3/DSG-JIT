"""CLI entry point for DSG-JIT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent so we can resolve dsg-jit root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_experiment(name: str) -> int:
    """Run an experiment script by name (e.g. exp01_mini_world)."""
    experiments_dir = _REPO_ROOT / "dsg-jit" / "experiments"
    candidates = list(experiments_dir.glob(f"{name}*.py"))
    if not candidates:
        print(f"Error: No experiment matching '{name}' found in {experiments_dir}")
        return 1
    if len(candidates) > 1:
        print(f"Error: Multiple experiments match '{name}': {[c.name for c in candidates]}")
        return 1
    script = candidates[0]
    import runpy
    try:
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)


def _feedback_cmd(_args: argparse.Namespace) -> int:
    """Run the feedback questionnaire."""
    from dsg_jit.cli.feedback import show_questionnaire_popup
    return 0 if show_questionnaire_popup() else 1


def _run_cmd(args: argparse.Namespace) -> int:
    """Run an experiment and optionally show feedback questionnaire afterward."""
    code = _run_experiment(args.experiment)
    if code == 0 and not args.no_feedback:
        from dsg_jit.cli.feedback import show_questionnaire_popup
        show_questionnaire_popup()
    return code


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dsg-jit",
        description="DSG-JIT: Differentiable Scene Graphs with JAX-based factor graphs and SLAM tooling.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # feedback: show questionnaire
    feedback_parser = subparsers.add_parser("feedback", help="Submit feedback to help improve DSG-JIT")
    feedback_parser.set_defaults(func=_feedback_cmd)

    # run: run an experiment
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("experiment", nargs="?", default="exp01_mini_world",
                            help="Experiment name (e.g. exp01_mini_world, exp06_dynamic_trajectory)")
    run_parser.add_argument("--no-feedback", action="store_true",
                            help="Skip feedback questionnaire after running")
    run_parser.set_defaults(func=_run_cmd)

    args = parser.parse_args()

    if args.command and hasattr(args, "func"):
        return args.func(args)

    # No subcommand: show help
    parser.print_help()
    print("\nTip: Run 'dsg-jit feedback' to share your feedback!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
