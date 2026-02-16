"""CLI entry point for DSG-JIT."""
from __future__ import annotations

import argparse
import sys


def _feedback_cmd(_args: argparse.Namespace) -> int:
    from dsg_jit.cli.feedback import show_questionnaire_popup
    return 0 if show_questionnaire_popup() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dsg-jit",
        description="DSG-JIT CLI",
    )
    sub = parser.add_subparsers(dest="command")

    fb = sub.add_parser("feedback", help="Run the feedback questionnaire")
    fb.set_defaults(func=_feedback_cmd)

    args = parser.parse_args()

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    print("\nTip: Run 'dsg-jit feedback' to share feedback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
