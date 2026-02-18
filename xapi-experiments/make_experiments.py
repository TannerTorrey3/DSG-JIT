#!/usr/bin/env python3
"""
Generate 5 DATASIM input specs (tiny, small, medium, large, huge) from the
shipped simple.json base. Varies cohort size, time window, and seed to target
~1k, 5k, 10k, 50k, 100k statements. Keeps xAPI profile/model structure intact.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Experiment definitions: (name, target statements, num actors, time window days, seed)
EXPERIMENTS = [
    ("tiny", 1_000, 5, 1, 42),
    ("small", 5_000, 15, 2, 42),
    ("medium", 10_000, 30, 2, 42),
    ("large", 50_000, 120, 3, 42),
    ("huge", 100_000, 250, 4, 42),
]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Prefer bundled base so we don't require a datasim clone
BASE_DIR = SCRIPT_DIR / "base"
DEFAULT_BASE_INPUT = BASE_DIR / "simple.json"
# Fallback if no bundled base: datasim clone (optional)
DATASIM_BASE_INPUT = REPO_ROOT / "datasim" / "dev-resources" / "input" / "simple.json"
SPECS_DIR = SCRIPT_DIR / "specs"
DATASIM_SIMPLE_JSON_URL = "https://raw.githubusercontent.com/yetanalytics/datasim/master/dev-resources/input/simple.json"


def make_actor(index: int) -> dict:
    """Create a single xAPI Agent for personae."""
    return {
        "name": f"Sim Actor {index}",
        "mbox": f"mailto:sim-actor-{index}@example.org",
        "role": "Trainee",
    }


def make_personae_array(n_actors: int) -> list[dict]:
    """Build personae-array with one Group containing n_actors members."""
    members = [make_actor(i) for i in range(n_actors)]
    return [
        {
            "objectType": "Group",
            "name": "trainees",
            "member": members,
        }
    ]


def make_parameters(
    target_max: int,
    window_days: int,
    seed: int,
) -> dict:
    """Build parameters with start/end window and max statements."""
    # Fixed base start (UTC)
    start = "2019-11-18T00:00:00.000000Z"
    # End = start + window_days (simplified: use same time, next day(s))
    from datetime import datetime, timedelta
    base = datetime(2019, 11, 18, 0, 0, 0)
    end_dt = base + timedelta(days=window_days)
    end = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    return {
        "start": start,
        "end": end,
        "from": start,
        "timezone": "America/New_York",
        "seed": seed,
        "max": target_max,
        "maxRestarts": 10,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 5 DATASIM experiment specs from a base input JSON.",
    )
    parser.add_argument(
        "--base-input",
        type=Path,
        default=DEFAULT_BASE_INPUT,
        help=f"Path to combined DATASIM input (default: {DEFAULT_BASE_INPUT})",
    )
    parser.add_argument(
        "--specs-dir",
        type=Path,
        default=SPECS_DIR,
        help=f"Directory to write spec JSON files (default: {SPECS_DIR})",
    )
    args = parser.parse_args()

    base_path = args.base_input.resolve()
    if not base_path.is_file():
        # Try fallback: datasim clone
        if DATASIM_BASE_INPUT.resolve().is_file():
            base_path = DATASIM_BASE_INPUT.resolve()
        else:
            # Download from datasim repo (no clone needed)
            BASE_DIR.mkdir(parents=True, exist_ok=True)
            downloaded = BASE_DIR / "simple.json"
            try:
                print(f"Downloading base input from DATASIM repo...", file=sys.stderr)
                urllib.request.urlretrieve(DATASIM_SIMPLE_JSON_URL, downloaded)
                base_path = downloaded.resolve()
            except Exception as e:
                print(f"Error: base input not found and download failed: {e}", file=sys.stderr)
                print(f"  Tried: {args.base_input}", file=sys.stderr)
                print("  Option 1: Clone datasim and run from repo root: git clone https://github.com/yetanalytics/datasim.git", file=sys.stderr)
                print("  Option 2: Ensure network access and re-run (downloads from GitHub).", file=sys.stderr)
                return 1

    specs_dir = args.specs_dir.resolve()
    specs_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)

    if "profiles" not in base or "parameters" not in base:
        print("Error: base input must contain 'profiles' and 'parameters'.", file=sys.stderr)
        return 1

    for name, target_max, n_actors, window_days, seed in EXPERIMENTS:
        spec = {
            "profiles": base["profiles"],
            "personae-array": make_personae_array(n_actors),
            "models": base.get("models", []),
            "parameters": make_parameters(target_max, window_days, seed),
        }
        out_path = specs_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        print(f"Wrote {out_path} (target ~{target_max} statements, {n_actors} actors, {window_days} day(s), seed={seed})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
