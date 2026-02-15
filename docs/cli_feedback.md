# CLI & Feedback Questionnaire Workflow

## Overview

DSG-JIT includes a CLI entry point and an optional feedback questionnaire to collect user input. The questionnaire can appear in two ways:

1. **On import** – When you `import dsg_jit` in an interactive Python session
2. **Via CLI** – When you run `dsg-jit feedback` or `dsg-jit run` (after experiments)

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User runs Python                              │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
     import dsg_jit                    dsg-jit feedback
     (interactive)                     (CLI command)
              │                               │
              ▼                               ▼
     maybe_prompt_feedback_on_import   show_questionnaire_popup
              │                               │
              ▼                               ▼
     ┌─────────────────────────────────────────────┐
     │  Checks: TTY? No DSG_JIT_NO_FEEDBACK?       │
     │         Not pytest/CI? Rate limit OK?       │
     └─────────────────┬───────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │ Yes                   │ No
           ▼                       ▼
     Show questionnaire      Skip (silent)
           │
           ▼
     Save to ~/.dsg_jit/feedback_*.json
```

---

## When the Questionnaire Appears (On Import)

The questionnaire **will** appear when:

- You run `import dsg_jit` in an **interactive terminal** (TTY)
- The `DSG_JIT_NO_FEEDBACK` environment variable is **not** set
- It has been at least 7 days since the last prompt (or never prompted before)

The questionnaire **will not** appear when:

- Running under **pytest** (tests)
- Running in **CI** (`CI` env var set)
- `DSG_JIT_NO_FEEDBACK=1` is set
- Output is **piped** or **non-interactive** (e.g. `python script.py | less`)
- You were prompted within the last 7 days

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `dsg-jit` | Show help and tip about feedback |
| `dsg-jit feedback` | Run the feedback questionnaire explicitly |
| `dsg-jit run [experiment]` | Run an experiment (e.g. `exp01_mini_world`), then optionally show feedback |
| `dsg-jit run --no-feedback` | Run experiment without showing feedback afterward |

---

## Disabling the On-Import Prompt

Set the environment variable:

```bash
export DSG_JIT_NO_FEEDBACK=1
```

Or in Python before importing:

```python
import os
os.environ["DSG_JIT_NO_FEEDBACK"] = "1"
import dsg_jit  # No questionnaire
```

---

## Feedback Storage

- Feedback is stored in `~/.dsg_jit/`
- Each response is saved as `feedback_YYYYMMDD_HHMMSS.json`
- Rate-limiting uses `last_import_prompt` in the same directory

---

## Branch Strategy

- **`dev`** – CLI and feedback questionnaire development
- **`main`** – Stable release; merge from `dev` only after validation

All CLI and feedback work stays on `dev` until ready to merge into `main`.
