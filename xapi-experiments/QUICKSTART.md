# Quick Start: Generate xAPI Test Data

**TL;DR**: Docker + one script. No datasim repo clone or Java/Clojure needed.

```bash
# From repo root (only need Docker + Python)
./xapi-experiments/run_experiments.sh
```

**Output**: `xapi-experiments/outputs/*.ndjson` (tiny → huge, ~1k → ~100k statements)

**Minimal requirements**: Docker, Python 3.10+

No Docker? See `SETUP.md` for the Java + Clojure + datasim clone path.
