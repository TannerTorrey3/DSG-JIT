# Team Setup Guide: xAPI DATASIM Experiments

Quick onboarding for team members to generate xAPI test datasets for dsg-jit. **You do not need access to the datasim repo**—we use it read-only (Docker or one-time download).

## Minimal setup (recommended): Docker only

From the repo root:

```bash
# Only requirement: Docker. No Java, Clojure, or cloning datasim.
./xapi-experiments/run_experiments.sh
```

- Script will create specs if missing (downloads base input from GitHub once if needed).
- Uses the official image `yetanalytics/datasim:latest`; no local datasim clone or build.

## Prerequisites check (minimal path)

```bash
docker --version   # Any recent Docker
python3 --version  # 3.10+
```

## Alternative: No Docker (Java + Clojure + datasim clone)

If you cannot use Docker:

```bash
# Check Java and Clojure
java -version
clojure -Sdescribe
python3 --version
```

### One-time setup without Docker

1. **Clone DATASIM** (read-only; no need to push changes):
   ```bash
   git clone https://github.com/yetanalytics/datasim.git
   ```

2. **Install Java + Clojure** (e.g. macOS with Homebrew):
   ```bash
   brew install openjdk clojure
   ```

3. **Run** (script will run `make bundle` in datasim once):
   ```bash
   ./xapi-experiments/run_experiments.sh
   ```

## Initial run (first time)

From the repo root:

```bash
./xapi-experiments/run_experiments.sh
```

This will:
- Generate 5 experiment specs if missing (download base from GitHub if needed)
- Use Docker if available (no clone/Java/Clojure), or build and run DATASIM locally
- Write NDJSON to `xapi-experiments/outputs/`

**With Docker**: First run pulls the image once, then ~1–5 minutes. **Without Docker**: First run builds DATASIM (~5–10 min), then faster.

## Daily usage

After initial setup:

```bash
./xapi-experiments/run_experiments.sh
```

Regenerates all five datasets (~1–5 minutes with Docker).

## Output Files

After running, you'll have:

```
xapi-experiments/outputs/
├── tiny.ndjson    (~1k statements,   ~500KB)
├── small.ndjson   (~5k statements,   ~2.5MB)
├── medium.ndjson  (~10k statements,  ~5MB)
├── large.ndjson   (~50k statements,  ~25MB)
└── huge.ndjson    (~100k statements, ~50MB)
```

## Troubleshooting

**Use Docker to avoid Java/Clojure:** Install Docker and run the script again; it will prefer the Docker path.

**"DATASIM repo not found"**
- You're on the non-Docker path. Either install Docker (recommended) or clone datasim: `git clone https://github.com/yetanalytics/datasim.git`

**"Java Runtime not found"**
- You're running without Docker. Install Java or use Docker.

**"Unable to access jarfile"**
- Local build didn’t complete. Use Docker, or run `cd datasim && make bundle` (requires Clojure CLI).

**"base input not found and download failed"**
- Network issue or GitHub unreachable. Ensure you can open https://raw.githubusercontent.com/yetanalytics/datasim/master/dev-resources/input/simple.json, or clone datasim and run from repo root so the script can use the clone’s file.

## Using the Generated Data

### Load into your system:
```bash
# Example: Send to an LRS endpoint
cat xapi-experiments/outputs/tiny.ndjson | while read line; do
  curl -X POST https://your-lrs.com/xapi/statements \
    -H "Content-Type: application/json" \
    -H "Authorization: Basic ..." \
    -d "$line"
done
```

### Process with Python:
```python
import json

with open('xapi-experiments/outputs/tiny.ndjson') as f:
    for line in f:
        statement = json.loads(line)
        # Process statement...
```

### Count statements:
```bash
wc -l xapi-experiments/outputs/*.ndjson
```

## Questions?

See `xapi-experiments/README.md` for detailed documentation.
