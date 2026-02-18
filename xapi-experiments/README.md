# xAPI DATASIM Experiments

Generate scaled xAPI Statement datasets (NDJSON) for testing **dsg-jit** (or any xAPI consumer) using [yetanalytics/datasim](https://github.com/yetanalytics/datasim). You do **not** need edit access to the datasim repo—we only use it as a read-only source (Docker image or one-time base input download).

## Minimal setup (recommended): Docker only

From the **repo root**:

```bash
# Only requirement: Docker. No Java, Clojure, or cloning the datasim repo.
./xapi-experiments/run_experiments.sh
```

- The script generates experiment specs if missing (downloads base input from GitHub once if needed).
- If Docker is available, it uses the official image `yetanalytics/datasim:latest` (pull happens automatically). No local datasim clone or build.

## How to run

From the **repo root**:

```bash
./xapi-experiments/run_experiments.sh
```

- **With Docker**: Uses `yetanalytics/datasim:latest`. No Java/Clojure/datasim clone.
- **Without Docker**: Requires a local clone of datasim + Java (JDK 8+) + Clojure CLI; script will build DATASIM and run.

Specs are created automatically (from a downloaded or existing base input). Outputs go to `xapi-experiments/outputs/*.ndjson`.

## Requirements

- **Minimal**: Docker + Python 3.10+ (for spec generation).
- **Without Docker**: macOS or Linux, Java (JDK 8+), [Clojure CLI](https://clojure.org/guides/getting_started), Python 3.10+, and clone [datasim](https://github.com/yetanalytics/datasim) into repo root as `datasim/`.

After the run, the script prints statement count and file size for each output.

## What gets produced

| Output | Target size | Path |
|--------|-------------|------|
| tiny   | ~1k stmts   | `xapi-experiments/outputs/tiny.ndjson` |
| small  | ~5k stmts   | `xapi-experiments/outputs/small.ndjson` |
| medium | ~10k stmts  | `xapi-experiments/outputs/medium.ndjson` |
| large  | ~50k stmts  | `xapi-experiments/outputs/large.ndjson` |
| huge   | ~100k stmts | `xapi-experiments/outputs/huge.ndjson` |

- **Specs:** `xapi-experiments/specs/*.json` — DATASIM combined input (profiles, personae, models, parameters).
- **Outputs:** `xapi-experiments/outputs/*.ndjson` — one xAPI Statement per line (NDJSON).

## NDJSON and reproducibility

- **NDJSON:** Newline-delimited JSON; each line is one valid JSON object (one xAPI Statement). Same as “JSON Lines”.
- **Reproducibility:** All specs use a fixed `seed` (42) and the same cmi5 profile/model structure. Re-running with the same DATASIM version and specs produces the same statement counts and content.

## Notes

- **No editing of datasim**: We only consume DATASIM (Docker image or base input from GitHub). All experiment logic and specs live in this repo.
- Specs: If `xapi-experiments/specs/` is missing, the script runs `make_experiments.py`, which uses `xapi-experiments/base/simple.json` if present, else a datasim clone’s `dev-resources/input/simple.json`, else a one-time download from the datasim repo.
- Without Docker: `DATASIM_DIR=/path/to/datasim ./xapi-experiments/run_experiments.sh`. With Docker: `DATASIM_IMAGE=your-registry/datasim:tag ./xapi-experiments/run_experiments.sh` to override the image.
