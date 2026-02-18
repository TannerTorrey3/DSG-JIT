#!/usr/bin/env bash
# Run DATASIM to generate xAPI Statement NDJSON. Prefers Docker (no Java/Clojure/datasim clone).
# Falls back to local datasim clone + make bundle if Docker not available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASIM_DIR="${DATASIM_DIR:-$REPO_ROOT/datasim}"
SPECS_DIR="$SCRIPT_DIR/specs"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"
BUNDLE_DIR="$DATASIM_DIR/target/bundle"
RUN_SH="$BUNDLE_DIR/bin/run.sh"
CLI_JAR="$BUNDLE_DIR/datasim_cli.jar"
DATASIM_IMAGE="${DATASIM_IMAGE:-yetanalytics/datasim:latest}"

cd "$REPO_ROOT"

# ---- Ensure specs exist (no datasim clone needed: downloads base from GitHub if missing) ----
if [[ ! -d "$SPECS_DIR" ]] || [[ -z "$(find "$SPECS_DIR" -maxdepth 1 -name '*.json' 2>/dev/null)" ]]; then
  echo "Specs missing; generating (may download base input once)..." >&2
  python3 "$SCRIPT_DIR/make_experiments.py" --specs-dir "$SPECS_DIR" || exit 1
fi

mkdir -p "$OUTPUTS_DIR"

# ---- Prefer Docker: no clone, no Java/Clojure ----
USE_DOCKER=false
if command -v docker &>/dev/null; then
  if docker image inspect "$DATASIM_IMAGE" &>/dev/null 2>&1; then
    USE_DOCKER=true
  elif docker pull "$DATASIM_IMAGE" 2>/dev/null; then
    USE_DOCKER=true
  fi
fi

if [[ "$USE_DOCKER" == true ]]; then
  echo "Using Docker (yetanalytics/datasim). No local datasim repo or Java needed." >&2
  SPECS_ABS="$(cd "$SPECS_DIR" && pwd)"
  echo "---- Running experiments ----"
  for spec in tiny small medium large huge; do
    spec_path="$SPECS_DIR/${spec}.json"
    out_path="$OUTPUTS_DIR/${spec}.ndjson"
    if [[ ! -f "$spec_path" ]]; then
      echo "Error: Spec not found: $spec_path" >&2
      exit 1
    fi
    echo "Generating $spec..."
    docker run --rm -v "$SPECS_ABS:/specs:ro" -i "$DATASIM_IMAGE" generate -i "/specs/${spec}.json" > "$out_path" || {
      echo "Error: DATASIM generate failed for $spec" >&2
      exit 1
    }
    count="$(wc -l < "$out_path")"
    size="$(wc -c < "$out_path")"
    echo "  $spec: $count statements, $size bytes"
  done
else
  # ---- Local: require datasim clone + Java + Clojure ----
  if [[ ! -d "$DATASIM_DIR" ]]; then
    echo "Error: DATASIM repo not found at $DATASIM_DIR (Docker not used)." >&2
    echo "Minimal setup: install Docker, then run this script again (no clone/Java/Clojure)." >&2
    echo "Or clone and build: git clone https://github.com/yetanalytics/datasim.git $DATASIM_DIR" >&2
    exit 1
  fi
  if ! command -v java &>/dev/null || ! java -version &>/dev/null; then
    echo "Error: Java Runtime not found. Install Java or use Docker for minimal setup." >&2
    exit 1
  fi
  if [[ ! -f "$CLI_JAR" ]]; then
    echo "Building DATASIM bundle (make bundle)..." >&2
    (cd "$DATASIM_DIR" && make bundle) || {
      echo "Error: 'make bundle' failed. Use Docker for minimal setup, or install Clojure CLI." >&2
      exit 1
    }
  fi
  if [[ ! -f "$CLI_JAR" ]]; then
    echo "Error: DATASIM CLI JAR not found at $CLI_JAR" >&2
    exit 1
  fi
  echo "---- Running experiments ----"
  for spec in tiny small medium large huge; do
    spec_path="$SPECS_DIR/${spec}.json"
    out_path="$OUTPUTS_DIR/${spec}.ndjson"
    if [[ ! -f "$spec_path" ]]; then
      echo "Error: Spec not found: $spec_path" >&2
      exit 1
    fi
    spec_abs="$(cd "$(dirname "$spec_path")" && pwd)/$(basename "$spec_path")"
    echo "Generating $spec..."
    (cd "$BUNDLE_DIR" && ./bin/run.sh generate -i "$spec_abs") > "$out_path" || {
      echo "Error: DATASIM generate failed for $spec" >&2
      exit 1
    }
    count="$(wc -l < "$out_path")"
    size="$(wc -c < "$out_path")"
    echo "  $spec: $count statements, $size bytes"
  done
fi

echo ""
echo "---- Summary ----"
for spec in tiny small medium large huge; do
  out_path="$OUTPUTS_DIR/${spec}.ndjson"
  if [[ -f "$out_path" ]]; then
    count="$(wc -l < "$out_path")"
    size="$(wc -c < "$out_path")"
    printf "  %-6s %s statements  %s bytes  %s\n" "$spec" "$count" "$size" "$out_path"
  fi
done
