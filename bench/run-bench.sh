#!/usr/bin/env bash
# Reproducible llama-bench matrix for Inkling 2 Mini on a single Intel Arc GPU.
# Usage: BENCH=/path/to/llama-bench MODEL=/path/to/model.gguf [ENV="VAR=1 ..."] bench/run-bench.sh > results/<name>.md
set -euo pipefail
: "${BENCH:?path to llama-bench}" "${MODEL:?path to gguf}"
echo "# $(basename "$MODEL") — $(date -Iseconds)"
echo; echo '```'; "$BENCH" --version 2>&1 | tail -2; echo '```'; echo
env ${ENV:-} "$BENCH" -m "$MODEL" -ngl 99 -fa 1 -p 512,4096 -n 128 -ub 512,1024,2048 -r 3 -o md
