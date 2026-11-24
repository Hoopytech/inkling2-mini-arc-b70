#!/usr/bin/env bash
# Build the vLLM 0.33 XPU runtime (~1-3 h, no GPU needed) and the oneAPI builder image derived from it.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$INKLING_DIR/.work/qwen-build"
python3 scripts/build-engine.py --tag "$INKLING2_RUNTIME" --output benchmark-results/inkling-runtime
cd "$INKLING_DIR"
docker build --build-arg "BASE_IMAGE=$INKLING2_RUNTIME" --tag "$INKLING2_BUILDER" \
  -f .work/qwen-build/benchmarks/experiments/m17-oneapi-recompile/Dockerfile.compiler \
  .work/qwen-build/benchmarks/experiments/m17-oneapi-recompile
docker images "$INKLING2_RUNTIME"; docker images "$INKLING2_BUILDER"
