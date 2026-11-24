#!/usr/bin/env bash
# Compile the recipe's SYCL kernels and run its numerical checks on the GPU (needs the GPU to itself).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$INKLING_DIR"
bash scripts/check-sycl.sh                                              # paged attention, MoE, prefill
INKLING2_EXL3_TILE_ROWS=32 bash scripts/check-sycl.sh exl3 --tokens 256 512  # EXL3 experts vs real weights
bash scripts/build-loop-control.sh
bash scripts/check-coding-kernels.sh                                    # QKV, prep, DSpark taps, reasoning budget
echo "OK: all checks passed"
