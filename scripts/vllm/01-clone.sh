#!/usr/bin/env bash
# Clone the recipe and its build sources at the tested commits and apply our host-compatibility patches.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PATCHES="$HERE/../../patches/vllm-recipe"
source "$HERE/env.sh"

git clone https://github.com/Danmoreng/intel-b70-inkling2-mini-vllm.git "$INKLING_DIR"
git -C "$INKLING_DIR" checkout -q --detach "$RECIPE_COMMIT"
git -C "$INKLING_DIR" apply "$PATCHES/0001-recipe-arc-host-compat.patch"

mkdir -p "$INKLING_DIR/.work"
git clone -q https://github.com/Danmoreng/intel-b70-qwen38-vllm.git "$INKLING_DIR/.work/qwen-build"
git -C "$INKLING_DIR/.work/qwen-build" checkout -q --detach "$QWEN_BUILD_COMMIT"

# Same fetch the recipe's check-sycl.sh does, so the C++20 patch can be applied before the first compile.
T="$INKLING_DIR/.work/trellis-serve"
git init -q "$T"
git -C "$T" fetch -q --no-tags --depth 1 https://github.com/sybil-solutions/trellis-serve.git "$TRELLIS_COMMIT"
git -C "$T" checkout -q --detach "$TRELLIS_COMMIT"
git -C "$T" apply "$PATCHES/0002-trellis-cxx20-torch215.patch"
echo "OK: $INKLING_DIR"
