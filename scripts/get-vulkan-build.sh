#!/usr/bin/env bash
# Unsloth's llama.cpp prebuild (Vulkan, Linux x64). Supports the root-layout GGUFs.
set -euo pipefail
TAG="${TAG:-b12631-mix-acc0843}"
DEST="${DEST:-./llama-vulkan}"
mkdir -p "$DEST"
curl -L --fail "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-linux-x64-vulkan.tar.gz" | tar -xz -C "$DEST"
"$DEST/llama-server" --list-devices
