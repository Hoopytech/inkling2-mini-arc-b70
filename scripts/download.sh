#!/usr/bin/env bash
# Download Inkling 2 Mini GGUFs from Hugging Face (unsloth/Inkling-2-Mini-GGUF) and verify sha256.
#   LAYOUT=unsloth  -> root files, for Unsloth's llama.cpp prebuilds (Vulkan path)
#   LAYOUT=mainline -> Mainline_Support files, for upstream llama.cpp (SYCL path)
set -euo pipefail
QUANT="${QUANT:-UD-Q4_K_XL}"
LAYOUT="${LAYOUT:-mainline}"
DEST="${DEST:-./models}"
REPO="unsloth/Inkling-2-Mini-GGUF"
mkdir -p "$DEST"

if [[ "$LAYOUT" == "mainline" ]]; then
  remote="Mainline_Support/Inkling-2-Mini-${QUANT}.gguf_file"
  local_name="Inkling-2-Mini-${QUANT}.mainline.gguf"
else
  remote="Inkling-2-Mini-${QUANT}.gguf"
  local_name="Inkling-2-Mini-${QUANT}.gguf"
fi

fetch() {  # fetch <remote path> <local file>
  echo ">> $1"
  curl -L --fail -C - -o "$DEST/$2" "https://huggingface.co/$REPO/resolve/main/$1"
  expected=$(curl -s -X POST -H 'Content-Type: application/json' \
    -d "{\"paths\":[\"$1\"]}" "https://huggingface.co/api/models/$REPO/paths-info/main" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)[0]["lfs"]["oid"])')
  actual=$(sha256sum "$DEST/$2" | cut -d' ' -f1)
  [[ "$expected" == "$actual" ]] && echo "   sha256 OK" || { echo "   sha256 MISMATCH ($actual != $expected)"; exit 1; }
}

fetch "$remote" "$local_name"
[[ "${WITH_MMPROJ:-0}" == "1" ]] && fetch "mmproj-F16.gguf" "mmproj-F16.gguf"
echo "Done: $DEST/$local_name"
