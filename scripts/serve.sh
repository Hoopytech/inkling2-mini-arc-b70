#!/usr/bin/env bash
# Serve Inkling 2 Mini on one Intel Arc Pro B70 (32 GB) with llama.cpp Vulkan.
# Defaults are the best measured configuration (see README): 384K context, KV q8_0, 1 slot.
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-./llama-vulkan}"
MODEL="${MODEL:-./models/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
CTX="${CTX:-393216}"        # see the context table in the README before raising it
KV="${KV:-q8_0}"            # q8_0 | q4_0 | f16
UBATCH="${UBATCH:-512}"     # 1024 = +10% prefill but a larger compute buffer (less context)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

exec "$LLAMA_DIR/llama-server" -m "$MODEL" --jinja \
  -ngl 99 -fa on -np 1 -c "$CTX" -ub "$UBATCH" -ctk "$KV" -ctv "$KV" \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --host "$HOST" --port "$PORT" "$@"
