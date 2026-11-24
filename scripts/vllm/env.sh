# Shared settings for the vLLM path. Source it: `source scripts/vllm/env.sh`
export INKLING_DIR="${INKLING_DIR:-$HOME/inkling-vllm}"            # checkout of the B70 recipe
export RECIPE_COMMIT=0a93121                                           # Danmoreng/intel-b70-inkling2-mini-vllm
export QWEN_BUILD_COMMIT=0356a370da5f73f6b0a4eab613d7103a2356b513      # Danmoreng/intel-b70-qwen38-vllm
export TRELLIS_COMMIT=8f8e2fccf0c82c9715b774812f30d9d52f78a5fb         # sybil-solutions/trellis-serve
export EXL3_REPO=grant-ai/Inkling-2-Mini-EXL3-4bpw
export EXL3_REV=c266f76b68429255e1d6ec151d0aa15f48f6abcc

# Images built by 02-build-images.sh
export INKLING2_RUNTIME="${INKLING2_RUNTIME:-local/b70-qwen38-vllm:inkling-runtime}"
export INKLING2_BUILDER="${INKLING2_BUILDER:-local/inkling2-oneapi:2026.1.1}"
# Render node of the B70 (check `ls -l /dev/dri/by-path/`)
export INKLING2_RENDER_NODE="${INKLING2_RENDER_NODE:-/dev/dri/renderD128}"

# Serving: no daily stop, bind address, and the tuned profile measured in results/2026-11-23-vllm-exl3.md
export INKLING2_NO_DEADLINE=1
export INKLING2_VLLM_HOST="${INKLING2_VLLM_HOST:-0.0.0.0}"
export INKLING2_VLLM_PORT="${INKLING2_VLLM_PORT:-8093}"
export INKLING2_PREFILL_PROFILE="${INKLING2_PREFILL_PROFILE:-optimized}"   # +43-46 % prefill on our card
export INKLING2_DRAFT_MAX="${INKLING2_DRAFT_MAX:-3}"                       # +7-9 % decode vs the default 7
