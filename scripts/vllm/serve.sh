#!/usr/bin/env bash
# Run the tuned server in the foreground (use the systemd unit to keep it running).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
cd "$INKLING_DIR"
docker stop --time 10 inkling2-vllm-exl3 >/dev/null 2>&1 || true
exec bash scripts/serve-coding-agent.sh dspark "$@"
