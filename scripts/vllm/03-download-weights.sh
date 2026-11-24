#!/usr/bin/env bash
# Download the EXL3 checkpoint into $INKLING_DIR/models/exl3 (real files: the repo is bind-mounted into
# the container, so a symlink to another disk would not resolve). ~16 GiB, sha256-verified.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
DEST="$INKLING_DIR/models/exl3"; mkdir -p "$DEST"; cd "$DEST"
curl -s "https://huggingface.co/api/models/$EXL3_REPO/tree/$EXL3_REV?recursive=1" | python3 -c '
import sys, json
for f in json.load(sys.stdin):
    if f["type"] == "file":
        print(f["path"], (f.get("lfs") or {}).get("oid", "-"))' > .files.txt
bad=0
while read -r path oid; do
  mkdir -p "$(dirname "$path")"
  curl -sL --fail -C - -o "$path" "https://huggingface.co/$EXL3_REPO/resolve/$EXL3_REV/$path"
  if [[ "$oid" != "-" && "$(sha256sum "$path" | cut -d' ' -f1)" != "$oid" ]]; then echo "SHA MISMATCH $path"; bad=1; fi
done < .files.txt
rm -f .files.txt
(( bad == 0 )) && echo "OK: $(du -sh "$DEST" | cut -f1) in $DEST"
exit $bad
