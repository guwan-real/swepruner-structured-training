#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-$REPO_ROOT/training/wheelhouse}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
mkdir -p "$DEST"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Run this script on an internet-connected Linux x86_64 host matching the B200 server." >&2
  exit 1
fi
python -m pip download --dest "$DEST" --index-url "$TORCH_INDEX_URL" "torch==2.8.0"
python -m pip download --dest "$DEST" -r "$REPO_ROOT/training/requirements-runtime.txt"
echo "Wheelhouse ready: $DEST"

