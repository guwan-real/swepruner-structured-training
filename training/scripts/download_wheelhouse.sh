#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-$REPO_ROOT/training/wheelhouse}"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
mkdir -p "$DEST"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "Run this script on an internet-connected Linux x86_64 host matching the B200 server." >&2
  exit 1
fi
python -m pip download --dest "$DEST" --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
python -m pip download --dest "$DEST" -r "$REPO_ROOT/training/requirements-runtime.txt"
echo "Wheelhouse ready: $DEST"
