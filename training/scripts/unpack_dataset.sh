#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="${1:-$REPO_ROOT/training/assets/swepruner_real_dataset_2k_seed42.tar.gz}"
DATA_DIR="${2:-$REPO_ROOT/training/data}"
EXPECTED="25b83b5bab239599aa8b49021260d24e4e11becacd10e6759f3ea25da60d26bf"

ACTUAL="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "Dataset archive checksum mismatch: $ACTUAL" >&2
  exit 1
fi
mkdir -p "$DATA_DIR"
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
echo "Dataset ready: $DATA_DIR/upload_bundle_2k"

