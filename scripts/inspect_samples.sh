#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="${1:-$ROOT/artifacts/samples_for_review/review_samples.md}"
sed -n '1,240p' "$FILE"

