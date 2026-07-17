#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/python_runtime.sh"

if [[ -f data_sources/swe_smith/tasks.jsonl ]]; then
  "$PYTHON_BIN" -m swepruner_dataset_builder build --source swe_smith --tasks data_sources/swe_smith/tasks.jsonl --output artifacts/swe_smith --config config/default.toml --seed 42 --num-workers 4 --resume
fi
if [[ -f data_sources/swe_gym/tasks.jsonl ]]; then
  "$PYTHON_BIN" -m swepruner_dataset_builder build --source swe_gym --tasks data_sources/swe_gym/tasks.jsonl --output artifacts/swe_gym --config config/default.toml --seed 42 --num-workers 4 --resume
fi
if [[ -f data_sources/swe_pruner/train.jsonl ]]; then
  "$PYTHON_BIN" -m swepruner_dataset_builder build --source swe_pruner_original --tasks data_sources/swe_pruner/train.jsonl --output artifacts/swe_pruner_original --config config/default.toml --seed 42 --resume
fi
"$PYTHON_BIN" -m swepruner_dataset_builder create-manifest --artifacts-root artifacts --output artifacts/combined_manifest.json --config config/default.toml --seed 42
