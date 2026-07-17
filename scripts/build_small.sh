#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/python_runtime.sh"

"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m swepruner_dataset_builder inspect-input --source swe_smith --tasks tests/fixtures/data_sources/swe_smith/tasks.jsonl
"$PYTHON_BIN" -m swepruner_dataset_builder build --source swe_smith --tasks tests/fixtures/data_sources/swe_smith/tasks.jsonl --output artifacts/fixture_demo/swe_smith --config config/default.toml --seed 42 --task-limit 100 --num-workers 4 --offline --resume
"$PYTHON_BIN" -m swepruner_dataset_builder create-manifest --artifacts-root artifacts/fixture_demo --output artifacts/fixture_demo/combined_manifest.json --config config/default.toml --seed 42
"$PYTHON_BIN" -m swepruner_dataset_builder validate --dataset artifacts/fixture_demo/swe_smith
"$PYTHON_BIN" -m swepruner_dataset_builder export-swepruner --input artifacts/fixture_demo/swe_smith/pruning_sft.jsonl --mapping config/swepruner_mapping.json --output artifacts/fixture_demo/swe_smith/swe_pruner_compatible.jsonl
