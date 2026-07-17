#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/python_runtime.sh"

"$PYTHON_BIN" -m swepruner_dataset_builder prepare-real-data \
  --root data_sources \
  --smith-limit 100 \
  --gym-limit 20 \
  --pruner-limit 100 \
  --max-repos-per-source 5 \
  --seed 42

"$PYTHON_BIN" -m swepruner_dataset_builder build \
  --source swe_pruner_original \
  --tasks data_sources/swe_pruner/first_phase.jsonl \
  --output artifacts/swe_pruner_original \
  --config config/real_first_phase.toml \
  --task-limit 100 \
  --seed 42 \
  --resume

"$PYTHON_BIN" -m swepruner_dataset_builder build \
  --source swe_smith \
  --tasks data_sources/swe_smith/tasks.jsonl \
  --output artifacts/swe_smith \
  --config config/real_first_phase.toml \
  --task-limit 100 \
  --num-workers 4 \
  --seed 42 \
  --offline \
  --resume

"$PYTHON_BIN" -m swepruner_dataset_builder build \
  --source swe_gym \
  --tasks data_sources/swe_gym/tasks.jsonl \
  --output artifacts/swe_gym \
  --config config/real_first_phase.toml \
  --task-limit 20 \
  --num-workers 4 \
  --seed 42 \
  --offline \
  --resume

"$PYTHON_BIN" -m swepruner_dataset_builder create-manifest \
  --artifacts-root artifacts \
  --output artifacts/combined_manifest.json \
  --config config/real_first_phase.toml \
  --seed 42

"$PYTHON_BIN" -m swepruner_dataset_builder validate --dataset artifacts/swe_pruner_original
"$PYTHON_BIN" -m swepruner_dataset_builder validate --dataset artifacts/swe_smith
"$PYTHON_BIN" -m swepruner_dataset_builder validate --dataset artifacts/swe_gym

"$PYTHON_BIN" -m swepruner_dataset_builder export-swepruner \
  --input artifacts/swe_smith/pruning_sft.jsonl \
  --mapping config/swepruner_mapping.json \
  --output artifacts/swe_smith/swe_pruner_compatible.jsonl

"$PYTHON_BIN" -m swepruner_dataset_builder export-swepruner \
  --input artifacts/swe_gym/pruning_sft.jsonl \
  --mapping config/swepruner_mapping.json \
  --output artifacts/swe_gym/swe_pruner_compatible.jsonl
