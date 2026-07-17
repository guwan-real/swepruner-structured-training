#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  GPU_LIST="$1"
  shift
else
  GPU_LIST="${GPU_IDS:-0,1}"
fi

if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU list must look like 0 or 0,1,3; got: $GPU_LIST" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

echo "Launching M2 on physical GPU(s): $CUDA_VISIBLE_DEVICES ($NUM_GPUS process(es))"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/training/data/upload_bundle_2k}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/training/offline_assets/code-pruner}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$REPO_ROOT/training/offline_assets/code-pruner}"
BACKBONE_PATH="${BACKBONE_PATH:-$REPO_ROOT/training/offline_assets/qwen3-reranker-config}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/training_outputs/m2_structural}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m training.train \
  --config "$REPO_ROOT/training/configs/m2_structural.json" \
  --data-root "$DATA_ROOT" \
  --backbone-path "$BACKBONE_PATH" \
  --backbone-config-only \
  --tokenizer-path "$TOKENIZER_PATH" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
