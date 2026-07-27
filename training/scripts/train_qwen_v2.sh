#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 Q0|Q1|Q2|Q3 [GPU_LIST] [training.train arguments]" >&2
  exit 2
fi
PRESET="${1,,}"
shift
if [[ $# -gt 0 && "$1" != --* ]]; then GPU_LIST="$1"; shift; else GPU_LIST="${GPU_IDS:-0,1}"; fi
if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then echo "Invalid GPU list: $GPU_LIST" >&2; exit 2; fi

case "$PRESET" in
  q0) OVERRIDES=(--set crf_keep_weight=1.0 --set token_ce_weight=0.0 --set retention_loss_weight=0.0 --set catastrophic_loss_weight=0.0); USE_AUG=0 ;;
  q1) OVERRIDES=(--set crf_keep_weight=0.2 --set token_ce_weight=1.0 --set retention_loss_weight=0.0 --set catastrophic_loss_weight=0.0); USE_AUG=0 ;;
  q2) OVERRIDES=(--set crf_keep_weight=0.2 --set token_ce_weight=1.0 --set retention_loss_weight=0.2 --set catastrophic_loss_weight=0.0); USE_AUG=0 ;;
  q3) OVERRIDES=(--set crf_keep_weight=0.2 --set token_ce_weight=1.0 --set retention_loss_weight=0.2 --set catastrophic_loss_weight=0.5); USE_AUG=1 ;;
  *) echo "Unknown preset: $PRESET" >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/training/data/upload_bundle_2k}"
MODEL_PATH="${BACKBONE_PATH:-$REPO_ROOT/training/offline_assets/qwen3-reranker}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
AUGMENTATION_DATA="${AUGMENTATION_DATA:-$DATA_ROOT/augmentations/command_outputs_v2.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/training_outputs/qwen_v2/$PRESET}"
if [[ ! -f "$MODEL_PATH/config.json" ]] || ! compgen -G "$MODEL_PATH/*.safetensors" >/dev/null; then
  echo "Full Qwen3-Reranker assets are missing under $MODEL_PATH" >&2; exit 1
fi
AUGMENTATION_ARGS=()
if [[ "$USE_AUG" == "1" ]]; then
  if [[ ! -s "$AUGMENTATION_DATA" ]]; then echo "Q3 requires $AUGMENTATION_DATA" >&2; exit 1; fi
  AUGMENTATION_ARGS=(--augmentation-data "$AUGMENTATION_DATA")
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Launching ${PRESET^^}: full-parameter FP32 Qwen on GPU(s) $GPU_LIST"
torchrun --standalone --nproc_per_node="$NUM_GPUS" -m training.train \
  --config "$REPO_ROOT/training/configs/qwen_full_v2.json" \
  --data-root "$DATA_ROOT" \
  --backbone-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --init-mode backbone \
  --output-dir "$OUTPUT_DIR" \
  "${AUGMENTATION_ARGS[@]}" \
  --set "experiment_name=${PRESET}" \
  "${OVERRIDES[@]}" \
  "$@"
