#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: train_ablation.sh PRESET [GPU_LIST] [training.train arguments]

Primary matrix:
  b0               keep only
  b1               keep + relation
  b2               keep + relation + role
  b3               full M2

Single-objective additions:
  keep_role        keep + role
  keep_relation    keep + relation
  keep_rank        keep + rank
  keep_document    keep + document

Leave-one-out from full M2:
  full_no_role
  full_no_relation
  full_no_rank
  full_no_document
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PRESET="$1"
shift
if [[ $# -gt 0 && "$1" != --* ]]; then
  GPU_LIST="$1"
  shift
else
  GPU_LIST="${GPU_IDS:-0,1}"
fi

case "$PRESET" in
  b0)
    BASE_CONFIG="m1_data_only.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.0,"relation":0.0,"rank":0.0,"document":0.0}'
    ;;
  b1|keep_relation)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.0,"relation":0.1,"rank":0.0,"document":0.0}'
    ;;
  b2)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.25,"relation":0.1,"rank":0.0,"document":0.0}'
    ;;
  b3)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.25,"relation":0.1,"rank":0.1,"document":0.05}'
    ;;
  keep_role)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.25,"relation":0.0,"rank":0.0,"document":0.0}'
    ;;
  keep_rank)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.0,"relation":0.0,"rank":0.1,"document":0.0}'
    ;;
  keep_document)
    BASE_CONFIG="m1_data_only.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.0,"relation":0.0,"rank":0.0,"document":0.05}'
    ;;
  full_no_role)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.0,"relation":0.1,"rank":0.1,"document":0.05}'
    ;;
  full_no_relation)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.25,"relation":0.0,"rank":0.1,"document":0.05}'
    ;;
  full_no_rank)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.25,"relation":0.1,"rank":0.0,"document":0.05}'
    ;;
  full_no_document)
    BASE_CONFIG="m2_structural.json"
    LOSS_WEIGHTS='{"keep":1.0,"role":0.25,"relation":0.1,"rank":0.1,"document":0.0}'
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown ablation preset: $PRESET" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU list must look like 0 or 0,1,3; got: $GPU_LIST" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/training/data/upload_bundle_2k}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/training/offline_assets/qwen3-reranker}"
BACKBONE_PATH="${BACKBONE_PATH:-$REPO_ROOT/training/offline_assets/qwen3-reranker}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/training_outputs/ablations/$PRESET}"

if [[ ! -f "$BACKBONE_PATH/config.json" ]] || ! compgen -G "$BACKBONE_PATH/*.safetensors" >/dev/null; then
  echo "Full Qwen3-Reranker weights are missing under $BACKBONE_PATH" >&2
  echo "Run: bash training/scripts/download_assets.sh" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Launching ablation $PRESET on physical GPU(s): $GPU_LIST ($NUM_GPUS process(es))"
echo "Initialization: pretrained Qwen3-Reranker backbone (SWE-Pruner checkpoint is not loaded)"
echo "Loss weights: $LOSS_WEIGHTS"
echo "Output: $OUTPUT_DIR"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m training.train \
  --config "$REPO_ROOT/training/configs/$BASE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --backbone-path "$BACKBONE_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --init-mode backbone \
  --output-dir "$OUTPUT_DIR" \
  --set "experiment_name=$PRESET" \
  --set "loss_weights=$LOSS_WEIGHTS" \
  "$@"
