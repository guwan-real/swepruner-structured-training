#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PRESET [GPU_LIST] [training.train arguments]" >&2
  echo "Example: $0 b1 4,5 --set epochs=5" >&2
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

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/training_outputs/full_backbone_ablations/$PRESET}"

echo "Full-backbone continued training from the official SWE-Pruner checkpoint"
echo "Preset: $PRESET"

exec bash "$REPO_ROOT/training/scripts/train_ablation.sh" "$PRESET" "$GPU_LIST" \
  --set backbone_training_mode=full \
  --set gradient_checkpointing=false \
  --set "experiment_name=full_backbone_${PRESET}" \
  "$@"
