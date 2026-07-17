#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-swepruner-eval}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.12 pip
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130

if [[ "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-8}" python -m pip install flash-attn --no-build-isolation
fi

echo "Created conda environment: $ENV_NAME"
echo "Next: conda activate $ENV_NAME"
