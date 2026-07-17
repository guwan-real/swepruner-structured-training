#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:?Usage: install_offline_conda.sh CONDA_ENV WHEELHOUSE}"
WHEELHOUSE="${2:?Usage: install_offline_conda.sh CONDA_ENV WHEELHOUSE}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install --no-index --find-links "$WHEELHOUSE" "torch==2.8.0"
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REPO_ROOT/training/requirements-runtime.txt"
python - <<'PY'
import torch, transformers, safetensors
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "safetensors", safetensors.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this conda environment")
PY

