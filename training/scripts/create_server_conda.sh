#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-swepruner-train}"
WHEELHOUSE="${2:-$(pwd)/training/wheelhouse}"
SOURCE_ENV="${3:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load the server's conda module first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment already exists: $ENV_NAME" >&2
  exit 1
fi

if [[ -n "$SOURCE_ENV" ]]; then
  echo "Cloning local conda environment '$SOURCE_ENV' to '$ENV_NAME'..."
  conda create -y -n "$ENV_NAME" --clone "$SOURCE_ENV"
else
  create_args=(-y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip)
  if [[ "${CONDA_OFFLINE:-0}" == "1" ]]; then
    create_args+=(--offline)
  fi
  echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
  conda create "${create_args[@]}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/install_offline_conda.sh" "$ENV_NAME" "$WHEELHOUSE"

echo "Environment ready. Activate it with: conda activate $ENV_NAME"
