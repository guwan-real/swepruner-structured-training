#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-$REPO_ROOT/training/offline_assets}"
mkdir -p "$DEST/code-pruner" "$DEST/qwen3-reranker-config"

python - <<'PY' "$DEST"
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(sys.argv[1])
snapshot_download(
    repo_id="ayanami-kitasan/code-pruner",
    local_dir=root / "code-pruner",
    allow_patterns=[
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "README.md",
    ],
)
config = hf_hub_download(repo_id="Qwen/Qwen3-Reranker-0.6B", filename="config.json")
(root / "qwen3-reranker-config" / "config.json").write_bytes(Path(config).read_bytes())
print(f"Offline model assets ready under {root}")
PY

