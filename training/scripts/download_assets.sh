#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-$REPO_ROOT/training/offline_assets}"
mkdir -p "$DEST/code-pruner" "$DEST/qwen3-reranker" "$DEST/qwen3-reranker-config"

python - <<'PY' "$DEST"
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
qwen_revision = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
snapshot_download(
    repo_id="ayanami-kitasan/code-pruner",
    local_dir=root / "code-pruner",
    allow_patterns=[
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "README.md",
    ],
)
snapshot_download(
    repo_id="Qwen/Qwen3-Reranker-0.6B",
    revision=qwen_revision,
    local_dir=root / "qwen3-reranker",
    allow_patterns=[
        "model.safetensors", "model-*.safetensors", "model.safetensors.index.json",
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json", "README.md",
    ],
)
(root / "qwen3-reranker-config" / "config.json").write_bytes(
    (root / "qwen3-reranker" / "config.json").read_bytes()
)
print(f"Offline model assets ready under {root}")
PY
