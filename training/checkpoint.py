from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _checkpoint_file(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source
    for name in ("best_model.pt", "model.safetensors", "pytorch_model.bin"):
        candidate = source / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no supported checkpoint found under {source}")


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Any = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint does not contain a state dict: {path}")
    return state


def _normalize_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model."):
        candidate = key[len("model.") :]
        if candidate.startswith(("backbone.", "fusion_layers.", "fusion_norms.", "compression_head.", "word_embeddings", "embedding_layer.", "role_head.", "relation_head.")):
            key = candidate
    return key


def load_official_initialization(model: torch.nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    path = _checkpoint_file(checkpoint)
    raw = _load_state(path)
    normalized = {_normalize_key(str(key)): value for key, value in raw.items()}
    normalized.pop("embedding_layer.weight", None)
    expected = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    for key, value in normalized.items():
        if key not in expected:
            continue
        if tuple(value.shape) != tuple(expected[key].shape):
            mismatched.append(f"{key}: checkpoint={tuple(value.shape)} model={tuple(expected[key].shape)}")
            continue
        matched[key] = value
    core_keys = [
        key for key in expected
        if not key.startswith(("role_head.", "relation_head.")) and key != "word_embeddings"
    ]
    core_numel = sum(expected[key].numel() for key in core_keys)
    matched_numel = sum(expected[key].numel() for key in core_keys if key in matched)
    coverage = matched_numel / max(1, core_numel)
    missing_core = [key for key in core_keys if key not in matched]
    if mismatched or coverage < 0.98:
        raise RuntimeError(
            f"official checkpoint is incompatible: coverage={coverage:.4f}, "
            f"missing_core={missing_core[:20]}, mismatched={mismatched[:20]}"
        )
    result = model.load_state_dict(matched, strict=False)
    return {
        "checkpoint": str(path.resolve()),
        "core_parameter_coverage": coverage,
        "matched_keys": len(matched),
        "missing_allowed": sorted(result.missing_keys),
        "unexpected_ignored": sorted(set(normalized) - set(matched)),
    }


def save_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "metadata": metadata}, temporary)
    os.replace(temporary, destination)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

