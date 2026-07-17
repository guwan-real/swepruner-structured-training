from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


CORE_PREFIXES = ("backbone.", "fusion_layers.", "fusion_norms.", "compression_head.")
REQUIRED_GROUPS = CORE_PREFIXES
DROP_PREFIXES = ("word_embeddings", "embedding_layer.", "role_head.", "relation_head.")
WEIGHT_SUFFIXES = (".bin", ".ckpt", ".pt", ".pth", ".safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained M1/M2 checkpoint for the official SWE-Pruner service"
    )
    parser.add_argument("--checkpoint", required=True, help="best_model.pt, last_model.pt, or its run directory")
    parser.add_argument("--tokenizer-dir", required=True, help="official code-pruner tokenizer directory")
    parser.add_argument("--backbone-config-dir", required=True, help="local Qwen backbone config directory")
    parser.add_argument("--output-dir", required=True, help="new or empty destination directory")
    parser.add_argument("--label", required=True, help="experiment label recorded in export_manifest.json")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "preserve"), default="bfloat16")
    return parser.parse_args()


def checkpoint_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path
    for name in ("best_model.pt", "last_model.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no best_model.pt or last_model.pt under {path}")


def normalize_key(raw_key: str) -> str:
    key = raw_key
    while key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("model.") and key[len("model.") :].startswith(CORE_PREFIXES + DROP_PREFIXES):
        key = key[len("model.") :]
    return key


def target_dtype(name: str) -> torch.dtype | None:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return None


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        metadata = payload["metadata"]
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError(f"checkpoint does not contain a tensor state dict: {path}")
    return payload, metadata


def export_state(
    raw_state: dict[str, torch.Tensor], dtype_name: str
) -> tuple[dict[str, torch.Tensor], list[str]]:
    dtype = target_dtype(dtype_name)
    exported: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for raw_key, tensor in raw_state.items():
        key = normalize_key(str(raw_key))
        if key.startswith(DROP_PREFIXES):
            dropped.append(str(raw_key))
            continue
        if not key.startswith(CORE_PREFIXES):
            dropped.append(str(raw_key))
            continue
        value = tensor.detach().cpu()
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype=dtype)
        # clone breaks shared storage aliases that safetensors intentionally rejects.
        exported[f"model.{key}"] = value.contiguous().clone()
    missing_groups = [prefix for prefix in REQUIRED_GROUPS if not any(key.startswith(f"model.{prefix}") for key in exported)]
    if missing_groups:
        raise RuntimeError(f"checkpoint is missing official inference groups: {missing_groups}")
    if not exported:
        raise RuntimeError("no official inference weights were exported")
    return exported, sorted(dropped)


def ensure_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def copy_tokenizer(source: Path, destination: Path) -> list[str]:
    if not (source / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"tokenizer_config.json is missing under {source}")
    copied: list[str] = []
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.name == "config.json" or item.name.endswith(WEIGHT_SUFFIXES):
            continue
        shutil.copy2(item, destination / item.name)
        copied.append(item.name)
    return copied


def copy_backbone_config(source: Path, destination: Path) -> list[str]:
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"backbone config.json is missing under {source}")
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[str] = []
    for item in sorted(source.iterdir()):
        if item.is_file() and not item.name.endswith(WEIGHT_SUFFIXES):
            shutil.copy2(item, destination / item.name)
            copied.append(item.name)
    return copied


def model_config(metadata: dict[str, Any], backbone_path: Path, dtype_name: str) -> dict[str, Any]:
    train = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
    return {
        "architectures": ["SwePrunerForCodeCompression"],
        "backbone_model_name_or_path": str(backbone_path.resolve()),
        "bottleneck": int(train.get("bottleneck", 256)),
        "compression_head_type": "crf",
        "compression_loss_type": "focal",
        "dropout": float(train.get("dropout", 0.4)),
        "early_layer_ratio": float(train.get("early_layer_ratio", 0.25)),
        "middle_layer_ratio": float(train.get("middle_layer_ratio", 0.5)),
        "model_type": "swepruner",
        "num_fusion_layers": int(train.get("num_fusion_layers", 1)),
        "num_heads": int(train.get("num_heads", 8)),
        "torch_dtype": "bfloat16" if dtype_name == "bfloat16" else "float32",
        "transformers_version": "4.57.1",
        "use_multi_layer_fusion": bool(train.get("use_multi_layer_fusion", True)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = checkpoint_file(args.checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    backbone_source = Path(args.backbone_config_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_empty_directory(output_dir)

    raw_state, metadata = load_checkpoint(checkpoint)
    state, dropped = export_state(raw_state, args.dtype)
    tokenizer_files = copy_tokenizer(tokenizer_dir, output_dir)
    backbone_destination = output_dir / "backbone_config"
    backbone_files = copy_backbone_config(backbone_source, backbone_destination)
    config = model_config(metadata, backbone_destination, args.dtype)
    write_json(output_dir / "config.json", config)

    temporary = output_dir / "model.safetensors.tmp"
    save_file(
        state,
        str(temporary),
        metadata={"format": "pt", "source": "swepruner-structured-training", "label": args.label},
    )
    os.replace(temporary, output_dir / "model.safetensors")
    manifest = {
        "format": "official-swe-pruner-huggingface",
        "label": args.label,
        "source_checkpoint": str(checkpoint),
        "source_epoch": metadata.get("epoch"),
        "source_global_step": metadata.get("global_step"),
        "source_metrics": metadata.get("metrics", {}),
        "dtype": args.dtype,
        "exported_key_count": len(state),
        "exported_parameter_count": sum(tensor.numel() for tensor in state.values()),
        "dropped_training_only_keys": dropped,
        "tokenizer_files": tokenizer_files,
        "backbone_config_files": backbone_files,
        "backbone_config_path_is_absolute": True,
    }
    write_json(output_dir / "export_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
