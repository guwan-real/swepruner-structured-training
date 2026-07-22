from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    strategy: str
    structural_heads: bool
    experiment_name: str = ""
    seed: int = 42
    max_length: int = 8192
    aux_max_length: int = 2048
    epochs: int = 3
    samples_per_epoch: int = 1566
    replay_ratio: float = 0.20
    per_device_batch_size: int = 2
    aux_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate_backbone: float = 2e-5
    learning_rate_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    backbone_training_mode: str = "last_n"
    num_finetune_layers: int = 2
    bottleneck: int = 256
    dropout: float = 0.4
    num_fusion_layers: int = 1
    num_heads: int = 8
    use_multi_layer_fusion: bool = True
    early_layer_ratio: float = 0.25
    middle_layer_ratio: float = 0.50
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    relation_batch_every: int = 1
    ranking_batch_every: int = 1
    ranking_margin: float = 0.20
    threshold: float = 0.50
    num_workers: int = 2
    gradient_log_every: int = 10
    save_optimizer: bool = False
    instruction: str = "Given a query, judge if the document(code) is related to query."
    loss_weights: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.strategy not in {"data_only", "structural_pruner"}:
            raise ValueError(f"unknown strategy: {self.strategy}")
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in [0, 1]")
        positive = {
            "max_length": self.max_length,
            "epochs": self.epochs,
            "samples_per_epoch": self.samples_per_epoch,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        required = {"keep", "role", "relation", "rank", "document"}
        if set(self.loss_weights) != required:
            raise ValueError(f"loss_weights must have exactly {sorted(required)}")
        if self.loss_weights["keep"] <= 0:
            raise ValueError("keep loss must be enabled for every experiment")
        if any(weight < 0 for weight in self.loss_weights.values()):
            raise ValueError("loss weights must be non-negative")
        if self.gradient_log_every < 0:
            raise ValueError("gradient_log_every must be non-negative")
        if self.backbone_training_mode not in {"last_n", "full"}:
            raise ValueError("backbone_training_mode must be 'last_n' or 'full'")
        if self.strategy == "data_only":
            if self.structural_heads:
                raise ValueError("M1 data_only must not create structural heads")
            if any(self.loss_weights[name] != 0 for name in ("role", "relation", "rank")):
                raise ValueError("M1 structural losses must be zero")
        if self.strategy == "structural_pruner" and not self.structural_heads:
            raise ValueError("M2 structural_pruner requires structural heads")
        if self.strategy == "structural_pruner" and not any(
            self.loss_weights[name] > 0 for name in ("role", "relation", "rank")
        ):
            raise ValueError("structural_pruner requires at least one structural objective")

    def objective_enabled(self, name: str) -> bool:
        return self.loss_weights[name] > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        key, value = item.split("=", 1)
        if key not in raw:
            raise ValueError(f"unknown config override: {key}")
        try:
            raw[key] = json.loads(value)
        except json.JSONDecodeError:
            raw[key] = value
    config = TrainConfig(**raw)
    config.validate()
    return config


PARITY_FIELDS = (
    "seed",
    "max_length",
    "aux_max_length",
    "epochs",
    "samples_per_epoch",
    "replay_ratio",
    "per_device_batch_size",
    "aux_batch_size",
    "gradient_accumulation_steps",
    "learning_rate_backbone",
    "learning_rate_heads",
    "weight_decay",
    "warmup_ratio",
    "backbone_training_mode",
    "num_finetune_layers",
    "bottleneck",
    "dropout",
    "num_fusion_layers",
    "num_heads",
    "use_multi_layer_fusion",
    "early_layer_ratio",
    "middle_layer_ratio",
    "attention_implementation",
    "relation_batch_every",
    "ranking_batch_every",
    "gradient_log_every",
)
