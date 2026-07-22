from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .config import TrainConfig


class CRFLayer(nn.Module):
    def __init__(self, num_tags: int = 2):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def loss(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gold = self._score(emissions, tags, mask)
        normalizer = self._normalizer(emissions, mask)
        return (normalizer - gold) / mask.sum(dim=1).float().clamp(min=1)

    def _score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0, None]).squeeze(1)
        for index in range(1, tags.shape[1]):
            valid = mask[:, index]
            emit = emissions[:, index].gather(1, tags[:, index, None]).squeeze(1)
            transition = self.transitions[tags[:, index], tags[:, index - 1]]
            score = score + (emit + transition) * valid
        last = tags.gather(1, (mask.sum(dim=1).long() - 1).clamp(min=0)[:, None]).squeeze(1)
        return score + self.end_transitions[last]

    def _normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        alpha = self.start_transitions + emissions[:, 0]
        for index in range(1, emissions.shape[1]):
            scores = alpha[:, None, :] + self.transitions[None, :, :]
            updated = torch.logsumexp(scores, dim=2) + emissions[:, index]
            alpha = torch.where(mask[:, index, None], updated, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]
        backpointers = []
        for index in range(1, emissions.shape[1]):
            scores = score[:, None, :] + self.transitions[None, :, :]
            best_score, best_tag = scores.max(dim=2)
            updated = best_score + emissions[:, index]
            score = torch.where(mask[:, index, None], updated, score)
            backpointers.append(best_tag)
        best = (score + self.end_transitions).argmax(dim=1)
        path = [best]
        for pointer in reversed(backpointers):
            path.append(pointer.gather(1, path[-1][:, None]).squeeze(1))
        return torch.stack(list(reversed(path)), dim=1)


class CRFCompressionHead(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 2, dtype=torch.float32),
        )
        self.crf = CRFLayer(2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(hidden)


def _load_backbone(path: str, config_only: bool, implementation: str) -> nn.Module:
    if config_only:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        config._attn_implementation = implementation
        model = AutoModel.from_config(config, trust_remote_code=True)
        return model.to(dtype=torch.bfloat16)
    return AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=implementation,
        device_map=None,
    )


def _transformer_layers(backbone: nn.Module) -> Sequence[nn.Module]:
    candidates = ("layers", "model.layers", "transformer.h", "encoder.layer")
    for path in candidates:
        value: Any = backbone
        try:
            for attribute in path.split("."):
                value = getattr(value, attribute)
        except AttributeError:
            continue
        if isinstance(value, (nn.ModuleList, list, tuple)) and len(value) > 0:
            return value
    raise ValueError("cannot locate transformer layers in backbone")


class StructuralPrunerModel(nn.Module):
    """Official TokenScorer-compatible core plus optional M2 heads."""

    def __init__(
        self,
        backbone_path: str,
        tokenizer: Any,
        config: TrainConfig,
        num_roles: int,
        num_relations: int,
        backbone_config_only: bool,
    ):
        super().__init__()
        self.config = config
        self.backbone = _load_backbone(backbone_path, backbone_config_only, config.attention_implementation)
        hidden_size = int(self.backbone.config.hidden_size)
        num_layers = int(self.backbone.config.num_hidden_layers)
        self.use_multi_layer_fusion = config.use_multi_layer_fusion
        self.early_layer_idx = max(1, int(num_layers * config.early_layer_ratio))
        self.middle_layer_idx = max(1, int(num_layers * config.middle_layer_ratio))
        self.final_layer_idx = num_layers
        self.fused_hidden_size = hidden_size * 3 if self.use_multi_layer_fusion else hidden_size
        self.word_embeddings = self.backbone.get_input_embeddings().weight
        self.token_yes_id = int(tokenizer.convert_tokens_to_ids("yes"))
        self.token_no_id = int(tokenizer.convert_tokens_to_ids("no"))
        if self.token_yes_id < 0 or self.token_no_id < 0:
            raise ValueError("tokenizer cannot resolve official yes/no scoring tokens")
        self.dropout = nn.Dropout(config.dropout)
        self.fusion_layers = nn.ModuleList([
            nn.MultiheadAttention(self.fused_hidden_size, config.num_heads, batch_first=True)
            for _ in range(config.num_fusion_layers)
        ])
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(self.fused_hidden_size) for _ in range(config.num_fusion_layers)
        ])
        self.compression_head = CRFCompressionHead(
            self.fused_hidden_size, config.bottleneck, config.dropout
        )
        self.role_head = (
            nn.Sequential(nn.LayerNorm(self.fused_hidden_size), nn.Linear(self.fused_hidden_size, num_roles))
            if config.objective_enabled("role") else None
        )
        self.relation_head = (
            nn.Sequential(nn.LayerNorm(self.fused_hidden_size), nn.Linear(self.fused_hidden_size, num_relations))
            if config.objective_enabled("relation") else None
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        layers = _transformer_layers(self.backbone)
        if config.backbone_training_mode == "full":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        else:
            if config.num_finetune_layers > len(layers):
                raise ValueError("num_finetune_layers exceeds backbone depth")
            for layer in layers[len(layers) - config.num_finetune_layers :]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone.config, "use_cache"):
                self.backbone.config.use_cache = False

    def _backbone_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        states = outputs.hidden_states
        if self.use_multi_layer_fusion:
            fused = torch.cat(
                [states[self.early_layer_idx].float(), states[self.middle_layer_idx].float(), states[self.final_layer_idx].float()],
                dim=-1,
            )
        else:
            fused = states[-1].float()
        hidden = fused
        padding_mask = attention_mask.eq(0)
        for attention, norm in zip(self.fusion_layers, self.fusion_norms):
            value, _ = attention(hidden, hidden, hidden, key_padding_mask=padding_mask, need_weights=False)
            hidden = norm(hidden + value)
        return self.dropout(hidden), outputs.last_hidden_state.float()

    def _document_logprob(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        indices = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        pooled = last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), indices]
        weights = self.backbone.get_input_embeddings().weight.float()
        no_score = (pooled * weights[self.token_no_id]).sum(dim=-1)
        yes_score = (pooled * weights[self.token_yes_id]).sum(dim=-1)
        return F.log_softmax(torch.stack([no_score, yes_score], dim=1), dim=1)[:, 1]

    def _pooled_relation(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.relation_head is None:
            raise RuntimeError("relation head is disabled")
        indices = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
        return self.relation_head(pooled)

    def _score_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, last = self._backbone_hidden(input_ids, attention_mask)
        return self._document_logprob(last, attention_mask).exp()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        relation_input_ids: torch.Tensor | None = None,
        relation_attention_mask: torch.Tensor | None = None,
        positive_input_ids: torch.Tensor | None = None,
        positive_attention_mask: torch.Tensor | None = None,
        negative_input_ids: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden, last = self._backbone_hidden(input_ids, attention_mask)
        emissions = self.compression_head(hidden)
        result: dict[str, torch.Tensor | None] = {
            "emissions": emissions,
            "token_logits": emissions[:, :, 1] - emissions[:, :, 0],
            "document_logprob": self._document_logprob(last, attention_mask),
            "role_logits": self.role_head(hidden) if self.role_head is not None else None,
            "relation_logits": self.relation_head(hidden) if self.relation_head is not None else None,
            "aux_relation_logits": None,
            "positive_scores": None,
            "negative_scores": None,
        }
        if relation_input_ids is not None:
            rel_hidden, _ = self._backbone_hidden(relation_input_ids, relation_attention_mask)
            result["aux_relation_logits"] = self._pooled_relation(rel_hidden, relation_attention_mask)
        if positive_input_ids is not None and negative_input_ids is not None:
            result["positive_scores"] = self._score_inputs(positive_input_ids, positive_attention_mask)
            result["negative_scores"] = self._score_inputs(negative_input_ids, negative_attention_mask)
        return result
