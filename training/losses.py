from __future__ import annotations

import torch
import torch.nn.functional as F


def pack_code(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    code_mask: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = code_mask.sum(dim=1)
    max_length = int(lengths.max().item())
    packed_emissions = emissions.new_zeros((emissions.shape[0], max_length, emissions.shape[-1]))
    packed_labels = labels.new_zeros((labels.shape[0], max_length))
    packed_mask = torch.zeros((labels.shape[0], max_length), dtype=torch.bool, device=labels.device)
    packed_confidence = confidence.new_zeros((labels.shape[0], max_length))
    for batch_index in range(emissions.shape[0]):
        positions = code_mask[batch_index].nonzero(as_tuple=False).squeeze(1)
        length = positions.numel()
        packed_emissions[batch_index, :length] = emissions[batch_index, positions]
        packed_labels[batch_index, :length] = labels[batch_index, positions].clamp(min=0)
        packed_mask[batch_index, :length] = True
        packed_confidence[batch_index, :length] = confidence[batch_index, positions]
    return packed_emissions, packed_labels, packed_mask, packed_confidence


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(-100)
    if not valid.any():
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits[valid], labels[valid], reduction="none")
    selected = weights[valid].clamp(min=0.0)
    return (loss * selected).sum() / selected.sum().clamp(min=1e-6)


def main_losses(model: torch.nn.Module, outputs: dict, batch: dict) -> dict[str, torch.Tensor]:
    packed = pack_code(
        outputs["emissions"], batch["keep_labels"], batch["code_mask"], batch["confidence_weights"]
    )
    emissions, tags, mask, confidence = packed
    per_sample = model.compression_head.crf.loss(emissions, tags, mask)
    sample_weights = (confidence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    keep = (per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)
    probability = outputs["document_logprob"].exp().clamp(1e-6, 1 - 1e-6)
    document = F.binary_cross_entropy(probability, batch["document_labels"])
    role_logits = outputs.get("role_logits")
    relation_logits = outputs.get("relation_logits")
    role = (
        weighted_cross_entropy(role_logits, batch["role_labels"], batch["confidence_weights"])
        if role_logits is not None else keep * 0.0
    )
    relation = (
        weighted_cross_entropy(relation_logits, batch["relation_labels"], batch["confidence_weights"])
        if relation_logits is not None else keep * 0.0
    )
    return {"keep": keep, "document": document, "role": role, "relation_line": relation}


def auxiliary_relation_loss(logits: torch.Tensor | None, batch: dict | None, reference: torch.Tensor) -> torch.Tensor:
    if logits is None or batch is None:
        return reference * 0.0
    losses = F.cross_entropy(logits, batch["labels"], reduction="none")
    weights = batch["weights"].clamp(min=0.0)
    return (losses * weights).sum() / weights.sum().clamp(min=1e-6)


def ranking_loss(
    positive: torch.Tensor | None,
    negative: torch.Tensor | None,
    batch: dict | None,
    margin: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if positive is None or negative is None or batch is None:
        return reference * 0.0
    losses = F.relu(margin - positive + negative)
    weights = batch["weights"].clamp(min=0.0)
    return (losses * weights).sum() / weights.sum().clamp(min=1e-6)

