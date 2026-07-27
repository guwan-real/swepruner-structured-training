from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TrainConfig


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


def _token_keep_losses(
    outputs: dict,
    batch: dict,
    reference: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = batch["code_mask"] & batch["keep_labels"].ne(-100)
    if not valid.any():
        zero = reference * 0.0
        return zero, zero, zero
    logits = outputs["emissions"].float()
    labels = batch["keep_labels"]
    confidence = batch["confidence_weights"].float()
    selected_labels = labels[valid]
    counts = torch.bincount(selected_labels, minlength=2).float().clamp(min=1)
    class_weights = (selected_labels.numel() / (2.0 * counts)).clamp(max=config.max_token_class_weight)
    per_token = F.cross_entropy(logits[valid], selected_labels, weight=class_weights, reduction="none")
    selected_confidence = confidence[valid].clamp(min=0)
    token_ce = (per_token * selected_confidence).sum() / selected_confidence.sum().clamp(min=1e-6)

    probabilities = logits.softmax(dim=-1)[..., 1]
    retention_values = []
    catastrophic_values = []
    for batch_index in range(logits.shape[0]):
        sample_valid = valid[batch_index]
        weights = confidence[batch_index][sample_valid].clamp(min=0)
        denominator = weights.sum().clamp(min=1e-6)
        predicted_ratio = (probabilities[batch_index][sample_valid] * weights).sum() / denominator
        target_ratio = (labels[batch_index][sample_valid].float() * weights).sum() / denominator
        retention_values.append(F.smooth_l1_loss(predicted_ratio, target_ratio))
        floor = batch["minimum_keep_ratios"][batch_index].to(predicted_ratio.device)
        catastrophic_values.append(F.relu(floor - predicted_ratio).square())
    return token_ce, torch.stack(retention_values).mean(), torch.stack(catastrophic_values).mean()


def main_losses(
    model: torch.nn.Module,
    outputs: dict,
    batch: dict,
    config: TrainConfig,
) -> dict[str, torch.Tensor]:
    packed = pack_code(
        outputs["emissions"], batch["keep_labels"], batch["code_mask"], batch["confidence_weights"]
    )
    emissions, tags, mask, confidence = packed
    per_sample = model.compression_head.crf.loss(emissions, tags, mask)
    sample_weights = (confidence * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    keep_crf = (per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1e-6)
    token_ce, retention, catastrophic = _token_keep_losses(outputs, batch, keep_crf, config)
    keep = (
        config.crf_keep_weight * keep_crf
        + config.token_ce_weight * token_ce
        + config.retention_loss_weight * retention
        + config.catastrophic_loss_weight * catastrophic
    )
    probability = outputs["document_logprob"].exp().clamp(1e-6, 1 - 1e-6)
    # Probability-form BCE is intentionally blocked by CUDA autocast. Keep the
    # existing objective, but evaluate this numerically sensitive term in FP32.
    with torch.autocast(device_type=probability.device.type, enabled=False):
        document = F.binary_cross_entropy(
            probability.float(), batch["document_labels"].float()
        )
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
    return {
        "keep": keep,
        "keep_crf": keep_crf,
        "keep_token_ce": token_ce,
        "keep_retention": retention,
        "keep_catastrophic": catastrophic,
        "document": document,
        "role": role,
        "relation_line": relation,
    }


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
