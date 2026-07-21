from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from .checkpoint import load_official_initialization, save_model_checkpoint, write_json
from .config import TrainConfig, load_train_config
from .data import (
    BatchEncoder,
    MainDataset,
    OFFICIAL_SOURCE,
    OffsetJsonlDataset,
    ReplayDistributedSampler,
    ROLE_NAMES,
    ShardedSequentialSampler,
    load_split_ids,
)
from .losses import auxiliary_relation_loss, main_losses, ranking_loss
from .model import StructuralPrunerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M1 data-only or M2 structural SWE-Pruner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--backbone-config-only", action="store_true")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def setup_distributed(allow_cpu: bool) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for tiny debugging")
    return rank, world_size, local_rank, device


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def next_batch(iterator: Any, loader: DataLoader) -> tuple[dict, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def make_scheduler(optimizer: torch.optim.Optimizer, warmup: int, total: int):
    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def unwrap(model: torch.nn.Module) -> StructuralPrunerModel:
    return model.module if isinstance(model, DistributedDataParallel) else model


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def reduce_counts(counts: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return counts


def parameter_gradient_norm(parameters: list[torch.nn.Parameter], device: torch.device) -> float:
    squared = torch.zeros((), dtype=torch.float32, device=device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
    return float(squared.sqrt().item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    structural: bool,
) -> dict[str, Any]:
    model.eval()
    # keep_tp, keep_fp, keep_fn, keep_total, predicted_keep, core_tp, core_total,
    # support_tp, support_total, drop_pred_correct, prune_pred_total,
    # hard_keep, hard_total, document_correct, document_total, relation_correct, relation_total
    counts = torch.zeros(16, dtype=torch.float64, device=device)
    curve_thresholds = [index / 20 for index in range(1, 20)]
    curve = torch.zeros((len(curve_thresholds), 4), dtype=torch.float64, device=device)
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        token_probability = outputs["emissions"].softmax(dim=-1)[:, :, 1]
        document_prediction = outputs["document_logprob"].exp().ge(0.5)
        counts[13] += document_prediction.eq(batch["document_labels"].bool()).sum()
        counts[14] += batch["document_labels"].numel()
        if structural and outputs["relation_logits"] is not None:
            valid_relation = batch["relation_labels"].ne(-100)
            counts[14] += 0  # keep fixed metric slots stable
            counts[15] += outputs["relation_logits"].argmax(-1)[valid_relation].eq(batch["relation_labels"][valid_relation]).sum()
        for batch_index in range(batch["input_ids"].shape[0]):
            valid = batch["code_mask"][batch_index]
            line_ids = batch["line_indices"][batch_index][valid]
            probabilities = token_probability[batch_index][valid]
            keep_labels = batch["keep_labels"][batch_index][valid]
            role_labels = batch["role_labels"][batch_index][valid]
            hard_flags = batch["hard_negative_mask"][batch_index][valid]
            for line_id in line_ids.unique(sorted=True):
                selected = line_ids.eq(line_id)
                probability = probabilities[selected].mean()
                target = keep_labels[selected][0].bool()
                role = int(role_labels[selected][0].item())
                hard = bool(hard_flags[selected].any().item())
                prediction = probability >= threshold
                counts[0] += prediction and target
                counts[1] += prediction and not target
                counts[2] += (not prediction) and target
                counts[3] += 1
                counts[4] += prediction
                if role == 2:
                    counts[5] += prediction
                    counts[6] += 1
                if role == 1:
                    counts[7] += prediction
                    counts[8] += 1
                if role == 0 and not prediction:
                    counts[9] += 1
                if role >= 0 and not prediction:
                    counts[10] += 1
                if hard:
                    counts[11] += prediction
                    counts[12] += 1
                for curve_index, value in enumerate(curve_thresholds):
                    curve_prediction = probability >= value
                    curve[curve_index, 0] += curve_prediction
                    curve[curve_index, 1] += 1
                    if role == 2:
                        curve[curve_index, 2] += curve_prediction
                        curve[curve_index, 3] += 1
    reduce_counts(counts)
    if dist.is_initialized():
        dist.all_reduce(curve, op=dist.ReduceOp.SUM)
    precision = counts[0] / (counts[0] + counts[1]).clamp(min=1)
    recall = counts[0] / (counts[0] + counts[2]).clamp(min=1)
    curve_records = [
        {
            "threshold": value,
            "keep_ratio": float((row[0] / row[1].clamp(min=1)).item()),
            "core_recall": float((row[2] / row[3].clamp(min=1)).item()),
        }
        for value, row in zip(curve_thresholds, curve)
    ]
    metrics = {
        "line_keep_f1": float((2 * precision * recall / (precision + recall).clamp(min=1e-12)).item()),
        "line_keep_precision": float(precision.item()),
        "line_keep_recall": float(recall.item()),
        "core_recall": float((counts[5] / counts[6].clamp(min=1)).item()),
        "support_recall": float((counts[7] / counts[8].clamp(min=1)).item()),
        "drop_precision": float((counts[9] / counts[10].clamp(min=1)).item()),
        "hard_negative_false_keep_rate": float((counts[11] / counts[12].clamp(min=1)).item()),
        "predicted_keep_ratio": float((counts[4] / counts[3].clamp(min=1)).item()),
        "document_accuracy": float((counts[13] / counts[14].clamp(min=1)).item()),
        "threshold": threshold,
        "threshold_curve": curve_records,
    }
    for target_ratio in (0.50, 0.55):
        nearest = min(curve_records, key=lambda row: abs(row["keep_ratio"] - target_ratio))
        suffix = int(target_ratio * 100)
        metrics[f"core_recall_at_keep_ratio_{suffix}"] = nearest["core_recall"]
        metrics[f"actual_keep_ratio_at_{suffix}"] = nearest["keep_ratio"]
        metrics[f"threshold_at_keep_ratio_{suffix}"] = nearest["threshold"]
    model.train()
    return metrics


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config, args.overrides)
    active_objectives = [name for name, weight in config.loss_weights.items() if weight > 0]
    rank, world_size, local_rank, device = setup_distributed(args.allow_cpu)
    seed_everything(config.seed, rank)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    train_ids = load_split_ids(data_root, "train")
    validation_ids = load_split_ids(data_root, "validation")
    main_path = data_root / "combined" / "pruning_sft.jsonl"
    train_dataset = MainDataset(main_path, train_ids)
    validation_dataset = MainDataset(main_path, validation_ids)
    relation_paths = [data_root / source / "block_relation.jsonl" for source in ("swe_smith", "swe_gym")]
    ranking_paths = [data_root / source / "block_ranking.jsonl" for source in ("swe_smith", "swe_gym")]
    relation_dataset = OffsetJsonlDataset(relation_paths, train_ids, "relation")
    ranking_dataset = OffsetJsonlDataset(ranking_paths, train_ids, "negative_type")
    relation_values = {"NONE"}
    for row in train_dataset.rows:
        if row["dataset_source"] != OFFICIAL_SOURCE:
            relation_values.update(map(str, row["line_relation_types"]))
    relation_values.update(relation_dataset.label_values)
    relation_names = tuple(sorted(relation_values))
    role_to_id = {name: index for index, name in enumerate(ROLE_NAMES)}
    relation_to_id = {name: index for index, name in enumerate(relation_names)}
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "right"
    encoder = BatchEncoder(
        tokenizer, config.max_length, config.aux_max_length, config.instruction, role_to_id, relation_to_id
    )
    model = StructuralPrunerModel(
        args.backbone_path,
        tokenizer,
        config,
        len(ROLE_NAMES),
        len(relation_names),
        args.backbone_config_only,
    )
    init_report = load_official_initialization(model, args.init_checkpoint)
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    train_sampler = ReplayDistributedSampler(
        train_dataset, config.samples_per_epoch, config.replay_ratio, config.seed, rank, world_size
    )
    validation_sampler = ShardedSequentialSampler(len(validation_dataset), rank, world_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        sampler=train_sampler,
        collate_fn=encoder.collate_main,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.per_device_batch_size,
        sampler=validation_sampler,
        collate_fn=encoder.collate_main,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    relation_loader = ranking_loader = None
    relation_sampler = ranking_sampler = None
    if config.objective_enabled("relation"):
        relation_sampler = DistributedSampler(relation_dataset, world_size, rank, shuffle=True, seed=config.seed)
        relation_loader = DataLoader(
            relation_dataset, batch_size=config.aux_batch_size, sampler=relation_sampler,
            collate_fn=encoder.collate_relation, num_workers=config.num_workers, pin_memory=True,
        )
    if config.objective_enabled("rank"):
        ranking_sampler = DistributedSampler(ranking_dataset, world_size, rank, shuffle=True, seed=config.seed + 1)
        ranking_loader = DataLoader(
            ranking_dataset, batch_size=config.aux_batch_size, sampler=ranking_sampler,
            collate_fn=encoder.collate_ranking, num_workers=config.num_workers, pin_memory=True,
        )
    core_model = unwrap(model)
    backbone_ids = {id(parameter) for parameter in core_model.backbone.parameters()}
    backbone_parameters = [parameter for parameter in core_model.parameters() if id(parameter) in backbone_ids and parameter.requires_grad]
    head_parameters = [parameter for parameter in core_model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    gradient_groups = {
        "backbone": backbone_parameters,
        "fusion": [
            parameter
            for module in (core_model.fusion_layers, core_model.fusion_norms)
            for parameter in module.parameters()
            if parameter.requires_grad
        ],
        "keep_head": [parameter for parameter in core_model.compression_head.parameters() if parameter.requires_grad],
        "role_head": (
            [parameter for parameter in core_model.role_head.parameters() if parameter.requires_grad]
            if core_model.role_head is not None else []
        ),
        "relation_head": (
            [parameter for parameter in core_model.relation_head.parameters() if parameter.requires_grad]
            if core_model.relation_head is not None else []
        ),
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.learning_rate_backbone, "weight_decay": config.weight_decay},
            {"params": head_parameters, "lr": config.learning_rate_heads, "weight_decay": 0.0},
        ]
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    scheduler = make_scheduler(optimizer, int(total_optimizer_steps * config.warmup_ratio), total_optimizer_steps)
    if rank == 0:
        run_manifest = {
            "config": config.to_dict(),
            "active_objectives": active_objectives,
            "world_size": world_size,
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
            "relation_rows": len(relation_dataset),
            "ranking_rows": len(ranking_dataset),
            "role_names": ROLE_NAMES,
            "relation_names": relation_names,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "official_initialization": init_report,
        }
        write_json(output_dir / "run_manifest.json", run_manifest)
    best_f1 = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        if relation_sampler is not None:
            relation_sampler.set_epoch(epoch)
        if ranking_sampler is not None:
            ranking_sampler.set_epoch(epoch)
        relation_iterator = iter(relation_loader) if relation_loader is not None else None
        ranking_iterator = iter(ranking_loader) if ranking_loader is not None else None
        model.train()
        started = time.time()
        sums = {name: 0.0 for name in ("total", "keep", "role", "relation", "rank", "document")}
        gradient_sums = {name: 0.0 for name in gradient_groups}
        gradient_samples = 0
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{config.epochs}",
            disable=rank != 0,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for micro_step, main_batch in enumerate(progress):
            main_batch = move(main_batch, device)
            relation_batch = ranking_batch = None
            if relation_loader is not None and micro_step % config.relation_batch_every == 0:
                relation_batch, relation_iterator = next_batch(relation_iterator, relation_loader)
                relation_batch = move(relation_batch, device)
            if ranking_loader is not None and micro_step % config.ranking_batch_every == 0:
                ranking_batch, ranking_iterator = next_batch(ranking_iterator, ranking_loader)
                ranking_batch = move(ranking_batch, device)
            kwargs = {}
            if relation_batch is not None:
                kwargs.update(relation_input_ids=relation_batch["input_ids"], relation_attention_mask=relation_batch["attention_mask"])
            if ranking_batch is not None:
                kwargs.update(
                    positive_input_ids=ranking_batch["positive_input_ids"],
                    positive_attention_mask=ranking_batch["positive_attention_mask"],
                    negative_input_ids=ranking_batch["negative_input_ids"],
                    negative_attention_mask=ranking_batch["negative_attention_mask"],
                )
            update_now = (micro_step + 1) % config.gradient_accumulation_steps == 0 or micro_step + 1 == len(train_loader)
            sync_context = contextlib.nullcontext() if update_now or not isinstance(model, DistributedDataParallel) else model.no_sync()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model(input_ids=main_batch["input_ids"], attention_mask=main_batch["attention_mask"], **kwargs)
                    losses = main_losses(core_model, outputs, main_batch)
                    relation_aux = auxiliary_relation_loss(outputs["aux_relation_logits"], relation_batch, losses["keep"])
                    relation = 0.5 * (losses["relation_line"] + relation_aux)
                    rank_loss = ranking_loss(
                        outputs["positive_scores"], outputs["negative_scores"], ranking_batch,
                        config.ranking_margin, losses["keep"],
                    )
                    weighted = (
                        config.loss_weights["keep"] * losses["keep"]
                        + config.loss_weights["role"] * losses["role"]
                        + config.loss_weights["relation"] * relation
                        + config.loss_weights["rank"] * rank_loss
                        + config.loss_weights["document"] * losses["document"]
                    )
                    scaled = weighted / config.gradient_accumulation_steps
                scaled.backward()
            values = {
                "total": weighted, "keep": losses["keep"], "role": losses["role"],
                "relation": relation, "rank": rank_loss, "document": losses["document"],
            }
            for name, value in values.items():
                sums[name] += float(value.detach().item())
            if update_now:
                next_global_step = global_step + 1
                log_gradients = config.gradient_log_every > 0 and (
                    next_global_step == 1 or next_global_step % config.gradient_log_every == 0
                )
                if log_gradients:
                    for name, parameters in gradient_groups.items():
                        gradient_sums[name] += parameter_gradient_norm(parameters, device)
                    gradient_samples += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if rank == 0:
                progress.set_postfix(
                    step=global_step,
                    loss=f"{float(weighted.detach().item()):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
        if rank == 0:
            print(
                f"epoch {epoch + 1}/{config.epochs} training complete; evaluating validation split...",
                flush=True,
            )
        metrics = evaluate(model, validation_loader, device, config.threshold, config.structural_heads)
        train_metrics = {f"train_{name}": value / max(1, len(train_loader)) for name, value in sums.items()}
        if gradient_samples:
            train_metrics.update({
                f"grad_{name}_norm": value / gradient_samples
                for name, value in gradient_sums.items()
            })
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "seconds": time.time() - started,
            "active_objectives": active_objectives,
            **train_metrics,
            **metrics,
        }
        if rank == 0:
            append_jsonl(output_dir / "metrics.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            metadata = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "metrics": metrics,
                "config": config.to_dict(),
                "role_names": ROLE_NAMES,
                "relation_names": relation_names,
                "official_initialization": init_report,
            }
            save_model_checkpoint(output_dir / "last_model.pt", core_model, metadata)
            if metrics["line_keep_f1"] > best_f1:
                best_f1 = metrics["line_keep_f1"]
                save_model_checkpoint(output_dir / "best_model.pt", core_model, metadata)
                write_json(output_dir / "best_metrics.json", metrics)
        if dist.is_initialized():
            dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
