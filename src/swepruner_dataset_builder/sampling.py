from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .deduplication import code_fingerprints
from .io_utils import stable_id
from .labeling import LabelingResult
from .leakage_check import hashes_present
from .python_parser import RepositoryIndex, top_level_block
from .schemas import Block, BlockLabel, RelationType, Role, TaskRecord


def _estimate_tokens(code: str, tokenizer_path: str | None) -> tuple[int, str]:
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            return len(tokenizer.encode(code)), "transformers_local"
        except (ImportError, OSError, ValueError):
            pass
    return max(1, (len(code) + 3) // 4), "char_approximation"


def _line_label(index: RepositoryIndex, result: LabelingResult, file_path: str, line: int) -> tuple[str, int, float, list[str], str]:
    candidates: list[tuple[Block, BlockLabel]] = []
    for block in index.files[file_path].blocks:
        if block.start_line <= line <= block.end_line:
            candidates.append((block, result.labels[block.block_id]))
    rank = {Role.DROP: 0, Role.SUPPORT: 1, Role.CORE: 2}
    if not candidates:
        return Role.DROP, 0, 0.70, ["unmapped_line_drop"], RelationType.NONE
    block, label = max(
        candidates,
        key=lambda item: (rank[Role(item[1].role)], item[1].confidence, -(item[0].end_line - item[0].start_line)),
    )
    return str(label.role), 1 if label.role in {Role.CORE, Role.SUPPORT} else 0, round(label.confidence, 4), list(label.provenance), str(label.relation_type)


def _block_view(block: Block) -> dict:
    return {
        "block_id": block.block_id,
        "file_path": block.relative_file_path,
        "symbol": block.symbol,
        "start_line": block.start_line,
        "end_line": block.end_line,
        "code": block.source_text,
    }


def _nearest_drop_unit(
    index: RepositoryIndex,
    result: LabelingResult,
    anchor_unit: Block,
) -> Block | None:
    units: dict[str, Block] = {}
    parsed = index.files[anchor_unit.relative_file_path]
    for block in parsed.blocks:
        label = result.labels.get(block.block_id)
        if not label or label.role != Role.DROP:
            continue
        unit = top_level_block(index, block)
        if unit.block_id != anchor_unit.block_id:
            units[unit.block_id] = unit
    if not units:
        return None
    return min(
        units.values(),
        key=lambda unit: (
            min(abs(unit.end_line - anchor_unit.start_line), abs(unit.start_line - anchor_unit.end_line)),
            unit.start_line,
            unit.block_id,
        ),
    )


def _core_centered_windows(anchor_unit: Block, core_blocks: list[Block], max_lines: int) -> list[tuple[int, int]]:
    if anchor_unit.end_line - anchor_unit.start_line + 1 <= max_lines:
        return [(anchor_unit.start_line, anchor_unit.end_line)]
    spans = sorted(
        (max(anchor_unit.start_line, block.start_line), min(anchor_unit.end_line, block.end_line))
        for block in core_blocks
        if block.end_line >= anchor_unit.start_line and block.start_line <= anchor_unit.end_line
    )
    clusters: list[list[int]] = []
    for start, end in spans:
        if end < start:
            continue
        if clusters and max(clusters[-1][1], end) - clusters[-1][0] + 1 <= max_lines:
            clusters[-1][1] = max(clusters[-1][1], end)
        else:
            clusters.append([start, end])
    windows: list[tuple[int, int]] = []
    for core_start, core_end in clusters:
        if core_end - core_start + 1 > max_lines:
            core_end = min(anchor_unit.end_line, core_start + max_lines - 1)
        spare = max_lines - (core_end - core_start + 1)
        start = max(anchor_unit.start_line, core_start - spare // 2)
        end = min(anchor_unit.end_line, start + max_lines - 1)
        start = max(anchor_unit.start_line, end - max_lines + 1)
        window = (start, end)
        if window not in windows:
            windows.append(window)
    return windows


def build_task_outputs(
    task: TaskRecord,
    index: RepositoryIndex,
    result: LabelingResult,
    config: dict,
    seed: int,
    tokenizer_path: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    budget = config["budget"]
    sampling = config["sampling"]
    samples: list[dict] = []
    relations: list[dict] = []
    rankings: list[dict] = []
    max_variants = min(int(sampling["max_samples_per_task"]), int(budget["hard_negative_count"]))
    for anchor_symbol_id in result.anchor_symbol_ids:
        symbol = index.symbol(anchor_symbol_id)
        if not symbol:
            continue
        symbol_block = index.block(symbol.block_id)
        if not symbol_block:
            continue
        anchor_unit = top_level_block(index, symbol_block)
        core_blocks = [index.block(block_id) for block_id in result.core_block_ids]
        core_blocks = [block for block in core_blocks if block and block.enclosing_symbol_id == anchor_symbol_id]
        negatives = result.hard_negatives.get(anchor_symbol_id, [])
        variants = [*negatives, None]
        seen_windows: set[tuple[int, int, str]] = set()
        anchor_windows = _core_centered_windows(
            anchor_unit,
            core_blocks,
            int(budget["max_lines"]),
        )
        for base_start, base_end in anchor_windows:
          for negative in variants:
            if len(samples) >= max_variants:
                break
            negative_block = index.block(negative.block_id) if negative else None
            negative_unit = top_level_block(index, negative_block) if negative_block else None
            if negative is None and (base_start, base_end) == (anchor_unit.start_line, anchor_unit.end_line):
                negative_unit = _nearest_drop_unit(index, result, anchor_unit)
            if negative_unit and negative_unit.relative_file_path != anchor_unit.relative_file_path:
                continue
            start = min(base_start, negative_unit.start_line if negative_unit else base_start)
            end = max(base_end, negative_unit.end_line if negative_unit else base_end)
            parsed = index.files[anchor_unit.relative_file_path]
            selected_lines = list(parsed.lines[start - 1:end])
            while selected_lines and selected_lines[-1] == "":
                selected_lines.pop()
            if not selected_lines:
                continue
            end = start + len(selected_lines) - 1
            window_key = (start, end, negative.negative_type if negative else "easy")
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)
            code = "\n".join(selected_lines)
            if (
                hashes_present(code, result.added_fragment_hashes)
                or hashes_present(task.issue_text, result.added_fragment_hashes)
            ):
                continue
            estimated_tokens, tokenizer_mode = _estimate_tokens(code, tokenizer_path)
            if (
                end - start + 1 > int(budget["max_lines"])
                or len(code) > int(budget["max_chars"])
                or estimated_tokens > int(budget["max_input_tokens"])
            ):
                continue
            line_numbers = list(range(start, end + 1))
            line_values = [_line_label(index, result, anchor_unit.relative_file_path, line) for line in line_numbers]
            roles = [value[0] for value in line_values]
            if Role.CORE not in roles or Role.DROP not in roles or roles.count(Role.CORE) < int(budget["min_core_lines"]):
                continue
            sample_id = stable_id("sample", task.dataset_source, task.task_id, anchor_symbol_id, start, end,
                                  negative.block_id if negative else "none", seed)
            selected_blocks = [
                block for block in parsed.blocks if start <= block.start_line and block.end_line <= end
            ]
            metadata = {
                "core_block_ids": [block.block_id for block in selected_blocks if result.labels[block.block_id].role == Role.CORE],
                "support_block_ids": [block.block_id for block in selected_blocks if result.labels[block.block_id].role == Role.SUPPORT],
                "drop_block_ids": [block.block_id for block in selected_blocks if result.labels[block.block_id].role == Role.DROP],
                "hard_negative_types": [negative.negative_type] if negative else [],
                "hard_negative": asdict(negative) if negative else None,
                "tokenizer_mode": tokenizer_mode,
                "estimated_tokens": estimated_tokens,
                "api_used": any("api_" in item for value in line_values for item in value[3]),
                "base_commit": task.base_commit,
                "patch_hash": result.patch_hash,
                "builder_version": config.get("builder_version", "0.1.0"),
                "input_revision": "buggy_base",
                "patch_mapping_verified": result.patch_mapping_verified,
                "patch_old_locations": result.patch_old_locations,
                "forbidden_added_hashes": result.added_fragment_hashes,
                "core_line_numbers": [line for line, role in zip(line_numbers, roles) if role == Role.CORE],
                "block_spans": [
                    {"block_id": block.block_id, "start_line": block.start_line, "end_line": block.end_line,
                     "role": str(result.labels[block.block_id].role)} for block in selected_blocks
                ],
            }
            sample = {
                "sample_id": sample_id,
                "task_id": task.task_id,
                "dataset_source": task.dataset_source,
                "query": task.issue_text,
                "repo_name": task.repo_name,
                "file_path": anchor_unit.relative_file_path,
                "anchor_symbol": symbol.qualname,
                "code": code,
                "line_numbers": line_numbers,
                "line_roles": roles,
                "line_keep_labels": [value[1] for value in line_values],
                "line_confidences": [value[2] for value in line_values],
                "line_provenance": [value[3] for value in line_values],
                "line_relation_types": [value[4] for value in line_values],
                "document_label": 1,
                "metadata": metadata,
            }
            metadata["dedup_hashes"] = code_fingerprints(code)
            samples.append(sample)
            anchor = core_blocks[0] if core_blocks else symbol_block
            relevant = []
            for relation in result.relations:
                if relation.source_block_id in result.core_block_ids:
                    candidate_id = relation.target_block_id
                elif relation.target_block_id in result.core_block_ids:
                    candidate_id = relation.source_block_id
                else:
                    continue
                candidate = index.block(candidate_id)
                if not candidate:
                    continue
                relevant.append((relation, candidate))
                label = result.labels[candidate_id]
                relations.append({
                    "sample_id": sample_id, "task_id": task.task_id, "dataset_source": task.dataset_source,
                    "query": task.issue_text, "anchor_block": _block_view(anchor),
                    "candidate_block": _block_view(candidate), "relation": str(relation.relation),
                    "candidate_role": str(label.role), "confidence": label.confidence,
                    "provenance": label.provenance, "graph_distance": relation.graph_distance,
                    "api_reviewed": any(item.startswith("api_") for item in label.provenance),
                })
            if negative_block:
                relations.append({
                    "sample_id": sample_id, "task_id": task.task_id, "dataset_source": task.dataset_source,
                    "query": task.issue_text, "anchor_block": _block_view(anchor),
                    "candidate_block": _block_view(negative_block), "relation": RelationType.NONE,
                    "candidate_role": Role.DROP, "confidence": config["confidence"].get("lexical_hard_negative", 0.75),
                    "provenance": [f"hard_negative:{negative.negative_type}"], "graph_distance": negative.graph_distance,
                    "api_reviewed": negative.api_reviewed,
                })
                positives = [(rel, candidate) for rel, candidate in relevant if result.labels[candidate.block_id].role == Role.SUPPORT]
                if positives:
                    positive_relation, positive = max(positives, key=lambda item: result.labels[item[1].block_id].confidence)
                    rankings.append({
                        "sample_id": sample_id, "task_id": task.task_id, "dataset_source": task.dataset_source,
                        "query": task.issue_text, "anchor_block": _block_view(anchor),
                        "positive_block": _block_view(positive), "hard_negative_block": _block_view(negative_block),
                        "positive_relation": str(positive_relation.relation), "negative_type": negative.negative_type,
                        "positive_confidence": result.labels[positive.block_id].confidence,
                        "negative_confidence": config["confidence"].get("lexical_hard_negative", 0.75),
                    })
            if len(samples) >= max_variants:
                break
    return samples, relations, rankings


def build_document_negative(task: TaskRecord, index: RepositoryIndex, config: dict, seed: int) -> dict | None:
    patch_files = set()
    for text in (task.mutation_patch, task.patch):
        if text:
            from .patch_parser import parse_unified_diff
            patch_files.update(parse_unified_diff(text).files)
    choices = [parsed for path, parsed in sorted(index.files.items()) if path not in patch_files and not parsed.syntax_error and parsed.lines]
    if not choices:
        return None
    parsed = choices[0]
    max_lines = int(config["budget"]["max_lines"])
    lines = list(parsed.lines[:max_lines])
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    code = "\n".join(lines)
    count = len(lines)
    estimated, mode = _estimate_tokens(code, None)
    sample_id = stable_id("docneg", task.dataset_source, task.task_id, parsed.relative_path, seed)
    return {
        "sample_id": sample_id, "task_id": task.task_id, "dataset_source": task.dataset_source,
        "query": task.issue_text, "repo_name": task.repo_name, "file_path": parsed.relative_path,
        "anchor_symbol": "", "code": code, "line_numbers": list(range(1, count + 1)),
        "line_roles": [Role.DROP] * count, "line_keep_labels": [0] * count,
        "line_confidences": [0.90] * count, "line_provenance": [["document_negative:unrelated_file"] for _ in lines],
        "line_relation_types": [RelationType.NONE] * count, "document_label": 0,
        "metadata": {
            "core_block_ids": [], "support_block_ids": [], "drop_block_ids": [],
            "hard_negative_types": [], "negative_document_type": "same_repo_unrelated_file",
            "tokenizer_mode": mode, "estimated_tokens": estimated, "api_used": False,
            "base_commit": task.base_commit, "patch_hash": "", "builder_version": config.get("builder_version", "0.1.0"),
            "input_revision": "buggy_base", "dedup_hashes": code_fingerprints(code),
        },
    }
