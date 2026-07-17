from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .hard_negatives import mine_hard_negatives
from .io_utils import stable_hash
from .leakage_check import added_line_hashes
from .patch_parser import ParsedPatch, parse_unified_diff
from .python_parser import RepositoryIndex, smallest_block_for_line
from .schemas import Block, BlockLabel, HardNegative, Relation, RelationType, Role, TaskRecord
from .traceback_parser import project_frames


@dataclass
class LabelingResult:
    labels: dict[str, BlockLabel]
    core_block_ids: list[str]
    anchor_symbol_ids: list[str]
    hard_negatives: dict[str, list[HardNegative]]
    patch_hash: str
    patch_old_locations: list[dict]
    added_fragment_hashes: list[str]
    patch_mapping_verified: bool
    relations: list[Relation] = field(default_factory=list)


def _confidence_for_relation(relation: str, confidence: dict[str, float]) -> float:
    mapping = {
        RelationType.DEF_USE: "direct_local_def_use",
        RelationType.CONTROL: "direct_control_dependency",
        RelationType.EXCEPTION: "exception_pair",
        RelationType.IMPORT: "import_dependency",
        RelationType.TYPE: "type_dependency",
        RelationType.ATTRIBUTE: "attribute_dependency",
        RelationType.CALL: "one_hop_direct_call",
        RelationType.CALLED_BY: "one_hop_caller",
        RelationType.INHERITANCE: "inheritance_or_override",
        RelationType.OVERRIDE: "inheritance_or_override",
        RelationType.DECORATOR: "import_dependency",
    }
    return float(confidence.get(mapping.get(RelationType(relation), "one_hop_direct_call"), 0.70))


def _resolve_file(index: RepositoryIndex, path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized in index.files:
        return normalized
    matches = [known for known in index.files if known.endswith("/" + normalized) or normalized.endswith("/" + known)]
    return matches[0] if len(matches) == 1 else ""


def _map_evidence(index: RepositoryIndex, parsed: ParsedPatch) -> tuple[list[tuple[Block, object]], bool]:
    mapped: list[tuple[Block, object]] = []
    verified = True
    for evidence in parsed.evidence:
        file_path = _resolve_file(index, evidence.file_path)
        source = index.files.get(file_path)
        if not source or source.syntax_error:
            verified = False
            continue
        line = evidence.old_line
        if evidence.kind == "deleted" and evidence.old_text.strip():
            current = source.lines[line - 1].strip() if 0 < line <= len(source.lines) else ""
            if current != evidence.old_text.strip():
                matches = [i + 1 for i, value in enumerate(source.lines) if value.strip() == evidence.old_text.strip()]
                if len(matches) == 1:
                    line = matches[0]
                else:
                    verified = False
                    continue
        line = min(max(1, line), max(1, len(source.lines)))
        block = smallest_block_for_line(index, file_path, line)
        if block:
            mapped.append((block, evidence))
        else:
            verified = False
    return mapped, verified


def label_task(task: TaskRecord, index: RepositoryIndex, confidence: dict[str, float]) -> LabelingResult:
    labels = {
        block_id: BlockLabel(Role.DROP, float(confidence.get("easy_negative", 0.70)), ["static_default_drop"], RelationType.NONE)
        for block_id in index.blocks
    }
    patch_texts = [text for text in (task.mutation_patch, task.patch) if text.strip()]
    parsed_patches = [
        parse_unified_diff(text, "mutation_original_line" if text == task.mutation_patch and task.mutation_patch else "patch_old_line")
        for text in patch_texts
    ]
    core_ids: set[str] = set()
    old_locations: list[dict] = []
    all_verified = True
    for parsed in parsed_patches:
        mapped, verified = _map_evidence(index, parsed)
        all_verified = all_verified and verified
        for block, evidence in mapped:
            if evidence.kind == "addition_context":
                score = float(confidence.get("patch_addition_nearest_context", 0.90))
                provenance = "patch_addition_nearest_context"
            else:
                key = "mutation_original_line" if evidence.provenance == "mutation_original_line" else "patch_deleted_or_modified_line"
                score = float(confidence.get(key, 1.0))
                provenance = evidence.provenance
            labels[block.block_id].apply(Role.CORE, score, provenance, RelationType.PATCH)
            labels[block.block_id].apply(Role.CORE, score, "ast_block_expansion", RelationType.PATCH)
            core_ids.add(block.block_id)
            old_locations.append({
                "file_path": block.relative_file_path,
                "old_line": evidence.old_line,
                "kind": evidence.kind,
                "hunk_header": evidence.hunk_header,
            })
    frames = project_frames(task.traceback, task.repo_path, set(index.files))
    for frame_index, frame in enumerate(frames):
        block = smallest_block_for_line(index, frame.file_path, frame.line)
        if not block:
            continue
        if frame_index == len(frames) - 1:
            labels[block.block_id].apply(
                Role.CORE, float(confidence.get("traceback_final_project_frame", 0.95)),
                "traceback_final_project_frame", RelationType.TRACEBACK,
            )
            core_ids.add(block.block_id)
        else:
            labels[block.block_id].apply(
                Role.SUPPORT, float(confidence.get("dynamic_traceback_neighbor", 0.85)),
                "dynamic_traceback_neighbor", RelationType.TRACEBACK,
            )
    anchor_symbols = sorted({index.blocks[block_id].enclosing_symbol_id for block_id in core_ids if block_id in index.blocks})
    frontier = set(core_ids)
    for symbol_id in anchor_symbols:
        symbol = index.symbol(symbol_id)
        if symbol:
            signature = index.block(symbol.signature_block_id)
            if signature:
                labels[signature.block_id].apply(
                    Role.SUPPORT, float(confidence.get("function_signature", 0.85)),
                    "function_signature", RelationType.TYPE,
                )
                frontier.add(signature.block_id)
    relevant_relations: list[Relation] = []
    for relation in index.relations:
        candidate_id = ""
        if relation.source_block_id in frontier:
            candidate_id = relation.target_block_id
        elif relation.target_block_id in frontier:
            candidate_id = relation.source_block_id
        if not candidate_id or candidate_id in core_ids or candidate_id not in labels:
            continue
        score = _confidence_for_relation(str(relation.relation), confidence)
        provenance = relation.provenance[0] if relation.provenance else str(relation.relation).lower()
        labels[candidate_id].apply(Role.SUPPORT, score, provenance, str(relation.relation))
        relevant_relations.append(relation)
    hard: dict[str, list[HardNegative]] = {}
    for symbol_id in anchor_symbols:
        hard[symbol_id] = mine_hard_negatives(index, symbol_id, labels, task.issue_text)
    combined_patch = "\n".join(patch_texts)
    return LabelingResult(
        labels=labels,
        core_block_ids=sorted(core_ids),
        anchor_symbol_ids=anchor_symbols,
        hard_negatives=hard,
        patch_hash=stable_hash(combined_patch, 64),
        patch_old_locations=old_locations,
        added_fragment_hashes=added_line_hashes(combined_patch),
        patch_mapping_verified=all_verified and bool(core_ids),
        relations=relevant_relations,
    )

