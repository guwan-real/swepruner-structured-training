from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .api_cache import ApiCache
from .api_provider import CachedLLMClient, OpenAICompatibleProvider, review_candidate
from .config import load_blacklist
from .io_utils import read_json, read_jsonl, stable_hash, write_json, write_jsonl
from .issue_generation import generate_issue
from .labeling import LabelingResult, label_task
from .leakage_check import check_leakage
from .patch_parser import parse_unified_diff
from .python_parser import RepositoryIndex, build_repository_index
from .reporting import build_report
from .review import write_review_samples
from .sampling import build_document_negative, build_task_outputs
from .schemas import BuildFailure, Role, TaskRecord
from .splitting import split_samples, write_split_manifests
from .task_adapters import load_task_records, normalize_swepruner_rows


def _task_state_path(output: Path, task: TaskRecord, fingerprint: str) -> Path:
    return output / ".state" / f"{stable_hash([task.task_id, fingerprint], 48)}.json"


def _buggy_context(task: TaskRecord, index: RepositoryIndex) -> str:
    for patch in (task.mutation_patch, task.patch):
        parsed = parse_unified_diff(patch)
        for evidence in parsed.evidence:
            source = index.files.get(evidence.file_path)
            if source:
                start = max(0, evidence.old_line - 8)
                end = min(len(source.lines), evidence.old_line + 7)
                return "\n".join(source.lines[start:end])
    return ""


def _priority_paths(task: TaskRecord) -> list[str]:
    paths: set[str] = set()
    for patch in (task.mutation_patch, task.patch):
        for evidence in parse_unified_diff(patch).evidence:
            if evidence.file_path.endswith(".py"):
                paths.add(evidence.file_path)
    return sorted(paths)


def _api_review(
    task: TaskRecord,
    index: RepositoryIndex,
    labeling: LabelingResult,
    client: CachedLLMClient,
    config: dict,
) -> list[dict]:
    api = config["api"]
    primary_model = os.environ.get("LLM_MODEL_PRIMARY", str(api["model_primary"]))
    reviewer_model = os.environ.get("LLM_MODEL_REVIEWER", str(api["model_reviewer"]))
    core = index.block(labeling.core_block_ids[0]) if labeling.core_block_ids else None
    if not core:
        return []
    candidates = []
    for block_id, label in labeling.labels.items():
        if label.role == Role.CORE or label.confidence >= float(api["local_skip_threshold"]):
            continue
        if label.role == Role.SUPPORT or any(item.block_id == block_id for values in labeling.hard_negatives.values() for item in values):
            candidates.append((block_id, label))
    decisions: list[dict] = []
    for block_id, label in sorted(candidates, key=lambda item: item[1].confidence)[:int(api["max_candidates_per_task"])]:
        candidate = index.block(block_id)
        if not candidate:
            continue
        payload = {
            "instruction": "Judge whether candidate code is SUPPORT or DROP for understanding the bug. Return strict JSON only.",
            "block_id": block_id, "issue_text": task.issue_text, "dataset_source": task.dataset_source,
            "file_path": candidate.relative_file_path, "core_symbol": core.symbol,
            "core_code": core.source_text[:3000], "candidate_symbol": candidate.symbol,
            "candidate_code": candidate.source_text[:4000],
            "relation_metadata": {"relation": str(label.relation_type), "graph_distance": 1,
                                  "local_role": str(label.role), "local_confidence": label.confidence},
        }
        try:
            decision = review_candidate(client, payload, str(label.role), label.confidence,
                                        primary_model, reviewer_model, config)
            label.apply(decision["role"], decision["confidence"], f"api_primary:{primary_model}", decision["relation"])
            if len(decision["models"]) > 1:
                label.provenance.append(f"api_reviewer:{reviewer_model}")
            decisions.append({"task_id": task.task_id, "dataset_source": task.dataset_source, **decision})
        except Exception as exc:
            label.confidence = max(0.0, label.confidence - 0.10)
            decisions.append({"task_id": task.task_id, "dataset_source": task.dataset_source,
                              "block_id": block_id, "role": str(label.role), "confidence": label.confidence,
                              "relation": str(label.relation_type), "error": str(exc)[:500]})
    return decisions


def _process_task(
    task: TaskRecord,
    index: RepositoryIndex,
    config: dict,
    seed: int,
    tokenizer_path: str | None,
    client: CachedLLMClient | None,
    use_api: bool,
) -> dict:
    if not task.issue_text.strip():
        if not client or not use_api:
            raise ValueError("query is empty and API issue generation is unavailable")
        generated = generate_issue(task, _buggy_context(task, index), client,
                                   os.environ.get("LLM_MODEL_PRIMARY", config["api"]["model_primary"]), config)
        task.issue_text = generated["issue_text"]
    labeling = label_task(task, index, config["confidence"])
    if not labeling.core_block_ids:
        raise RuntimeError("patch/traceback did not map to any buggy-base AST block")
    if not labeling.patch_mapping_verified:
        raise RuntimeError("patch mapping to buggy-base AST blocks is not verified")
    decisions = _api_review(task, index, labeling, client, config) if client and use_api else []
    samples, relations, rankings = build_task_outputs(task, index, labeling, config, seed, tokenizer_path)
    if not samples:
        raise RuntimeError("no sample satisfied CORE/DROP and budget constraints")
    patch_text = task.mutation_patch or task.patch
    safe_samples = []
    for sample in samples:
        leakage = check_leakage(task.issue_text, sample["code"], patch_text)
        if leakage.risk == "high":
            continue
        safe_samples.append(sample)
    if not safe_samples:
        raise RuntimeError("all samples failed leakage checks")
    if (int(stable_hash([task.task_id, seed], 8), 16) / 0xFFFFFFFF) < float(config["sampling"]["document_negative_ratio"]):
        negative = build_document_negative(task, index, config, seed)
        if negative:
            safe_samples.append(negative)
    valid_ids = {sample["sample_id"] for sample in safe_samples}
    return {
        "samples": safe_samples,
        "relations": [row for row in relations if row["sample_id"] in valid_ids],
        "rankings": [row for row in rankings if row["sample_id"] in valid_ids],
        "decisions": decisions,
        "stats": {"ast_failures": len(index.ast_failures), "patch_mapping_verified": labeling.patch_mapping_verified},
    }


def build_dataset(
    source: str,
    tasks_path: str | Path,
    output_dir: str | Path,
    config: dict,
    seed: int = 42,
    task_limit: int | None = None,
    num_workers: int = 1,
    resume: bool = False,
    use_api: bool = False,
    offline: bool = False,
    tokenizer_path: str | None = None,
) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if source == "swe_pruner_original":
        rows, row_failures = normalize_swepruner_rows(tasks_path, task_limit)
        write_jsonl(output / "pruning_sft.jsonl", rows)
        write_jsonl(output / "failed_tasks.jsonl", row_failures)
        report = build_report(output)
        return {"output": str(output), "samples": len(rows), "failures": len(row_failures), "report": report}
    records, input_failures = load_task_records(source, tasks_path)
    if task_limit is not None:
        records = records[:task_limit]
    blacklist_path = Path(config["config_path"]).parent / "eval_repo_blacklist.txt"
    blacklist = load_blacklist(blacklist_path)
    excluded = [record for record in records if record.repo_name in blacklist]
    records = [record for record in records if record.repo_name not in blacklist]
    failures = input_failures + [
        BuildFailure(record.task_id, source, record.repo_path, "input", "EvalRepoBlacklisted",
                     f"repository excluded by {blacklist_path}", False) for record in excluded
    ]
    cache = ApiCache(output.parent / "api_cache.sqlite")
    client = None
    if use_api and not offline:
        client = CachedLLMClient(OpenAICompatibleProvider.from_env(config), cache, config)
    api_mode = bool(use_api and not offline)
    fingerprint = stable_hash([
        config["fingerprint"], config.get("builder_version"), seed, tokenizer_path,
        api_mode,
        os.environ.get("LLM_BASE_URL", config["api"]["base_url"]) if api_mode else "offline",
        os.environ.get("LLM_MODEL_PRIMARY", config["api"]["model_primary"]) if api_mode else "offline",
        os.environ.get("LLM_MODEL_REVIEWER", config["api"]["model_reviewer"]) if api_mode else "offline",
        config["api"]["prompt_version"] if api_mode else "offline",
    ])
    results: dict[str, dict] = {}
    pending: list[TaskRecord] = []
    for record in records:
        state = _task_state_path(output, record, fingerprint)
        if resume and state.exists():
            results[record.task_id] = read_json(state)
        else:
            pending.append(record)
    max_repo_files = int(config["analysis"]["max_repo_files"])
    path_counts = Counter(record.repo_path for record in pending)
    shared_priority_paths: dict[str, set[str]] = defaultdict(set)
    for record in pending:
        if path_counts[record.repo_path] > 1:
            shared_priority_paths[record.repo_path].update(_priority_paths(record))
    shared_indexes = {
        repo_path: build_repository_index(repo_path, max_repo_files, sorted(priority_paths))
        for repo_path, priority_paths in sorted(shared_priority_paths.items())
    }
    def process(record: TaskRecord) -> tuple[TaskRecord, dict]:
        index = shared_indexes.get(record.repo_path)
        if index is None:
            index = build_repository_index(record.repo_path, max_repo_files, _priority_paths(record))
        return record, _process_task(record, index, config, seed, tokenizer_path, client, use_api and not offline)
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        futures = {executor.submit(process, record): record for record in pending}
        for future in as_completed(futures):
            record = futures[future]
            try:
                _, value = future.result()
                results[record.task_id] = value
                state = _task_state_path(output, record, fingerprint)
                state.parent.mkdir(parents=True, exist_ok=True)
                write_json(state, value)
            except Exception as exc:
                failures.append(BuildFailure(record.task_id, source, record.repo_path, "sampling",
                                             type(exc).__name__, str(exc)[:1000]))
    for record in records:
        result = results.get(record.task_id)
        if result and not result.get("stats", {}).get("patch_mapping_verified", False):
            failures.append(BuildFailure(
                record.task_id,
                source,
                record.repo_path,
                "labeling",
                "UnverifiedPatchMapping",
                "patch mapping to buggy-base AST blocks is not verified",
            ))
            del results[record.task_id]
    ordered_results = [results[record.task_id] for record in records if record.task_id in results]
    samples = sorted([row for result in ordered_results for row in result["samples"]], key=lambda row: row["sample_id"])
    relations = sorted([row for result in ordered_results for row in result["relations"]], key=lambda row: (row["sample_id"], row["candidate_block"]["block_id"]))
    rankings = sorted([row for result in ordered_results for row in result["rankings"]], key=lambda row: row["sample_id"])
    decisions = sorted([row for result in ordered_results for row in result["decisions"]], key=lambda row: (row["task_id"], row["block_id"]))
    write_jsonl(output / "pruning_sft.jsonl", samples)
    write_jsonl(output / "block_relation.jsonl", relations)
    write_jsonl(output / "block_ranking.jsonl", rankings)
    write_jsonl(output / "failed_tasks.jsonl", [failure.to_dict() for failure in failures])
    write_jsonl(output / "api_decisions.jsonl", decisions)
    api_stats = cache.stats()
    cache.close()
    patch_failures = sum(not result.get("stats", {}).get("patch_mapping_verified", False) for result in ordered_results)
    report = build_report(output, {
        "task_count": len(records) + len(input_failures), "successful_task_count": len(ordered_results),
        "failed_task_count": len(failures), "excluded_eval_repositories": sorted({record.repo_name for record in excluded}),
        "ast_parse_failure_rate": sum(result.get("stats", {}).get("ast_failures", 0) for result in ordered_results) / max(1, len(ordered_results)),
        "patch_mapping_failure_rate": patch_failures / max(1, len(ordered_results)),
        "api_request_count": api_stats["requests"], "api_success_count": api_stats["success"],
        "api_failure_count": api_stats["failed"], "api_cache_hit_rate": api_stats["cache_hit_rate"],
        "api_input_tokens": api_stats["input_tokens"], "api_output_tokens": api_stats["output_tokens"],
    })
    return {"output": str(output), "samples": len(samples), "relations": len(relations), "rankings": len(rankings),
            "successful_tasks": len(ordered_results), "failures": len(failures), "api": api_stats, "report": report}


def create_manifest(artifacts_root: str | Path, output_path: str | Path, config: dict, seed: int = 42) -> dict:
    root = Path(artifacts_root).resolve()
    sources = ["swe_pruner_original", "swe_smith", "swe_gym"]
    all_samples: list[dict] = []
    source_manifest: dict[str, dict] = {}
    for source in sources:
        path = root / source / "pruning_sft.jsonl"
        rows = list(read_jsonl(path)) if path.exists() else []
        all_samples.extend(rows)
        source_manifest[source] = {
            "path": str(path), "sample_count": len(rows),
            "repo_count": len({row.get("repo_name") for row in rows if row.get("repo_name")}),
        }
    manifests = split_samples(all_samples, seed, config)
    write_split_manifests(root, manifests)
    review = write_review_samples(root, all_samples, int(config["sampling"]["review_sample_count"]), seed)
    combined = {
        "builder_version": config.get("builder_version", "0.1.0"), "seed": seed,
        "sources": source_manifest,
        "recommended_sampling_weights": {"swe_pruner_original": 0.30, "swe_smith": 0.55, "swe_gym": 0.15},
        "weights_are_loader_configuration_not_oversampling": True,
        "splits": {name: {"repo_count": len(value["repositories"]), "sample_count": len(value["sample_ids"])}
                   for name, value in manifests.items()},
        "review_samples": review,
    }
    write_json(output_path, combined)
    for source in sources:
        report_path = root / source / "report.json"
        if report_path.exists():
            report = read_json(report_path)
            report["split_repo_counts"] = {
                name: len(set(value["repositories"]) & {row.get("repo_name") for row in all_samples if row.get("dataset_source") == source})
                for name, value in manifests.items()
            }
            write_json(report_path, report)
    return combined
