from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, stable_id
from .schemas import BuildFailure, TaskRecord, repo_name_from_path


def _first(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _tests(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def load_task_records(
    source: str, tasks_path: str | Path
) -> tuple[list[TaskRecord], list[BuildFailure]]:
    path = Path(tasks_path).resolve()
    records: list[TaskRecord] = []
    failures: list[BuildFailure] = []
    for index, row in enumerate(read_jsonl(path), 1):
        task_id = str(_first(row, "task_id", "instance_id", "id", default=f"row-{index}"))
        raw_repo_path = str(_first(row, "repo_path", "checkout_path", "local_repo_path"))
        repo_path = Path(raw_repo_path).expanduser() if raw_repo_path else Path()
        if raw_repo_path and not repo_path.is_absolute():
            repo_path = (path.parent / repo_path).resolve()
        patch = str(_first(row, "patch", "gold_patch", "patch_text"))
        mutation = str(_first(row, "mutation_patch", "mutation", "mutant_patch"))
        error = ""
        if not raw_repo_path:
            error = "repo_path is required"
        elif not repo_path.is_dir():
            error = f"repo_path does not exist: {repo_path}"
        elif not (patch.strip() or mutation.strip()):
            error = "patch or mutation_patch is required"
        if error:
            failures.append(
                BuildFailure(task_id, source, str(repo_path), "input", "InvalidTask", error)
            )
            continue
        repo_name = str(_first(row, "repo_name", "repo", "repository")) or repo_name_from_path(str(repo_path))
        records.append(
            TaskRecord(
                task_id=task_id,
                dataset_source=source,
                repo_name=repo_name,
                repo_path=str(repo_path),
                base_commit=str(_first(row, "base_commit", "base_sha", "commit")),
                issue_text=str(_first(row, "issue_text", "problem_statement", "issue", "query")),
                patch=patch,
                mutation_patch=mutation,
                traceback=str(_first(row, "traceback", "trace", "test_log", "logs")),
                failing_tests=_tests(_first(row, "failing_tests", "fail_to_pass", "tests", default=[])),
                test_command=str(_first(row, "test_command", "test_cmd")),
            )
        )
    return records, failures


def inspect_tasks(source: str, tasks_path: str | Path) -> dict[str, Any]:
    records, failures = load_task_records(source, tasks_path)
    return {
        "source": source,
        "tasks_path": str(Path(tasks_path).resolve()),
        "valid_tasks": len(records),
        "invalid_tasks": len(failures),
        "repositories": sorted({record.repo_name for record in records}),
        "tasks_with_issue": sum(bool(record.issue_text.strip()) for record in records),
        "tasks_with_traceback": sum(bool(record.traceback.strip()) for record in records),
        "tasks_with_mutation": sum(bool(record.mutation_patch.strip()) for record in records),
        "failures": [failure.to_dict() for failure in failures[:20]],
    }


def normalize_swepruner_rows(path: str | Path, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path), 1):
        if limit is not None and len(output) >= limit:
            break
        query = str(_first(row, "query", "issue_text", "problem_statement"))
        code = str(_first(row, "code", "document", "text"))
        labels = _first(row, "line_keep_labels", "labels", "line_labels", default=[])
        kept_frags = row.get("kept_frags")
        lines = code.splitlines()
        if not labels and isinstance(kept_frags, list):
            kept = {int(value) for value in kept_frags if isinstance(value, int) or str(value).isdigit()}
            labels = [1 if line_number in kept else 0 for line_number in range(1, len(lines) + 1)]
        if not isinstance(labels, list):
            labels = []
        labels = [int(bool(value)) for value in labels]
        if not query.strip() or not code or len(lines) != len(labels):
            failures.append({"row": index, "error": "query/code/labels missing or length mismatch"})
            continue
        sample_id = str(row.get("sample_id") or stable_id("spr", index, query, code))
        output.append(
            {
                "sample_id": sample_id,
                "task_id": str(row.get("task_id") or sample_id),
                "dataset_source": "swe_pruner_original",
                "query": query,
                "repo_name": str(row.get("repo_name", "")),
                "file_path": str(row.get("file_path", "")),
                "anchor_symbol": str(row.get("anchor_symbol", "")),
                "code": code,
                "line_numbers": list(range(1, len(lines) + 1)),
                "line_roles": ["SUPPORT" if value else "DROP" for value in labels],
                "line_keep_labels": labels,
                "line_confidences": [1.0] * len(lines),
                "line_provenance": [["swe_pruner_original_label"] for _ in lines],
                "line_relation_types": ["NONE"] * len(lines),
                "document_label": int(row.get("document_label", 0 if row.get("is_negative") else 1)),
                "metadata": {
                    "core_block_ids": [],
                    "support_block_ids": [],
                    "drop_block_ids": [],
                    "hard_negative_types": [],
                    "tokenizer_mode": "original",
                    "estimated_tokens": max(1, len(code) // 4),
                    "api_used": False,
                    "base_commit": str(row.get("base_commit", "")),
                    "patch_hash": "",
                    "builder_version": "0.1.0",
                    "input_revision": "original_dataset",
                    "original_score": row.get("score"),
                    "original_is_negative": bool(row.get("is_negative", False)),
                    "original_kept_frags": kept_frags if isinstance(kept_frags, list) else [],
                },
            }
        )
    return output, failures
