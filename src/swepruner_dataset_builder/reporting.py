from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

from .io_utils import read_jsonl, write_json


def _distribution(values: list, buckets: int = 10) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        number = min(0.999999, max(0.0, float(value)))
        lower = int(number * buckets) / buckets
        counter[f"{lower:.1f}-{lower + 1 / buckets:.1f}"] += 1
    return dict(sorted(counter.items()))


def build_report(artifact_dir: str | Path, extra: dict | None = None) -> dict:
    root = Path(artifact_dir)
    samples = list(read_jsonl(root / "pruning_sft.jsonl")) if (root / "pruning_sft.jsonl").exists() else []
    relations = list(read_jsonl(root / "block_relation.jsonl")) if (root / "block_relation.jsonl").exists() else []
    rankings = list(read_jsonl(root / "block_ranking.jsonl")) if (root / "block_ranking.jsonl").exists() else []
    failures = list(read_jsonl(root / "failed_tasks.jsonl")) if (root / "failed_tasks.jsonl").exists() else []
    decisions = list(read_jsonl(root / "api_decisions.jsonl")) if (root / "api_decisions.jsonl").exists() else []
    roles = [role for sample in samples for role in sample.get("line_roles", [])]
    confidences = [value for sample in samples for value in sample.get("line_confidences", [])]
    provenance = [item for sample in samples for items in sample.get("line_provenance", []) for item in items]
    relation_types = [value for sample in samples for value in sample.get("line_relation_types", [])]
    hard_types = [value for sample in samples for value in sample.get("metadata", {}).get("hard_negative_types", [])]
    keep_ratios = [sum(sample.get("line_keep_labels", [])) / max(1, len(sample.get("line_keep_labels", []))) for sample in samples]
    role_counter = Counter(roles)
    total_lines = len(roles)
    api_agreements = [item.get("agreement") for item in decisions if item.get("agreement") is not None]
    report = {
        "task_count": len({sample.get("task_id") for sample in samples}) + len({item.get("task_id") for item in failures}),
        "successful_task_count": len({sample.get("task_id") for sample in samples}),
        "failed_task_count": len({item.get("task_id") for item in failures}),
        "repo_count": len({sample.get("repo_name") for sample in samples if sample.get("repo_name")}),
        "pruning_sft_sample_count": len(samples), "relation_sample_count": len(relations),
        "ranking_sample_count": len(rankings),
        "core_lines": role_counter["CORE"], "core_ratio": role_counter["CORE"] / total_lines if total_lines else 0,
        "support_lines": role_counter["SUPPORT"], "support_ratio": role_counter["SUPPORT"] / total_lines if total_lines else 0,
        "drop_lines": role_counter["DROP"], "drop_ratio": role_counter["DROP"] / total_lines if total_lines else 0,
        "keep_ratio_distribution": _distribution(keep_ratios),
        "average_code_lines": mean([len(sample.get("line_numbers", [])) for sample in samples]) if samples else 0,
        "average_characters": mean([len(sample.get("code", "")) for sample in samples]) if samples else 0,
        "average_estimated_tokens": mean([sample.get("metadata", {}).get("estimated_tokens", 0) for sample in samples]) if samples else 0,
        "truncated_sample_count": sum(bool(sample.get("metadata", {}).get("truncated")) for sample in samples),
        "ast_parse_failure_rate": 0, "patch_mapping_failure_rate": 0,
        "traceback_parse_success_rate": 0,
        "api_request_count": len(decisions), "api_success_count": sum(not item.get("error") for item in decisions),
        "api_failure_count": sum(bool(item.get("error")) for item in decisions), "api_cache_hit_rate": 0,
        "api_input_tokens": 0, "api_output_tokens": 0,
        "primary_reviewer_agreement_rate": sum(bool(value) for value in api_agreements) / len(api_agreements) if api_agreements else 0,
        "provenance_distribution": dict(Counter(provenance)),
        "relation_type_distribution": dict(Counter(relation_types)),
        "hard_negative_type_distribution": dict(Counter(hard_types)),
        "confidence_distribution": _distribution(confidences),
        "leakage_check_failure_count": sum(item.get("error_type") == "LeakageError" for item in failures),
        "split_repo_counts": {"train": 0, "validation": 0, "test": 0},
    }
    if extra:
        report.update(extra)
    write_json(root / "report.json", report)
    return report

