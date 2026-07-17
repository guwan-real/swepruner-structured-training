from __future__ import annotations

from .api_provider import CachedLLMClient
from .leakage_check import check_leakage
from .schemas import TaskRecord


def generate_issue(task: TaskRecord, buggy_context: str, client: CachedLLMClient, model: str, config: dict) -> dict:
    payload = {
        "instruction": (
            "Generate a user-observable bug report as strict JSON with issue_text, actual_behavior, "
            "expected_behavior, and leakage_risk. Do not reveal a patch, replacement code, target line, or fix method."
        ),
        "dataset_source": task.dataset_source,
        "failing_tests": task.failing_tests,
        "traceback": task.traceback[-6000:],
        "buggy_context": buggy_context[:4000],
        "issue_text": "",
        "candidate_code": buggy_context[:4000],
        "relation_metadata": {"operation": "issue_generation"},
    }
    value = client.request(model, config["api"]["prompt_version"] + ":issue", payload)
    required = {"issue_text", "actual_behavior", "expected_behavior", "leakage_risk"}
    if not required.issubset(value) or value["leakage_risk"] not in {"low", "medium", "high"}:
        raise ValueError("invalid generated issue schema")
    leakage = check_leakage(str(value["issue_text"]), "", task.mutation_patch or task.patch)
    if value["leakage_risk"] == "high" or leakage.risk == "high":
        raise ValueError("generated issue failed leakage check")
    return {key: str(value[key]) for key in required}

