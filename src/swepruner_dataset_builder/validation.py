from __future__ import annotations

import json
import re
from pathlib import Path

from .io_utils import read_json, read_jsonl
from .leakage_check import hashes_present


KEY_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}")


def validate_dataset(path: str | Path) -> dict:
    source = Path(path)
    errors: list[dict] = []
    seen: set[str] = set()
    rows = 0
    for line_number, row in enumerate(read_jsonl(source), 1):
        rows += 1
        sample_id = str(row.get("sample_id", ""))
        def add(code: str, message: str) -> None:
            errors.append({"line": line_number, "sample_id": sample_id, "code": code, "message": message})
        if not sample_id or sample_id in seen:
            add("sample_id", "sample_id is missing or duplicated")
        seen.add(sample_id)
        code = str(row.get("code", ""))
        count = len(code.splitlines())
        arrays = ["line_numbers", "line_roles", "line_keep_labels", "line_confidences", "line_provenance", "line_relation_types"]
        for name in arrays:
            if not isinstance(row.get(name), list) or len(row.get(name, [])) != count:
                add("length", f"{name} length does not match code lines")
        roles = row.get("line_roles", [])
        keep = row.get("line_keep_labels", [])
        for index, (role, label) in enumerate(zip(roles, keep)):
            expected = 1 if role in {"CORE", "SUPPORT"} else 0
            if role not in {"CORE", "SUPPORT", "DROP"} or label != expected:
                add("role_keep", f"invalid role/keep at local line {index + 1}")
        if not str(row.get("query", "")).strip():
            add("query", "query is empty")
        if int(row.get("document_label", 1)) == 1:
            if row.get("dataset_source") != "swe_pruner_original" and ("CORE" not in roles or "DROP" not in roles):
                add("positive_roles", "positive sample must contain CORE and DROP")
            metadata = row.get("metadata", {})
            if metadata.get("input_revision") != "buggy_base" and row.get("dataset_source") != "swe_pruner_original":
                add("revision", "input is not marked as buggy_base")
            if not metadata.get("patch_mapping_verified") and row.get("dataset_source") != "swe_pruner_original":
                add("patch_mapping", "patch mapping is not verified")
            core_numbers = set(metadata.get("core_line_numbers", []))
            actual_core = {line for line, role in zip(row.get("line_numbers", []), roles) if role == "CORE"}
            if core_numbers and core_numbers != actual_core:
                add("core_truncated", "CORE line metadata does not match sample")
            for span in metadata.get("block_spans", []):
                if int(span.get("start_line", 0)) <= 0 or int(span.get("end_line", 0)) < int(span.get("start_line", 0)):
                    add("block_span", "invalid AST block span")
            forbidden = metadata.get("forbidden_added_hashes", [])
            if hashes_present(code, forbidden) or hashes_present(str(row.get("query", "")), forbidden):
                add("leakage", "patch addition fragment detected")
        serialized = json.dumps(row, ensure_ascii=False)
        if KEY_RE.search(serialized):
            add("api_key", "possible API key found in artifact")
        if "diff --git" in code or ("--- a/" in code and "+++ b/" in code):
            add("gold_patch", "sample code appears to contain a complete patch")
    return {"path": str(source.resolve()), "rows": rows, "valid": not errors, "error_count": len(errors), "errors": errors}


def validate_artifact_dir(path: str | Path) -> dict:
    root = Path(path)
    result = validate_dataset(root / "pruning_sft.jsonl")
    ranking_path = root / "block_ranking.jsonl"
    if ranking_path.exists():
        for line_number, row in enumerate(read_jsonl(ranking_path), 1):
            if not row.get("positive_block") or not row.get("hard_negative_block") or not row.get("negative_type"):
                result["errors"].append({"line": line_number, "code": "ranking", "message": "ranking row lacks real positive or hard negative"})
    decisions_path = root / "api_decisions.jsonl"
    if decisions_path.exists():
        for line_number, row in enumerate(read_jsonl(decisions_path), 1):
            if row.get("role") not in {"SUPPORT", "DROP"} or not 0 <= float(row.get("confidence", -1)) <= 1:
                result["errors"].append({"line": line_number, "code": "api_schema", "message": "invalid API decision"})
    result["error_count"] = len(result["errors"])
    result["valid"] = result["error_count"] == 0
    return result
