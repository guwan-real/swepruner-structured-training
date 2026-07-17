from __future__ import annotations

from pathlib import Path

from .io_utils import read_json, read_jsonl, write_jsonl


def export_swepruner(input_path: str | Path, mapping_path: str | Path, output_path: str | Path) -> dict:
    mapping = read_json(mapping_path)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("mapping must be a non-empty JSON object")
    rows: list[dict] = []
    for row in read_jsonl(input_path):
        code = row.get("code", "")
        labels = row.get("line_keep_labels", [])
        if len(str(code).splitlines()) != len(labels):
            raise ValueError(f"label length mismatch for {row.get('sample_id', '<unknown>')}")
        exported = {}
        for source_field, target_field in mapping.items():
            if source_field not in row:
                raise ValueError(f"source field missing: {source_field}")
            exported[str(target_field)] = row[source_field]
        rows.append(exported)
    write_jsonl(output_path, rows)
    return {"input": str(input_path), "output": str(output_path), "rows": len(rows), "mapping": mapping}


def export_swepruner_official(input_path: str | Path, output_path: str | Path) -> dict:
    rows: list[dict] = []
    for row in read_jsonl(input_path):
        code = str(row.get("code", ""))
        labels = row.get("line_keep_labels", [])
        if len(code.splitlines()) != len(labels):
            raise ValueError(f"label length mismatch for {row.get('sample_id', '<unknown>')}")
        original_score = row.get("metadata", {}).get("original_score")
        score = float(row.get("document_label", 0) if original_score is None else original_score)
        rows.append({
            "query": str(row.get("query", "")),
            "code": code,
            "kept_frags": [line for line, keep in enumerate(labels, 1) if int(keep) == 1],
            "score": score,
        })
    write_jsonl(output_path, rows)
    return {"input": str(input_path), "output": str(output_path), "rows": len(rows),
            "format": "swe_pruner_official"}
