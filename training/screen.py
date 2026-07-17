from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .config import PARITY_FIELDS, load_train_config

EXPECTED_ARCHIVE_SHA256 = "25b83b5bab239599aa8b49021260d24e4e11becacd10e6759f3ea25da60d26bf"
KEY_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen offline training code and data")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--archive", default="training/assets/swepruner_real_dataset_2k_seed42.tar.gz")
    args = parser.parse_args()
    root = Path(args.data_root)
    archive = Path(args.archive)
    errors: list[str] = []
    archive_hash = sha256(archive) if archive.exists() else "missing"
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        errors.append(f"dataset archive sha256 mismatch: {archive_hash}")
    m1 = load_train_config("training/configs/m1_data_only.json")
    m2 = load_train_config("training/configs/m2_structural.json")
    for field in PARITY_FIELDS:
        if getattr(m1, field) != getattr(m2, field):
            errors.append(f"M1/M2 parity mismatch: {field}")
    split_ids: dict[str, set[str]] = {}
    expected_splits = {"train": 1566, "validation": 207, "test": 228}
    for split, expected in expected_splits.items():
        manifest = json.loads((root / "splits" / f"{split}_manifest.json").read_text(encoding="utf-8"))
        split_ids[split] = set(manifest["sample_ids"])
        if len(split_ids[split]) != expected:
            errors.append(f"{split} split count mismatch")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[left] & split_ids[right]:
            errors.append(f"split overlap: {left}/{right}")
    main_path = root / "combined" / "pruning_sft.jsonl"
    sample_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    relation_values: set[str] = set()
    main_count = 0
    with main_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if KEY_RE.search(line):
                errors.append(f"possible key in main dataset line {line_number}")
            row = json.loads(line)
            main_count += 1
            sample_id = row.get("sample_id")
            if sample_id in sample_ids:
                errors.append(f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            source_counts[row["dataset_source"]] += 1
            line_count = len(row["code"].splitlines())
            for field in ("line_keep_labels", "line_roles", "line_relation_types", "line_confidences", "line_numbers"):
                if len(row.get(field, [])) != line_count:
                    errors.append(f"line mismatch {sample_id}:{field}")
            if row["dataset_source"] != "swe_pruner_original" and row["document_label"] == 1:
                if not row.get("metadata", {}).get("patch_mapping_verified"):
                    errors.append(f"unverified positive patch mapping: {sample_id}")
            relation_values.update(map(str, row.get("line_relation_types", [])))
    if main_count != 2001 or len(sample_ids) != 2001:
        errors.append(f"main dataset count mismatch: rows={main_count}, ids={len(sample_ids)}")
    expected_sources = {"swe_pruner_original": 910, "swe_smith": 696, "swe_gym": 395}
    if dict(source_counts) != expected_sources:
        errors.append(f"source counts mismatch: {dict(source_counts)}")
    auxiliary_counts = {"relation": 0, "ranking": 0}
    for source in ("swe_smith", "swe_gym"):
        for kind, filename, label in (
            ("relation", "block_relation.jsonl", "relation"),
            ("ranking", "block_ranking.jsonl", "negative_type"),
        ):
            for row in rows(root / source / filename):
                auxiliary_counts[kind] += 1
                if row.get("sample_id") not in sample_ids:
                    errors.append(f"orphan {kind} sample_id: {row.get('sample_id')}")
                if kind == "relation":
                    relation_values.add(str(row[label]))
    training_text = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in Path("training").rglob("*") if path.is_file() and path.suffix in {".py", ".sh", ".json", ".md", ".txt"}
    )
    if KEY_RE.search(training_text):
        errors.append("possible API key in training source")
    report = {
        "valid": not errors,
        "errors": errors[:50],
        "archive_sha256": archive_hash,
        "main_rows": main_count,
        "source_counts": dict(source_counts),
        "split_counts": {name: len(value) for name, value in split_ids.items()},
        "auxiliary_counts": auxiliary_counts,
        "relation_labels": sorted(relation_values),
        "m1_m2_parity_fields": len(PARITY_FIELDS),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

