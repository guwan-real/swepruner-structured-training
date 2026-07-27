from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROFILES = (
    ("local_context", 0.45, "local", 0.80),
    ("lexical_hard_negative", 0.40, "lexical", 0.75),
    ("mixed_noise", 0.55, "mixed", 0.75),
    ("wide_context", 0.65, "wide", 0.70),
)
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand grounded command outputs with deterministic context views."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_hash(*values: object) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def content_signature(row: dict[str, Any]) -> bytes:
    payload = "\x00".join(
        (
            str(row.get("query", "")),
            str(row.get("code", "")),
            json.dumps(row.get("line_keep_labels", []), separators=(",", ":")),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def query_tokens(query: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(query)}


def lexical_score(line: str, tokens: set[str]) -> int:
    if not tokens:
        return 0
    return len(tokens.intersection(token.lower() for token in TOKEN_PATTERN.findall(line)))


def distance_to_anchor(index: int, anchors: list[int], length: int) -> int:
    if anchors:
        return min(abs(index - anchor) for anchor in anchors)
    return abs(index - length // 2)


def select_indices(
    row: dict[str, Any],
    profile: str,
    ratio: float,
    strategy: str,
    seed: int,
) -> tuple[int, ...]:
    lines = str(row["code"]).splitlines()
    labels = [int(value) for value in row["line_keep_labels"]]
    positives = [index for index, value in enumerate(labels) if value == 1]
    negatives = [index for index, value in enumerate(labels) if value == 0]
    target = max(len(positives), int(math.ceil(len(lines) * ratio)))
    if negatives and target >= len(lines):
        target = len(lines) - 1
    negative_budget = max(0, target - len(positives))
    tokens = query_tokens(str(row.get("query", "")))
    sample_id = str(row.get("sample_id", ""))

    if strategy == "local":
        ranked = sorted(
            negatives,
            key=lambda index: (
                distance_to_anchor(index, positives, len(lines)),
                stable_hash(seed, sample_id, profile, index),
            ),
        )
    elif strategy == "lexical":
        ranked = sorted(
            negatives,
            key=lambda index: (
                -lexical_score(lines[index], tokens),
                distance_to_anchor(index, positives, len(lines)),
                stable_hash(seed, sample_id, profile, index),
            ),
        )
    elif strategy == "mixed":
        ranked = sorted(
            negatives,
            key=lambda index: (
                stable_hash(seed, sample_id, profile, index) % 3,
                distance_to_anchor(index, positives, len(lines)),
                stable_hash(seed, profile, sample_id, index),
            ),
        )
    else:
        ranked = sorted(
            negatives,
            key=lambda index: (
                distance_to_anchor(index, positives, len(lines)) // 3,
                stable_hash(seed, profile, index, sample_id),
            ),
        )

    return tuple(sorted(positives + ranked[:negative_budget]))


def subset_row(
    row: dict[str, Any],
    indices: tuple[int, ...],
    profile: str,
    floor_multiplier: float,
) -> dict[str, Any]:
    result = copy.deepcopy(row)
    original_lines = str(row["code"]).splitlines()
    original_length = len(original_lines)
    result["code"] = "\n".join(original_lines[index] for index in indices)

    for key, value in row.items():
        if isinstance(value, list) and len(value) == original_length:
            result[key] = [copy.deepcopy(value[index]) for index in indices]

    source_sample_id = str(row["sample_id"])
    result["sample_id"] = f"{source_sample_id}__v3_{profile}"
    labels = [int(value) for value in result["line_keep_labels"]]
    positive_ratio = sum(labels) / max(1, len(labels))
    original_floor = float(row.get("minimum_keep_ratio", 0.0))
    result["minimum_keep_ratio"] = max(
        original_floor,
        min(0.65, positive_ratio * floor_multiplier),
    )

    provenance = result.get("line_provenance")
    if isinstance(provenance, list):
        marker = f"command_output_v3:{profile}"
        for item in provenance:
            if isinstance(item, list) and marker not in item:
                item.append(marker)

    metadata = copy.deepcopy(row.get("metadata", {}))
    metadata.update(
        {
            "augmentation_version": "v3",
            "augmentation_profile": profile,
            "source_sample_id": source_sample_id,
            "source_line_count": original_length,
            "selected_line_count": len(indices),
            "selection_ratio": len(indices) / max(1, original_length),
            "positive_ratio": positive_ratio,
        }
    )
    result["metadata"] = metadata
    return result


def validate_row(row: dict[str, Any], line_number: int) -> None:
    lines = str(row["code"]).splitlines()
    if not lines:
        raise ValueError(f"row {line_number}: empty code")
    for field in (
        "line_keep_labels",
        "line_roles",
        "line_confidences",
        "line_provenance",
    ):
        if len(row[field]) != len(lines):
            raise ValueError(
                f"row {line_number}: {field} has {len(row[field])} values "
                f"for {len(lines)} lines"
            )
    if not row.get("parent_sample_id"):
        raise ValueError(f"row {line_number}: missing parent_sample_id")


def main() -> None:
    args = parse_args()
    input_rows = read_jsonl(args.input)
    for line_number, row in enumerate(input_rows, 1):
        validate_row(row, line_number)

    rows: list[dict[str, Any]] = []
    content_signatures: set[bytes] = set()
    for row in input_rows:
        signature = content_signature(row)
        if signature in content_signatures:
            continue
        content_signatures.add(signature)
        rows.append(row)

    if args.target_rows < len(rows):
        raise ValueError(
            f"target rows ({args.target_rows}) is smaller than input ({len(rows)})"
        )

    output_rows = list(rows)
    seen: dict[str, set[tuple[int, ...]]] = {
        str(row["sample_id"]): {tuple(range(len(str(row["code"]).splitlines())))}
        for row in rows
    }
    ordered_rows = sorted(
        rows,
        key=lambda row: stable_hash(args.seed, row.get("sample_id", "")),
    )
    profile_counts: Counter[str] = Counter()

    for round_index in range(len(PROFILES)):
        for row_index, row in enumerate(ordered_rows):
            if len(output_rows) >= args.target_rows:
                break
            profile, ratio, strategy, floor_multiplier = PROFILES[
                (row_index + round_index) % len(PROFILES)
            ]
            indices = select_indices(
                row,
                profile=profile,
                ratio=ratio,
                strategy=strategy,
                seed=args.seed,
            )
            sample_id = str(row["sample_id"])
            if not indices or indices in seen[sample_id]:
                continue
            seen[sample_id].add(indices)
            candidate = subset_row(
                row,
                indices=indices,
                profile=profile,
                floor_multiplier=floor_multiplier,
            )
            signature = content_signature(candidate)
            if signature in content_signatures:
                continue
            content_signatures.add(signature)
            output_rows.append(candidate)
            profile_counts[profile] += 1
        if len(output_rows) >= args.target_rows:
            break

    if len(output_rows) < args.target_rows:
        raise RuntimeError(
            f"could only construct {len(output_rows)} unique rows; "
            f"requested {args.target_rows}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)

    type_counts = Counter(str(row.get("sample_type", "unknown")) for row in output_rows)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "rows": len(output_rows),
                "input_rows": len(input_rows),
                "base_rows": len(rows),
                "deduplicated_base_rows": len(input_rows) - len(rows),
                "derived_rows": len(output_rows) - len(rows),
                "profiles": profile_counts,
                "types": type_counts,
                "bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
