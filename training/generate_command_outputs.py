from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
from urllib.request import Request, urlopen

OFFICIAL_SOURCE = "swe_pruner_original"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded command-output pruning samples")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-parent-samples", type=int, default=0)
    parser.add_argument("--max-grep-lines", type=int, default=80)
    parser.add_argument("--api-max-samples", type=int, default=0)
    parser.add_argument("--api-url", default=os.environ.get("LLM_API_URL", ""))
    parser.add_argument("--api-model", default=os.environ.get("LLM_API_MODEL", "qwen-chat"))
    parser.add_argument("--api-keys-env", default="LLM_API_KEYS")
    parser.add_argument("--api-cache")
    return parser.parse_args()


def stable_id(parent: str, sample_type: str) -> str:
    digest = hashlib.sha256(f"{parent}\0{sample_type}".encode()).hexdigest()[:24]
    return f"cmd_{sample_type}_{digest}"


def derived_row(
    parent: dict[str, Any],
    sample_type: str,
    command: str,
    lines: list[str],
    labels: list[int],
    roles: list[str],
    confidences: list[float],
    minimum_keep_ratio: float,
) -> dict[str, Any]:
    if not lines or not (len(lines) == len(labels) == len(roles) == len(confidences)):
        raise ValueError("generated command-output arrays do not align")
    return {
        "sample_id": stable_id(str(parent["sample_id"]), sample_type),
        "parent_sample_id": str(parent["sample_id"]),
        "task_id": str(parent.get("task_id", "")),
        "repo_name": str(parent.get("repo_name", "")),
        "dataset_source": "grounded_command_output",
        "sample_type": sample_type,
        "query": str(parent["query"]),
        "code": "\n".join(lines) + "\n",
        "document_label": 1,
        "minimum_keep_ratio": minimum_keep_ratio,
        "line_numbers": list(range(1, len(lines) + 1)),
        "line_keep_labels": labels,
        "line_roles": roles,
        "line_relation_types": ["NONE"] * len(lines),
        "line_confidences": confidences,
        "line_provenance": [["grounded_command_output"] for _ in lines],
        "metadata": {
            "builder_version": "command-output-v1",
            "command": command,
            "parent_file_path": parent.get("file_path"),
            "api_used": False,
        },
    }


def numbered_view(parent: dict[str, Any]) -> dict[str, Any] | None:
    source_lines = str(parent.get("code", "")).splitlines()
    if not source_lines:
        return None
    lines = [f"{number:6}\t{text}" for number, text in zip(parent["line_numbers"], source_lines)]
    return derived_row(
        parent,
        "numbered_view",
        f"nl -ba {parent.get('file_path', 'file')}",
        lines,
        list(map(int, parent["line_keep_labels"])),
        list(map(str, parent["line_roles"])),
        list(map(float, parent["line_confidences"])),
        0.15,
    )


def grep_view(parent: dict[str, Any], max_lines: int, seed: int) -> dict[str, Any] | None:
    source_lines = str(parent.get("code", "")).splitlines()
    labels = list(map(int, parent.get("line_keep_labels", [])))
    if not source_lines or len(source_lines) != len(labels):
        return None
    positive = {index for index, label in enumerate(labels) if label == 1}
    if not positive:
        return None
    context = {
        neighbor
        for index in positive
        for neighbor in range(max(0, index - 2), min(len(source_lines), index + 3))
    }
    rng = random.Random(f"{seed}:{parent['sample_id']}:grep")
    negatives = [index for index in range(len(source_lines)) if index not in context]
    rng.shuffle(negatives)
    selected = sorted(context | set(negatives[: max(0, max_lines - len(context))]))[:max_lines]
    file_path = str(parent.get("file_path", "file"))
    lines = [f"{file_path}:{parent['line_numbers'][index]}:{source_lines[index]}" for index in selected]
    output_labels = [1 if index in context else 0 for index in selected]
    roles = [
        str(parent["line_roles"][index]) if index in positive else ("SUPPORT" if index in context else "DROP")
        for index in selected
    ]
    confidences = [1.0 if index in positive else (0.75 if index in context else 0.85) for index in selected]
    return derived_row(
        parent, "grep_results", f"rg -n '<issue-related-pattern>' {file_path}",
        lines, output_labels, roles, confidences, 0.20,
    )


def file_listing(parent: dict[str, Any], repo_paths: dict[str, set[str]], seed: int) -> dict[str, Any] | None:
    repo = str(parent.get("repo_name", ""))
    metadata = parent.get("metadata") if isinstance(parent.get("metadata"), dict) else {}
    patch_paths = {
        str(location["file_path"])
        for location in metadata.get("patch_old_locations", [])
        if isinstance(location, dict) and location.get("file_path")
    }
    if parent.get("file_path"):
        patch_paths.add(str(parent["file_path"]))
    pool = sorted(repo_paths.get(repo, set()) | patch_paths)
    if len(pool) < 4 or not patch_paths:
        return None
    rng = random.Random(f"{seed}:{parent['sample_id']}:find")
    negatives = [path for path in pool if path not in patch_paths]
    rng.shuffle(negatives)
    selected = list(dict.fromkeys(sorted(patch_paths) + negatives[: max(0, 40 - len(patch_paths))]))
    target_keep = max(len(patch_paths), int(0.30 * len(selected) + 0.999))
    parents = {str(Path(value).parent) for value in patch_paths}
    top_levels = {Path(value).parts[0] for value in patch_paths if Path(value).parts}

    def similarity(path: str) -> tuple[int, int, str]:
        return (
            int(str(Path(path).parent) in parents),
            int(bool(Path(path).parts) and Path(path).parts[0] in top_levels),
            path,
        )

    ranked = sorted((path for path in selected if path not in patch_paths), key=similarity, reverse=True)
    support = set(ranked[: max(0, target_keep - len(patch_paths))])
    labels = [1 if path in patch_paths or path in support else 0 for path in selected]
    roles = ["CORE" if path in patch_paths else ("SUPPORT" if path in support else "DROP") for path in selected]
    confidence = [1.0 if path in patch_paths else (0.7 if path in support else 0.85) for path in selected]
    return derived_row(parent, "file_listing", "find . -type f", selected, labels, roles, confidence, 0.30)


def load_cache(path: Path | None) -> dict[str, list[int]]:
    if path is None or not path.exists():
        return {}
    return {
        str(row["sample_id"]): [int(value) for value in row.get("support_line_numbers", [])]
        for row in read_jsonl(path)
    }


def request_support(row: dict[str, Any], url: str, model: str, keys: list[str]) -> list[int]:
    numbered = "\n".join(f"{index + 1}: {line}" for index, line in enumerate(row["code"].splitlines()))
    prompt = (
        "Identify only additional SUPPORT lines in this real command output that may help solve the issue. "
        "Do not invent lines. Return strict JSON as {\"support_line_numbers\":[1,2]}.\n\n"
        f"ISSUE:\n{row['query']}\n\nCOMMAND OUTPUT:\n{numbered}"
    )
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    last_error: Exception | None = None
    for key in keys:
        try:
            request = Request(url, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode())
            content = str(result["choices"][0]["message"]["content"])
            parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
            return [int(value) for value in parsed.get("support_line_numbers", [])]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"all API keys failed: {last_error}")


def apply_support(row: dict[str, Any], support_lines: list[int]) -> None:
    maximum = max(1, int(len(row["line_keep_labels"]) * 0.20))
    valid = sorted({value for value in support_lines if 1 <= value <= len(row["line_keep_labels"])})[:maximum]
    for line_number in valid:
        index = line_number - 1
        if row["line_roles"][index] != "CORE":
            row["line_keep_labels"][index] = 1
            row["line_roles"][index] = "SUPPORT"
            row["line_confidences"][index] = 0.65
            row["line_provenance"][index].append("api_support")
    row["metadata"]["api_used"] = bool(valid)


def main() -> None:
    args = parse_args()
    source = Path(args.data_root) / "combined" / "pruning_sft.jsonl"
    parents = list(read_jsonl(source))
    parents.sort(key=lambda row: str(row["sample_id"]))
    if args.max_parent_samples > 0:
        parents = parents[: args.max_parent_samples]
    repo_paths: dict[str, set[str]] = defaultdict(set)
    for row in parents:
        repo = str(row.get("repo_name", ""))
        if row.get("file_path"):
            repo_paths[repo].add(str(row["file_path"]))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for location in metadata.get("patch_old_locations", []):
            if isinstance(location, dict) and location.get("file_path"):
                repo_paths[repo].add(str(location["file_path"]))

    generated: list[dict[str, Any]] = []
    for parent in parents:
        for item in (
            numbered_view(parent),
            grep_view(parent, args.max_grep_lines, args.seed),
            file_listing(parent, repo_paths, args.seed),
        ):
            if item is not None:
                generated.append(item)

    cache_path = Path(args.api_cache) if args.api_cache else None
    cache = load_cache(cache_path)
    if args.api_max_samples:
        keys = [value.strip() for value in os.environ.get(args.api_keys_env, "").split(",") if value.strip()]
        if not args.api_url or not keys:
            raise RuntimeError("API enrichment requires --api-url and keys in the configured environment variable")
        candidates = [row for row in generated if row["sample_type"] in {"grep_results", "file_listing"}]
        for row in candidates[: args.api_max_samples]:
            support = cache.get(row["sample_id"])
            if support is None:
                support = request_support(row, args.api_url, args.api_model, keys)
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"sample_id": row["sample_id"], "support_line_numbers": support}) + "\n")
            apply_support(row, support)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for row in generated:
        counts[row["sample_type"]] += 1
    print(json.dumps({"output": str(output.resolve()), "rows": len(generated), "types": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
