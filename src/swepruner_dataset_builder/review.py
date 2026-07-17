from __future__ import annotations

import random
from pathlib import Path

from .io_utils import write_jsonl


def write_review_samples(root: str | Path, samples: list[dict], count: int = 20, seed: int = 42) -> dict:
    target = Path(root) / "samples_for_review"
    target.mkdir(parents=True, exist_ok=True)
    ordered = sorted(samples, key=lambda item: item["sample_id"])
    chosen = random.Random(seed).sample(ordered, min(count, len(ordered))) if ordered else []
    write_jsonl(target / "review_samples.jsonl", chosen)
    markdown = ["# Dataset review samples", "", f"Seed: `{seed}`. Selected: `{len(chosen)}`.", ""]
    for sample in chosen:
        metadata = sample.get("metadata", {})
        markdown.extend([
            f"## {sample['sample_id']}", "",
            f"- dataset_source: `{sample.get('dataset_source', '')}`",
            f"- task_id: `{sample.get('task_id', '')}`",
            f"- repo/file: `{sample.get('repo_name', '')}/{sample.get('file_path', '')}`",
            f"- anchor: `{sample.get('anchor_symbol', '')}`",
            f"- hard negatives: `{', '.join(metadata.get('hard_negative_types', [])) or 'none'}`",
            f"- API used: `{metadata.get('api_used', False)}`",
            f"- patch old locations: `{metadata.get('patch_old_locations', [])}`",
            "", "### Query", "", sample.get("query", ""), "", "### Labeled code", "", "```text",
        ])
        for number, line, role, keep, confidence, provenance in zip(
            sample.get("line_numbers", []), sample.get("code", "").splitlines(), sample.get("line_roles", []),
            sample.get("line_keep_labels", []), sample.get("line_confidences", []), sample.get("line_provenance", []),
        ):
            markdown.append(f"{number:5d} {role:7s} keep={keep} conf={confidence:.2f} {provenance} | {line}")
        markdown.extend(["```", ""])
    (target / "review_samples.md").write_text("\n".join(markdown), encoding="utf-8")
    return {"requested": count, "selected": len(chosen), "shortfall": max(0, count - len(chosen))}

