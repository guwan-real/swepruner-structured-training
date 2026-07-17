from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .deduplication import code_fingerprints
from .io_utils import stable_hash, write_json


class _UnionFind:
    def __init__(self, values: set[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def split_samples(samples: list[dict], seed: int, config: dict) -> dict[str, dict]:
    repos = {str(sample.get("repo_name", "")) for sample in samples if sample.get("repo_name")}
    union = _UnionFind(repos)
    owners: dict[tuple[str, str], str] = {}
    for sample in samples:
        repo = str(sample.get("repo_name", ""))
        if not repo:
            continue
        hashes = sample.get("metadata", {}).get("dedup_hashes") or code_fingerprints(sample.get("code", ""))
        for kind, value in hashes.items():
            key = (kind, value)
            if key in owners and owners[key] != repo:
                union.union(repo, owners[key])
            else:
                owners[key] = repo
    components: dict[str, set[str]] = defaultdict(set)
    for repo in repos:
        components[union.find(repo)].add(repo)
    ordered = sorted(components.values(), key=lambda group: stable_hash([seed, sorted(group)], 64))
    split = config["split"]
    count = len(ordered)
    test_count = max(1, round(count * float(split["test_ratio"]))) if count >= 3 else 0
    validation_count = max(1, round(count * float(split["validation_ratio"]))) if count >= 3 else (1 if count == 2 else 0)
    while test_count + validation_count >= count and test_count:
        test_count -= 1
    assignment: dict[str, str] = {}
    for index, group in enumerate(ordered):
        name = "train"
        if index >= count - test_count:
            name = "test"
        elif index >= count - test_count - validation_count:
            name = "validation"
        for repo in group:
            assignment[repo] = name
    manifests = {name: {"split": name, "seed": seed, "repositories": [], "sample_ids": []}
                 for name in ("train", "validation", "test")}
    for repo, name in assignment.items():
        manifests[name]["repositories"].append(repo)
    for sample in samples:
        name = assignment.get(str(sample.get("repo_name", "")), "train")
        manifests[name]["sample_ids"].append(sample["sample_id"])
    for manifest in manifests.values():
        manifest["repositories"].sort()
        manifest["sample_ids"].sort()
    return manifests


def write_split_manifests(root: str | Path, manifests: dict[str, dict]) -> None:
    target = Path(root) / "splits"
    for name, manifest in manifests.items():
        write_json(target / f"{name}_manifest.json", manifest)

