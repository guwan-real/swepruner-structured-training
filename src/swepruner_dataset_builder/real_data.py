from __future__ import annotations

import hashlib
import heapq
import http.client
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_jsonl, stable_hash, write_json, write_jsonl
from .patch_parser import parse_unified_diff


SWE_PRUNER_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=18g_kWeyvd8EICEDZcKylEEf8mnOFhwdi&export=download&confirm=t"
)
SWE_SMITH_DATASET = "SWE-bench/SWE-smith-py"
SWE_GYM_DATASET = "SWE-Gym/SWE-Gym"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def _request_json(url: str, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "swepruner-dataset-builder/0.1.0"})
    value: Any = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            if attempt == 4:
                raise
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
            TimeoutError,
            ConnectionError,
            OSError,
        ):
            if attempt == 4:
                raise
        delay = min(8, 2 ** attempt)
        print(f"[prepare] transient dataset API failure; retrying in {delay}s", flush=True)
        time.sleep(delay)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def fetch_hf_rows(dataset: str, offset: int, length: int = 100) -> tuple[int, list[dict[str, Any]]]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": length}
    )
    value = _request_json(f"{HF_ROWS_URL}?{query}")
    rows = [item["row"] for item in value.get("rows", []) if isinstance(item, dict) and isinstance(item.get("row"), dict)]
    return int(value.get("num_rows_total", len(rows))), rows


def _eligible_python_patch(row: dict[str, Any]) -> bool:
    patch = str(row.get("patch", ""))
    parsed = parse_unified_diff(patch)
    return bool(parsed.files) and all(path.endswith(".py") for path in parsed.files)


def collect_diverse_rows(dataset: str, target: int, max_repos: int, seed: int) -> list[dict[str, Any]]:
    total, first = fetch_hf_rows(dataset, 0, 100)
    scan_count = max(16, max_repos * 4)
    max_offset = max(0, total - 100)
    offsets = sorted({0, *[round(index * max_offset / max(1, scan_count - 1)) for index in range(scan_count)]})
    pages: list[dict[str, Any]] = list(first)
    for offset in offsets:
        if offset == 0:
            continue
        _, rows = fetch_hf_rows(dataset, offset, 100)
        pages.extend(rows)
    unique_pages: dict[str, dict[str, Any]] = {}
    for row in pages:
        instance_id = str(row.get("instance_id", ""))
        if (
            instance_id
            and instance_id not in unique_pages
            and _eligible_python_patch(row)
            and str(row.get("problem_statement", "")).strip()
        ):
            unique_pages[instance_id] = row
    pages = sorted(
        unique_pages.values(),
        key=lambda row: stable_hash([seed, dataset, row.get("instance_id", "")], 64),
    )
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pages:
        repo = str(row.get("repo", ""))
        if repo:
            by_repo[repo].append(row)
    chosen_repos = sorted(
        by_repo,
        key=lambda repo: stable_hash([seed, dataset, repo], 64),
    )[:max_repos]
    if not chosen_repos:
        return []
    per_repo = max(1, math.ceil(target / len(chosen_repos)))
    selected: list[dict[str, Any]] = []
    for round_index in range(per_repo):
        for repo in chosen_repos:
            if round_index < len(by_repo[repo]):
                selected.append(by_repo[repo][round_index])
                if len(selected) >= target * 2:
                    return selected
    for row in pages:
        if row not in selected and str(row.get("repo", "")) in chosen_repos:
            selected.append(row)
            if len(selected) >= target * 2:
                break
    return selected


def parse_smith_repo(value: str) -> tuple[str, str]:
    encoded = value.split("/", 1)[-1]
    if "." not in encoded or "__" not in encoded:
        raise ValueError(f"unrecognized SWE-smith repo identifier: {value}")
    repository, commit = encoded.rsplit(".", 1)
    owner, name = repository.split("__", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"unsafe GitHub repository identifier: {value}")
    return f"{owner}/{name}", commit


def _slug(value: str, limit: int = 96) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return f"{safe[:limit]}_{stable_hash(value, 12)}"


def _run(command: list[str], cwd: Path | None = None, input_text: str | None = None) -> str:
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    process = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    if process.returncode:
        message = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command[:4])}: {message[-1500:]}")
    return process.stdout


class GitWorktreeManager:
    def __init__(self, root: Path):
        self.root = root
        self.sources = root / "_git_sources"
        self.sources.mkdir(parents=True, exist_ok=True)

    def source(self, repo: str, commit: str) -> Path:
        owner, name = repo.split("/", 1)
        path = self.sources / f"{owner}__{name}"
        if not path.exists():
            _run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(path)]
            )
        check = subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode:
            _run(["git", "-C", str(path), "fetch", "--depth=1", "origin", commit])
        return path

    def worktree(self, repo: str, commit: str, destination: Path) -> Path:
        source = self.source(repo, commit)
        if destination.exists():
            git_pointer = destination / ".git"
            if git_pointer.exists():
                return destination
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "-C", str(source), "worktree", "add", "--detach", str(destination), commit])
        return destination

    def discard_worktree(self, repo: str, destination: Path) -> None:
        owner, name = repo.split("/", 1)
        source = self.sources / f"{owner}__{name}"
        if source.exists():
            subprocess.run(
                ["git", "-C", str(source), "worktree", "remove", "--force", str(destination)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "swepruner-dataset-builder/0.1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_swepruner_rows(source: Path, destination: Path, limit: int, seed: int) -> dict[str, int]:
    targets = {False: limit - limit // 2, True: limit // 2}
    heaps: dict[bool, list[tuple[int, int, dict[str, Any]]]] = {False: [], True: []}
    rejected_out_of_range = 0
    rejected_inconsistent_negative = 0
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"query", "code", "kept_frags", "score"}.issubset(row):
                continue
            category = bool(row.get("is_negative", False))
            kept_frags = row.get("kept_frags")
            if category:
                if kept_frags:
                    rejected_inconsistent_negative += 1
                    continue
            else:
                line_count = len(row["code"].splitlines(keepends=True))
                if not kept_frags or any(
                    not isinstance(line_number, int)
                    or isinstance(line_number, bool)
                    or line_number < 1
                    or line_number > line_count
                    for line_number in kept_frags
                ):
                    rejected_out_of_range += 1
                    continue
            key = int(stable_hash([seed, row["query"], row["code"], index], 16), 16)
            item = (-key, index, row)
            heap = heaps[category]
            if len(heap) < targets[category]:
                heapq.heappush(heap, item)
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected = [item for heap in heaps.values() for item in heap]
    selected.sort(key=lambda item: (-item[0], item[1]))
    rows = [item[2] for item in selected]
    if len(rows) < limit:
        raise RuntimeError(f"only {len(rows)} balanced SWE-Pruner rows available for requested {limit}")
    write_jsonl(destination, rows)
    return {
        "selected_rows": len(rows),
        "positive_rows": sum(not bool(row.get("is_negative", False)) for row in rows),
        "negative_rows": sum(bool(row.get("is_negative", False)) for row in rows),
        "rejected_positive_label_rows": rejected_out_of_range,
        "rejected_inconsistent_negative_rows": rejected_inconsistent_negative,
    }


def _prepare_smith(
    root: Path,
    manager: GitWorktreeManager,
    candidates: Iterable[dict[str, Any]],
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in candidates:
        if len(tasks) >= target:
            break
        instance_id = str(row.get("instance_id", ""))
        repo_path: Path | None = None
        repo = ""
        try:
            repo, commit = parse_smith_repo(str(row.get("repo", "")))
            repo_path = root / "swe_smith" / "repos" / _slug(instance_id)
            manager.worktree(repo, commit, repo_path)
            head = _run(["git", "-C", str(repo_path), "rev-parse", "HEAD"]).strip()
            if not head.lower().startswith(commit.lower()):
                raise RuntimeError(f"existing worktree HEAD {head} does not match expected commit {commit}")
            existing_diff = _run(["git", "-C", str(repo_path), "diff", "--binary", "--", "."])
            if not existing_diff.strip():
                _run(["git", "-C", str(repo_path), "apply", "--whitespace=nowarn", "-"], input_text=str(row["patch"]))
            reverse_patch = _run(["git", "-C", str(repo_path), "diff", "--binary", "-R", "--", "."])
            if not reverse_patch.strip():
                raise RuntimeError("mutation patch produced no buggy working-tree diff")
            tasks.append(
                {
                    "task_id": instance_id,
                    "dataset_source": "swe_smith",
                    "repo_name": repo,
                    "repo_path": str(repo_path.resolve()),
                    "base_commit": commit,
                    "issue_text": str(row.get("problem_statement", "")),
                    "patch": "",
                    "mutation_patch": reverse_patch,
                    "traceback": "",
                    "failing_tests": list(row.get("FAIL_TO_PASS") or []),
                    "test_command": "",
                    "source_dataset": SWE_SMITH_DATASET,
                    "official_bug_patch_hash": stable_hash(str(row.get("patch", "")), 64),
                    "patch_direction": "buggy_to_original_for_labeling",
                }
            )
            print(f"[prepare] SWE-smith {len(tasks)}/{target}: {instance_id}", flush=True)
        except Exception as exc:
            if repo_path is not None and repo:
                manager.discard_worktree(repo, repo_path)
            failures.append({"task_id": instance_id, "source": "swe_smith", "error": str(exc)[:2000]})
            print(f"[prepare] SWE-smith skipped: {instance_id}: {str(exc)[:180]}", flush=True)
    return tasks, failures


def _prepare_gym(
    root: Path,
    manager: GitWorktreeManager,
    candidates: Iterable[dict[str, Any]],
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shared: dict[tuple[str, str], Path] = {}
    for row in candidates:
        if len(tasks) >= target:
            break
        instance_id = str(row.get("instance_id", ""))
        try:
            repo = str(row.get("repo", ""))
            commit = str(row.get("base_commit", ""))
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
                raise ValueError("invalid SWE-Gym repo or base_commit")
            key = (repo, commit)
            if key not in shared:
                destination = root / "swe_gym" / "repos" / _slug(f"{repo}@{commit}")
                shared[key] = manager.worktree(repo, commit, destination)
            _run(
                ["git", "-C", str(shared[key]), "apply", "--check", "--whitespace=nowarn", "-"],
                input_text=str(row.get("patch", "")),
            )
            tasks.append(
                {
                    "task_id": instance_id,
                    "dataset_source": "swe_gym",
                    "repo_name": repo,
                    "repo_path": str(shared[key].resolve()),
                    "base_commit": commit,
                    "issue_text": str(row.get("problem_statement", "")),
                    "patch": str(row.get("patch", "")),
                    "mutation_patch": "",
                    "traceback": "",
                    "failing_tests": list(row.get("FAIL_TO_PASS") or []),
                    "test_command": "",
                    "source_dataset": SWE_GYM_DATASET,
                    "test_patch_hash": stable_hash(str(row.get("test_patch", "")), 64),
                    "patch_direction": "buggy_to_fixed",
                }
            )
            print(f"[prepare] SWE-Gym {len(tasks)}/{target}: {instance_id}", flush=True)
        except Exception as exc:
            failures.append({"task_id": instance_id, "source": "swe_gym", "error": str(exc)[:2000]})
            print(f"[prepare] SWE-Gym skipped: {instance_id}: {str(exc)[:180]}", flush=True)
    return tasks, failures


def prepare_real_data(
    root_dir: str | Path,
    smith_limit: int = 100,
    gym_limit: int = 20,
    pruner_limit: int = 100,
    seed: int = 42,
    max_repos_per_source: int = 5,
) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "preparation_report.json"
    smith_tasks_path = root / "swe_smith" / "tasks.jsonl"
    gym_tasks_path = root / "swe_gym" / "tasks.jsonl"
    pruner_path = root / "swe_pruner" / "train.jsonl"
    pruner_sample_path = root / "swe_pruner" / "first_phase.jsonl"
    if report_path.exists() and smith_tasks_path.exists() and gym_tasks_path.exists() and pruner_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        sources = existing.get("sources", {})
        smith_ready = int(sources.get("swe_smith", {}).get("prepared_tasks", 0)) >= smith_limit
        gym_ready = int(sources.get("swe_gym", {}).get("prepared_tasks", 0)) >= gym_limit
        pruner_ready = int(sources.get("swe_pruner_original", {}).get("validated_rows", 0)) >= pruner_limit
        smith_rows = list(read_jsonl(smith_tasks_path))
        gym_rows = list(read_jsonl(gym_tasks_path))
        smith_unique_ready = len({row.get("task_id") for row in smith_rows}) >= smith_limit
        gym_unique_ready = len({row.get("task_id") for row in gym_rows}) >= gym_limit
        task_paths = [Path(row["repo_path"]) for row in [*smith_rows, *gym_rows]]
        if (
            smith_ready
            and gym_ready
            and pruner_ready
            and smith_unique_ready
            and gym_unique_ready
            and all(path.exists() for path in task_paths)
        ):
            sample_stats = sample_swepruner_rows(pruner_path, pruner_sample_path, pruner_limit, seed)
            existing["sources"]["swe_pruner_original"].update(sample_stats)
            existing["sources"]["swe_pruner_original"]["first_phase_path"] = str(pruner_sample_path)
            write_json(report_path, existing)
            existing["reused"] = True
            return existing
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print("[prepare] downloading/verifying SWE-Pruner official JSONL", flush=True)
    _download(SWE_PRUNER_URL, pruner_path)
    valid_pruner = 0
    with pruner_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if {"query", "code", "kept_frags", "score"}.issubset(row):
                valid_pruner += 1
            if valid_pruner >= pruner_limit:
                break
    if valid_pruner < pruner_limit:
        raise RuntimeError(f"SWE-Pruner source contains only {valid_pruner} valid rows before requested limit")
    sample_stats = sample_swepruner_rows(pruner_path, pruner_sample_path, pruner_limit, seed)
    print(f"[prepare] SWE-Pruner validated first {valid_pruner} rows", flush=True)
    smith_candidates_path = raw / "swe_smith_candidates.jsonl"
    gym_candidates_path = raw / "swe_gym_candidates.jsonl"
    candidate_metadata_path = raw / "candidate_selection.json"
    candidate_metadata = {
        "seed": seed,
        "max_repos_per_source": max_repos_per_source,
        "smith_limit": smith_limit,
        "gym_limit": gym_limit,
        "smith_dataset": SWE_SMITH_DATASET,
        "gym_dataset": SWE_GYM_DATASET,
    }
    cached_metadata = (
        json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
        if candidate_metadata_path.exists()
        else None
    )
    if cached_metadata == candidate_metadata and smith_candidates_path.exists() and gym_candidates_path.exists():
        smith_candidates = list(read_jsonl(smith_candidates_path))
        gym_candidates = list(read_jsonl(gym_candidates_path))
        if len(smith_candidates) < smith_limit or len(gym_candidates) < gym_limit:
            cached_metadata = None
        else:
            print("[prepare] reusing deterministic cached dataset candidates", flush=True)
    if cached_metadata != candidate_metadata:
        print("[prepare] selecting diverse SWE-smith rows from official dataset", flush=True)
        smith_candidates = collect_diverse_rows(SWE_SMITH_DATASET, smith_limit, max_repos_per_source, seed)
        print("[prepare] selecting diverse SWE-Gym rows from official dataset", flush=True)
        gym_candidates = collect_diverse_rows(SWE_GYM_DATASET, gym_limit, max_repos_per_source, seed)
        write_jsonl(smith_candidates_path, smith_candidates)
        write_jsonl(gym_candidates_path, gym_candidates)
        write_json(candidate_metadata_path, candidate_metadata)
    manager = GitWorktreeManager(root)
    smith_tasks, smith_failures = _prepare_smith(root, manager, smith_candidates, smith_limit)
    gym_tasks, gym_failures = _prepare_gym(root, manager, gym_candidates, gym_limit)
    write_jsonl(root / "swe_smith" / "tasks.jsonl", smith_tasks)
    write_jsonl(root / "swe_gym" / "tasks.jsonl", gym_tasks)
    report = {
        "seed": seed,
        "sources": {
            "swe_pruner_original": {
                "url": SWE_PRUNER_URL,
                "path": str(pruner_path),
                "sha256": _sha256(pruner_path),
                "validated_rows": valid_pruner,
                "requested_rows": pruner_limit,
                "first_phase_path": str(pruner_sample_path),
                **sample_stats,
            },
            "swe_smith": {
                "dataset": SWE_SMITH_DATASET,
                "prepared_tasks": len(smith_tasks),
                "requested_tasks": smith_limit,
                "repo_count": len({task["repo_name"] for task in smith_tasks}),
                "failures": smith_failures,
            },
            "swe_gym": {
                "dataset": SWE_GYM_DATASET,
                "prepared_tasks": len(gym_tasks),
                "requested_tasks": gym_limit,
                "repo_count": len({task["repo_name"] for task in gym_tasks}),
                "failures": gym_failures,
            },
        },
    }
    write_json(report_path, report)
    return report
