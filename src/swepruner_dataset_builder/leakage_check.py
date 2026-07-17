from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .patch_parser import parse_unified_diff


INSTRUCTION_RE = re.compile(
    r"\b(change|replace|edit|modify)\b.{0,40}\b(to|with)\b|把.{0,30}(改成|替换为)|第\s*\d+\s*行",
    re.IGNORECASE,
)


def normalize_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def fragment_hash(text: str) -> str:
    return hashlib.sha256(normalize_fragment(text).encode("utf-8")).hexdigest()


def added_line_hashes(patch: str) -> list[str]:
    parsed = parse_unified_diff(patch)
    return sorted({fragment_hash(line) for line in parsed.added_lines if len(normalize_fragment(line)) >= 12})


@dataclass(slots=True)
class LeakageResult:
    safe: bool
    risk: str
    reasons: list[str]
    added_hashes: list[str]


def check_leakage(issue_text: str, code: str, patch: str) -> LeakageResult:
    parsed = parse_unified_diff(patch)
    reasons: list[str] = []
    issue_normalized = normalize_fragment(issue_text)
    code_lines = {normalize_fragment(line) for line in code.splitlines()}
    for added in parsed.added_lines:
        fragment = normalize_fragment(added)
        if len(fragment) < 12:
            continue
        if fragment in issue_normalized:
            reasons.append("issue_contains_patch_addition")
        if fragment in code_lines:
            reasons.append("code_contains_patch_addition")
    if INSTRUCTION_RE.search(issue_text):
        reasons.append("issue_contains_explicit_edit_instruction")
    risk = "high" if any(reason.startswith("code_") or reason.startswith("issue_contains_patch") for reason in reasons) else ("medium" if reasons else "low")
    return LeakageResult(not reasons, risk, sorted(set(reasons)), added_line_hashes(patch))


def hashes_present(text: str, forbidden_hashes: list[str]) -> bool:
    forbidden = set(forbidden_hashes)
    return any(fragment_hash(line) in forbidden for line in text.splitlines() if len(normalize_fragment(line)) >= 12)

