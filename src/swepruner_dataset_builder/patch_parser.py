from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import PatchEvidence


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def normalize_patch_path(path: str) -> str:
    value = path.strip().split("\t", 1)[0]
    if value in {"/dev/null", ""}:
        return ""
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return value.lstrip("./")


@dataclass(slots=True)
class ParsedPatch:
    evidence: list[PatchEvidence] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)
    files: set[str] = field(default_factory=set)


def parse_unified_diff(text: str, provenance: str = "patch_old_line") -> ParsedPatch:
    result = ParsedPatch()
    old_path = ""
    new_path = ""
    current_path = ""
    old_line = 0
    new_line = 0
    hunk_header = ""
    in_hunk = False
    pending_addition: tuple[str, int, str] | None = None
    change_has_deletion = False

    def flush_change() -> None:
        nonlocal pending_addition, change_has_deletion
        if pending_addition is not None and not change_has_deletion:
            path, line, header = pending_addition
            result.evidence.append(
                PatchEvidence(
                    path,
                    line,
                    "addition_context",
                    "",
                    header,
                    "patch_addition_context",
                )
            )
        pending_addition = None
        change_has_deletion = False

    for raw in text.splitlines():
        if raw.startswith("--- "):
            flush_change()
            old_path = normalize_patch_path(raw[4:])
            in_hunk = False
            continue
        if raw.startswith("+++ "):
            new_path = normalize_patch_path(raw[4:])
            current_path = old_path or new_path
            if current_path:
                result.files.add(current_path)
            continue
        match = HUNK_RE.match(raw)
        if match:
            flush_change()
            old_line = int(match.group(1))
            new_line = int(match.group(3))
            hunk_header = raw
            current_path = old_path or new_path
            in_hunk = True
            continue
        if not in_hunk or not current_path or not raw:
            continue
        marker, content = raw[0], raw[1:]
        if marker == " ":
            flush_change()
            old_line += 1
            new_line += 1
        elif marker == "-" and not raw.startswith("---"):
            change_has_deletion = True
            result.evidence.append(
                PatchEvidence(current_path, old_line, "deleted", content, hunk_header, provenance)
            )
            old_line += 1
        elif marker == "+" and not raw.startswith("+++"):
            result.added_lines.append(content)
            if pending_addition is None:
                pending_addition = (current_path, max(1, old_line), hunk_header)
            new_line += 1
        elif marker == "\\":
            continue
        else:
            flush_change()
            in_hunk = False
    flush_change()
    return result
