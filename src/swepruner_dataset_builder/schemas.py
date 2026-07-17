from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Role(StrEnum):
    CORE = "CORE"
    SUPPORT = "SUPPORT"
    DROP = "DROP"


class RelationType(StrEnum):
    PATCH = "PATCH"
    CALL = "CALL"
    CALLED_BY = "CALLED_BY"
    DEF_USE = "DEF_USE"
    CONTROL = "CONTROL"
    TYPE = "TYPE"
    IMPORT = "IMPORT"
    ATTRIBUTE = "ATTRIBUTE"
    INHERITANCE = "INHERITANCE"
    OVERRIDE = "OVERRIDE"
    EXCEPTION = "EXCEPTION"
    TRACEBACK = "TRACEBACK"
    DECORATOR = "DECORATOR"
    NONE = "NONE"


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    dataset_source: str
    repo_name: str
    repo_path: str
    base_commit: str = ""
    issue_text: str = ""
    patch: str = ""
    mutation_patch: str = ""
    traceback: str = ""
    failing_tests: list[str] = field(default_factory=list)
    test_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PatchEvidence:
    file_path: str
    old_line: int
    kind: str
    old_text: str = ""
    hunk_header: str = ""
    provenance: str = "patch_old_line"


@dataclass(slots=True)
class Block:
    block_id: str
    block_type: str
    relative_file_path: str
    start_line: int
    end_line: int
    parent_block_id: str | None
    enclosing_symbol_id: str
    source_text: str
    symbol: str
    defined_names: list[str] = field(default_factory=list)
    used_names: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)

    def to_dict(self, include_source: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_source:
            value.pop("source_text", None)
        return value


@dataclass(slots=True)
class Symbol:
    symbol_id: str
    name: str
    qualname: str
    kind: str
    relative_file_path: str
    start_line: int
    end_line: int
    parent_symbol_id: str | None
    block_id: str
    signature_block_id: str
    class_qualname: str = ""


@dataclass(slots=True)
class Relation:
    source_block_id: str
    target_block_id: str
    relation: str
    confidence: float
    provenance: list[str]
    graph_distance: int = 1


@dataclass(slots=True)
class ParsedFile:
    relative_path: str
    absolute_path: str
    source: str
    lines: list[str]
    blocks: list[Block]
    symbols: list[Symbol]
    syntax_error: str = ""


@dataclass(slots=True)
class BlockLabel:
    role: str = Role.DROP
    confidence: float = 0.70
    provenance: list[str] = field(default_factory=lambda: ["static_default_drop"])
    relation_type: str = RelationType.NONE

    def apply(self, role: str, confidence: float, provenance: str, relation_type: str) -> None:
        rank = {Role.DROP: 0, Role.SUPPORT: 1, Role.CORE: 2}
        current = Role(self.role)
        incoming = Role(role)
        if rank[incoming] > rank[current] or (incoming == current and confidence > self.confidence):
            self.role = incoming
            self.confidence = float(confidence)
            self.relation_type = relation_type
        if provenance not in self.provenance:
            if self.provenance == ["static_default_drop"] and incoming != Role.DROP:
                self.provenance.clear()
            self.provenance.append(provenance)


@dataclass(slots=True)
class HardNegative:
    block_id: str
    negative_type: str
    lexical_similarity: float
    graph_distance: int
    same_file: bool
    same_class: bool
    covered_by_test: bool = False
    api_reviewed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BuildFailure:
    task_id: str
    dataset_source: str
    repo_path: str
    stage: str
    error_type: str
    error_message: str
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_name_from_path(path: str) -> str:
    return Path(path).resolve().name

